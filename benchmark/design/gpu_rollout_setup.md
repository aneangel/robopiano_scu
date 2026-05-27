# GPU-Accelerated Rollout for `rp1m_simulator/simulator.py`

Investigation of how to move the `simulate_rp1m_rollout(...)` per-substep
MuJoCo loop onto an RTX 5080 (Ubuntu 22.04, CUDA 12.x, driver 570.86) so the
planner+rollout eval lands sub-60 s per MAESTRO song.

## 1. Current rollout interface

Signature (`rp1m_simulator/simulator.py:857-861`):

```python
def simulate_rp1m_rollout(
    trajectory: RP1MTrajectory,
    config: RolloutConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
```

Default cadence (`simulator.py:64-65`): `dataset_timestep = 0.05` s,
`simulation_timestep = 0.005` s → `substeps = 10` per source step
(`simulator.py:883`) → MuJoCo runs at 200 Hz.

Inner loop (`simulator.py:988-1055`). Per source step, per substep, it does:

1. **Locate task/physics** via wrapped-env walk (`simulator.py:1017`,
   `_locate_task_physics_piano` at `simulator.py:394-402`). Python attr walk;
   not the cost driver but called every substep.
2. **Write `qpos`** for ~52 hand joints — `_set_hand_qpos` loops
   `physics.bind(joint).qpos = float(value)` over each joint
   (`simulator.py:414-421`). This is dm_control's slow per-joint binding path.
3. **Write `qvel`** the same way (`_set_hand_qvel` at
   `simulator.py:424-434`) when `set_hand_qvel=True`.
4. **`physics.forward()`** (`simulator.py:1021-1022`) — full MuJoCo FK + contact
   prepass on CPU.
5. **`env.step(control)`** (`simulator.py:1045`) — the dm_control wrapper that
   internally calls `mj_step` once per call, plus the reward/observation
   composition pipeline.
6. **Activation read** (`simulator.py:1048`,
   `_capture_piano_activation` at `simulator.py:526-532`) — pulls
   `task.piano.activation[:88]` via attribute walk every substep.

Hand-state mode (default) does the qpos override at **every** substep
(`simulator.py:1010-1023`); action mode skips that. Cost is dominated by
`env.step` plus the per-substep qpos/qvel writes.

## 2. Existing MJX integration

`tin/mjx_env.py` is the only MJX touch point in the repo.

- It wraps RoboPianist via `suite.load(...)` (`mjx_env.py:48-77`), exports the
  MJCF (`mjx_env.py:112-115`), prunes non-fingertip collisions
  (`mjx_env.py:159-167`), and `mjx.put_model(...)`s the result
  (`mjx_env.py:118-124`).
- It builds a **batched** vmapped step:
  `jax.jit(jax.vmap(lambda data, ctrl: mjx.step(self.mx, data.replace(ctrl=ctrl))))`
  (`mjx_env.py:136`).
- `step(...)` returns observations, rewards (its own F1 reward at
  `mjx_env.py:218-227`), and dones (`mjx_env.py:149-157`).

Consumers: only `tin/online_rl.py:200-209` and `tin/maestro_eval.py:354-356`
(RL training and an unrelated MAESTRO RL eval). **Nothing in
`rp1m_simulator/`, `benchmark/`, `Impromptu/`, or `Bagatelle/` references
`mjx`** (confirmed via grep). So the rollout path used by
`run_full_evaluation.py` is 100% CPU dm_control today.

Caveat: `MJXBatchedEnv` is **action-mode only** — it never overrides `qpos`
between steps, so it cannot serve hand-state rollouts that the current
planner pipeline uses (`config.mode = "hand_state"` rewrites qpos every
substep). Goals are also tracked via a precomputed sequence
(`mjx_env.py:169-179`), not the RP1M trajectory format.

## 3. MJX rollout options

### (a) Replace `env.step` with `mjx.step` in a JIT'd scan

Build a `jit(scan(...))` over the substep loop that takes
`(qpos_target_seq, ctrl_seq)` and emits per-substep piano activations. Reuse
`mjx_env.py`'s xml export + `mjx.put_model` plumbing; add a custom step that
overwrites `data.qpos` (hand slice) before `mjx.step` for hand-state mode.

- Effort: ~3–5 days. Need to (i) map dm_control joint handles to qpos indices
  once at startup (similar to `_find_hand_slices` at `mjx_env.py:181-201`),
  (ii) port `_apply_mujoco_options` / `_apply_post_reset_hand_anchor_y_offset`
  to MJX equivalents, (iii) extract piano activation indices from the model
  rather than via `task.piano.activation` Python attr (which only exists on
  dm_control-side).
- Speedup: 10–30× wall on a single song when the inner loop is a JIT'd `scan`.
  More if batched (multiple seeds/songs at once), but that doesn't help the
  per-song latency directly.

### (b) `mjx.put_data` once, step in MJX, read back at the end

Trivially expressible — basically (a) without unrolling into a scan. Each
Python-level call to `jit(mjx.step)` still works but you pay JAX dispatch
overhead per call (~50–200 µs). For 60 000 calls that is 3–12 s of dispatch
overhead alone, which is most of the budget.

- Effort: 1–2 days (smaller surface).
- Speedup: 3–5× realistic; bottleneck shifts from MuJoCo to JAX dispatch.

### (c) CPU process parallelism (status quo)

`benchmark/run_full_evaluation.py:1-19` already fans songs out via
`ProcessPoolExecutor` (`--parallelism N`). This improves **throughput** but
**not per-song wall**, which is what the goal explicitly asks for.

- Effort: 0 days.
- Speedup: 0× per-song wall; N× throughput on N cores.

Recommendation if pursued: **(a)**. (b) is a partial win and (c) doesn't
address the requirement.

## 4. JAX-CUDA install (Linux + Python 3.10, CUDA 12, driver 570.86)

Recommended (bundled CUDA via pip, no system CUDA required):

```bash
pip install --upgrade "jax[cuda12]"
```

This pulls `jaxlib` + `nvidia-cudnn-cu12` + `nvidia-cublas-cu12` etc. into the
env. Driver 570.86 supports CUDA 12.x runtimes.

Verify:

```bash
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
python -c "import mujoco.mjx as m; print(m.__version__)"
```

Recommended env vars (already partially set in `tin/mjx_env.py:8-10`):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab full VRAM up front
export XLA_PYTHON_CLIENT_ALLOCATOR=platform  # incremental alloc; safer w/ 16GB
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=0
# Optional perf: export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
```

**RTX 5080 caveat (Blackwell, SM 12.0):** as of mid-2025 the prebuilt
`jaxlib==0.4.x` wheels target CUDA 12.4 and ship PTX up to SM 9.0. On SM 12.0
JAX will JIT PTX → SASS at first run (slow startup, but functional). If
first-run compile is unacceptable, build `jaxlib` from source with
`--cuda_compute_capabilities=12.0` or use the
`nvidia-jax` NGC container.

## 5. Validation harness (A/B CPU vs GPU)

Single short song A/B:

1. Pick an existing benchmark song with a ready trajectory. The repo already
   uses `Impromptu/scripts/plan_trajectory.py` + `retest_impromptu_rp1m_simulator.py`
   end-to-end in `benchmark/run_full_evaluation.py:40-43`. Pre-plan one short
   MAESTRO test song (≤ 60 s) so we A/B only the rollout stage.
2. Run CPU rollout: `retest_impromptu_rp1m_simulator.py ...` → records
   `rollout.npz` (`source_played_piano`) and `summary.json`
   (`against_goals.key_f1`) per `benchmark/design/rollout_eval_plan.md:43-56`.
3. Run GPU rollout: same trajectory, MJX backend → same artifacts.
4. **Correctness check**: `np.allclose(cpu.source_played_piano,
   gpu.source_played_piano, atol=5e-3)` for each substep; and
   `abs(cpu_key_f1 - gpu_key_f1) < 0.005` from `summary.json`. The activation
   sign (key press vs not) at the `threshold=0.5` boundary is what matters for
   F1 — small numerical drift below the threshold is expected.
5. Re-use `benchmark/run_ab_compare.py` shape for the per-song row diff.

Expect non-bitwise-identical piano states (MJX uses different LCP iterations
on contacts than the CPU solver), but key-press F1 within ~0.01 once
`mujoco_iterations` and `mujoco_ls_iterations` are matched.

## 6. Per-song wall target math

Measured CPU rollout cost: **~0.4–0.6 s wall per second of music** (Mac CPU).
A 5-min MAESTRO song = 300 s music → **120–180 s rollout wall** on CPU today.

- 10× MJX speedup → 12–18 s rollout wall per 5-min song. **Hits the sub-60 s
  target with margin.**
- 30× MJX speedup → 4–6 s rollout wall.

Per-substep budget on GPU: 5-min song × 10 substeps × 200 Hz × 60 = 60 000
sim steps. At a typical MJX inner-step cost of ~50 µs/step on a single env
post-JIT, that is ~3 s of pure MuJoCo. Realistic wall after Python overhead
will land 5–15 s, consistent with the 10–30× estimate.

## 7. Planning dominates — GPU rollout alone is ~13% of total

From `benchmark/results/bottleneck_report.md:7-23`:

- Total wall: **131.7 s** on the profiled run.
- Bagatelle IK (`scipy.optimize.least_squares` in
  `Bagatelle/src/bagatelle/kinematics.py:585-673`): **96.2 %** of wall.
- MuJoCo FK during planning (`physics.forward` in
  `kinematics.py:264-283`): **67.6 %** (overlaps with IK; the IK residual
  drives forward calls).
- Rollout itself is **not** in the top-10; from the bottleneck report it is
  bundled with "Other" and is well under 5 %.

**If rollout is ~13 % of end-to-end wall, a 10× rollout speedup saves ~11.7 %
of total wall.** A 30× speedup saves ~12.6 %. Either way **end-to-end gain is
capped at ~13 %** unless planning is also accelerated.

GPU planning (porting `solve_press_pose` to a JAX Levenberg-Marquardt solver
over a batched MJX FK) is the only path to a real >2× end-to-end speedup. The
existing infrastructure in `tin/mjx_env.py` already proves the MJCF→MJX
pipeline works for this exact model — a JAX FK kernel would reuse it. The IK
residual at `Bagatelle/src/bagatelle/kinematics.py:641` is a closed-form
function of `fingertip_positions_for_qpos` (`kinematics.py:283`), so the port
is mechanical but moderate effort (~2 weeks): vmap LM over candidate seeds,
keep the rest of the planner unchanged.

## Concrete 5-step install plan (RTX 5080)

1. `conda activate robopiano` on the Ubuntu box.
2. `pip install --upgrade "jax[cuda12]"` then verify
   `python -c "import jax; print(jax.devices())"` shows `CudaDevice(id=0)`.
3. Export the four env vars from §4 (`XLA_PYTHON_CLIENT_PREALLOCATE=false`,
   `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, `TF_GPU_ALLOCATOR=cuda_malloc_async`,
   `CUDA_VISIBLE_DEVICES=0`) — add to `benchmark/activate_env.sh`.
4. Smoke test MJX on the existing wrapper:
   `python -c "from tin.mjx_env import MJXBatchedEnv; ..."` with a 1-env
   instance and 10 steps. First run will JIT-compile (expect 30–120 s on SM
   12.0); subsequent runs are fast.
5. On the first successful step, snapshot `jax.devices()` and the JIT-compile
   wall to `benchmark/results/gpu_smoke.json` for future regression tracking.

## Go / No-go

**No-go on GPU rollout alone.** Rollout is ≤ 13 % of end-to-end wall per
`bottleneck_report.md`; even an optimistic 30× rollout speedup only buys ~13 %.
That is not worth ~1 week of MJX-port effort plus debugging hand-state-mode
qpos overrides, MJX contact solver mismatches, and Blackwell first-run JIT
quirks.

**Go on GPU planning.** The 96 % IK hotspot
(`Bagatelle/src/bagatelle/kinematics.py:585-673`) is the only thing whose
acceleration moves the end-to-end needle. The `tin/mjx_env.py` infrastructure
already gives us a model export path; reusing it for a vmapped LM solver over
candidate seeds is the right next investigation. Treat GPU rollout as a
secondary objective to bundle **after** GPU planning, since the MJX
infrastructure built for planning trivially covers rollout too.
