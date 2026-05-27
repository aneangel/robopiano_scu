# GPU IK Port Plan: `solve_press_pose` to JAX / MJX

Owner: planning team. Target hardware: RTX 5080 (16 GB), JAX 0.6.2 + CUDA 12.9.
Status today: IK is 96.2 % of wall (`benchmark/results/bottleneck_report.md:10-15`).
Mean F1 = 0.580 on the 177-song test split; mean per-song wall on Mac = 66 s for
30 s of music (1.93 s wall / s music).

## 1. Anatomy of `solve_press_pose` (what we port)

The function lives at `Bagatelle/src/bagatelle/kinematics.py:651-1012`. Five
nested closures dominate; only some are MuJoCo-bound.

### 1.1 `residual(values) -> np.ndarray`
`kinematics.py:742-772`. Hot path. Calls
`self.fingertip_positions_for_qpos(q)` (line 744), which is
`_set_qpos(q); current_fingertips()` — one `physics.forward()` plus a site
xpos read (`kinematics.py:295-319`). After that the residual is **pure numpy**:
weighted (a) fingertip error, (b) smoothness `q - previous`, (c) neutral
`q - neutral`, (d) optional inactive Z-clearance, (e) optional avoid-mispresses
proximity term, (f) optional wrong-key XY avoidance. MuJoCo dependency = the
one FK call.

### 1.2 `jacobian(values) -> np.ndarray`
`kinematics.py:847-864`. Calls `self.fingertip_jacobian(finger_indices)`
(line 852), which loops `mujoco.mj_jacSite` over the active fingertip site IDs
(`kinematics.py:321-347`). Returns shape `(n_active, 3, 46)`. Optional
`_avoid_mispresses_jacobian` (lines 802-845) chains in the inactive-fingertip
Jacobian piece. MuJoCo dependency = one `mj_jacSite` call per active +
inactive fingertip site (10 calls max).

### 1.3 `scipy.optimize.least_squares` call
`kinematics.py:870-893` (`solve_from_seed`). TRF method with analytical
Jacobian when `ik_analytical_jacobian=True` (`config.py:53`). Bounds from
`joint_lower / joint_upper`. `max_nfev = ik_max_nfev` (default 120,
`config.py:31`). Pure scipy; no MuJoCo dependency beyond what `residual` and
`jacobian` import.

### 1.4 Multistart loop
`kinematics.py:941-996`. First solve from `x0` (warm-start cache or previous
pose, line 712). If validation requires contact metrics, or first solve fails
and `ik_multistart_on_failure=True`, also try `extra_seeds` from
`_ranked_press_pose_initial_seeds` plus an optional Rhapsody RP1M seed
(line 963). At most `1 + ik_multistart_seed_count` solves
(default 1+2 = 3 LM solves per waypoint), but the contact-perfect early-exit
at line 953 short-circuits when the first solve already lands every key.

### 1.5 Portability summary
| Block | Pure math | MuJoCo dep |
|---|---|---|
| residual body (lines 742-772) | yes | 1 `physics.forward()` for fingertips |
| jacobian body (lines 847-864) | identity blocks pure | `n_fingers` `mj_jacSite` |
| `_avoid_mispresses_jacobian` | chain-rule math pure | reuses `fingertip_jacobian` |
| LM driver (lines 870-893) | pure scipy | none |
| multistart (lines 941-996) | pure python | calls residual/jac transitively |

**One MuJoCo touchpoint matters: `fingertip_positions_for_qpos` (FK) and
`fingertip_jacobian` (site Jacobian). Both are equivalent to `mjx.kinematics`
+ `mjx.kinematics_jacobian` — see §3.**

## 2. Three porting options

### Option A: JIT-compiled scalar IK
JAX-port residual + Jacobian; run `jaxopt.LevenbergMarquardt` per waypoint
serially. One JIT compile up-front (30-120 s per
`benchmark/design/gpu_rollout_setup.md:222`), then ~2-5 ms per LM iter on
GPU vs ~25 ms on Mac CPU. **3-10× planning speedup**. Touches ~2 files.
Risk: jaxopt LM is less battle-tested than scipy TRF and doesn't natively
support bounds — must encode as soft penalty or use `ProjectedGradient`.

### Option B: vmap'd batched IK
JIT the full residual+jac+LM pipeline, then `jax.vmap` it. Two variants:
(b1) **waypoint-batched** across all 150-300 waypoints — blocked by the
smoothness dependency on the previous pose, so requires solving in chunks
(seed each chunk with the CPU previous pose, fix-up pass). (b2)
**seed-batched** across `assignment_top_k * ik_multistart_seed_count`
seeds per waypoint (4-12 wide) — solves each waypoint serially but
evaluates all seeds in parallel; replaces the multistart loop at
`kinematics.py:983-996`. Expected: **30-100×** for (b2), **100-300×** for
(b1). Touches ~4-5 files. Risk: batching reshapes multistart selection.

### Option C: Hybrid (scipy outer, JAX inner)
Keep `scipy.least_squares` on CPU; replace `residual` and `jacobian` with
JIT'd JAX functions. Each LM iter incurs a host↔device copy
(46 float32 = 184 B), but JIT-dispatch overhead dominates at this
granularity. Expected: **5-15×**. Touches ~1-2 files. Lowest risk.

### Recommendation
**Option C is the fastest first win** (1 day, low risk). **Option B
seed-batching is the right second step** to unlock the F1 knobs in §6.
Skip Option A — same per-iter dispatch cost as C, less familiar driver.

## 3. The fingertip Jacobian on MJX

`mujoco-mjx` (Google DeepMind, not installed locally per `pip show` returning
"Package(s) not found" in the Mac env) does **not** expose a one-shot
`mj_jacSite` equivalent in MJX 0.6.x. The standard idiom in the MJX
community is to compute the site Jacobian via `jax.jacrev` on the FK
function:

```python
def fingertip_xyz_for_q(q, mx, data_init):
    data = data_init.replace(qpos=data_init.qpos.at[hand_qpos_idx].set(q))
    data = mjx.kinematics(mx, data)   # MJX rigid-body FK
    return data.site_xpos[fingertip_site_ids]   # (10, 3)

fingertip_jac_fn = jax.jit(jax.jacrev(fingertip_xyz_for_q, argnums=0))
```

This produces `(n_fingertips, 3, n_hand_dofs=46)` — the same tensor as
`fingertip_jacobian` at `kinematics.py:321-347`. `jacrev` is the right
choice here because the input dim (46) > output dim (30 = 10*3), so
reverse-mode is ~1.5× the cost of one forward pass. Empirically this should
match or beat the CPU `mj_jacSite` loop after the first JIT.

`mjx.kinematics(mx, data)` runs only the position-stage FK (no constraint
solve, no actuator forces) — exactly what we need. The piano keys are not
in our `q`-dependent subset; their qpos stays at the default value.

Setup steps:
1. `mjx.put_model(self._mj_model)` already done in `tin/mjx_env.py:124`.
2. `mjx.make_data(mx).replace(qpos=...)` already done at `mjx_env.py:130`.
3. The hand-joint qpos indices we need are `self._joint_qpos_indices`
   (`kinematics.py:179-199`).
4. Site IDs are `self._fingertip_site_ids` (`kinematics.py:181-205`).

**Local-Mac caveat:** `mujoco.mjx` is **not** importable in the Mac env
(verified above). All JAX-IK development happens on the RTX 5080 box. The
plan below ensures the port is a no-op on Mac when `--use-jax-ik` is off.

## 4. What `tin/mjx_env.py` already gives us

`tin/mjx_env.py:80-228` (`MJXBatchedEnv`) already does the hard MJCF→MJX
work:

- **Model loading**: `mjx.put_model(self._mj_model)` at line 124.
- **Data init**: `mjx.make_data(mx).replace(qpos=...)` at line 130.
- **Hand-joint slice resolution**: `_find_hand_slices` at lines 181-201,
  resolving `rh_*` / `lh_*` qpos addresses identically to
  `kinematics.py:179-199`.
- **JIT pipeline proof**: `self._jit_step = jax.jit(jax.vmap(...))` at
  line 136 — proves JIT+vmap works on this exact model.

What it does **not** give us: no site-position read on `data`, no FK-only
path (only `mjx.step`), no Jacobian helper. **~40 % of MJX scaffolding
reusable; FK + site-Jacobian path is new code.**

## 5. Validation harness (50-LOC test)

`benchmark/validate_jax_ik.py` (new). For one short song from the
mix subset:

```python
# 1. Plan with CPU IK
cpu_traj = run_plan(song_path, extra_args=["--ik-analytical-jacobian"])

# 2. Plan with JAX IK
gpu_traj = run_plan(song_path, extra_args=["--use-jax-ik"])

# 3. Compare per-waypoint qpos
delta = np.abs(cpu_traj.qpos - gpu_traj.qpos)
assert delta.max() < 1e-2, f"max qpos drift {delta.max():.4f}"
assert delta.mean() < 1e-3, f"mean qpos drift {delta.mean():.4f}"

# 4. Rollout F1 parity
f1_cpu = rollout_f1(cpu_traj, song_path)
f1_gpu = rollout_f1(gpu_traj, song_path)
assert abs(f1_cpu - f1_gpu) < 0.02, f"F1 drift {f1_cpu:.3f} vs {f1_gpu:.3f}"
```

Tolerances are loose because the LM optimum is a basin, not a point:
multiple `q` values produce indistinguishable key contacts. F1 ±0.02 is
the run-to-run tolerance from `benchmark/results/AB_COMPARISON_FINAL.md`.
Run on 5 short songs from `benchmark/results/eval_runs/`: ~10 min/cycle.

## 6. Expected F1 lift from a GPU IK budget unlock

Today's config (`benchmark/run_full_evaluation.py:53-67`):
- `--assignment-top-k 2` → 2 candidate assignments per waypoint
- `--ik-max-nfev 60` → ≤ 60 LM iters per solve
- `--ik-multistart-on-failure` default (`config.py:39` = True)
- `--ik-multistart-seed-count 2` (`config.py:40`) → 1+2 = 3 seeds when retry triggers

With a 30-100× GPU speedup (Option B seed-batching), we can afford:
- `--ik-max-nfev 240+` (4× budget) — closes the "96.7 % of waypoints hit
  the nfev cap" failure mode at `benchmark/design/warmstart_refinements.md:158`.
- `--assignment-top-k 4+` (2× candidates) — the assignment top-K sweep at
  `benchmark/design/ab_topk_tuned.md` saw +0.01 to +0.04 F1 going 1→2 and
  diminishing but positive returns 2→4.
- `--ik-multistart-on-failure` with `ik_multistart_seed_count=6` — measured
  +0.03 on synthetic chord-heavy songs in
  `benchmark/results/ab_all_levers.md`.

Math: each independent lever has measured F1 lift in {0.01, 0.02, 0.04}.
If they were independent and additive, +0.07. If they overlap (the typical
case for IK quality knobs that target the same chord-density failure mode),
**realistic combined lift = +0.03 to +0.07** on the test split → mean F1
ranges 0.61-0.65 → clears the 0.60 floor with margin of 0.01-0.05.

This is the headline lift number that justifies the port.

## 7. Concrete implementation plan

Steps in order. File paths absolute. "Risk" is technical risk; "dep" is
what blocks it.

**Step 1: Spike-test `mjx.kinematics` + site read on RTX 5080 box.**
Write a 30-LOC scratch script that loads the Bagatelle MJCF via
`tin/mjx_env.py:107-115`, calls `mjx.kinematics(mx, data)`, reads
`data.site_xpos[fingertip_site_ids]`, and compares to CPU
`physics.bind(fingertip_sites).xpos` at 10 random qpos values. Pass criterion:
max diff < 1e-5. **Time: 2 h. Risk: low. Dep: GPU box access.**

**Step 2: Create `Bagatelle/src/bagatelle/kinematics_jax.py` (new).**
Skeleton:
- `class JaxIKKernel`: holds `mx`, `data_init`, `hand_qpos_idx`,
  `fingertip_site_ids`, JIT'd `_residual_fn(q, target_positions, ...)`,
  JIT'd `_jacobian_fn` via `jax.jacrev`. Mirrors the CPU `residual` /
  `jacobian` closures at `kinematics.py:742-864` term-by-term. Only the
  three always-on terms (a)(b)(c) in v1; the optional terms (d)(e)(f) gate
  back to scipy.
- `solve_press_pose_jax(seed, finger_indices, target_positions,
  previous, neutral, cfg)`: wraps `scipy.least_squares` (Option C) with
  `residual=jit_residual_wrapper, jac=jit_jacobian_wrapper`. The wrappers
  do `np.asarray(fn(jnp.asarray(q)))` per call.
- `JaxIKKernel.is_available()`: returns False on Mac (no MJX); True on the
  Ubuntu GPU box.
- **Time: 1 day. Risk: medium (first JAX code in repo; jaxopt vs scipy
  driver choice). Dep: Step 1.**

**Step 3: Plumb `--use-jax-ik` through to `kinematics.py`.**
- Add `use_jax_ik: bool = False` to `BagatelleConfig` (`config.py:53`).
- In `kinematics.py:651-660` (top of `solve_press_pose`), if
  `cfg.use_jax_ik` and the kernel is available and only terms (a)(b)(c)
  are active, route to `solve_press_pose_jax`. Else fall back to the
  existing CPU path. This preserves bit-exact behavior when the flag is off.
- **Time: 2 h. Risk: low. Dep: Step 2.**

**Step 4: Plumb `--use-jax-ik` through `plan_trajectory.py` and
`Impromptu/src/impromptu/config.py:109`.**
- One-line additions to `Bagatelle/scripts/plan_trajectory.py:59` (next to
  `--ik-max-nfev`) and `Impromptu/src/impromptu/config.py` /
  `Impromptu/src/impromptu/planner.py:58`.
- **Time: 1 h. Risk: low. Dep: Step 3.**

**Step 5: Write `benchmark/validate_jax_ik.py` (per §5).**
- **Time: 3 h. Risk: low. Dep: Step 4.**

**Step 6: Run validation on 5 mix songs; iterate until parity.**
Common bugs: float32↔float64 at the wrapper boundary (scipy wants
float64); site-id ordering (model-order vs `left-then-right`; map at
kernel boundary). Bound enforcement stays in scipy via `bounds=`.
**Time: 1 day. Risk: medium-high (most port pain). Dep: Step 5.**

**Step 7: Benchmark wall time on the 25-song subset.**
- Run `benchmark/run_full_evaluation.py` with and without `--use-jax-ik`.
- Compare against `benchmark/results/FINAL_SPEED_QUALITY_REPORT.md`.
- **Time: 4 h compute + 1 h analysis. Risk: low. Dep: Step 6.**

**Step 8 (stretch): Move to Option B seed-batching.**
- `JaxIKKernel.solve_batched(seeds, finger_indices, target_positions, ...)`
  vmaps the LM over the seed axis. Requires a JAX-native LM implementation
  (jaxopt or hand-rolled) since scipy is not vmappable.
- Replaces the multistart loop at `kinematics.py:983-996` with a single
  batched call.
- **Time: 3 days. Risk: high (jaxopt LM correctness, bound handling).
  Dep: Step 7.**

**Total to working A/B (Steps 1-7): ~3 days. Stretch to batched (Step 8):
+3 days = 1 week.**

## 8. Pragmatic minimum: port only `fingertip_jacobian`

Best single-day win. Today `fingertip_jacobian` at `kinematics.py:321-347`
runs `mj_jacSite` ten times per waypoint in a Python loop. Replace with
one JIT'd `jax.jacrev(fingertip_xyz_for_q)` call producing all 10
`(3, 46)` Jacobians in one GPU launch.

CPU: `scipy.least_squares`, residual, multistart, config plumbing. GPU:
the Jacobian only. Code scope: add `_jax_jacobian_fn` member to
`BagatelleKinematics`, guarded by `cfg.use_jax_ik`; fall back to
`mj_jacSite` loop when off. One file touched. Expected: **2-5× planning,
F1 unchanged.** Validates MJX on the Bagatelle model with ~1 day of risk
before a 1-week full port.

## 9. The two paths

### Pragmatic 1-day path (`fingertip_jacobian` only)
- Step 1 (spike, 2 h) → mini Step 2 (~50 LOC `_jax_jacobian_fn`, 2 h) →
  inline into `kinematics.py:321-347` (1 h) → validate on 1 song
  (1 h) → benchmark wall on 5 songs (2 h).
- **Win: 2-5× planning, F1 unchanged. Risk: low. Cost: ~1 day.**

### Full 1-week path (Option C → Option B)
- Steps 1-7 above (3 days) → seed-batching (Step 8, 3 days) → unlock budget
  knobs for F1 (§6, ~1 day evaluation).
- **Win: 30-100× planning, +0.03-0.07 F1 lift expected. Risk: medium.
  Cost: 1 week.**

### Decision criteria
- **Pick pragmatic-1-day if:** the next deliverable is "make the
  benchmark report show a 2× speedup" or "validate MJX works on this
  model before investing more". This is the safe scout.
- **Pick full-1-week if:** the F1 floor of 0.60 is a blocker that must
  close in the next sprint, AND the GPU box is available for at least 4
  uninterrupted days. F1 lift is the only justification for the extra
  4-6 days; if F1 is already adequate, the wall-time savings alone do
  not justify the increment over the 1-day path.
- **Default order:** start with pragmatic-1-day. If the 2-5× speedup
  observed there matches the projection, commit to the full path next
  sprint. If the spike reveals JIT compile overhead is larger than
  projected (e.g. 5+ min per cold start on Blackwell SM 12.0), the
  decision flips to "deepen the CPU-only optimizations" (analytical Jac
  already shipped, keyset cache already shipped, warm-start already
  shipped — next CPU lever is qpos-vectorized FK across waypoints).
