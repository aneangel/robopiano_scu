# Rhapsody Warm-Start for Bagatelle scipy LM Solver

## 1. Summary

Viable on Mac CPU: the `RhapsodyIKSolver` (`Rhapsody/src/rhapsody/solver.py:37`) is a pure-PyTorch two-MLP model that already defaults to `device="cpu"` and runs on Apple Silicon without CUDA. The integration is already 90% scaffolded in `Bagatelle/src/bagatelle/kinematics.py` and `Impromptu/scripts/plan_trajectory.py` — the work is to (a) produce a usable checkpoint locally, (b) flip `rhapsody_ik_enabled=True` from the benchmark harness, and (c) tighten the existing fallback so a bad prediction never blocks the neutral/previous seed retry. Biggest risk is that no Rhapsody checkpoint exists on this Mac and the documented smoke command (`Rhapsody/README.md:29-42`) points at the WAVE `rp1m.zarr` dataset which is not present locally; we have to either copy a checkpoint or stand up a tiny RP1M slice on the Mac.

## 2. Rhapsody solver API

Class entrypoint, from `Rhapsody/src/rhapsody/solver.py:37-61`:

```
class RhapsodyIKSolver:
    def __init__(self, *, normalizer, policy, fk_model, device="cpu") -> None: ...
    @classmethod
    def from_checkpoint(cls, path, *, device="cpu") -> "RhapsodyIKSolver":
        _config, normalizer, policy, fk_model, _state = load_checkpoint(path, device=device)
        return cls(normalizer=..., policy=..., fk_model=..., device=device)
```

Single-shot inference, `Rhapsody/src/rhapsody/solver.py:63-129`:

```
def solve(self, target_fingertips, *, active_mask=None, previous_qpos=None,
          refinement_steps=0, refinement_lr=0.05) -> IKSolution:
    # target_fingertips: (10, 3) or (30,) float32, NaN-fill-tolerant
    # active_mask:       (10,)  float32, 1.0 = finger has a target
    # previous_qpos:     (46,)  float32 hand state
    ...
    qpos_norm = self.policy(target_norm, mask_t, prev_norm)
    # optional Adam refinement against the FK surrogate
    return IKSolution(qpos=..., predicted_fingertips=..., active_mask=...,
                      mean_error_m=..., max_error_m=..., refinement_steps=...)
```

Constants confirm shapes: `HAND_STATE_DIM = 46`, `NUM_FINGERS = 10`, `FINGERTIP_COORD_DIM = 3` (`Rhapsody/src/rhapsody/constants.py:1-4`).

Architecture is two MLPs only (`Rhapsody/src/rhapsody/models.py:26-114`): `ResidualIKPolicy` (Linear + 256/256 SiLU residual head, tanh-bounded action_scale=2.0) and `ForwardKinematicsSurrogate` (same shape, qpos→fingertips). No CUDA-specific ops, no custom kernels — safe for CPU.

`solve_batch`: referenced in `Maestroso/scripts/run_acceleration_sweep.py:279-286` and `Maestroso/README.md:34-39`, but **not implemented** on `RhapsodyIKSolver`. The sweep code already has a compat shim (`solve_rhapsody_batch_compat`, lines 270-309) that loops `solve()` when `hasattr(solver, "solve_batch")` is false. For our per-anchor warm-start use case we just call `solve()` one anchor at a time, so this is a non-issue.

## 3. Checkpoint plan

Disk scan: no Rhapsody `.pt` / `.pth` / `.ckpt` anywhere under `/Users/aangeles/robopiano/` (only Sonata diffusion/transformer checkpoints exist). The canonical smoke command from `Rhapsody/README.md:29-42` is:

```
python Rhapsody/scripts/train_rpik.py \
  --rp1m-root /WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr \
  --max-songs 1 --num-demos 1 --frame-stride 20 --max-pairs 256 \
  --previous-mode random --fk-epochs 2 --bc-epochs 2 --policy-epochs 2 \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/smoke
```

The script defaults already include `--device cpu` and `--batch-size 256` (`Rhapsody/scripts/train_rpik.py:40,48`), so the smoke command runs on Mac as-is once an RP1M slice is reachable. Plan:

1. Scp one RP1M zarr song (~tens of MB at `--num-demos 1 --frame-stride 20`) to e.g. `~/data/rp1m_smoke.zarr`.
2. Re-run the smoke command with `--rp1m-root ~/data/rp1m_smoke.zarr --output-dir benchmark/results/rhapsody_smoke`. Drop `--max-pairs` to 128 and `--policy-epochs` to 1 if we need to fit under ~10 min on M-series CPU; the smoke recipe is already the lightest documented config.
3. Wire `--rhapsody-ik-checkpoint benchmark/results/rhapsody_smoke/rhapsody_rpik.pt` into the benchmark planner invocation.

The smoke checkpoint is intentionally low quality. We only need it good enough that the predicted qpos beats neutral as an LM warm start; the residual gate at `Bagatelle/src/bagatelle/kinematics.py:571-573` already rejects worse-than-previous seeds, so a poor smoke model degrades to the current baseline rather than corrupting outputs.

## 4. Bagatelle seed strategy

Existing seed ranker, `Bagatelle/src/bagatelle/kinematics.py:398-471`, currently builds a finite list of `(previous, neutral)` rows crossed with a forearm-tx grid sweep (`ik_multistart_forearm_tx_grid: int = 5`, `ik_multistart_seed_count: int = 2` per `Bagatelle/src/bagatelle/config.py:40-41`), scores each by fingertip distance under FK, and returns the top `seed_count`. It does not accept an external warm-start hint.

`solve_press_pose` (`kinematics.py:585`) currently:

1. Solves once from `x0 = previous` at line 744.
2. Calls `self._rhapsody_press_pose_seed(...)` at line 746 (already exists, `kinematics.py:546-583`).
3. If `rhapsody_seed is not None` or the first solve failed or static-contact validation is on, it builds `extra_seeds = [rhapsody_seed, *_ranked_press_pose_initial_seeds(...)]` and solves from each (line 752-775).

So the Rhapsody seed is **already injected** and is **already first** in the extra-seed list — but only after the previous-pose solve has burned its ~710 nfev. To capture the ~9x nfev win the BENCHMARK_REPORT projects, the warm start must run **before** the first solve, not after.

## 5. Patch sketch

File: `Bagatelle/src/bagatelle/kinematics.py`, function `solve_press_pose` around lines 740-755.

Diff hint (about 25 lines):

```
@@ kinematics.py around line 743
-        try:
-            result = with_static_contact_metrics(solve_from_seed(x0, seed_index=0))
-            best = result
-            rhapsody_seed = self._rhapsody_press_pose_seed(assignments, previous, cfg)
-            try_extra_seeds = bool(validate_static_contacts) or (
-                not bool(result.success) and bool(getattr(cfg, "ik_multistart_on_failure", True))
-            ) or rhapsody_seed is not None
+        try:
+            # Warm-start with Rhapsody if enabled; falls back to previous-pose
+            # if the prediction fails the residual gate inside _rhapsody_press_pose_seed.
+            rhapsody_seed = self._rhapsody_press_pose_seed(assignments, previous, cfg)
+            primary_seed = rhapsody_seed if rhapsody_seed is not None else x0
+            result = with_static_contact_metrics(
+                solve_from_seed(primary_seed, seed_index=0)
+            )
+            best = result
+            try_extra_seeds = bool(validate_static_contacts) or (
+                not bool(result.success) and bool(getattr(cfg, "ik_multistart_on_failure", True))
+            )
             if not try_extra_seeds:
                 return result
             extra_seeds: list[np.ndarray] = []
-            if rhapsody_seed is not None:
-                extra_seeds.append(self.clip_qpos(rhapsody_seed))
+            # Previous-pose becomes a fallback seed only if Rhapsody was the primary.
+            if rhapsody_seed is not None:
+                extra_seeds.append(self.clip_qpos(x0))
             extra_seeds.extend(self._ranked_press_pose_initial_seeds(...))
```

No CLI change is needed: `--enable-rhapsody-ik`, `--rhapsody-ik-checkpoint`, etc. already exist in `Impromptu/scripts/plan_trajectory.py:161-179` and propagate through `Impromptu/src/impromptu/config.py:121` and `Impromptu/src/impromptu/planner.py:72` into `BagatelleConfig.rhapsody_ik_enabled` (`Bagatelle/src/bagatelle/config.py:52`). The benchmark harness just needs to pass these to its `plan_trajectory.py` invocation.

Note: there is a parallel warm-start branch in `Impromptu/src/impromptu/ik_solver.py:150-162` that overwrites `initial` but never uses it (the local variable is shadowed by `target_positions` later in the same function). That branch should be removed or fixed in the same patch.

## 6. Mac-CPU performance plan

Expected per-call cost: a forward pass through ResidualIKPolicy is `Linear(76→46) + 256-256 SiLU MLP(76→46) + tanh`, total ~80k FLOPs; on Apple M-series CPU one call should be sub-millisecond. The current LM solve runs ~398 ms per anchor with 710 nfev (`benchmark/results/BENCHMARK_REPORT.md:53-58`), so even a generous 5 ms inference budget pays for itself the moment it cuts nfev below ~700.

No public latency number is published in `Rhapsody/README.md`, so add a 30-line standalone microbenchmark to confirm before trusting projections:

```
# benchmark/bench_rhapsody_solve.py — sketch
import time, numpy as np, torch
from rhapsody.solver import RhapsodyIKSolver
solver = RhapsodyIKSolver.from_checkpoint(
    "benchmark/results/rhapsody_smoke/rhapsody_rpik.pt", device="cpu")
target = np.random.uniform(-0.05, 0.05, (10, 3)).astype(np.float32)
mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32)
prev = np.zeros((46,), dtype=np.float32)
for _ in range(5): solver.solve(target, active_mask=mask, previous_qpos=prev)  # warm
N = 200
t0 = time.perf_counter()
for _ in range(N): solver.solve(target, active_mask=mask, previous_qpos=prev)
print(f"mean={(time.perf_counter()-t0)/N*1e3:.2f} ms")
```

Pass criterion: mean < 5 ms with `refinement_steps=0`. If it lands >50 ms, the warm-start is net negative and we drop the integration or switch to `refinement_steps=0` + cached normalization tensors.

## 7. Safety and fallback

The fallback structure already exists in `Bagatelle/src/bagatelle/kinematics.py:546-583`:

- `_rhapsody_press_pose_seed` returns `None` if (a) any FK call throws, (b) any seed-fingertip distance is non-finite, (c) `max(seed_errors) > rhapsody_ik_seed_max_active_error` (default 0.08 m), or (d) `rhapsody_ik_seed_require_previous_improvement=True` and the seed's mean active-finger error does not strictly improve on the previous-pose error.

This is already a 5x-of-success-threshold residual gate (0.08 m vs `residual_success_threshold = 0.02` m, `config.py:35`), exactly as requested. The patch above keeps this gate as the only entry condition for using Rhapsody as the primary seed; if `_rhapsody_press_pose_seed` returns `None`, we fall back to `x0 = previous`, which is the current baseline behavior.

Additional safety check to add: after the primary solve from the Rhapsody seed, if `result.max_residual > 5 * cfg.residual_success_threshold`, force the multistart path on regardless of `ik_multistart_on_failure`. This guards the case where the Rhapsody seed passes the input-side gate but LM still converges to a worse local minimum than `previous` would have.

## 8. Risks and open questions

1. **No local checkpoint.** Smoke training requires an RP1M zarr slice on the Mac. Quickest unblock: copy a single song's worth of RP1M into `~/data/rp1m_smoke.zarr` (the smoke recipe uses `--max-songs 1 --num-demos 1`). Falls back to: skip warm-start, do not flip the flag.
2. **Coordinate-frame mismatch.** Rhapsody was trained on RP1M fingertip coordinates with left/right swap and a y-offset (`Bagatelle/src/bagatelle/kinematics.py:526-544`, default `rhapsody_ik_y_offset=0.08289646`). The transform is already coded, but the constant must match the actual Rhapsody training frame. Worth printing seed errors before/after the transform in the smoke benchmark to confirm.
3. **Smoke checkpoint quality.** With `--fk-epochs 2 --bc-epochs 2 --policy-epochs 2`, the predicted poses may rarely pass the `rhapsody_ik_seed_max_active_error=0.08` gate. If gate-rejection rate exceeds, say, 50%, the warm-start contributes nothing on average. Mitigation: temporarily raise the gate to 0.15 m during smoke validation, then re-tighten once a real checkpoint exists.
4. **Dead branch in `Impromptu/src/impromptu/ik_solver.py:150-162`** computes `initial` from `kin.rhapsody_seed_for_fingertips` but never threads it into `least_squares`. Either remove it (we standardize on the Bagatelle-side warm start) or wire it through. Recommend removal — Bagatelle's `solve_press_pose` is the single canonical IK entrypoint.
5. **`solve_batch` not implemented.** Maestroso GPU paths assume it (`Maestroso/scripts/run_acceleration_sweep.py:279-286`). Out of scope for this Mac-CPU warm-start, but flag for the Maestroso authors so the compat shim stays in place.
