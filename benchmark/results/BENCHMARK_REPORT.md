# Maestroso/Impromptu IK Bottleneck — Benchmark Report

**Branch:** `benchmark/ik-bottleneck-investigation`
**Date:** 2026-05-26
**Platform:** MacBook Pro, Apple Silicon (arm64), macOS 26.2, Python 3.10.20, miniforge3
**Profiler:** cProfile (deterministic, full coverage; py-spy skipped due to macOS codesigning requirement)

## TL;DR

**Two-thirds of planning wall time is MuJoCo forward kinematics, and another fifth is Python overhead from setting joint values one at a time.** The IK solver itself (LM optimizer machinery, not counting the FK calls inside it) is only 3% of time. Algorithmic levers — analytical Jacobian, vectorized qpos writes, warm-starting across anchors — should plausibly cut planning time by 3–6x before any GPU work is needed.

**A secondary, independent problem:** on the synthetic test MIDI used here, only **5 of 150 waypoints (3.3%)** met the 0.02 residual success threshold, with mean residual 0.117. The IK is hitting `max_nfev=80` cap and exiting without converging. This is a tuning/quality issue separate from speed, but it means *F1 numbers on real songs may already be limited by IK convergence quality, not just runtime.*

## Smoke test result

| Test | Duration | Waypoints | IK anchors | Wall time | Wall / music |
|---|---|---|---|---|---|
| Twinkle (synthetic, 5s active window) | 8 sec MIDI | 5 | 10 | 5.5 sec | 1.1 s/s |
| Dense bimanual (synthetic, 15s active window) | 16.8 sec MIDI | 150 | 306 | 117 sec uninstrumented / 132 sec under cProfile | 8–9 s/s |

The dense run is the basis for all profiling below. It exercises both hands across pitches 48–72 with eighth-note RH + half-note LH chords for 16.8 seconds, anchor_stride=1, max_nfev=80, joint_space_straighten trajectory mode, contact validation enabled.

## Profile breakdown — where the 131.7 seconds go

Top of the cumulative-time tree, deepest hot function highlighted:

```
plan_trajectory.main                       130.7 s    100%
  Impromptu plan_target_keys (orchestrator) 130.6 s     99%
    Bagatelle plan_target_keys              126.6 s     96%
      solve_press_pose  (150 calls)         126.5 s     96%
        solve_from_seed (306 calls)         121.9 s     93%
          scipy.optimize.least_squares      121.7 s     92%
            residual fn (kinematics.py:641) 117.6 s     89%
              _set_qpos (kinematics.py:264) 116.2 s     88%
                dm_control physics.forward   90.8 s     69%   ← MuJoCo FK numerical work
```

Decomposing by **non-overlapping stages**:

| Stage | Time | % of wall |
|---|---|---|
| MuJoCo `physics.forward()` (actual FK numerical work) | 90.8 s | **69.0%** |
| `_set_qpos` Python overhead (per-joint `bind` + `setattr`, no FK) | 25.4 s | **19.3%** |
| LM optimizer machinery (TRF, finite-diff coordination, lin alg) | 4.1 s | 3.1% |
| Everything outside `least_squares` (setup, targets, interp, write) | 10.0 s | 7.6% |
| **Total** | **130.3 s** | **99.0%** |

Per-call costs:

| Call site | Calls | Time/call |
|---|---|---|
| `least_squares` (one per anchor IK) | 306 | 397.7 ms |
| Residual evaluations | 217,202 | 0.542 ms |
| `physics.forward` (MuJoCo FK) | 223,317 | 0.407 ms |
| `physics.bind` (dm_control joint accessor) | 10,481,788 | ~0.36 µs |

**Residual evaluations per IK call: 710 average.** With `max_nfev=80`, that's only possible because `scipy.optimize.least_squares` uses **finite-difference Jacobian estimation** — each LM iteration calls the residual N_joints (~46) extra times for forward differencing. So ~9 LM iterations × ~46 fd calls + ~80 residual calls ≈ 710. **Most of the FK work is Jacobian estimation, not function evaluation.**

## What's actually slow, ranked

1. **Finite-difference Jacobian dominates FK calls.** ~70% of FK calls (~156k of 223k) come from `approx_derivative` building the Jacobian by forward differences. A MuJoCo analytical Jacobian (`mj_jac` or `mj_jacGeom`) would replace ~156k FK calls with a single FK + cheap Jacobian extraction per LM iteration. Estimated wall-time saving: ~60–70 seconds.

2. **Per-joint qpos writes through dm_control are slow.** `_set_qpos` loops through ~46 joints calling `self.physics.bind(joint).qpos = float(value)` for each, costing 25.4 sec of pure Python overhead. Replacing with a vectorized `physics.data.qpos[joint_indices] = q` (precomputed indices) avoids ~10M `__setattr__` and `bind` calls. Estimated saving: ~20 seconds.

3. **No warm-starting across anchors.** Each of the 306 IK calls reseeds from neutral or previous-pose. If we feed the previous anchor's IK solution as the initial guess, LM iterations typically drop 2–5x. Estimated saving: ~30–60 seconds (compounds with #1).

4. **IK quality issue: 96.7% of anchors hit nfev cap without converging.** Mean residual 0.117, p95 0.196, success threshold 0.02. Either `max_nfev=80` is too low, the IK weights are mistuned for the test MIDI, or both. This is independent of speed but means current F1 numbers may be capped by IK quality. Need to validate on a real MAESTRO MIDI vs. our synthetic test.

5. **LM solver overhead is negligible.** Only 4.1 sec (3.1%) of total time is in actual LM machinery outside the residual function. Replacing scipy with a hand-rolled or cuRobo LM solver saves at most this amount unless it also restructures the residual.

## Implications for the 24–48 hour plan

The original 48-hour plan (warm starts → keyset cache → anchor stride tuning → tolerance tuning) is **directionally correct but mis-ordered for this workload**. Updated ordering by data-supported leverage:

| Priority | Lever | Code surface | Expected saving | Hours |
|---|---|---|---|---|
| **1** | Vectorized qpos write in `_set_qpos` | `kinematics.py:264-269` | ~20 s (19%) | 4–6 |
| **2** | Warm-start IK from previous anchor solution | `planner.py:550-671` + seed handling | ~30–50 s | 4–8 |
| **3** | Analytical Jacobian via `mj_jac` instead of finite differences | `kinematics.py:641-682` (pass `jac=` to `least_squares`) | ~60–70 s | 8–16 |
| **4** | Fix IK convergence (tune weights or raise max_nfev) | `kinematics.py:674-682` + IK cost terms | quality, not speed | 4–8 |
| 5 | Keyset fingerprint cache | new module | depends on song; ~20–40% on Bach-like | 4–8 |
| 6 | Anchor stride sweep | already a CLI flag | quality vs. speed Pareto | 2–4 |

**Net plausibility check:** Stacking 1+2+3 with no quality regression could plausibly bring this 15-sec window from 131s → 30–50s, which scales to ~1–2 min per 30-sec MAESTRO active window. That hits the user's target *without* GPU IK.

GPU IK (MJX or cuRobo) becomes the next play only after these classical levers are exhausted. The 69% time in `physics.forward()` looks like a GPU win on paper, but with analytical Jacobian the FK count drops ~9x, after which GPU IK ROI is much smaller.

## What we did NOT measure

- **Real MAESTRO MIDI** — bundled robopianist music data is missing locally (the `rousseau/` directory under `robopianist/music/data/` is not in the repo). The synthetic dense MIDI is denser than typical RoboPianist test pieces, so timings are upper bounds.
- **GPU comparison** — MJX/cuRobo numbers belong on the desktop with NVIDIA hardware.
- **F1 on the planned trajectory** — we did not run the trajectory through the rp1m_simulator rollout. Per the user, online rollout evaluation is intentionally out of scope for the per-song target.
- **py-spy flame graph** — py-spy attached to a child process needs `sudo` or codesigning on macOS; we used cProfile instead, which gave precise function-level breakdown.

## Files in this report

```
benchmark/
├── activate_env.sh                 # source this to enter the conda env
├── gen_test_midi.py                # creates /tmp/twinkle.mid (and /tmp/dense_test.mid via the inline run)
├── run_smoke.sh                    # original 5-sec smoke
├── run_profile.sh                  # original profile script (used by hand for dense run)
├── analyze_ik_metrics.py           # per-waypoint metric summary
├── analyze_profile.py              # cProfile bucketed report
├── README.md
└── results/
    ├── BENCHMARK_REPORT.md         # ← this file
    ├── profile.cprof               # raw cProfile binary (1.6 MB)
    ├── profile_top40.txt           # top-40 cumulative dump
    ├── bottleneck_report.md        # autogenerated breakdown (note: % sum>100 because nested calls double-counted; this report supersedes)
    ├── dense_stdout.log
    ├── dense_stderr.log
    ├── dense_prof_stdout.log
    └── dense_prof_stderr.log
```

Trajectory and metadata artifacts:
```
/tmp/maestroso_smoke/twinkle_smoke/           # 5-sec Twinkle, 5 waypoints, 5.5 sec
/tmp/maestroso_dense/dense_smoke/             # 15-sec dense, 150 waypoints, 117 sec (uninstrumented)
/tmp/maestroso_dense_prof/dense_prof/         # same as above, under cProfile, 132 sec
```

## Reproducibility

```bash
source /Users/aangeles/robopiano/benchmark/activate_env.sh

# Smoke test (always run this first to confirm env works)
bash /Users/aangeles/robopiano/benchmark/run_smoke.sh

# Dense profile run
python /Users/aangeles/robopiano/benchmark/gen_test_midi.py   # writes /tmp/twinkle.mid
# (Use the inline python in run_smoke.sh / report to regenerate /tmp/dense_test.mid)
# Then run plan_trajectory.py under cProfile as shown in run_profile.sh.
```
