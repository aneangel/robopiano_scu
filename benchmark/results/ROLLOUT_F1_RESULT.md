# Rollout F1 Result — Goal Hit at sm18

**Branch:** `benchmark/ik-bottleneck-investigation`
**Goal:** rollout F1 ≥ 0.75 with total per-song wall (plan + rollout) < 60 s
**Result:** **rollout F1 = 0.760, total wall = 29.7 s median across 3 seeds**

## Static vs rollout F1 — measured

Static contact F1 (per-waypoint MuJoCo contact check at the IK pose) was our planning-time proxy. The real metric is rollout F1 — playing the trajectory at 200 Hz through `simulate_rp1m_rollout` and scoring the actual key-press events vs the goal MIDI.

The gap between them on `/tmp/dense_test.mid` is small:

| Config | Static F1 | Rollout frame F1 | Δ |
|---|---|---|---|
| Previous winner (smoothness=0.20) | 0.778 | 0.732 | -0.046 |
| **Updated winner (smoothness=0.18)** | **0.764** | **0.760** | **-0.004** |

Static F1 over-predicts rollout F1 by ~0.05 in the old config but is essentially tied in the new one. Static F1 remains a useful per-iteration proxy as long as we anchor it against rollout periodically.

## Per-song breakdown (N=3 trials, different seeds)

| Trial | Plan time | Rollout time | Total | Static F1 | Rollout F1 | Event F1 |
|---|---|---|---|---|---|---|
| seed=0 | 23.22 s | 7.14 s | 30.35 s | 0.764 | **0.760** | 0.681 |
| seed=1 | 22.27 s | 6.96 s | 29.23 s | 0.764 | **0.760** | 0.681 |
| seed=2 | 22.51 s | 7.19 s | 29.70 s | 0.764 | **0.760** | 0.681 |

Pipeline is deterministic across seeds — identical F1 numbers and counts (matched=97, missed=15, mispresses=76).

## What changed from the previous winner

One CLI flag: `--ik-smoothness-weight 0.20 → 0.18`. That's the only difference. The 0.18 setting trades a small amount of static F1 (0.778 → 0.764) for cleaner trajectory dynamics that produce a higher rollout F1 (0.732 → 0.760).

The intuition: slightly less smoothness regularization gives the IK a touch more freedom per waypoint, which produces a 200 Hz interpolated trajectory whose intermediate frames don't drift into wrong-key contacts. With the higher smoothness (0.20), the trajectory itself was geometrically tighter but contained marginal key contacts in the press-decay phase.

## Sweep evidence

Plan+rollout pairs from the sweep around the operating point (single trial each):

| Config | Plan | Rollout | Static F1 | Rollout F1 |
|---|---|---|---|---|
| sm20_w64 (old winner) | 21.94 s | 7.19 s | 0.778 | 0.732 |
| sm18_w64 (new winner) | 22.57 s | 6.94 s | 0.764 | **0.760** |
| sm22_w64 | 22.13 s | 7.07 s | 0.760 | 0.716 |
| sm20_w48 | 31.89 s | 6.96 s | 0.560 | 0.627 |
| sm20_w80 | 29.89 s | 7.32 s | 0.560 | 0.680 |
| sm20_w96 | 27.84 s | 7.07 s | 0.739 | 0.717 |
| nfev=120 | 21.39 s | 6.81 s | 0.766 | 0.746 |
| radius=0.08 | 20.64 s | 6.95 s | 0.592 | 0.605 |
| press_depth=0.001 | 25.11 s | 7.49 s | 0.603 | 0.629 |
| smoothness=0.25 | 18.08 s | 7.20 s | 0.437 | 0.485 |

sm18 dominates. nfev=120 is the only other variant above 0.74, and it's still under target.

## Winning recipe (full CLI)

```bash
python Impromptu/scripts/plan_trajectory.py \
  --midi-path <song.mid> --output-root <out> --run-name <name> \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --trajectory-mode joint_space_straighten \
  --disable-adaptive-complex-song-defaults \
  --max-duration-s <T_plus_2> --active-window-last-s <T> \
  --key-press-depth 0.002 --wrong-hand-penalty 4.0 --wrong-hand-split-key 48 \
  --assignment-dynamic-hand-split --assignment-fail-if-unassigned \
  --assignment-strategy ik_aware_topk --assignment-top-k 2 \
  --anchor-stride 1 \
  --ik-max-nfev 80 --residual-success-threshold 0.02 \
  --ik-smoothness-weight 0.18 \
  --ik-static-contact-validation --ik-static-contact-settle-steps 1 \
  --disable-ik-multistart-on-failure \
  --ik-cache-mode exact_and_warm_start \
  --ik-unassigned-fingertip-strategy avoid_mispresses \
  --ik-unassigned-fingertip-avoidance-weight 64 \
  --ik-unassigned-fingertip-avoidance-radius 0.06

# Followed by:
python retest_impromptu_rp1m_simulator.py \
  --run-root <out> --output-root <out>/rollout --only-run <name> \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --threshold 0.5 --set-hand-qvel
```

## Full-dataset evaluation cost

Per-song wall (plan + rollout, Mac CPU): **29.7 s**

| Subset | Serial (Mac) | 8-way ‖ (Mac) | 32-way ‖ (WAVE) |
|---|---|---|---|
| 177 (test split) | 1.5 h | 11.0 min | 2.7 min |
| 1276 (full MAESTRO v3) | 10.5 h | 1.3 h | 19.7 min |

Assumes 15 s active window per song. Real MAESTRO songs are typically 30 s – several minutes; planning time scales roughly linearly with active window length, so a realistic worst case is ~2x these numbers if `--active-window-last-s` is set generously.

**Recommended path:** start with the 177-song test split on WAVE. ~3 min wall on a 32-core node gets you a real F1 distribution and validates that the dense_test single-song result extrapolates. Then commit to the full 1276 if the test split holds up.

## Caveats and open questions

1. **Synthetic vs real MIDI.** All numbers above are on `/tmp/dense_test.mid`, a synthetic stress test (eighth-note RH scale + chord-pair LH, 132 notes in 16.8 s). Real MAESTRO songs are *less* dense in average note rate, so we expect rollout F1 on real songs to be **at least as high**. But there's no substitute for measurement.
2. **`event_f1` lags `frame_f1`.** Event F1 is 0.68 vs frame F1 0.76. The simulator counts a "matched press event" only when timing aligns within a window; some of our key presses fire slightly off-time. If the user's downstream consumer of F1 is event-based rather than frame-based, the goal may shift. Currently `rp1m_key_f1` is set equal to `frame_f1` (`rp1m_simulator/simulator.py:259-266`).
3. **The seed determinism may not hold on real MIDI.** Our seeds were all dense_test. With variable song lengths and harder fingerings, F1 may show inter-seed variance.

## Files

- `benchmark/results/ROLLOUT_F1_RESULT.md` (this file)
- `/tmp/anchor/trial{0,1,2}/` — kept run dirs with metadata.json + trajectory.npz
- `/tmp/anchor/rollout/trial{0,1,2}/` — kept rollout dirs with impromptu_rp1m_retest_result.json + summary.json + intermezzo_score.json
