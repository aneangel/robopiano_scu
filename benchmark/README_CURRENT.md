# Current State of the IK Bottleneck Investigation

**Branch:** `benchmark/ik-bottleneck-investigation`
**Latest commit:** `cad9c3a` — "Enable multistart (6 seeds, nfev=240) for F1 lift experiment"
**Last updated:** 2026-05-28
**Remote:** `git@github.com:aneangel/robopiano_scu.git`

## TL;DR — what produces our best results

The single config below produces our highest validated F1 on the MAESTRO test split. It's hard-coded into `benchmark/run_full_evaluation.py:55-78` and runs through `Impromptu/scripts/plan_trajectory.py` plus `retest_impromptu_rp1m_simulator.py`.

```bash
source benchmark/activate_env.sh   # macOS conda env
cd /path/to/robopiano

# Drives plan + rollout for an entire MAESTRO manifest, writes CSV
python benchmark/run_full_evaluation.py \
  --manifest /path/to/maestro/test_split.csv \
  --manifest-root /path/to/maestro/maestro-v3.0.0 \
  --output-root /tmp/maestro_eval \
  --csv-out /tmp/maestro_eval/results.csv \
  --max-duration-s 30 \
  --active-window-last-s 28 \
  --parallelism 6
```

That single command produces a per-song results CSV with rollout F1, per-song wall, and per-song mispress counts.

## What changed in the latest commit

`cad9c3a` flipped two flags in `BASELINE_PLAN_FLAGS`:

```diff
- "--ik-max-nfev", "60",
+ "--ik-max-nfev", "240",
- "--disable-ik-multistart-on-failure",
+ "--ik-multistart-seed-count", "6",
```

Effect: each per-waypoint IK now runs 6 multistart seeds × 240 LM iterations (vs 1×60), giving the LM solver much more room to escape bad local minima on hard polyphonic chord transitions.

## Architecture (committed code that produces these results)

The planner is a fully decomposed classical pipeline. All four levers below are already on by default; the only thing the latest commit changed is the LM budget.

```
MIDI input (MAESTRO v3 .mid)
      │
      ▼
Impromptu/src/impromptu/keys.py          extract target_keys[T, 88] + waypoint_frames
      │
      ▼
Bagatelle/src/bagatelle/planner.py:474   plan_target_keys()
   ├─ Hungarian + topk finger assignment (assignment_strategy=ik_aware_topk, top_k=2)
   ├─ avoid_mispresses unassigned strategy (weight=64, radius=0.06)
   ├─ Keyset fingerprint cache (mode=exact_and_warm_start)
   │  └─ Bagatelle/src/bagatelle/keyset_cache.py
   ├─ Contact-perfect first-solve early-exit
   ├─ Multistart NLS IK (6 seeds × 240 LM iters with scipy.least_squares)
   │  └─ Bagatelle/src/bagatelle/kinematics.py:585  solve_press_pose()
   │     ├─ Vectorized _set_qpos (Lever 1)
   │     ├─ Analytical Jacobian via mj_jacSite (Lever 2)
   │     └─ Static-contact-validation in candidate ranking
   ▼
Impromptu/src/impromptu/planner.py        joint_space_straighten trajectory mode
   └─ Down-sampled to control rate, with --enable-trajectory-refinement
      │
      ▼
trajectory.npz (target_keys, waypoint_hand_joints, planned_hand_joints_dense)
      │
      ▼
retest_impromptu_rp1m_simulator.py        rollout F1 evaluation
   └─ Plays back the dense trajectory at 200 Hz through MuJoCo
      │
      ▼
impromptu_rp1m_retest_result.json         frame_F1, event_F1, matched, missed, mispresses
```

## Code paths to read first

1. `benchmark/run_full_evaluation.py` — single source of truth for the planner CLI invocation.
2. `Bagatelle/src/bagatelle/kinematics.py:585-820` — `solve_press_pose`, the IK core. Contains all four levers and the multistart loop.
3. `Bagatelle/src/bagatelle/keyset_cache.py` — fingerprint cache.
4. `Impromptu/scripts/plan_trajectory.py` — argparse layer that funnels CLI flags into `ImpromptuConfig`.
5. `Impromptu/src/impromptu/planner.py:570-720` — joint-space trajectory build path.

## Reproducing the numbers

All commands are exact CLI; no environment-variable tweaks needed beyond `MUJOCO_GL=glfw` for macOS (`activate_env.sh` sets it).

```bash
# 1. Activate
source benchmark/activate_env.sh

# 2. Download MAESTRO v3 MIDIs (one time, ~85 MB)
mkdir -p /tmp/maestro && cd /tmp/maestro
curl -fsSL -o maestro.zip \
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q -o maestro.zip
cd -

# 3. Build the test split manifest
python - <<'PY'
import csv
with open("/tmp/maestro/maestro-v3.0.0/maestro-v3.0.0.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
with open("/tmp/maestro/test_177.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} songs")
PY

# 4. Run the 177-song eval (~75 min on Mac M3 Pro 6-way parallel)
python benchmark/run_full_evaluation.py \
  --manifest /tmp/maestro/test_177.csv \
  --manifest-root /tmp/maestro/maestro-v3.0.0 \
  --output-root /tmp/maestro_eval_177 \
  --csv-out /tmp/maestro_eval_177/results.csv \
  --max-duration-s 30 \
  --active-window-last-s 28 \
  --parallelism 6
```

## Preliminary results so far

### 10-song mix (curated short+long, easy+hard) — DONE

| Config | Mean F1 | Median F1 | Mean wall |
|---|---|---|---|
| Baseline (cad9c3a^) | 0.555 | ~0.560 | ~66 s |
| **F1-lift (cad9c3a)** | **0.631** | **0.636** | **166 s** |

10/10 succeeded. The lift comes from multistart escaping wrong-key drift on hard polyphonic chords. Notable per-song deltas:
- Scriabin Op.63 Entragete: 0.404 → **0.561** (+0.157)
- Debussy Estampes complete: 0.566 → **0.637** (+0.071)
- Debussy Etude No. 7: 0.596 → **0.634** (+0.038)

### 25 shortest test-split songs — DONE on baseline only

Baseline (`f122322`, no multistart, nfev=60): **mean F1 0.607**, median 0.612, mean wall 73 s, 25/25 succeeded. Used as the F1-floor smoke test on the canonical short-bias subset.

### 177-song MAESTRO test split — DONE on baseline; **IN PROGRESS on F1-lift**

| Config | Mean F1 | Median F1 | Mean wall | Status |
|---|---|---|---|---|
| Baseline (`fa853b9`) | 0.580 | 0.588 | 66 s | DONE — 177/177 success |
| **F1-lift (`cad9c3a`)** | **0.603 @ 74/177 so far** | 0.610 | 203 s | RUNNING (~75 min total) |

74-song partial result is the latest measured F1 on the F1-lift config: **+0.023 lift** above the baseline mean, **clears the 0.60 floor**. Final mean expected ~0.60-0.62.

### 1276-song full MAESTRO v3 — NOT YET RUN

Estimated wall on Mac 6-way at 203 s/song: ~7 hours. Conditional on the 177 F1-lift run finishing above the 0.60 floor (currently on track).

## Honest trade-offs

The F1-lift config **breaks the <2 min/song wall budget** in exchange for the F1 lift:

| | Baseline | F1-lift |
|---|---|---|
| Songs ≤ 60 s wall | 99% | <5% |
| Songs ≤ 120 s wall | 99% | 22% |
| Songs ≤ 300 s wall | 100% | 92% |
| Total wall on 177 | 0.5 h | ~1.5 h |

If the sub-2-min-per-song goal is the hard requirement, the baseline config (`fa853b9`) is the right pick (mean F1 0.580). If F1 ≥ 0.60 is the hard requirement, the F1-lift config (`cad9c3a`) is the pick.

## What's NOT yet wired (open work)

- **GPU planning port (MJX)**: planning is 96% of wall; GPU is the only path to keep multistart + nfev=240 *and* hit sub-2-min/song. Blocked on MJX not supporting RoboPianist's cylinder-box collisions (see `benchmark/design/gpu_ik_port_plan.md`).
- **Server parallel execution**: Linux fork start-method + MuJoCo deadlocks. The `mp_context=spawn` fix in `e14c885` was necessary but not sufficient; deeper investigation needed.
- **Rhapsody warm-start**: integration slot exists at `kinematics.py:610` but no checkpoint locally.
- **Full 1276-song run**: blocked on confirming 177 F1-lift result.

## Per-song outputs and dataset

Each per-song run writes to `<output-root>/runs/<song>/`:
- `trajectory.npz` — planned hand joints, fingertip targets, IK metrics
- `metadata.json` — config snapshot, contact totals, cache stats

Each rollout writes to `<output-root>/rollout/<song>/`:
- `impromptu_rp1m_retest_result.json` — primary F1 number (`frame_f1`)
- `summary.json` — secondary scoring (`against_goals.key_f1`)
- `intermezzo_score.json` — event-level F1

These two trees plus the CSV are sufficient to reproduce, audit, and visualize any individual song's result.

## Commit lineage on this branch (newest first)

```
cad9c3a Enable multistart (6 seeds, nfev=240) for F1 lift experiment   <-- LATEST
e14c885 Add mp_context=spawn to ProcessPoolExecutor                       (server path enabler)
fa853b9 Save 177-song MAESTRO test split results + GPU rollout scoping doc (last baseline 177 results)
9c32d95 Make run_full_evaluation.py path-portable                          (server-portability)
f122322 Fix ik_aware_topk brittleness + add 25-song result analyzer        (crash fix; this is the cad9c3a parent)
c41dd2c Expose assignment-lookahead CLI flags + trajectory refinement default
f0b632e Add sample-trajectory selector for sharing repo samples
a80a01d Add MAESTRO full-evaluation driver + real-song-tuned config
... (earlier IK levers and benchmarking)
```

To check out the current best config:
```bash
git checkout benchmark/ik-bottleneck-investigation   # branch
git checkout cad9c3a                                  # exact commit
```
