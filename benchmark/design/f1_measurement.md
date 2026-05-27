# Computing the true key-press F1 of a planned trajectory

Goal: planning per song <60 s AND key-press F1 >= 0.65. Today the A/B harness
reports `Static F1 = NaN` because the metadata keys it reads
(`static_contact_hits_total`, ...) are never written. This document maps every
F1 implementation in the repo, what it consumes, and how to wire one of them
into `benchmark/run_ab_compare.py` immediately.

## 1. Canonical F1 implementations

Two implementations are used end-to-end in the pipeline.

**Frame-level key F1** (the "official" rollout metric):
- `Intermezzo/src/intermezzo/online_eval.py:198` — `score_rollout(*, target_keys, played_keys, dt, threshold, timing_tolerance_s)`. Computes per-frame TP/FP/FN over the 88 keys of a piano roll above `threshold`, returns `frame_precision`, `frame_recall`, `frame_f1`, plus event-level matching with `timing_tolerance_s`.
- `rp1m_simulator/simulator.py:245` — `key_metrics(target, played, *, threshold=0.5)`. The same algorithm; this is the one wrapped by `simulate_rp1m_rollout` and reported as `against_goals.key_f1` (`simulator.py:822-826`).

Both reduce to: `tp = (target & played).sum(); fp = (~target & played).sum(); fn = (target & ~played).sum(); f1 = 2*p*r/(p+r)`.

**Static-pose contact F1** (per-waypoint, no simulation):
- `Bagatelle/src/bagatelle/kinematics.py:329` — `static_contact_metrics(qpos, assigned_keys, *, threshold, settle_steps)` returns `{target_hit_count, wrong_key_count, missed_key_count, played_key_count}` by calling `activation_for_qpos` for `settle_steps` MuJoCo steps and bucketing keys above `threshold`.
- The Bagatelle planner already calls this when `--ik-static-contact-validation` is set (`kinematics.py:732-751`) and stores the three counts on every `IKResult` (`kinematics.py:65-68`). Per-waypoint values are visible in metadata under `waypoint_results[*]` (`planner.py:86-89`).

## 2. How either is driven today

**Rollout F1 from a trajectory.npz**: `retest_impromptu_rp1m_simulator.py:58-140` is the canonical driver. It reads `planned_hand_joints_dense` and `target_keys` from `trajectory.npz`, builds a hand-state rollout config, calls `simulate_rp1m_rollout`, then post-scores with `score_rollout`. CLI:

```bash
python retest_impromptu_rp1m_simulator.py \
  --run-root <PARENT_OF_RUN_DIRS> \
  --output-root <OUTPUT> \
  --only-run <RUN_DIR_NAME> \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --threshold 0.5 \
  --set-hand-qvel
```

It writes `<output>/<run>/impromptu_rp1m_retest_result.json` with `rp1m_key_f1`, `frame_f1`, and `event_f1`.

`Maestroso/scripts/run_accuracy_recovery_sweep.py:614-640` uses this script as a subprocess for every variant (`run_retest`). `Impromptu/scripts/render_hand_state_playback.py` is the same evaluation embedded in a longer rendering pipeline; it does not have a "F1 only" mode.

**No script aggregates static contact F1 today**. The data exists (see §5) but nothing sums it.

## 3. What rollout actually requires

Inputs (per song):
- `<run>/trajectory.npz` with `planned_hand_joints_dense` (shape `[T*S, 46]`) and `target_keys` (shape `[T, 88+]`).
- An installed RoboPianist env. `_load_env` is called once; MuJoCo is `egl` headless. On Mac the user already has this working.
- The MIDI is reconstructed inside `simulate_rp1m_rollout` from `goals` via `write_goals_proto` (`simulator.py:286-312`), so no extra MIDI input is needed.

Cost: `simulate_rp1m_rollout` does `T*S` MuJoCo control steps with hand qpos restoration each step (`simulator.py:1010-1023`). For our 16.8 s test MIDI at 50 ms control × 10 substeps that's ~3360 substeps. On a Mac CPU running MuJoCo CPU each substep is roughly 5-15 ms, so a single rollout is roughly **30-60 s wall**, plus a couple of seconds for env load. This is on the same order as the planning budget itself.

## 4. Static contact F1 from waypoint counts

The static counts already on every `IKResult` map naturally onto P/R/F1:
- `target_hit_count` -> TP (target key was assigned and actually depressed)
- `wrong_key_count` -> FP (unassigned key got depressed)
- `missed_key_count` -> FN (assigned key didn't depress)

Aggregating across all anchor waypoints gives a single static F1 number per plan: `F1 = TP / (TP + 0.5*(FP+FN))`.

Caveats:
1. It scores *static* poses after `settle_steps=1` MuJoCo substeps, not a dynamic rollout, so transitions, inertia, and finger interference between adjacent waypoints are invisible. It is therefore expected to be slightly higher than rollout F1 for the same plan.
2. It is exactly the signal the IK objective minimizes (`planner.py:191-197`, `kinematics.py:763-772`), so optimizing for it is by construction in-distribution.
3. It excludes any contact churn outside the anchor frames. With `--anchor-stride 1` (baseline) every control frame is an anchor, so this is the entire active window.

## 5. The IK_METRIC_COLUMNS constant and the missing aggregation

`Bagatelle/src/bagatelle/planner.py:26-34` currently:

```python
IK_METRIC_COLUMNS = (
    "success",
    "optimizer_success",
    "nfev",
    "optimizer_cost",
    "mean_assigned_distance",
    "max_assigned_distance",
    "residual_norm",
)
```

The row writer at `planner.py:111-124` only emits these seven values. The Impromptu variant (`Impromptu/src/impromptu/ik_solver.py:16-25`) adds an eighth `active_finger_count` column but no contact columns.

Note: `ik_metrics` is *not* written into Impromptu's `trajectory.npz` (the npz only has `ik_anchor_metrics`, the Impromptu-side anchor metrics — see `Impromptu/src/impromptu/planner.py:605` and verified by `np.load('/tmp/cache_smoke/cache_ws/trajectory.npz').keys()`). However, **the per-waypoint contact counts are already serialized into `metadata.json`** via `IKResult.to_dict()` (`kinematics.py:70-90`) inside `waypoint_results[:500]` (`planner.py:426`). Aggregation can read from there without touching the IK row schema.

If we want them on the matrix and on `trajectory.npz`, the diff is minimal and additive:

```python
# Bagatelle/src/bagatelle/planner.py:26
IK_METRIC_COLUMNS = (
    "success", "optimizer_success", "nfev", "optimizer_cost",
    "mean_assigned_distance", "max_assigned_distance", "residual_norm",
    "contact_target_hit_count",   # NEW
    "contact_wrong_key_count",    # NEW
    "contact_missed_key_count",   # NEW
)

# planner.py:111 _ik_metric_row(...)
return np.asarray([
    float(result.success), float(result.optimizer_success), float(result.nfev),
    float(result.optimizer_cost), mean_distance, float(result.max_residual),
    float(result.residual_norm),
    float(getattr(result, "contact_target_hit_count", 0)),
    float(getattr(result, "contact_wrong_key_count", 0)),
    float(getattr(result, "contact_missed_key_count", 0)),
], dtype=np.float32)
```

Plus three new top-level metadata keys in `_metadata_from_results` (`planner.py:417-422`):

```python
contact_hits = sum(int(r.contact_target_hit_count) for r in ik_results)
contact_wrongs = sum(int(r.contact_wrong_key_count) for r in ik_results)
contact_misses = sum(int(r.contact_missed_key_count) for r in ik_results)
metadata["static_contact_hits_total"] = contact_hits
metadata["static_contact_wrongs_total"] = contact_wrongs
metadata["static_contact_misses_total"] = contact_misses
```

These three keys are exactly what `benchmark/run_ab_compare.py:144-146` already tries to read. After this patch, `Static F1` stops being NaN with zero changes to the harness.

## 6. rp1m_simulator interface

`rp1m_simulator/README.md` documents `python -m rp1m_simulator rollout` for RP1M-zarr inputs, not for Impromptu's `planned_hand_joints_dense`. The actual interface we use is the Python API `simulate_rp1m_rollout(trajectory, config, output_dir)` plus `make_rp1m_trajectory_from_arrays`, which is exactly what `retest_impromptu_rp1m_simulator.py` does. Mac CPU runtime for our 16.8 s / 15 s active-window MIDI: ~30-60 s per song per rollout (see §3).

## 7. Recommendation

**Adopt (A) static contact F1 as the per-iteration A/B metric; reserve (B) for a periodic ground-truth check.**

Reasons:
1. It already exists per-waypoint; only a sum + three metadata keys are needed.
2. It is what the IK objective is being asked to minimize, so it is the right number to track while iterating on IK and assignment.
3. Adding ~30-60 s of rollout to every A/B trial doubles harness wall time and reintroduces "online rollout evaluation," which the user explicitly excluded for the per-iteration loop.
4. The user's threshold (F1 >= 0.65) is achievable with static F1 because Bagatelle's contact validation already encodes the same scoring — IK candidates are ranked by missed+wrong key counts (`planner.py:191-197`).

For the F1 >= 0.65 gate, optimize static F1, then verify with one rollout (B) per locked variant before declaring the goal met. If static and rollout F1 diverge by more than ~0.05 on the verification song, treat that as a separate "static-vs-rollout drift" investigation rather than gating planning on rollout.

## 8. Patch sketch (Recommendation A)

Two-line touch in `_metadata_from_results` (Bagatelle):

```python
# Bagatelle/src/bagatelle/planner.py, inside _metadata_from_results around line 418
metadata["static_contact_hits_total"] = int(
    sum(int(getattr(r, "contact_target_hit_count", 0)) for r in ik_results)
)
metadata["static_contact_wrongs_total"] = int(
    sum(int(getattr(r, "contact_wrong_key_count", 0)) for r in ik_results)
)
metadata["static_contact_misses_total"] = int(
    sum(int(getattr(r, "contact_missed_key_count", 0)) for r in ik_results)
)
```

These keys propagate into Impromptu's `metadata.json` already (Impromptu's `_metadata(...)` merges Bagatelle's metadata via `bagatelle_plan.metadata` in `planner.py:557-587`, but the totals must be hoisted to top level for the harness to read them — easiest: explicitly mirror in `Impromptu/src/impromptu/planner.py` right after line 587:

```python
for key in ("static_contact_hits_total",
            "static_contact_wrongs_total",
            "static_contact_misses_total"):
    if key in bagatelle_plan.metadata:
        metadata[key] = bagatelle_plan.metadata[key]
```

`benchmark/run_ab_compare.py` already does the right aggregation at lines 202-211:

```python
denom = tp + 0.5 * (fp + fn)
f1s.append(tp / denom)
agg.static_contact_f1 = float(np.mean(f1s)) if f1s else float("nan")
```

No harness change is required. After the Bagatelle/Impromptu patches above, `Static F1` immediately becomes a real number.

For verification runs (recommendation A's fallback to ground truth):

```bash
python retest_impromptu_rp1m_simulator.py \
  --run-root /tmp/ab_compare_run \
  --output-root /tmp/ab_compare_run/rollout_check \
  --only-run baseline_trial0 \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --threshold 0.5 --set-hand-qvel
cat /tmp/ab_compare_run/rollout_check/baseline_trial0/impromptu_rp1m_retest_result.json \
  | python -c "import json,sys; r=json.load(sys.stdin); \
    print('rp1m_key_f1=', r['rp1m_key_f1'], 'frame_f1=', r['frame_f1'])"
```

## Plan to start measuring F1 today

1. Add the three `static_contact_*_total` sums in `_metadata_from_results` (`Bagatelle/src/bagatelle/planner.py:417` block).
2. Mirror them in `Impromptu/src/impromptu/planner.py:587`.
3. Confirm `--ik-static-contact-validation` is in `BASELINE_FLAGS` (already at `benchmark/run_ab_compare.py:55`). Without it the three counts are zero.
4. Re-run `python benchmark/run_ab_compare.py --variant baseline --trials 3` — the `Static F1` column will now be populated. Anything below 0.65 means we have not yet hit the goal even by the proxy.
5. Once a variant reaches the static-F1 target, run `retest_impromptu_rp1m_simulator.py` once on that variant's trial-0 artifact (use `--keep-artifacts` when running the A/B) to verify rollout F1 has not diverged.
