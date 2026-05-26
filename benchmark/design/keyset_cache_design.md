# Keyset Fingerprint Cache Design

## 1. Summary

The keyset fingerprint cache memoizes `Kinematics.solve_press_pose`
(`kinematics.py:585`) keyed on (active-keys, hand-split, finger-assignment) so
MAESTRO songs revisiting the same chord shape reuse a previously-solved qpos
instead of rerunning scipy LM. It is safe because only solves whose
`residual_norm` already passes `cfg.residual_success_threshold` (default 0.02)
are cached, and warm-start hits still run the full IK solver. At a 30-40%
exact-hit rate against the profiled `306 anchor IK calls x 397.7 ms` baseline
that is ~35-50 sec saved per song, plus an expected ~10% wall-time reduction
from warm-started near-misses converging in fewer LM iterations.

## 2. Cache Key Design

`KeysetCacheKey` (`keyset_cache.py`, `@dataclass(frozen=True)`) holds three
co-indexed tuples sorted by active key index:

- `active_keys`: MIDI key indices (0..87) being pressed.
- `hand_assignment`: per-key 'L'/'R' label. Same notes split differently
  between hands MUST cache separately - qpos lies in disjoint joint regions.
- `finger_assignment`: per-key dense finger index (0..9). Different finger
  assignments produce different optimal poses because the IK residual at
  `kinematics.py:644` indexes `fingertips[finger_indices]`.

Deliberately NOT hashed: `previous_qpos` (seed at `planner.py:586`; would make
the cache miss-only; smoothness loss is tracked as a risk in Sec. 7);
`contact_targets` (deterministic function of `active_keys` for a fixed piano);
`neutral_qpos` (constant per song).

`KeysetCache.make_key(...)` sorts by `active_keys` for canonicalization so the
key is invariant to the order the assignment stage emits.

## 3. Lookup Behavior

`KeysetCache.lookup(key) -> (qpos_or_None, hit_kind)`:

```
        +-----------------------+
        | self.mode == "off"?   |
        +-----------+-----------+
                    | no
                    v
        +-----------------------+
        | exact match in dict?  |---yes--> return (qpos.copy(), 'exact')
        +-----------+-----------+          [skip IK entirely]
                    | no
                    v
        +-------------------------------+
        | mode == 'exact_and_warm_start'|
        | AND cache non-empty?          |
        +-----------+-------------------+
                    | yes
                    v
        +-------------------------------+
        | scan entries; pick highest    |
        | Jaccard sim over threshold    |---hit--> return (qpos.copy(),
        +-----------+-------------------+          'warm_start')
                    | none above threshold       [feed to scipy LM as seed]
                    v
                  return (None, 'miss')   [run IK with normal seeds]
```

Jaccard is computed only over `active_keys` as a `frozenset` (helper
`_jaccard_similarity`). Hand/finger assignment are ignored for near-miss
because warm starts only need to be close in fingertip target space. With
`jaccard_threshold=0.8`, a 5-note chord matches if 4 of 5 keys overlap.
Warm-start consumers MUST still call `insert()` after LM converges.

## 4. Insertion Policy

`KeysetCache.insert(key, qpos, residual_norm)`:
- Rejected if `residual_norm > max_residual_for_insert` (default 0.02, mirrors
  `cfg.residual_success_threshold` from `config.py:35`). Increments
  `stats.rejected_low_quality`.
- If `key` is already present, the lower-residual entry wins. Ties prefer the
  existing entry to avoid churn.
- Cache scope is one `KeysetCache` instance constructed at the start of
  `plan_target_keys` and discarded at song end. Nothing persists to disk.

## 5. Integration Plan

Patch `Bagatelle/src/bagatelle/planner.py` inside the waypoint loop at
`planner.py:550-671`. The existing terminal block (lines 665-671) reads:

```python
                press_targets = kin.key_press_targets(active_keys)
                if assignment.count:
                    assignment = replace(
                        assignment,
                        target_positions=press_targets[assignment.assigned_key_positions].astype(np.float32),
                    )
                result = kin.solve_press_pose(assignment, previous_qpos, neutral_qpos=neutral_qpos, config=cfg)
```

The cache call site, inserted immediately before `solve_press_pose` (line 671)
and a corresponding insert after the call:

```python
                # --- keyset cache lookup ---
                cache_key = None
                cache_qpos = None
                hit_kind = "miss"
                if keyset_cache.enabled and assignment.count:
                    hand_labels = tuple(
                        "L" if int(f) < 5 else "R"
                        for f in assignment.assigned_finger_indices.tolist()
                    )
                    cache_key = keyset_cache.make_key(
                        active_keys=assignment.assigned_keys,
                        hand_assignment=hand_labels,
                        finger_assignment=assignment.assigned_finger_indices,
                    )
                    cache_qpos, hit_kind = keyset_cache.lookup(cache_key)
                if hit_kind == "exact" and cache_qpos is not None:
                    fingertips = kin.fingertip_positions_for_qpos(cache_qpos)
                    result = kin._result_from_pose(
                        cache_qpos, fingertips, assignment,
                        optimizer_success=True, optimizer_status=0,
                        optimizer_message="keyset_cache_exact_hit",
                        optimizer_cost=0.0, nfev=0,
                        threshold=float(cfg.residual_success_threshold),
                    )
                else:
                    seed_override = cache_qpos if hit_kind == "warm_start" else None
                    result = kin.solve_press_pose(
                        assignment, previous_qpos, neutral_qpos=neutral_qpos,
                        config=cfg, warm_start_seed=seed_override,
                    )
                # --- keyset cache insert ---
                if cache_key is not None and hit_kind != "exact":
                    keyset_cache.insert(cache_key, result.pose, float(result.residual_norm))
```

This requires a small additive change in `solve_press_pose`
(`kinematics.py:585`) to accept an optional `warm_start_seed`, which when
provided is prepended to the seed list inside `_ranked_press_pose_initial_seeds`
(`kinematics.py:398`). The `ik_aware_topk` branch at `planner.py:555-633` gets
the analogous wrapper around the final solve on line 620 (and the per-candidate
scoring solves at line 584 are left uncached - they are exploratory).

CLI plumbing in `Bagatelle/scripts/plan_trajectory.py` (next to the existing
`--residual-success-threshold` on line 60):

```
    parser.add_argument(
        "--ik-cache-mode",
        default="off",
        choices=("off", "exact_only", "exact_and_warm_start"),
        help="Keyset fingerprint cache for IK calls. 'off' preserves current "
             "behavior. 'exact_only' returns cached qpos when the (active_keys,"
             " hand_split, finger_assignment) tuple has been solved before. "
             "'exact_and_warm_start' additionally uses near-misses (Jaccard "
             ">0.8) as scipy LM warm starts.",
    )
```

`plan_target_keys` instantiates `KeysetCache(mode=cfg.ik_cache_mode, ...)`
and writes `keyset_cache.report()` into `metadata.json` alongside the existing
IK metrics.

## 6. Validation Strategy

Two A/B comparisons on a held-out 5-song MAESTRO subset:

1. **Quality regression check.** Run the same songs with `--ik-cache-mode=off`
   and `--ik-cache-mode=exact_only`. Compare `static_contact_f1`, mean
   `residual_norm`, and `max_residual` over all waypoints. Acceptance: cached
   run is within +/- 0.5% of baseline F1 and mean residual is no worse than
   baseline by more than 1e-4.
2. **Warm-start sanity.** With `--ik-cache-mode=exact_and_warm_start`, log
   `nfev` per IK call. Warm-started solves should show lower median nfev than
   the cold-start runs while converging to the same residual.

A unit test in `Bagatelle/tests/test_keyset_cache.py` covers `make_key`
canonicalization (different input orders -> same key), `lookup` returning
`'miss'` when `mode='off'`, residual gating in `insert`, and the Jaccard
near-miss path producing a `'warm_start'` hit.

## 7. Risks

- **Smoothness coupling.** The IK residual at `kinematics.py:647` includes
  `(q - previous) * ik_smoothness_weight`. A cached pose was solved from a
  *different* `previous_qpos`, so trusting it bypasses that term. Mitigation:
  the cache only short-circuits when the keyset/hand/finger tuple is
  identical, and the smoothness-driven delta tends to be small for repeated
  chord shapes; warm-start mode lets the LM re-impose smoothness explicitly.
  Validation strategy (Sec. 6, item 1) catches regressions.
- **Stale cache across configuration changes.** If `ik_smoothness_weight` or
  the piano model change mid-run the cached qpos becomes invalid. Mitigation:
  cache is per-`plan_target_keys` call; nothing persists across runs.
- **Hand-label inference.** Inferring 'L'/'R' from finger index < 5 assumes the
  current 10-finger model; documented at the call site so a future split
  change is easy to find.
- **Warm-start divergence.** A bad warm start could push LM into a worse
  local minimum than `_ranked_press_pose_initial_seeds`
  (`kinematics.py:398`) would have. Mitigation: warm start is *prepended* to
  the existing seed list, not substituted; the multistart code keeps its
  fallbacks.
