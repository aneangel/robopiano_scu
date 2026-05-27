# Inter-Waypoint Warm-Start Refinements

Investigation of additional warm-start opportunities beyond the keyset cache
(`benchmark/design/keyset_cache_design.md`) that delivered the 118s -> 72s
(1.65x) speedup.

## 1. Current multistart structure

`Bagatelle/src/bagatelle/kinematics.py:778-817` is the multistart driver inside
`solve_press_pose`. Key block:

```python
# kinematics.py:778-800
result = with_static_contact_metrics(solve_from_seed(x0, seed_index=0))
best = result
rhapsody_seed = self._rhapsody_press_pose_seed(assignments, previous, cfg)
try_extra_seeds = bool(validate_static_contacts) or (
    not bool(result.success) and bool(getattr(cfg, "ik_multistart_on_failure", True))
) or rhapsody_seed is not None
if not try_extra_seeds:
    if cache is not None and cache_key is not None and bool(result.success):
        cache.insert(cache_key, result.pose, float(result.residual_norm))
    return result
extra_seeds: list[np.ndarray] = []
if rhapsody_seed is not None:
    extra_seeds.append(self.clip_qpos(rhapsody_seed))
extra_seeds.extend(
    self._ranked_press_pose_initial_seeds(
        previous=previous, neutral=neutral,
        finger_indices=finger_indices, target_positions=target_positions,
        config=cfg,
    )
)
```

`_ranked_press_pose_initial_seeds` (`kinematics.py:398-471`) generates up to
`ik_multistart_seed_count` (default 4) candidates from a forearm-tx grid, ranked
by forward-kinematics fingertip residual.

Per-candidate early-exit (`kinematics.py:811-814`):

```python
if not validate_static_contacts and bool(candidate.success):
    if cache is not None and cache_key is not None:
        cache.insert(cache_key, candidate.pose, float(candidate.residual_norm))
    return candidate
```

The first-solve early-exit (`kinematics.py:785-788`) only fires when
`try_extra_seeds` is False -- i.e. when contact validation is OFF and the first
solve already succeeded and no rhapsody seed exists.

## 2. Why multistart fires on every waypoint today

The current run sets `ik_static_contact_validation=True`, so
`validate_static_contacts` is True at `kinematics.py:782`, which forces
`try_extra_seeds=True` unconditionally. Every waypoint pays for the first solve
plus the rhapsody seed plus the ranked seeds. That explains the 306 LM solves /
150 waypoints (~2.04 solves per waypoint).

Per-candidate, the `validate_static_contacts` branch on line 811 is bypassed --
candidates are only compared by `contact_rank` (`kinematics.py:753-773`). There
is no short-circuit even when a candidate is contact-perfect.

## 3. Proposed early-exit: contact-perfect first solve

`result.success` is set at `kinematics.py:865`:
`bool(optimizer_success and max_residual <= threshold and unassigned_keys.size == 0)`.
Contact metrics (`contact_missed_key_count`, `contact_wrong_key_count`) come
from `static_contact_metrics` via `with_static_contact_metrics`
(`kinematics.py:735-751`). They are independent of `result.success`.

A contact-perfect first solve is one where:

- `result.success` is True (residual <= `residual_success_threshold`, all keys assigned)
- `result.contact_missed_key_count == 0`
- `result.contact_wrong_key_count == 0`

When all three hold, none of the extra seeds can produce a strictly better
`contact_rank` result (the rank components `missed`, `wrong`, `failure`, and
`max_residual` are at their floor, and a tied `score` would only break by
`residual_norm`). So the extra seeds are wasted work.

Proposed change at `kinematics.py:778-800`:

```python
result = with_static_contact_metrics(solve_from_seed(x0, seed_index=0))
best = result
# Contact-perfect first solve short-circuit: when validation is enabled and the
# first solve already produces a hit-all, no-mispress, sub-threshold pose, no
# extra seed can beat it under contact_rank, so skip the multistart.
contact_perfect = (
    bool(result.success)
    and int(result.contact_missed_key_count) == 0
    and int(result.contact_wrong_key_count) == 0
)
rhapsody_seed = (
    None if contact_perfect
    else self._rhapsody_press_pose_seed(assignments, previous, cfg)
)
try_extra_seeds = (
    (bool(validate_static_contacts) and not contact_perfect)
    or (not bool(result.success) and bool(getattr(cfg, "ik_multistart_on_failure", True)))
    or rhapsody_seed is not None
)
if not try_extra_seeds:
    if cache is not None and cache_key is not None and bool(result.success):
        cache.insert(cache_key, result.pose, float(result.residual_norm))
    return result
```

Three behavioral changes:

- Adds the `contact_perfect` gate so contact-validated runs can short-circuit.
- Skips the rhapsody seed when the first solve is already contact-perfect
  (rhapsody is a backup, not a refinement target).
- Preserves the existing failure-multistart and non-validated paths.

Expected impact: on the 150-waypoint Bagatelle song with contact validation on,
if even 50% of waypoints are contact-perfect on the first solve, total LM
solves drop from ~306 to ~225, roughly an 18% wall-clock reduction on top of
the keyset cache (assumes uniform per-solve cost). Combined with the warm-start
seed from the keyset cache (`kinematics.py:638-639`), waypoints with cache
near-misses now also get the early-exit benefit.

Quality risk is bounded: by construction, any pose we skip rerunning was
already inside the residual/contact thresholds. The risk is only that an extra
seed might have found a strictly lower `residual_norm` tiebreaker -- not
worth the multistart cost.

## 4. Cache prewarming (future work)

True prewarming is not viable: a cache entry depends on `previous_qpos`, which
is itself a function of song history. Pre-solving keysets in isolation would
seed them from neutral, which is exactly the bad case the keyset cache is
designed to repair.

Simpler alternative: for waypoint 0, run two seeds (neutral and rhapsody-if-
available), pick the contact-ranked best, and insert into the cache. This is
already approximated by the current multistart -- no code change needed.

## 5. Anchor-level warm-start (Impromptu)

`Impromptu/src/impromptu/ik_solver.py:244-302` already chains anchor solves via
`previous = result.pose.astype(np.float32)` on line 293. Each anchor IK seeds
LM with `initial_qpos=seed_rows[row]` (line 287) and regularizes smoothness
against the previous anchor pose (line 199 inside `solve_fingertip_frame`).
Rhapsody can override `initial` (lines 150-162) when enabled. There is no
unused warm-start slot here -- inter-anchor warm-start is fully wired.

The remaining anchor-side opportunity is the same contact-perfect early-exit
proposed above, but Impromptu does not run multistart at all
(`solve_fingertip_frame` is a single `least_squares` call, lines 204-213), so
no further work is needed.

## 6. max_nfev tuning (deferred)

`ik_max_nfev=80` with 96.7% of waypoints hitting the cap is a Lever-2 problem.
Two paths:

- (a) Raise `max_nfev` to 160-240. Improves convergence rate but scales each LM
  call linearly. Net effect unclear without measurement.
- (b) Land analytical Jacobian (Lever 2). Each LM step drops the finite-
  difference probe (~D residual evaluations per Jacobian), so 80 nfev with
  analytical J does more useful work than 80 nfev with finite-difference J.

Recommendation: do not re-tune `ik_max_nfev` until analytical Jacobian is in
place. Then sweep `{40, 80, 120}` with the contact-perfect early-exit active.

## 7. Patch sketch (primary recommendation)

`Bagatelle/src/bagatelle/kinematics.py:778` -- replace the first 12 lines of
the existing `try:` block with the snippet in section 3 above. Net diff is
~10 lines added, no signature changes, no config flag required (the new gate
is a strict subset of existing behavior).

## Summary

Primary recommendation: land the contact-perfect first-solve early-exit
(section 3, ~10-line patch). Estimated ~18% additional wall-clock reduction on
contact-validated runs, no quality regression by construction. Other items
(prewarming, anchor warm-start, `max_nfev`) are either already-wired or
deferred behind Lever 2.
