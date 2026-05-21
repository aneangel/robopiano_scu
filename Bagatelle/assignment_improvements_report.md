# Bagatelle Assignment Improvements Report

## Summary

- Added configurable Bagatelle fingertip assignment strategies while preserving the deterministic legacy previous-pose Hungarian path as the default.
- Added `legacy_previous_pose`, `composite_cost`, and `ik_aware_topk` assignment modes.
- Preserved backward compatibility by keeping the legacy assignment API available and defaulting new config to `legacy_previous_pose`.

## Files modified

- `Bagatelle/src/bagatelle/config.py`
- `Bagatelle/src/bagatelle/assignment.py`
- `Bagatelle/src/bagatelle/planner.py`
- `Bagatelle/scripts/plan_trajectory.py`
- `Bagatelle/tests/test_assignment_improved.py`
- `Bagatelle/tests/test_planner.py`
- `Bagatelle/README.md`
- `Bagatelle/assignment_improvements_report.md`

## Config options added

- `assignment_strategy`
- `assignment_distance_weight`
- `assignment_hand_zone_weight`
- `assignment_finger_zone_weight`
- `assignment_crossing_weight`
- `assignment_hold_weight`
- `assignment_reach_weight`
- `assignment_black_key_weight`
- `assignment_hard_hand_split`
- `assignment_middle_key`
- `assignment_wrong_hand_penalty`
- `assignment_reach_soft_limit`
- `assignment_top_k`
- `assignment_top_k_extra_penalty`
- `assignment_ik_residual_weight`
- `assignment_ik_max_residual_weight`
- `assignment_ik_failure_penalty`
- `assignment_motion_weight`
- `assignment_store_cost_components`

## Tests run

- `pytest Bagatelle/tests/test_assignment.py -q`
- `pytest Bagatelle/tests -q`
- `python Bagatelle/scripts/plan_trajectory.py --help`
- `python -m compileall Bagatelle/src/bagatelle Bagatelle/scripts Bagatelle/tests`

## Known limitations

- Composite heuristics are still hand-tuned.
- Top-K generation is approximate, not exact Murty K-best.
- IK-aware top-K is slower because it runs IK per candidate.
- Black-key heuristics depend on piano key indexing and should be validated against RoboPianist key numbering.

## Recommended next step

- Evaluate `legacy_previous_pose`, `composite_cost`, and `ik_aware_topk` on the same target-key or MIDI set using Bagatelle evaluation metrics.
