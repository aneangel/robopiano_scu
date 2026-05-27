from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BagatelleConfig:
    control_timestep: float = 0.05
    threshold: float = 0.5
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    seed: int = 0
    reduced_action_space: bool = True

    # IK objective weights. Values are residual multipliers before least-squares.
    ik_fingertip_weight: float = 1.0
    ik_key_front_weight: float = 1.0
    ik_key_width_weight: float = 1.0
    ik_key_height_weight: float = 1.0
    ik_smoothness_weight: float = 0.05
    ik_neutral_weight: float = 0.01
    ik_inactive_fingertip_clearance_weight: float = 0.0
    ik_inactive_fingertip_clearance: float = 0.02
    # `legacy` preserves the original IK objective. `avoid_mispresses`
    # lifts idle fingertips when they are close enough to non-target keys to
    # risk a wrong-key contact.
    ik_unassigned_fingertip_strategy: str = "legacy"
    ik_unassigned_fingertip_avoidance_weight: float = 0.5
    ik_unassigned_fingertip_avoidance_radius: float = 0.03
    ik_wrong_key_xy_avoidance_weight: float = 0.0
    ik_wrong_key_xy_avoidance_radius: float = 0.025
    ik_max_nfev: int = 120
    ik_ftol: float = 1e-5
    ik_xtol: float = 1e-5
    ik_gtol: float = 1e-5
    residual_success_threshold: float = 0.02
    # Retry failed press-pose IK from forearm seeds ranked by fingertip target
    # distance. This keeps the common online path single-solve, but avoids
    # accepting bad local minima on wide or polyphonic chords.
    ik_multistart_on_failure: bool = True
    ik_multistart_seed_count: int = 2
    ik_multistart_forearm_tx_grid: int = 5
    ik_static_contact_validation: bool = False
    ik_static_contact_settle_steps: int = 1
    ik_static_contact_wrong_key_weight: float = 1.0
    ik_static_contact_missed_key_weight: float = 2.0
    ik_static_contact_residual_weight: float = 10.0
    ik_static_contact_failure_weight: float = 25.0
    # When the first IK seed already produces a pose with success=True, zero
    # missed keys, and zero wrong-key contacts, skip multistart's extra seeds.
    # No quality regression by construction: extra seeds are ranked under
    # contact_rank, and a contact-perfect first solve already sits at the
    # rank floor (missed=wrong=0, max_residual <= threshold).
    ik_contact_perfect_early_exit: bool = True

    # Per-song keyset fingerprint cache for IK short-circuiting.
    # off               : no cache, baseline behavior
    # exact_only        : on exact (active_keys, finger_assignment) match, skip LM and reuse cached qpos
    # exact_and_warm_start : exact-match short-circuit + near-miss seeds returned as warm starts (not yet plumbed)
    ik_cache_mode: str = "off"
    ik_cache_jaccard_threshold: float = 0.8
    ik_cache_estimated_seconds_per_ik: float = 0.4

    # Optional RP1M-derived Rhapsody IK seed. This does not replace the
    # physics-backed IK/contact validation; it proposes an initial hand state
    # for the existing least-squares solve.
    rhapsody_ik_enabled: bool = False
    rhapsody_ik_checkpoint: str = ""
    rhapsody_ik_refinement_steps: int = 0
    rhapsody_ik_refinement_lr: float = 0.05
    rhapsody_ik_device: str = "cpu"
    rhapsody_ik_candidate_scoring: bool = False
    rhapsody_ik_coordinate_transform: str = "bagatelle_to_rp1m"
    rhapsody_ik_y_offset: float = 0.08289646
    rhapsody_ik_fill_inactive_from_previous: bool = True
    rhapsody_ik_seed_max_active_error: float = 0.08
    rhapsody_ik_seed_require_previous_improvement: bool = True

    # RoboPianist's fingering reward uses this same key contact heuristic.
    key_target_front_offset: float = 0.35
    key_target_top_offset: float = 0.5
    key_press_depth: float = 0.008

    # Optional sequence-aware finger assignment. Defaults preserve the original greedy Hungarian path.
    sequence_model_type: str = "none"
    sequence_model_checkpoint: str = ""
    cost_bias_alpha: float = 1.0
    assignment_lookahead_steps: int = 0
    assignment_lookahead_weight: float = 1.0

    # Legacy assignment shaping retained for backward compatibility.
    distance_weight: float = 1.0
    same_finger_bonus: float = 0.0
    reassignment_penalty: float = 0.0
    finger_crossing_penalty: float = 0.0
    wrong_hand_penalty: float = 0.0
    large_jump_penalty: float = 0.0
    same_key_same_finger_bonus: float = 0.0
    wrong_hand_split_key: int = 44
    assignment_dynamic_hand_split: bool = False
    assignment_dynamic_hand_split_min_span: int = 12
    assignment_dynamic_hand_split_min_keys: int = 3
    large_jump_distance_m: float = 0.06
    finger_crossing_slack_m: float = 0.005

    # Configurable assignment strategy.
    assignment_strategy: str = "legacy_previous_pose"

    # Composite-cost weights.
    assignment_distance_weight: float = 1.0
    assignment_hand_zone_weight: float = 0.0
    assignment_finger_zone_weight: float = 0.0
    assignment_crossing_weight: float = 0.0
    assignment_hold_weight: float = 0.0
    assignment_reach_weight: float = 0.0
    assignment_black_key_weight: float = 0.0

    # Optional hard/soft constraints.
    assignment_hard_hand_split: bool = False
    assignment_middle_key: int = 44
    assignment_wrong_hand_penalty: float = 1.0
    assignment_reach_soft_limit: float = 0.20

    # Candidate generation.
    assignment_top_k: int = 1
    assignment_top_k_extra_penalty: float = 1e-4

    # Whole-sequence assignment search. `assignment_candidates_per_step`
    # defaults to `assignment_top_k` when set to 0.
    assignment_beam_width: int = 4
    assignment_candidates_per_step: int = 0
    assignment_future_horizon: int = 0
    assignment_sequence_cost_discount: float = 0.9
    assignment_fail_if_unassigned: bool = False
    assignment_unassigned_penalty: float = 25.0

    # IK-aware selection.
    assignment_ik_residual_weight: float = 0.0
    assignment_ik_max_residual_weight: float = 0.0
    assignment_ik_failure_penalty: float = 10.0
    assignment_motion_weight: float = 0.0

    # Debugging / metadata.
    assignment_store_cost_components: bool = True

    # Inter-waypoint planning parameters.
    clearance_height: float = 0.02
    lift_fraction: float = 0.20
    descent_fraction: float = 0.35
    vertical_min: float = 0.0
    vertical_max: float = 0.06

    # Evaluation-only settle steps after restoring each direct pose.
    settle_steps: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
