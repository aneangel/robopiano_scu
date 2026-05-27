from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImpromptuConfig:
    control_timestep: float = 0.05
    threshold: float = 0.5
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    seed: int = 0
    reduced_action_space: bool = True

    # Fingertip trajectory planning.
    trajectory_mode: str = "joint_space_straighten"
    adaptive_complex_song_defaults: bool = True
    adaptive_active_mean_polyphony_threshold: float = 2.2
    adaptive_max_polyphony_threshold: int = 5
    interpolation_substeps: int = 10
    approach_s: float = 0.055
    hold_s: float = 0.008
    release_s: float = 0.025
    clearance_height: float = 0.04
    key_press_depth: float = 0.0035

    # Joint-space transition planning. The default planner keeps Bagatelle's
    # sparse press poses, then inserts straight-finger anchors between them.
    joint_space_straight_value: float = 0.0
    joint_space_release_fraction: float = 0.25
    joint_space_approach_fraction: float = 0.35
    joint_space_straighten_all_fingers: bool = False
    joint_space_preserve_sustained_fingers: bool = True
    joint_space_release_repeated_keys_across_gaps: bool = True
    joint_space_straighten_idle_fingers_at_waypoints: bool = True
    joint_space_lift_straight_anchors: bool = False
    joint_space_straight_lift_height: float = 0.02

    # Fingertip attraction/clearance shaping.
    inactive_clearance_height: float = 0.04
    inactive_clearance_weight: float = 1.0
    active_clearance_weight: float = 0.1
    approach_start_weight: float = 0.25
    hover_weight: float = 0.15
    release_end_weight: float = 0.05
    travel_weight: float = 0.05
    press_weight: float = 1.0
    press_lead_s: float = 0.0
    wrong_key_xy_radius: float = 0.018
    wrong_key_avoid_weight: float = 0.5

    # Bagatelle sparse assignment shaping reused during waypoint contact selection.
    assignment_distance_weight: float = 1.0
    same_finger_bonus: float = 0.0
    reassignment_penalty: float = 0.0
    finger_crossing_penalty: float = 0.0
    wrong_hand_penalty: float = 0.0
    wrong_hand_split_key: int = 44
    assignment_dynamic_hand_split: bool = False
    assignment_dynamic_hand_split_min_span: int = 12
    assignment_dynamic_hand_split_min_keys: int = 3
    large_jump_penalty: float = 0.0
    same_key_same_finger_bonus: float = 0.0

    # Optional Bagatelle assignment search. Defaults preserve the original
    # greedy previous-pose assignment, while complex-song adaptation can enable
    # bounded IK-aware candidate selection before Impromptu interpolates.
    assignment_strategy: str = "legacy_previous_pose"
    assignment_hand_zone_weight: float = 0.0
    assignment_finger_zone_weight: float = 0.0
    assignment_hold_weight: float = 0.0
    assignment_reach_weight: float = 0.0
    assignment_black_key_weight: float = 0.0
    assignment_hard_hand_split: bool = False
    assignment_middle_key: int = 44
    assignment_reach_soft_limit: float = 0.20
    assignment_top_k: int = 1
    assignment_top_k_extra_penalty: float = 1e-4
    assignment_beam_width: int = 4
    assignment_candidates_per_step: int = 0
    assignment_fail_if_unassigned: bool = False
    assignment_unassigned_penalty: float = 25.0
    assignment_ik_residual_weight: float = 0.0
    assignment_ik_max_residual_weight: float = 0.0
    assignment_ik_failure_penalty: float = 10.0
    assignment_motion_weight: float = 0.0

    # IK anchor selection.
    solve_contact_window_only: bool = False
    anchor_stride: int = 1
    include_midpoint_anchors: bool = True
    anchor_change_threshold: float = 0.01

    # IK objective.
    ik_fingertip_weight: float = 2.0
    ik_key_front_weight: float = 1.0
    ik_key_width_weight: float = 1.0
    ik_key_height_weight: float = 1.0
    ik_smoothness_weight: float = 0.05
    ik_neutral_weight: float = 0.005
    ik_inactive_fingertip_clearance_weight: float = 0.0
    ik_inactive_fingertip_clearance: float = 0.02
    ik_unassigned_fingertip_strategy: str = "legacy"
    ik_unassigned_fingertip_avoidance_weight: float = 0.5
    ik_unassigned_fingertip_avoidance_radius: float = 0.03
    ik_wrong_key_xy_avoidance_weight: float = 0.0
    ik_wrong_key_xy_avoidance_radius: float = 0.025
    ik_max_nfev: int = 160
    ik_ftol: float = 1e-5
    ik_xtol: float = 1e-5
    ik_gtol: float = 1e-5
    residual_success_threshold: float = 0.02
    ik_multistart_on_failure: bool = True
    ik_multistart_seed_count: int = 2
    ik_multistart_forearm_tx_grid: int = 5
    ik_static_contact_validation: bool = False
    ik_static_contact_settle_steps: int = 1
    ik_static_contact_wrong_key_weight: float = 1.0
    ik_static_contact_missed_key_weight: float = 2.0
    ik_static_contact_residual_weight: float = 10.0
    ik_static_contact_failure_weight: float = 25.0
    ik_analytical_jacobian: bool = True
    ik_contact_perfect_early_exit: bool = True
    ik_cache_mode: str = "off"
    ik_cache_jaccard_threshold: float = 0.8
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
    preserve_waypoint_press_qpos: bool = True
    waypoint_qpos_blend: float = 1.0
    key_target_front_offset: float = 0.35
    key_target_top_offset: float = 0.5

    # Optional local dense-qpos refinement after anchor IK interpolation.
    enable_trajectory_refinement: bool = False
    trajectory_refinement_window_frames: int = 24
    trajectory_refinement_max_nfev: int = 20
    trajectory_refinement_fingertip_weight: float = 1.0
    trajectory_refinement_velocity_weight: float = 0.005
    trajectory_refinement_acceleration_weight: float = 0.002
    trajectory_refinement_jerk_weight: float = 0.001
    trajectory_refinement_neutral_weight: float = 0.001
    trajectory_refinement_endpoint_weight: float = 0.05

    output_root: str = "/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
