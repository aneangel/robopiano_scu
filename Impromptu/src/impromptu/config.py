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
    interpolation_substeps: int = 10
    approach_s: float = 0.055
    hold_s: float = 0.008
    release_s: float = 0.025
    clearance_height: float = 0.04
    key_press_depth: float = 0.005

    # Joint-space transition planning. The default planner keeps Bagatelle's
    # sparse press poses, then inserts straight-finger anchors between them.
    joint_space_straight_value: float = 0.0
    joint_space_release_fraction: float = 0.25
    joint_space_approach_fraction: float = 0.35
    joint_space_straighten_all_fingers: bool = False
    joint_space_preserve_sustained_fingers: bool = True
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
    large_jump_penalty: float = 0.0
    same_key_same_finger_bonus: float = 0.0

    # IK anchor selection.
    solve_contact_window_only: bool = False
    anchor_stride: int = 1
    include_midpoint_anchors: bool = True
    anchor_change_threshold: float = 0.01

    # IK objective.
    ik_fingertip_weight: float = 2.0
    ik_smoothness_weight: float = 0.05
    ik_neutral_weight: float = 0.005
    ik_max_nfev: int = 160
    ik_ftol: float = 1e-5
    ik_xtol: float = 1e-5
    ik_gtol: float = 1e-5
    residual_success_threshold: float = 0.02
    preserve_waypoint_press_qpos: bool = True
    waypoint_qpos_blend: float = 1.0

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
