#!/usr/bin/env python
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Rhapsody" / "src",
    REPO_ROOT / "Variations" / "src",
    REPO_ROOT / "Variations",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from impromptu.active_window import crop_active_window  # noqa: E402
from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.keys import load_target_keys_npz  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Impromptu trajectories from Bagatelle sparse press poses.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi-path", default=None)
    source.add_argument("--target-keys-npz", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--preset", choices=("precision_safe",), default=None)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--active-window-last-s", type=float, default=None)
    parser.add_argument("--active-window-preroll-s", type=float, default=0.5)
    parser.add_argument("--active-window-postroll-s", type=float, default=0.25)
    parser.add_argument(
        "--trajectory-mode",
        choices=("joint_space_straighten", "dense_fingertip_ik"),
        default="joint_space_straighten",
    )
    parser.add_argument("--disable-adaptive-complex-song-defaults", action="store_true")
    parser.add_argument("--adaptive-active-mean-polyphony-threshold", type=float, default=2.2)
    parser.add_argument("--adaptive-max-polyphony-threshold", type=int, default=5)
    parser.add_argument("--interpolation-substeps", type=int, default=10)
    parser.add_argument("--approach-s", type=float, default=0.055)
    parser.add_argument("--hold-s", type=float, default=0.008)
    parser.add_argument("--release-s", type=float, default=0.025)
    parser.add_argument("--clearance-height", type=float, default=0.04)
    parser.add_argument("--key-press-depth", type=float, default=0.0035)
    parser.add_argument("--inactive-clearance-height", type=float, default=0.04)
    parser.add_argument("--inactive-clearance-weight", type=float, default=1.0)
    parser.add_argument("--active-clearance-weight", type=float, default=0.1)
    parser.add_argument("--approach-start-weight", type=float, default=0.25)
    parser.add_argument("--hover-weight", type=float, default=0.15)
    parser.add_argument("--release-end-weight", type=float, default=0.05)
    parser.add_argument("--travel-weight", type=float, default=0.05)
    parser.add_argument("--press-weight", type=float, default=1.0)
    parser.add_argument("--active-press-weight", type=float, default=None)
    parser.add_argument("--press-lead-s", type=float, default=0.0)
    parser.add_argument("--wrong-key-xy-radius", type=float, default=0.018)
    parser.add_argument("--wrong-key-avoid-weight", type=float, default=0.5)
    parser.add_argument(
        "--disable-attraction-forces",
        action="store_true",
        help="Zero clearance and wrong-key attraction residuals for dense fingertip IK comparisons.",
    )
    parser.add_argument("--joint-space-straight-value", type=float, default=0.0)
    parser.add_argument("--joint-space-release-fraction", type=float, default=0.25)
    parser.add_argument("--joint-space-approach-fraction", type=float, default=0.35)
    parser.add_argument("--joint-space-straighten-all-fingers", action="store_true")
    parser.add_argument("--no-joint-space-preserve-sustained-fingers", action="store_true")
    parser.add_argument("--joint-space-preserve-repeated-key-gaps", action="store_true")
    parser.add_argument("--no-joint-space-straighten-idle-waypoint-fingers", action="store_true")
    parser.add_argument("--joint-space-lift-straight-anchors", action="store_true")
    parser.add_argument("--joint-space-straight-lift-height", type=float, default=0.02)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--same-finger-bonus", type=float, default=0.0)
    parser.add_argument("--reassignment-penalty", type=float, default=0.0)
    parser.add_argument("--finger-crossing-penalty", type=float, default=0.0)
    parser.add_argument("--wrong-hand-penalty", type=float, default=0.0)
    parser.add_argument("--wrong-hand-split-key", type=int, default=44)
    parser.add_argument("--assignment-dynamic-hand-split", action="store_true")
    parser.add_argument("--assignment-dynamic-hand-split-min-span", type=int, default=12)
    parser.add_argument("--assignment-dynamic-hand-split-min-keys", type=int, default=3)
    parser.add_argument("--large-jump-penalty", type=float, default=0.0)
    parser.add_argument("--same-key-same-finger-bonus", type=float, default=0.0)
    parser.add_argument(
        "--assignment-strategy",
        choices=("legacy_previous_pose", "composite_cost", "ik_aware_topk", "sequence_beam"),
        default="legacy_previous_pose",
    )
    parser.add_argument("--assignment-hand-zone-weight", type=float, default=0.0)
    parser.add_argument("--assignment-finger-zone-weight", type=float, default=0.0)
    parser.add_argument("--assignment-hold-weight", type=float, default=0.0)
    parser.add_argument("--assignment-reach-weight", type=float, default=0.0)
    parser.add_argument("--assignment-black-key-weight", type=float, default=0.0)
    parser.add_argument("--assignment-hard-hand-split", action="store_true")
    parser.add_argument("--assignment-middle-key", type=int, default=44)
    parser.add_argument("--assignment-reach-soft-limit", type=float, default=0.20)
    parser.add_argument("--assignment-top-k", type=int, default=1)
    parser.add_argument("--assignment-top-k-extra-penalty", type=float, default=1e-4)
    parser.add_argument("--assignment-beam-width", type=int, default=4)
    parser.add_argument("--assignment-candidates-per-step", type=int, default=0)
    parser.add_argument("--assignment-fail-if-unassigned", action="store_true")
    parser.add_argument("--assignment-unassigned-penalty", type=float, default=25.0)
    parser.add_argument("--assignment-ik-residual-weight", type=float, default=0.0)
    parser.add_argument("--assignment-ik-max-residual-weight", type=float, default=0.0)
    parser.add_argument("--assignment-ik-failure-penalty", type=float, default=10.0)
    parser.add_argument("--assignment-motion-weight", type=float, default=0.0)
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--anchor-change-threshold", type=float, default=0.01)
    parser.add_argument("--solve-contact-window-only", action="store_true")
    parser.add_argument("--solve-all-stride-anchors", action="store_true")
    parser.add_argument("--include-midpoint-anchors", action="store_true", default=True)
    parser.add_argument("--no-include-midpoint-anchors", action="store_true")
    parser.add_argument("--ik-fingertip-weight", type=float, default=2.0)
    parser.add_argument("--ik-key-front-weight", type=float, default=1.0)
    parser.add_argument("--ik-key-width-weight", type=float, default=1.0)
    parser.add_argument("--ik-key-height-weight", type=float, default=1.0)
    parser.add_argument("--ik-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--ik-neutral-weight", type=float, default=0.005)
    parser.add_argument("--ik-inactive-fingertip-clearance-weight", type=float, default=0.0)
    parser.add_argument("--ik-inactive-fingertip-clearance", type=float, default=0.02)
    parser.add_argument(
        "--ik-unassigned-fingertip-strategy",
        choices=("legacy", "avoid_mispresses"),
        default="legacy",
    )
    parser.add_argument("--ik-unassigned-fingertip-avoidance-weight", type=float, default=0.5)
    parser.add_argument("--ik-unassigned-fingertip-avoidance-radius", type=float, default=0.03)
    parser.add_argument("--ik-wrong-key-xy-avoidance-weight", type=float, default=0.0)
    parser.add_argument("--ik-wrong-key-xy-avoidance-radius", type=float, default=0.025)
    parser.add_argument("--ik-max-nfev", type=int, default=160)
    parser.add_argument("--residual-success-threshold", type=float, default=0.02)
    parser.add_argument("--disable-ik-multistart-on-failure", action="store_true")
    parser.add_argument("--ik-multistart-seed-count", type=int, default=2)
    parser.add_argument("--ik-multistart-forearm-tx-grid", type=int, default=5)
    parser.add_argument("--ik-static-contact-validation", action="store_true")
    parser.add_argument("--ik-static-contact-settle-steps", type=int, default=1)
    parser.add_argument("--ik-static-contact-wrong-key-weight", type=float, default=1.0)
    parser.add_argument("--ik-static-contact-missed-key-weight", type=float, default=2.0)
    parser.add_argument("--ik-static-contact-residual-weight", type=float, default=10.0)
    parser.add_argument("--ik-static-contact-failure-weight", type=float, default=25.0)
    parser.add_argument("--enable-rhapsody-ik", action="store_true")
    parser.add_argument("--rhapsody-ik-checkpoint", default="")
    parser.add_argument("--rhapsody-ik-refinement-steps", type=int, default=0)
    parser.add_argument("--rhapsody-ik-refinement-lr", type=float, default=0.05)
    parser.add_argument("--rhapsody-ik-device", default="cpu")
    parser.add_argument(
        "--rhapsody-ik-candidate-scoring",
        action="store_true",
        help="Use Rhapsody seeds while scoring every IK-aware assignment candidate. Slower; default seeds only the selected pose.",
    )
    parser.add_argument(
        "--rhapsody-ik-coordinate-transform",
        choices=("bagatelle_to_rp1m", "none"),
        default="bagatelle_to_rp1m",
    )
    parser.add_argument("--rhapsody-ik-y-offset", type=float, default=0.08289646)
    parser.add_argument("--no-rhapsody-ik-fill-inactive-from-previous", action="store_true")
    parser.add_argument("--rhapsody-ik-seed-max-active-error", type=float, default=0.08)
    parser.add_argument("--no-rhapsody-ik-seed-require-previous-improvement", action="store_true")
    parser.add_argument("--key-target-front-offset", type=float, default=0.35)
    parser.add_argument("--key-target-top-offset", type=float, default=0.5)
    parser.add_argument("--enable-trajectory-refinement", action="store_true")
    parser.add_argument("--trajectory-refinement-window-frames", type=int, default=24)
    parser.add_argument("--trajectory-refinement-max-nfev", type=int, default=20)
    parser.add_argument("--trajectory-refinement-fingertip-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-refinement-velocity-weight", type=float, default=0.005)
    parser.add_argument("--trajectory-refinement-acceleration-weight", type=float, default=0.002)
    parser.add_argument("--trajectory-refinement-jerk-weight", type=float, default=0.001)
    parser.add_argument("--trajectory-refinement-neutral-weight", type=float, default=0.001)
    parser.add_argument("--trajectory-refinement-endpoint-weight", type=float, default=0.05)
    return parser


def _load_target_keys(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.midi_path:
        target_keys, meta = load_target_keys_from_midi(
            args.midi_path,
            control_timestep=float(args.control_timestep),
            max_steps=args.max_steps,
            max_duration_s=args.max_duration_s,
        )
        return target_keys, {
            "source_type": "midi",
            "midi_path": str(Path(args.midi_path).expanduser().resolve()),
            "midi": meta,
        }
    target_keys = load_target_keys_npz(args.target_keys_npz)
    return target_keys, {
        "source_type": "target_keys_npz",
        "target_keys_npz": str(Path(args.target_keys_npz).expanduser().resolve()),
    }


def main() -> None:
    args = build_parser().parse_args()
    target_keys, source_meta = _load_target_keys(args)
    crop = crop_active_window(
        target_keys,
        dt=float(args.control_timestep),
        threshold=float(args.threshold),
        active_window_last_s=args.active_window_last_s,
        active_window_preroll_s=float(args.active_window_preroll_s),
        active_window_postroll_s=float(args.active_window_postroll_s),
    )
    target_keys = crop.target_keys
    if args.solve_all_stride_anchors:
        solve_contact_window_only = False
    elif args.solve_contact_window_only:
        solve_contact_window_only = True
    else:
        solve_contact_window_only = False
    press_weight = args.active_press_weight if args.active_press_weight is not None else args.press_weight
    inactive_clearance_weight = 0.0 if args.disable_attraction_forces else float(args.inactive_clearance_weight)
    active_clearance_weight = 0.0 if args.disable_attraction_forces else float(args.active_clearance_weight)
    wrong_key_avoid_weight = 0.0 if args.disable_attraction_forces else float(args.wrong_key_avoid_weight)
    config = ImpromptuConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        environment_name=str(args.environment_name),
        seed=int(args.seed),
        trajectory_mode=str(args.trajectory_mode),
        adaptive_complex_song_defaults=not bool(args.disable_adaptive_complex_song_defaults),
        adaptive_active_mean_polyphony_threshold=float(args.adaptive_active_mean_polyphony_threshold),
        adaptive_max_polyphony_threshold=int(args.adaptive_max_polyphony_threshold),
        interpolation_substeps=int(args.interpolation_substeps),
        approach_s=float(args.approach_s),
        hold_s=float(args.hold_s),
        release_s=float(args.release_s),
        clearance_height=float(args.clearance_height),
        key_press_depth=float(args.key_press_depth),
        inactive_clearance_height=float(args.inactive_clearance_height),
        joint_space_straight_value=float(args.joint_space_straight_value),
        joint_space_release_fraction=float(args.joint_space_release_fraction),
        joint_space_approach_fraction=float(args.joint_space_approach_fraction),
        joint_space_straighten_all_fingers=bool(args.joint_space_straighten_all_fingers),
        joint_space_preserve_sustained_fingers=not bool(args.no_joint_space_preserve_sustained_fingers),
        joint_space_release_repeated_keys_across_gaps=not bool(args.joint_space_preserve_repeated_key_gaps),
        joint_space_straighten_idle_fingers_at_waypoints=not bool(
            args.no_joint_space_straighten_idle_waypoint_fingers
        ),
        joint_space_lift_straight_anchors=bool(args.joint_space_lift_straight_anchors),
        joint_space_straight_lift_height=float(args.joint_space_straight_lift_height),
        inactive_clearance_weight=float(inactive_clearance_weight),
        active_clearance_weight=float(active_clearance_weight),
        approach_start_weight=float(args.approach_start_weight),
        hover_weight=float(args.hover_weight),
        release_end_weight=float(args.release_end_weight),
        travel_weight=float(args.travel_weight),
        press_weight=float(press_weight),
        press_lead_s=float(args.press_lead_s),
        wrong_key_xy_radius=float(args.wrong_key_xy_radius),
        wrong_key_avoid_weight=float(wrong_key_avoid_weight),
        assignment_distance_weight=float(args.distance_weight),
        same_finger_bonus=float(args.same_finger_bonus),
        reassignment_penalty=float(args.reassignment_penalty),
        finger_crossing_penalty=float(args.finger_crossing_penalty),
        wrong_hand_penalty=float(args.wrong_hand_penalty),
        wrong_hand_split_key=int(args.wrong_hand_split_key),
        assignment_dynamic_hand_split=bool(args.assignment_dynamic_hand_split),
        assignment_dynamic_hand_split_min_span=int(args.assignment_dynamic_hand_split_min_span),
        assignment_dynamic_hand_split_min_keys=int(args.assignment_dynamic_hand_split_min_keys),
        large_jump_penalty=float(args.large_jump_penalty),
        same_key_same_finger_bonus=float(args.same_key_same_finger_bonus),
        assignment_strategy=str(args.assignment_strategy),
        assignment_hand_zone_weight=float(args.assignment_hand_zone_weight),
        assignment_finger_zone_weight=float(args.assignment_finger_zone_weight),
        assignment_hold_weight=float(args.assignment_hold_weight),
        assignment_reach_weight=float(args.assignment_reach_weight),
        assignment_black_key_weight=float(args.assignment_black_key_weight),
        assignment_hard_hand_split=bool(args.assignment_hard_hand_split),
        assignment_middle_key=int(args.assignment_middle_key),
        assignment_reach_soft_limit=float(args.assignment_reach_soft_limit),
        assignment_top_k=int(args.assignment_top_k),
        assignment_top_k_extra_penalty=float(args.assignment_top_k_extra_penalty),
        assignment_beam_width=int(args.assignment_beam_width),
        assignment_candidates_per_step=int(args.assignment_candidates_per_step),
        assignment_fail_if_unassigned=bool(args.assignment_fail_if_unassigned),
        assignment_unassigned_penalty=float(args.assignment_unassigned_penalty),
        assignment_ik_residual_weight=float(args.assignment_ik_residual_weight),
        assignment_ik_max_residual_weight=float(args.assignment_ik_max_residual_weight),
        assignment_ik_failure_penalty=float(args.assignment_ik_failure_penalty),
        assignment_motion_weight=float(args.assignment_motion_weight),
        anchor_stride=int(args.anchor_stride),
        anchor_change_threshold=float(args.anchor_change_threshold),
        solve_contact_window_only=bool(solve_contact_window_only),
        include_midpoint_anchors=not bool(args.no_include_midpoint_anchors),
        ik_fingertip_weight=float(args.ik_fingertip_weight),
        ik_key_front_weight=float(args.ik_key_front_weight),
        ik_key_width_weight=float(args.ik_key_width_weight),
        ik_key_height_weight=float(args.ik_key_height_weight),
        ik_smoothness_weight=float(args.ik_smoothness_weight),
        ik_neutral_weight=float(args.ik_neutral_weight),
        ik_inactive_fingertip_clearance_weight=float(args.ik_inactive_fingertip_clearance_weight),
        ik_inactive_fingertip_clearance=float(args.ik_inactive_fingertip_clearance),
        ik_unassigned_fingertip_strategy=str(args.ik_unassigned_fingertip_strategy),
        ik_unassigned_fingertip_avoidance_weight=float(args.ik_unassigned_fingertip_avoidance_weight),
        ik_unassigned_fingertip_avoidance_radius=float(args.ik_unassigned_fingertip_avoidance_radius),
        ik_wrong_key_xy_avoidance_weight=float(args.ik_wrong_key_xy_avoidance_weight),
        ik_wrong_key_xy_avoidance_radius=float(args.ik_wrong_key_xy_avoidance_radius),
        ik_max_nfev=int(args.ik_max_nfev),
        residual_success_threshold=float(args.residual_success_threshold),
        ik_multistart_on_failure=not bool(args.disable_ik_multistart_on_failure),
        ik_multistart_seed_count=int(args.ik_multistart_seed_count),
        ik_multistart_forearm_tx_grid=int(args.ik_multistart_forearm_tx_grid),
        ik_static_contact_validation=bool(args.ik_static_contact_validation),
        ik_static_contact_settle_steps=int(args.ik_static_contact_settle_steps),
        ik_static_contact_wrong_key_weight=float(args.ik_static_contact_wrong_key_weight),
        ik_static_contact_missed_key_weight=float(args.ik_static_contact_missed_key_weight),
        ik_static_contact_residual_weight=float(args.ik_static_contact_residual_weight),
        ik_static_contact_failure_weight=float(args.ik_static_contact_failure_weight),
        rhapsody_ik_enabled=bool(args.enable_rhapsody_ik),
        rhapsody_ik_checkpoint=str(args.rhapsody_ik_checkpoint),
        rhapsody_ik_refinement_steps=int(args.rhapsody_ik_refinement_steps),
        rhapsody_ik_refinement_lr=float(args.rhapsody_ik_refinement_lr),
        rhapsody_ik_device=str(args.rhapsody_ik_device),
        rhapsody_ik_candidate_scoring=bool(args.rhapsody_ik_candidate_scoring),
        rhapsody_ik_coordinate_transform=str(args.rhapsody_ik_coordinate_transform),
        rhapsody_ik_y_offset=float(args.rhapsody_ik_y_offset),
        rhapsody_ik_fill_inactive_from_previous=not bool(args.no_rhapsody_ik_fill_inactive_from_previous),
        rhapsody_ik_seed_max_active_error=float(args.rhapsody_ik_seed_max_active_error),
        rhapsody_ik_seed_require_previous_improvement=not bool(
            args.no_rhapsody_ik_seed_require_previous_improvement
        ),
        key_target_front_offset=float(args.key_target_front_offset),
        key_target_top_offset=float(args.key_target_top_offset),
        enable_trajectory_refinement=bool(args.enable_trajectory_refinement),
        trajectory_refinement_window_frames=int(args.trajectory_refinement_window_frames),
        trajectory_refinement_max_nfev=int(args.trajectory_refinement_max_nfev),
        trajectory_refinement_fingertip_weight=float(args.trajectory_refinement_fingertip_weight),
        trajectory_refinement_velocity_weight=float(args.trajectory_refinement_velocity_weight),
        trajectory_refinement_acceleration_weight=float(args.trajectory_refinement_acceleration_weight),
        trajectory_refinement_jerk_weight=float(args.trajectory_refinement_jerk_weight),
        trajectory_refinement_neutral_weight=float(args.trajectory_refinement_neutral_weight),
        trajectory_refinement_endpoint_weight=float(args.trajectory_refinement_endpoint_weight),
        output_root=str(Path(args.output_root).expanduser()),
    )
    plan = plan_target_keys(target_keys, config=config)
    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=args.run_name, prefix="impromptu")
    trajectory_path = run_dir / "trajectory.npz"
    metadata_path = run_dir / "metadata.json"
    payload = plan.npz_payload()
    payload.update(
        active_window_crop_start_frame=np.asarray(crop.start_frame, dtype=np.int64),
        active_window_crop_end_frame=np.asarray(crop.end_frame, dtype=np.int64),
        active_window_original_steps=np.asarray(crop.metadata["original_steps"], dtype=np.int64),
        active_window_cropped_steps=np.asarray(crop.metadata["cropped_steps"], dtype=np.int64),
    )
    atomic_save_npz(trajectory_path, **payload)
    metadata = {
        **source_meta,
        "run_dir": str(run_dir),
        "trajectory_npz": str(trajectory_path),
        "seed": int(args.seed),
        "preset": args.preset,
        "active_window": crop.metadata,
        **crop.metadata,
        **plan.metadata,
    }
    atomic_save_json(metadata_path, metadata)
    atomic_save_json(run_dir / "active_window_summary.json", crop.metadata)
    print(f"Wrote Impromptu trajectory: {run_dir}")
    print(f"waypoints={plan.waypoint_frames.size} anchors={plan.ik_anchor_frames_dense.size} ik_success={metadata['ik_success_count']}")


if __name__ == "__main__":
    main()
