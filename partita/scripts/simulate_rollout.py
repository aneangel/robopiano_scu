from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PARTITA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PARTITA_ROOT.parent
SRC_ROOT = PARTITA_ROOT / "src"
for import_root in [REPO_ROOT, SRC_ROOT]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

from partita.evaluation.rollout import (
    ACTION_MAPPINGS,
    ACTION_SOURCE_SCALES,
    calibrate_action_scale,
    rollout_recorded_rp1m_episode_with_robopianist,
    rollout_reconstructed_actions_with_robopianist,
)
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, load_json, save_json


_KNOWN_TASK_KWARGS: tuple[str, ...] = (
    "n_steps_lookahead",
    "n_seconds_lookahead",
    "trim_silence",
    "wrong_press_termination",
    "initial_buffer_time",
    "disable_fingering_reward",
    "disable_forearm_reward",
    "disable_colorization",
    "disable_hand_collisions",
    "energy_penalty_coef",
    "randomize_hand_positions",
    "gravity_compensation",
    "primitive_fingertip_collisions",
    "attachment_yaw",
    "change_color_on_activation",
)

_KNOWN_SUITE_LOAD_KWARGS: tuple[str, ...] = (
    "stretch",
    "shift",
    "recompile_physics",
    "legacy_step",
)


def _load_target_npz(data_dir: Path) -> dict[str, np.ndarray]:
    target = np.load(data_dir / "target_trajectory.npz")
    if "goals" not in target.files:
        raise RuntimeError("target_trajectory.npz does not contain goals; cannot synthesize rollout MIDI.")
    return {name: target[name] for name in target.files}


def _filter_kwargs(raw: dict[str, Any] | None, allowed: tuple[str, ...]) -> dict[str, Any]:
    if not raw:
        return {}
    return {key: value for key, value in raw.items() if key in allowed and value is not None}


def _load_calibration(rollout_dir: Path) -> dict[str, Any] | None:
    calibration_path = rollout_dir / "calibration.json"
    if not calibration_path.exists():
        return None
    try:
        return load_json(calibration_path)
    except Exception:
        return None


def _verdict_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply Phase 6 success criteria to a fidelity summary block.

    - Mean per-step ||hand_qpos - rp1m_hand_joints||_2 over the early window < 1e-3 rad.
    - Piano-state F1 vs RP1M piano_states over the full clip > 0.95.
    - F1 vs goals must improve materially over the historical ~0.07 baseline; we flag
      anything below 0.50 as "still drifting" so the validator surfaces a partial fix.
    """
    early = summary.get("hand_qpos_l2_mean_first_n")
    state_block = summary.get("against_rp1m_piano_states") or {}
    state_f1 = state_block.get("key_f1")
    goal_block = summary.get("against_goals") or {}
    goal_f1 = goal_block.get("key_f1")
    checks = {
        "hand_qpos_l2_mean_first_n_lt_1e-3": (
            None if early is None else bool(float(early) < 1e-3)
        ),
        "rp1m_piano_state_f1_gt_0_95": (
            None if state_f1 is None else bool(float(state_f1) > 0.95)
        ),
        "goal_f1_gt_0_50": (
            None if goal_f1 is None else bool(float(goal_f1) > 0.50)
        ),
    }
    decided = [v for v in checks.values() if v is not None]
    passed = bool(decided) and all(decided)
    return {
        "passed": passed,
        "checks": checks,
        "hand_qpos_l2_mean_first_n": early,
        "rp1m_piano_state_f1": state_f1,
        "goal_f1": goal_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Partita target/reconstructed trajectories in RoboPianist and render video.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--which",
        choices=["original", "reconstructed", "original-state", "reconstructed-state", "both", "all"],
        default="both",
        help=(
            "original/reconstructed replay actions; original-state/reconstructed-state restore "
            "hand joints and piano states frame-by-frame; all runs every mode."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument(
        "--action-source-scale",
        choices=list(ACTION_SOURCE_SCALES),
        default=None,
        help="Interpret input actions as normalized [-1, 1] values or actuator units.",
    )
    parser.add_argument(
        "--set-initial-state",
        action="store_true",
        help="Before action replay, restore the initial hand qpos from target_trajectory.npz hand_joints[0].",
    )
    parser.add_argument(
        "--calibrate-action-scale",
        action="store_true",
        help="Sweep action_source_scale x action_mapping against the RP1M target's first frames "
        "and persist the winner to <rollout>/calibration.json. Subsequent runs auto-load it.",
    )
    parser.add_argument(
        "--calibration-probe-steps",
        type=int,
        default=5,
        help="How many steps to probe per (scale, mapping) candidate during calibration.",
    )
    parser.add_argument(
        "--validate-fidelity",
        action="store_true",
        help="Run the original_target rollout with full state restore + calibrated scale and "
        "check the Phase 6 success criteria. Writes fidelity_summary.json and exits non-zero on failure.",
    )
    parser.add_argument(
        "--ignore-calibration",
        action="store_true",
        help="Skip auto-loading <rollout>/calibration.json even if it exists.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exp = experiment_name(config)
    root = output_root(config)
    data_dir = root / "data" / exp
    recon_dir = root / "reconstruction" / exp
    rollout_dir = ensure_dir(root / "rollout" / exp)
    selection = load_json(data_dir / "selection.json")
    target = _load_target_npz(data_dir)
    goals = np.asarray(target["goals"], dtype=np.float32)
    target_song_name = selection.get("target_song_name", selection.get("song_name"))
    threshold = float(selection.get("key_threshold", config.get("selection", {}).get("key_threshold", 0.5)))
    control_timestep = float(config.get("control_timestep", 0.05))
    rollout_cfg = config.get("rollout", {}) or {}
    reduced_action_space = bool(rollout_cfg.get("reduced_action_space", True))
    action_source_scale = str(
        args.action_source_scale
        if args.action_source_scale is not None
        else rollout_cfg.get("action_source_scale", "normalized_minus_one_to_one")
    )
    action_mapping = str(rollout_cfg.get("action_mapping", "as_is"))
    require_exact_action_dim = bool(rollout_cfg.get("require_exact_action_dim", True))
    restore_initial_state = False
    set_initial_state = bool(args.set_initial_state)
    prefer_canonical_midi = bool(rollout_cfg.get("prefer_canonical_midi", False))
    extra_task_kwargs = _filter_kwargs(rollout_cfg.get("task_kwargs"), _KNOWN_TASK_KWARGS)
    suite_load_kwargs = _filter_kwargs(rollout_cfg.get("suite_load_kwargs"), _KNOWN_SUITE_LOAD_KWARGS)

    reference_hand_joints = (
        np.asarray(target["hand_joints"], dtype=np.float32)
        if "hand_joints" in target
        else None
    )
    reference_piano_states = (
        np.asarray(target["piano_states"], dtype=np.float32)
        if "piano_states" in target
        else None
    )

    if action_source_scale not in ACTION_SOURCE_SCALES:
        raise ValueError(
            f"rollout.action_source_scale='{action_source_scale}' not one of {list(ACTION_SOURCE_SCALES)}"
        )
    if action_mapping not in ACTION_MAPPINGS:
        raise ValueError(
            f"rollout.action_mapping='{action_mapping}' not one of {list(ACTION_MAPPINGS)}"
        )

    if args.calibrate_action_scale:
        if reference_hand_joints is None:
            raise RuntimeError(
                "Calibration requires target_trajectory.npz to include hand_joints. "
                "Re-run partita data extraction with hand_joints recorded."
            )
        original_actions = np.load(recon_dir / "original_actions.npy")
        print(
            f"Calibrating action scale x mapping over {args.calibration_probe_steps} probe steps..."
        )
        calibration = calibrate_action_scale(
            actions=np.asarray(original_actions, dtype=np.float32),
            reference_hand_joints=reference_hand_joints,
            reference_piano_states=reference_piano_states,
            song_name=target_song_name,
            control_timestep=control_timestep,
            seed=0,
            reduced_action_space=reduced_action_space,
            extra_task_kwargs=extra_task_kwargs,
            suite_load_kwargs=suite_load_kwargs,
            prefer_canonical_midi=prefer_canonical_midi,
            probe_steps=int(args.calibration_probe_steps),
            output_dir=rollout_dir,
            label="calibration",
            key_threshold=threshold,
        )
        best = calibration.get("best") or {}
        print(
            "Calibration winner: "
            f"scale={best.get('action_source_scale')} mapping={best.get('action_mapping')} "
            f"hand_qpos_l2_mean={best.get('hand_qpos_l2_mean')}"
        )
        if best:
            action_source_scale = str(best.get("action_source_scale", action_source_scale))
            action_mapping = str(best.get("action_mapping", action_mapping))

    if not args.calibrate_action_scale and not args.ignore_calibration and args.action_source_scale is None:
        calibration = _load_calibration(rollout_dir)
        if calibration and calibration.get("best"):
            best = calibration["best"]
            action_source_scale = str(best.get("action_source_scale", action_source_scale))
            action_mapping = str(best.get("action_mapping", action_mapping))
            print(
                "Loaded calibration: "
                f"scale={action_source_scale} mapping={action_mapping} "
                f"hand_qpos_l2_mean={best.get('hand_qpos_l2_mean')}"
            )

    fidelity_validation_requested = args.validate_fidelity
    if fidelity_validation_requested:
        if reference_hand_joints is None:
            raise RuntimeError(
                "--validate-fidelity requires target_trajectory.npz to include hand_joints."
            )
        if args.which not in {"original", "both", "all"}:
            args.which = "original"
        restore_initial_state = True
        set_initial_state = False

    action_jobs: list[tuple[str, np.ndarray]] = []
    if args.which in {"original", "both", "all"}:
        action_jobs.append(("original_target", np.load(recon_dir / "original_actions.npy")))
    if args.which in {"reconstructed", "both", "all"}:
        action_jobs.append(("reconstructed", np.load(recon_dir / "reconstructed_actions.npy")))

    results = []
    for label, actions in action_jobs:
        initial_hand_joints = None
        if set_initial_state:
            if reference_hand_joints is None or reference_hand_joints.shape[0] == 0:
                raise RuntimeError("--set-initial-state requires target_trajectory.npz hand_joints.")
            initial_hand_joints = reference_hand_joints[0]
        print(
            f"Rendering {label} rollout ({actions.shape[0]} steps) "
            f"scale={action_source_scale} mapping={action_mapping} "
            f"restore={restore_initial_state} set_initial={set_initial_state}..."
        )
        result = rollout_reconstructed_actions_with_robopianist(
            actions=actions,
            goals=goals,
            song_name=target_song_name,
            output_dir=rollout_dir,
            label=label,
            control_timestep=control_timestep,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            render_every=args.render_every,
            seed=0,
            reduced_action_space=reduced_action_space,
            action_source_scale=action_source_scale,
            require_exact_action_dim=require_exact_action_dim,
            action_mapping=action_mapping,
            reference_hand_joints=reference_hand_joints,
            reference_piano_states=reference_piano_states,
            restore_initial_state=restore_initial_state,
            initial_hand_joints=initial_hand_joints,
            key_threshold=threshold,
            extra_task_kwargs=extra_task_kwargs,
            suite_load_kwargs=suite_load_kwargs,
            prefer_canonical_midi=prefer_canonical_midi,
        )
        results.append(result)
        print(f"  video: {result.get('video_path')}")
        print(f"  reward: {result.get('total_reward')} executed={result.get('actions_executed')} terminated={result.get('terminated')}")
        if "rollout_key_f1" in result:
            print(
                "  rollout_key_f1: "
                f"{result.get('rollout_key_f1')} "
                f"precision={result.get('rollout_key_precision')} "
                f"recall={result.get('rollout_key_recall')}"
            )
        fidelity = result.get("fidelity") or {}
        if fidelity.get("hand_qpos_available"):
            print(
                "  fidelity: "
                f"hand_qpos_l2_mean_first_n={fidelity.get('hand_qpos_l2_mean_first_n')} "
                f"hand_qpos_l2_max={fidelity.get('hand_qpos_l2_max')} "
                f"piano_state_iou_mean={fidelity.get('piano_state_iou_mean')}"
            )
        if result.get("audio_warning"):
            print(f"  audio_warning: {result.get('audio_warning')}")

    def _missing_state_result(label: str, missing_files: list[str]) -> dict[str, Any]:
        result = {
            "label": label,
            "song_name": target_song_name,
            "playback_mode": "recorded_hand_joints_and_piano_states",
            "status": "missing_files",
            "missing_files": missing_files,
            "steps_rendered": 0,
            "render_error": None,
            "video_path": None,
        }
        save_json(rollout_dir / f"{label}_playback.json", result)
        return result

    state_jobs: list[tuple[str, np.ndarray, np.ndarray]] = []
    if args.which in {"original-state", "all"}:
        if reference_hand_joints is None or reference_piano_states is None:
            missing = [
                name
                for name, arr in [
                    ("target_trajectory.npz:hand_joints", reference_hand_joints),
                    ("target_trajectory.npz:piano_states", reference_piano_states),
                ]
                if arr is None
            ]
            results.append(_missing_state_result("original_state", missing))
        else:
            state_jobs.append(("original_state", reference_hand_joints, reference_piano_states))
    if args.which in {"reconstructed-state", "all"}:
        hand_path = recon_dir / "reconstructed_hand_joints.npy"
        piano_path = recon_dir / "reconstructed_piano_states.npy"
        missing = [str(path) for path in [hand_path, piano_path] if not path.exists()]
        if missing:
            results.append(_missing_state_result("reconstructed_state", missing))
        else:
            state_jobs.append((
                "reconstructed_state",
                np.load(hand_path),
                np.load(piano_path),
            ))

    for label, hand_joints, piano_states in state_jobs:
        print(f"Rendering {label} playback ({hand_joints.shape[0]} steps)...")
        result = rollout_recorded_rp1m_episode_with_robopianist(
            hand_joints=hand_joints,
            piano_states=piano_states,
            goals=goals,
            song_name=target_song_name,
            output_dir=rollout_dir,
            label=label,
            control_timestep=control_timestep,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            render_every=args.render_every,
            seed=0,
            threshold=threshold,
        )
        results.append(result)
        print(f"  video: {result.get('video_path')}")
        print(
            "  state playback: "
            f"against_goals_f1={(result.get('against_goals') or {}).get('key_f1')} "
            f"against_rp1m_state_f1={(result.get('against_rp1m_piano_states') or {}).get('key_f1')}"
        )
    existing_results = []
    summary_path = rollout_dir / "rollout_summary.json"
    if summary_path.exists():
        try:
            existing = load_json(summary_path)
            existing_results = list(existing.get("results", []))
        except Exception:
            existing_results = []
    allowed_labels = {"original_target", "reconstructed", "original_state", "reconstructed_state"}
    by_label = {
        str(item.get("label")): item
        for item in existing_results
        if isinstance(item, dict) and str(item.get("label")) in allowed_labels
    }
    for item in results:
        by_label[str(item.get("label"))] = item
    combined = list(by_label.values())
    save_json(summary_path, {"experiment_name": exp, "results": combined})
    print(f"Saved rollout summary: {summary_path}")

    if fidelity_validation_requested:
        original_result = next(
            (item for item in results if str(item.get("label")) == "original_target"),
            None,
        )
        if original_result is None:
            raise RuntimeError(
                "--validate-fidelity could not locate the original_target rollout result."
            )
        fidelity = original_result.get("fidelity") or {}
        verdict = _verdict_from_summary(fidelity)
        fidelity_summary = {
            "experiment_name": exp,
            "song_name": target_song_name,
            "action_source_scale": original_result.get("action_source_scale"),
            "action_mapping": original_result.get("action_mapping"),
            "restore_initial_state": original_result.get("restore_initial_state"),
            "prefer_canonical_midi": original_result.get("prefer_canonical_midi"),
            "used_canonical_midi": original_result.get("used_canonical_midi"),
            "task_kwargs": original_result.get("task_kwargs"),
            "fidelity": fidelity,
            "verdict": verdict,
        }
        save_json(rollout_dir / "fidelity_summary.json", fidelity_summary)
        status = "PASS" if verdict["passed"] else "FAIL"
        print(f"Fidelity verdict: {status}")
        for name, value in verdict["checks"].items():
            print(f"  {name}: {value}")
        if not verdict["passed"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
