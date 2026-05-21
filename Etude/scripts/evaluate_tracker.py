from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from etude.data.trajectory_io import load_qpos_trajectory
from etude.evaluation.event_metrics import EventMetricsConfig, compute_event_metrics
from etude.evaluation.fingertip_metrics import compute_fingertip_assignment_metrics
from etude.evaluation.metrics import action_metrics, align_reference, error_profile_metrics, joint_metrics, note_metrics
from etude.evaluation.tracker_eval_utils import build_tracker_controller
from etude.robopianist.env_factory import make_robopianist_env
from etude.robopianist.state_mapping import resolve_mapping_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Etude tracker in RoboPianist.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backend", default="dm_control")
    parser.add_argument("--render-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    del args.backend
    del args.render_video

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.trajectory is None:
        raise ValueError("--trajectory is required until RP1M eval sampling is wired in")

    traj = load_qpos_trajectory(args.trajectory)
    env = make_robopianist_env()
    mapping = resolve_mapping_from_env(env)
    controller = build_tracker_controller(mapping, config, checkpoint_path=args.checkpoint)

    from etude.evaluation.rollout import rollout_controller

    rollout = rollout_controller(
        env,
        controller,
        mapping,
        traj["q_ref"],
        traj["qdot_ref"],
        metadata=traj.get("metadata"),
    )
    reference_indices = rollout.get("reference_indices")
    aligned_q_ref = align_reference(traj["q_ref"], reference_indices)
    aligned_qdot_ref = align_reference(traj["qdot_ref"], reference_indices)
    metrics = joint_metrics(
        rollout["q"],
        aligned_q_ref,
        rollout["qdot"],
        aligned_qdot_ref,
    )
    metrics.update(
        error_profile_metrics(
            rollout["q"],
            aligned_q_ref,
            prefix="tracking/joint",
            dt=float(np.asarray(rollout.get("env_dt", traj.get("dt", 0.005))).item()),
        )
    )
    metrics.update(action_metrics(rollout["actions"], getattr(mapping, "action_low", None), getattr(mapping, "action_high", None)))
    metrics.update(_fingertip_metrics(rollout, traj))
    metrics.update(_piano_metrics(rollout, traj))
    metrics.update(_timing_metrics(rollout))

    diagnostics = getattr(controller, "diagnostics", None)
    if callable(diagnostics):
        metrics.update(diagnostics())

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_root / "rollout.npz", **rollout)
    print(json.dumps(metrics, indent=2))


def _piano_metrics(rollout: dict[str, np.ndarray], traj: dict[str, Any]) -> dict[str, float]:
    predicted = rollout.get("key_state")
    target = traj.get("metadata", {}).get("target_keys")
    if predicted is None or target is None:
        return {}
    predicted_keys = np.asarray(predicted, dtype=np.float32)
    target_keys = align_reference(np.asarray(target, dtype=np.float32), rollout.get("reference_indices"))
    steps = min(predicted_keys.shape[0], target_keys.shape[0])
    if steps == 0:
        return {}
    predicted_keys = predicted_keys[:steps]
    target_keys = target_keys[:steps]
    metrics = note_metrics(predicted_keys, target_keys)
    metrics.update(
        compute_event_metrics(
            predicted_keys,
            target_keys,
            EventMetricsConfig(dt=float(traj.get("dt", 0.005))),
        )
    )
    return metrics


def _fingertip_metrics(rollout: dict[str, np.ndarray], traj: dict[str, Any]) -> dict[str, float]:
    current = rollout.get("fingertips")
    desired = rollout.get("desired_fingertips")
    active_finger_mask = rollout.get("active_finger_mask")
    if current is None or desired is None:
        desired = align_reference(traj.get("metadata", {}).get("desired_fingertips"), rollout.get("reference_indices"))
    if active_finger_mask is None:
        active_finger_mask = align_reference(
            traj.get("metadata", {}).get("active_finger_mask"),
            rollout.get("reference_indices"),
        )
    if current is None or desired is None:
        return {}
    metrics = compute_fingertip_assignment_metrics(
        current,
        desired,
        active_finger_mask=active_finger_mask,
    )
    metrics.update(
        error_profile_metrics(
            np.asarray(current, dtype=np.float32),
            np.asarray(desired, dtype=np.float32),
            prefix="tracking/fingertip",
            dt=float(np.asarray(rollout.get("env_dt", traj.get("dt", 0.005))).item()),
        )
    )
    return metrics


def _timing_metrics(rollout: dict[str, np.ndarray]) -> dict[str, float]:
    trajectory_dt = float(np.asarray(rollout.get("trajectory_dt", 0.0)).item())
    env_dt = float(np.asarray(rollout.get("env_dt", 0.0)).item())
    metrics = {
        "timing/reference_dt_s": trajectory_dt,
        "timing/env_dt_s": env_dt,
    }
    if env_dt > 0.0:
        metrics["timing/controller_rate_hz"] = 1.0 / env_dt
    if trajectory_dt > 0.0:
        metrics["timing/reference_rate_hz"] = 1.0 / trajectory_dt
    if env_dt > 0.0 and trajectory_dt > 0.0:
        metrics["timing/reference_repeat_factor"] = max(trajectory_dt / env_dt, 1.0)
    return metrics


if __name__ == "__main__":
    main()
