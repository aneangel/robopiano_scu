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
from etude.evaluation.metrics import action_metrics, joint_metrics, note_metrics
from etude.evaluation.tracker_eval_utils import build_tracker_controller
from etude.features.fingertip_phase_blocks import FingertipFeatureSpec, PhaseFeatureSpec
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
    metrics = joint_metrics(
        rollout["q"],
        traj["q_ref"][: rollout["q"].shape[0]],
        rollout["qdot"],
        traj["qdot_ref"][: rollout["qdot"].shape[0]],
    )
    metrics.update(action_metrics(rollout["actions"], getattr(mapping, "action_low", None), getattr(mapping, "action_high", None)))
    metrics.update(_fingertip_metrics(rollout, traj))
    metrics.update(_piano_metrics(rollout, traj))

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
    target_keys = np.asarray(target, dtype=np.float32)
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
    if current is None or desired is None:
        desired = traj.get("metadata", {}).get("desired_fingertips")
    if current is None or desired is None:
        return {}
    return compute_fingertip_assignment_metrics(
        current,
        desired,
        active_finger_mask=rollout.get("active_finger_mask", traj.get("metadata", {}).get("active_finger_mask")),
    )


if __name__ == "__main__":
    main()
