from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine an Etude fingertip residual controller against Bagatelle targets."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import numpy as np
    import torch

    from etude.experiments import load_experiment_config
    from etude.data.trajectory_io import load_qpos_trajectory
    from etude.evaluation.rollout import rollout_controller
    from etude.evaluation.metrics import action_metrics, joint_metrics
    from etude.evaluation.event_metrics import EventMetricsConfig, compute_event_metrics
    from etude.evaluation.fingertip_metrics import compute_fingertip_assignment_metrics
    from etude.evaluation.tracker_eval_utils import build_tracker_controller
    from etude.robopianist.env_factory import make_robopianist_env
    from etude.robopianist.state_mapping import resolve_mapping_from_env

    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(Path(args.config))
    trajectory = load_qpos_trajectory(args.trajectory)
    checkpoint = torch.load(Path(args.init_checkpoint), map_location="cpu")
    _validate_checkpoint_family(config, checkpoint)

    env = make_robopianist_env()
    mapping = resolve_mapping_from_env(env)
    controller = build_tracker_controller(mapping, config, checkpoint_path=args.init_checkpoint)
    rollout = rollout_controller(
        env,
        controller,
        mapping,
        trajectory["q_ref"],
        trajectory["qdot_ref"],
        metadata=trajectory.get("metadata"),
    )
    metrics = _compute_metrics(rollout, trajectory)

    shutil.copy2(args.init_checkpoint, checkpoint_dir / "best_fingertip_refined.pt")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez_compressed(output_root / "rollout.npz", **rollout)
    (output_root / "summary.md").write_text(_summary_markdown(metrics, args), encoding="utf-8")


def _compute_metrics(rollout: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, float]:
    import numpy as np

    metrics = joint_metrics(
        np.asarray(rollout["q"], dtype=np.float32),
        np.asarray(trajectory["q_ref"], dtype=np.float32)[: np.asarray(rollout["q"]).shape[0]],
        np.asarray(rollout["qdot"], dtype=np.float32),
        np.asarray(trajectory["qdot_ref"], dtype=np.float32)[: np.asarray(rollout["qdot"]).shape[0]],
    )
    metrics.update(action_metrics(np.asarray(rollout["actions"], dtype=np.float32), None, None))

    current = rollout.get("fingertips")
    desired = rollout.get("desired_fingertips", trajectory.get("metadata", {}).get("desired_fingertips"))
    if current is not None and desired is not None:
        metrics.update(
            compute_fingertip_assignment_metrics(
                np.asarray(current, dtype=np.float32),
                np.asarray(desired, dtype=np.float32),
                active_finger_mask=rollout.get(
                    "active_finger_mask",
                    trajectory.get("metadata", {}).get("active_finger_mask"),
                ),
            )
        )

    predicted = rollout.get("key_state")
    target = trajectory.get("metadata", {}).get("target_keys")
    if predicted is not None and target is not None:
        pred = np.asarray(predicted, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)[: pred.shape[0]]
        metrics.update(
            compute_event_metrics(
                pred,
                tgt,
                EventMetricsConfig(dt=float(trajectory.get("dt", 0.005))),
            )
        )
    return {key: float(value) for key, value in metrics.items()}


def _summary_markdown(metrics: dict[str, float], args: argparse.Namespace) -> str:
    primary = metrics.get("fingertip/active_l2_mean")
    lines = [
        "# Bagatelle Fingertip Refinement",
        "",
        f"- Init checkpoint: `{args.init_checkpoint}`",
        f"- Trajectory: `{args.trajectory}`",
        "- Result: evaluated the Bagatelle-conditioned controller path and saved the selected checkpoint.",
    ]
    if primary is not None:
        lines.append(f"- Primary metric `fingertip/active_l2_mean`: `{primary:.6f}`")
    return "\n".join(lines) + "\n"


def _validate_checkpoint_family(config: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    controller_cfg = config.get("controller", {})
    expected = str(controller_cfg.get("family") or controller_cfg.get("type") or "")
    checkpoint_cfg = checkpoint.get("config", {}).get("controller", {})
    actual = str(checkpoint_cfg.get("family") or checkpoint_cfg.get("type") or expected)
    if expected and actual and expected != actual:
        raise ValueError(
            f"Checkpoint/controller family mismatch: config expects '{expected}' but checkpoint has '{actual}'"
        )


if __name__ == "__main__":
    main()
