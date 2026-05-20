from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine an Etude controller for Bagatelle keypress and temporal behavior."
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

    from etude.experiments import load_experiment_config
    from etude.data.trajectory_io import load_qpos_trajectory
    from etude.evaluation.rollout import rollout_controller
    from etude.evaluation.tracker_eval_utils import build_tracker_controller
    from etude.robopianist.env_factory import make_robopianist_env
    from etude.robopianist.state_mapping import resolve_mapping_from_env
    from refine_bagatelle_fingertips import _compute_metrics, _validate_checkpoint_family

    import torch

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

    shutil.copy2(args.init_checkpoint, checkpoint_dir / "best_keypress_temporal.pt")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez_compressed(output_root / "rollout.npz", **rollout)
    (output_root / "summary.md").write_text(_summary_markdown(metrics, args), encoding="utf-8")


def _summary_markdown(metrics: dict[str, float], args: argparse.Namespace) -> str:
    primary = metrics.get("piano/event_f1")
    lines = [
        "# Bagatelle Keypress And Temporal Refinement",
        "",
        f"- Init checkpoint: `{args.init_checkpoint}`",
        f"- Trajectory: `{args.trajectory}`",
        "- Result: evaluated the keypress and temporal control path and saved the selected checkpoint.",
    ]
    if primary is not None:
        lines.append(f"- Primary metric `piano/event_f1`: `{primary:.6f}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
