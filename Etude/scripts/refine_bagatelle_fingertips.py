from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Bagatelle-conditioned residual correction head on top of an existing Etude follower."
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
    from etude.evaluation.tracker_eval_utils import build_tracker_controller_from_checkpoint_payload
    from etude.robopianist.env_factory import make_robopianist_env
    from etude.robopianist.state_mapping import resolve_mapping_from_env
    from etude.training.bagatelle_refinement import (
        build_refinement_checkpoint,
        build_refinement_controller,
        build_stage1_training_batch,
        create_or_validate_correction_model,
        determine_feature_dim,
        evaluate_refinement_candidate,
        load_refinement_state,
        train_refinement_epoch,
    )
    from etude.evaluation.tracker_eval_utils import resolve_device

    config = load_experiment_config(Path(args.config))
    trajectory = load_qpos_trajectory(args.trajectory)
    state = load_refinement_state(args.init_checkpoint)

    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_robopianist_env()
    mapping = resolve_mapping_from_env(env)
    base_controller = build_tracker_controller_from_checkpoint_payload(mapping, state.base_checkpoint)
    base_rollout = rollout_controller(
        env,
        base_controller,
        mapping,
        np.asarray(trajectory["q_ref"], dtype=np.float32),
        np.asarray(trajectory["qdot_ref"], dtype=np.float32),
        metadata=trajectory.get("metadata"),
    )
    feature_dim = determine_feature_dim(config, base_rollout, trajectory.get("metadata", {}))
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()
    correction_model = create_or_validate_correction_model(
        config=config,
        state=state,
        feature_dim=feature_dim,
        action_dim=mapping.action_dim,
    )
    device = torch.device(resolve_device(config.get("controller", {}), config))
    correction_model.to(device)

    env = make_robopianist_env()
    mapping = resolve_mapping_from_env(env)
    controller = build_refinement_controller(mapping, config, state, correction_model)
    best_metrics, best_rollout = evaluate_refinement_candidate(
        env,
        controller,
        mapping,
        trajectory,
        config,
        stage="fingertip",
    )
    best_state_dict = {key: value.detach().cpu().clone() for key, value in correction_model.state_dict().items()}
    history = [{"epoch": 0, "train_loss": 0.0, "metrics": best_metrics}]

    optimizer = torch.optim.AdamW(
        correction_model.parameters(),
        lr=float(config["training"].get("lr", 1.0e-5)),
        weight_decay=float(config["training"].get("weight_decay", 1.0e-5)),
    )
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    residual_l2_weight = float(config.get("training", {}).get("residual_l2_weight", 0.05))

    for epoch in range(1, int(config["training"].get("epochs", 5)) + 1):
        train_rollout = best_rollout if epoch == 1 else evaluate_refinement_candidate(
            env,
            controller,
            mapping,
            trajectory,
            config,
            stage="fingertip",
        )[1]
        features, targets = build_stage1_training_batch(
            controller,
            train_rollout,
            trajectory,
            mapping.action_dim,
            gain=float(config.get("refinement", {}).get("heuristic_gain", 0.25)),
            error_scale_cm=float(config.get("refinement", {}).get("error_scale_cm", 2.0)),
        )
        train_loss = train_refinement_epoch(
            correction_model,
            optimizer,
            features,
            targets,
            batch_size=batch_size,
            device=device,
            residual_l2_weight=residual_l2_weight,
        )
        metrics, rollout = evaluate_refinement_candidate(
            env,
            controller,
            mapping,
            trajectory,
            config,
            stage="fingertip",
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "metrics": metrics})
        if _is_better(metrics, best_metrics, primary="fingertip/active_l2_mean", mode="min", ties=(
            "piano/false_events",
            "control/action_l2",
            "tracking/joint_mse",
        )):
            best_metrics = metrics
            best_rollout = rollout
            best_state_dict = {key: value.detach().cpu().clone() for key, value in correction_model.state_dict().items()}

    correction_model.load_state_dict(best_state_dict)
    checkpoint = build_refinement_checkpoint(
        stage="fingertip",
        config=config,
        state=state,
        model=correction_model,
        input_dim=feature_dim,
        action_dim=mapping.action_dim,
        metrics=best_metrics,
    )
    torch.save(checkpoint, checkpoint_dir / "best_fingertip_refined.pt")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    (output_root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez_compressed(output_root / "rollout.npz", **best_rollout)
    (output_root / "summary.md").write_text(_summary_markdown(best_metrics, args, history), encoding="utf-8")

    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def _is_better(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    *,
    primary: str,
    mode: str,
    ties: tuple[str, ...],
) -> bool:
    candidate_primary = float(candidate.get(primary, float("inf") if mode == "min" else float("-inf")))
    incumbent_primary = float(incumbent.get(primary, float("inf") if mode == "min" else float("-inf")))
    if mode == "min":
        if candidate_primary < incumbent_primary:
            return True
        if candidate_primary > incumbent_primary:
            return False
    else:
        if candidate_primary > incumbent_primary:
            return True
        if candidate_primary < incumbent_primary:
            return False
    for key in ties:
        c = float(candidate.get(key, float("inf")))
        i = float(incumbent.get(key, float("inf")))
        if c < i:
            return True
        if c > i:
            return False
    return False


def _summary_markdown(metrics: dict[str, float], args: argparse.Namespace, history: list[dict[str, object]]) -> str:
    primary = metrics.get("fingertip/active_l2_mean")
    lines = [
        "# Bagatelle Fingertip Refinement",
        "",
        f"- Init checkpoint: `{args.init_checkpoint}`",
        f"- Trajectory: `{args.trajectory}`",
        f"- Candidate epochs evaluated: `{len(history)}`",
        "- Result: trained and selected a Bagatelle-conditioned residual correction head.",
    ]
    if primary is not None:
        lines.append(f"- Primary metric `fingertip/active_l2_mean`: `{primary:.6f}`")
    return "\n".join(lines) + "\n"


def _read_initial_dim(state, key: str) -> int:
    import torch

    payload = torch.load(Path(state.source_path), map_location="cpu")
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Refinement checkpoint missing integer {key}")
    return int(value)


if __name__ == "__main__":
    main()
