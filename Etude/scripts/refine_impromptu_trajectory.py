from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine an Etude controller checkpoint against an Impromptu-derived planned trajectory."
    )
    parser.add_argument("--config")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--trajectory")
    parser.add_argument("--output-root")
    parser.add_argument("--run-name")
    parser.add_argument("--job-manifest")
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()

    if args.job_manifest is not None or args.job_index is not None:
        if args.job_manifest is None or args.job_index is None:
            parser.error("--job-manifest and --job-index must be provided together")
        row = _load_manifest_row(Path(args.job_manifest), args.job_index)
        args.config = args.config or row.get("config")
        args.init_checkpoint = args.init_checkpoint or row.get("init_checkpoint")
        args.trajectory = args.trajectory or row.get("trajectory")
        args.output_root = args.output_root or row.get("output_root")
        args.run_name = args.run_name or row.get("run_name")

    missing = [
        name
        for name in ("config", "init_checkpoint", "trajectory", "output_root", "run_name")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import numpy as np
    import torch

    from etude.data.trajectory_io import load_qpos_trajectory
    from etude.evaluation.rollout import rollout_controller
    from etude.evaluation.tracker_eval_utils import build_tracker_controller_from_checkpoint_payload
    from etude.evaluation.tracker_eval_utils import resolve_device
    from etude.experiments import load_experiment_config
    from etude.robopianist.env_factory import make_robopianist_env
    from etude.robopianist.state_mapping import resolve_mapping_from_env
    from etude.training.bagatelle_refinement import (
        build_refinement_checkpoint,
        build_refinement_controller,
        build_stage2_training_batch,
        compute_rollout_objective_metrics,
        create_or_validate_correction_model,
        determine_feature_dim,
        evaluate_refinement_candidate,
        load_refinement_state,
        train_refinement_epoch,
    )

    config = load_experiment_config(Path(args.config))
    trajectory = load_qpos_trajectory(args.trajectory)
    state = load_refinement_state(args.init_checkpoint)

    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env_from_config(config)
    mapping = resolve_mapping_from_env(env)
    base_controller = build_tracker_controller_from_checkpoint_payload(mapping, state.base_checkpoint)
    bootstrap_rollout = rollout_controller(
        env,
        base_controller,
        mapping,
        np.asarray(trajectory["q_ref"], dtype=np.float32),
        np.asarray(trajectory["qdot_ref"], dtype=np.float32),
        metadata=trajectory.get("metadata"),
    )
    feature_dim = determine_feature_dim(config, bootstrap_rollout, trajectory.get("metadata", {}))
    _close_env(env)

    correction_model = create_or_validate_correction_model(
        config=config,
        state=state,
        feature_dim=feature_dim,
        action_dim=mapping.action_dim,
    )
    initial_state_dict = {key: value.detach().cpu().clone() for key, value in correction_model.state_dict().items()}
    device = torch.device(resolve_device(config.get("controller", {}), config))
    correction_model.to(device)

    env = _make_env_from_config(config)
    mapping = resolve_mapping_from_env(env)
    controller = build_refinement_controller(mapping, config, state, correction_model)
    best_metrics, best_rollout = evaluate_refinement_candidate(
        env,
        controller,
        mapping,
        trajectory,
        config,
        stage="keypress_temporal",
    )
    target_keys = trajectory.get("metadata", {}).get("target_keys")
    if target_keys is not None:
        best_metrics.update(
            compute_rollout_objective_metrics(
                best_rollout,
                np.asarray(target_keys, dtype=np.float32)[: best_rollout["q"].shape[0]],
                trajectory,
                config=config,
            )
        )
    best_epoch = 0
    best_state_dict = {key: value.detach().cpu().clone() for key, value in correction_model.state_dict().items()}
    history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": 0.0, "metrics": best_metrics}]

    optimizer = torch.optim.AdamW(
        correction_model.parameters(),
        lr=float(config["training"].get("lr", 1.0e-5)),
        weight_decay=float(config["training"].get("weight_decay", 1.0e-5)),
    )
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    residual_l2_weight = float(config.get("training", {}).get("residual_l2_weight", 0.05))
    epochs = int(config["training"].get("epochs", 10))

    for epoch in range(1, epochs + 1):
        train_rollout = best_rollout if epoch == 1 else evaluate_refinement_candidate(
            env,
            controller,
            mapping,
            trajectory,
            config,
            stage="keypress_temporal",
        )[1]
        features, targets = build_stage2_training_batch(
            controller,
            train_rollout,
            trajectory,
            mapping.action_dim,
            gain=float(config.get("refinement", {}).get("heuristic_gain", 0.35)),
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
            stage="keypress_temporal",
        )
        if target_keys is not None:
            metrics.update(
                compute_rollout_objective_metrics(
                    rollout,
                    np.asarray(target_keys, dtype=np.float32)[: rollout["q"].shape[0]],
                    trajectory,
                    config=config,
                )
            )
        history.append({"epoch": epoch, "train_loss": train_loss, "metrics": metrics})
        if _is_better(metrics, best_metrics, config):
            best_metrics = metrics
            best_rollout = rollout
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in correction_model.state_dict().items()}

    correction_model.load_state_dict(best_state_dict)
    weights_changed = _state_dict_changed(initial_state_dict, best_state_dict)
    best_metrics["refinement/best_epoch"] = float(best_epoch)
    best_metrics["refinement/weights_changed"] = float(1.0 if weights_changed else 0.0)
    best_metrics["refinement/history_length"] = float(len(history))

    checkpoint = build_refinement_checkpoint(
        stage="impromptu_playback",
        config=config,
        state=state,
        model=correction_model,
        input_dim=feature_dim,
        action_dim=mapping.action_dim,
        metrics=best_metrics,
    )
    torch.save(checkpoint, checkpoint_dir / "best_refined.pt")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    (output_root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez_compressed(output_root / "rollout.npz", **best_rollout)
    (output_root / "summary.md").write_text(
        _summary_markdown(best_metrics, args, history, weights_changed, best_epoch),
        encoding="utf-8",
    )

    _close_env(env)


def _load_manifest_row(path: Path, job_index: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if job_index < 0 or job_index >= len(rows):
        raise SystemExit(f"job_index {job_index} outside manifest range 0..{len(rows) - 1}")
    return rows[job_index]


def _is_better(candidate: dict[str, float], incumbent: dict[str, float], config: dict[str, Any]) -> bool:
    refinement = config.get("refinement", {}) if isinstance(config.get("refinement"), dict) else {}
    primary = str(refinement.get("selection_metric", "piano/event_f1"))
    mode = str(refinement.get("selection_mode", "max"))
    fallback = str(refinement.get("fallback_selection_metric", "tracking/joint_pos_rmse"))
    ties = tuple(refinement.get("tie_break_metrics", ())) if isinstance(refinement.get("tie_break_metrics"), list) else ()
    primary_key = primary if primary in candidate and primary in incumbent else fallback
    return _compare_metric_tuple(candidate, incumbent, primary_key, mode if primary_key == primary else "min", ties)


def _compare_metric_tuple(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    primary: str,
    mode: str,
    ties: tuple[str, ...],
) -> bool:
    candidate_primary = float(candidate.get(primary, float("-inf") if mode == "max" else float("inf")))
    incumbent_primary = float(incumbent.get(primary, float("-inf") if mode == "max" else float("inf")))
    if mode == "max":
        if candidate_primary > incumbent_primary:
            return True
        if candidate_primary < incumbent_primary:
            return False
    else:
        if candidate_primary < incumbent_primary:
            return True
        if candidate_primary > incumbent_primary:
            return False
    for key in ties:
        c = float(candidate.get(key, float("inf")))
        i = float(incumbent.get(key, float("inf")))
        if c < i:
            return True
        if c > i:
            return False
    return False


def _state_dict_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    import torch

    for key, before_value in before.items():
        after_value = after.get(key)
        if after_value is None:
            return True
        if not torch.equal(before_value.cpu(), after_value.cpu()):
            return True
    return False


def _summary_markdown(
    metrics: dict[str, float],
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    weights_changed: bool,
    best_epoch: int,
) -> str:
    lines = [
        "# Impromptu Trajectory Refinement",
        "",
        f"- Run name: `{args.run_name}`",
        f"- Init checkpoint: `{args.init_checkpoint}`",
        f"- Trajectory: `{args.trajectory}`",
        f"- Candidate epochs evaluated: `{len(history)}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Model weights updated: `{'true' if weights_changed else 'false'}`",
        "- Result: trained a residual correction head from the warm-start Etude checkpoint and selected the best rollout candidate.",
    ]
    for key in (
        "piano/event_f1",
        "piano/missed_events",
        "piano/false_events",
        "piano/timing_abs_error_mean_s",
        "tracking/joint_pos_rmse",
        "fingertip/active_l2_mean",
        "action/l2_mean",
        "rollout_objective/total",
    ):
        if key in metrics:
            lines.append(f"- `{key}`: `{float(metrics[key]):.6f}`")
    if not weights_changed:
        lines.append("")
        lines.append(
            "The selected candidate was epoch 0, so `checkpoints/best_refined.pt` is a refinement-format checkpoint "
            "containing an unchanged residual head, not a copied source checkpoint."
        )
    return "\n".join(lines) + "\n"


def _close_env(env: Any) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def _make_env_from_config(config: dict[str, Any]) -> Any:
    from etude.robopianist.env_factory import make_robopianist_env

    rollout_cfg = config.get("rollout", {}) if isinstance(config.get("rollout"), dict) else {}
    task = (
        rollout_cfg.get("task")
        or config.get("environment_name")
        or "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    )
    kwargs = {}
    if "max_steps" in rollout_cfg:
        kwargs["max_steps"] = rollout_cfg["max_steps"]
    return make_robopianist_env(str(task), **kwargs)


if __name__ == "__main__":
    main()
