from __future__ import annotations

import argparse
import csv
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from etude.experiments import load_experiment_config
from etude.controllers.factory import normalize_controller_family
from etude.data.feature_builder import FeatureSpec
from etude.data.rp1m_tracking_dataset import RP1MTrackingDataset
from etude.training.bc_trainer import eval_bc_epoch, train_bc_epoch
from etude.utils.import_utils import load_symbol
from etude.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Etude controller.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(Path(args.config))
    seed_everything(int(config.get("seed", 7)))

    dataset_root = Path(config["data"]["dataset_root"])
    splits_file = dataset_root / "splits.csv"
    has_splits = splits_file.exists()
    if has_splits:
        print(f"Using dataset splits from {splits_file}")
        train_split = "train"
        val_split = "val"
    else:
        print(
            f"WARNING: no splits.csv found at {splits_file}; training on the full manifest "
            "and selecting best.pt by train loss for backward compatibility."
        )
        train_split = None
        val_split = None

    dataset = RP1MTrackingDataset(
        config["data"]["dataset_root"],
        sequence_length=int(config["data"].get("sequence_length", 1)),
        feature_spec=_build_tracking_feature_spec(config),
        feature_mode=_feature_mode(config),
        feature_config=_feature_config(config),
        split=train_split,
        splits_file=splits_file if has_splits else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 256)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    val_loader = None
    if val_split is not None:
        val_dataset = RP1MTrackingDataset(
            config["data"]["dataset_root"],
            sequence_length=int(config["data"].get("sequence_length", 1)),
            feature_spec=_build_tracking_feature_spec(config),
            feature_mode=_feature_mode(config),
            feature_config=_feature_config(config),
            split=val_split,
            splits_file=splits_file,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["data"].get("batch_size", 256)),
            shuffle=False,
            num_workers=int(config["data"].get("num_workers", 0)),
        )

    sample = dataset[0]
    input_dim = int(sample["features"].shape[-1])
    action_dim = int(sample["actions"].shape[-1])
    model = _build_model(config, input_dim=input_dim, action_dim=action_dim)

    device = torch.device(config["training"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["training"].get("epochs", 30)) + 1):
        result = train_bc_epoch(model, loader, optimizer, device)
        val_loss = None
        if val_loader is not None:
            val_result = eval_bc_epoch(model, val_loader, device)
            val_loss = val_result.eval_loss
            print(f"epoch={epoch} train_loss={result.train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"epoch={epoch} train_loss={result.train_loss:.6f}")

        metadata = {
            "epoch": epoch,
            "train_loss": result.train_loss,
            "val_loss": val_loss,
            "selection_metric": "val_loss" if val_loss is not None else "train_loss",
            "selection_mode": "min",
        }
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            config=config,
            input_dim=input_dim,
            action_dim=action_dim,
            metadata=metadata,
        )
        if result.train_loss < best_train_loss:
            best_train_loss = result.train_loss
            _save_checkpoint(
                checkpoint_dir / "best_train.pt",
                model=model,
                config=config,
                input_dim=input_dim,
                action_dim=action_dim,
                metadata={**metadata, "selection_metric": "train_loss"},
            )
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                checkpoint_dir / "best_val.pt",
                model=model,
                config=config,
                input_dim=input_dim,
                action_dim=action_dim,
                metadata={**metadata, "selection_metric": "val_loss"},
            )
        best_alias_source = checkpoint_dir / ("best_val.pt" if val_loader is not None else "best_train.pt")
        if best_alias_source.exists():
            shutil.copy2(best_alias_source, checkpoint_dir / "best.pt")
        history.append({"epoch": epoch, "train_loss": result.train_loss, "val_loss": val_loss})

    _write_history(output_root / "training_history.csv", history)
    summary = {
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss if val_loader is not None else None,
        "best_checkpoint": str(checkpoint_dir / ("best_val.pt" if val_loader is not None else "best_train.pt")),
        "best_alias": str(checkpoint_dir / "best.pt"),
        "split_file": str(splits_file) if has_splits else None,
        "selection_note": "Supervised best checkpoints are preselection only; final selection should use rollout evaluation.",
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    input_dim: int,
    action_dim: int,
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "input_dim": input_dim,
            "action_dim": action_dim,
            **metadata,
        },
        path,
    )


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def _build_model(config: dict[str, Any], *, input_dim: int, action_dim: int) -> torch.nn.Module:
    controller_cfg = _as_dict(config.get("controller"))
    family = normalize_controller_family(
        str(controller_cfg.get("family") or controller_cfg.get("type") or "pd_residual")
    )
    model_module = controller_cfg.get("model_module")
    if not model_module:
        model_module = _default_model_module(controller_cfg, family)
    model_cls = load_symbol(str(model_module))
    if not isinstance(model_cls, type):
        raise TypeError(f"Resolved model_module is not a class: {model_module}")

    signature = inspect.signature(model_cls)
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name == "self":
            continue
        if name == "input_dim":
            kwargs[name] = input_dim
        elif name == "action_dim":
            kwargs[name] = action_dim
        elif name in controller_cfg and controller_cfg[name] is not None:
            kwargs[name] = controller_cfg[name]
    return model_cls(**kwargs)


def _feature_mode(config: dict[str, Any]) -> str:
    controller_cfg = _as_dict(config.get("controller"))
    family = normalize_controller_family(
        str(controller_cfg.get("family") or controller_cfg.get("type") or "pd_residual")
    )
    if family == "key_aware_residual":
        return "key_aware"
    if family == "fingertip_residual":
        return "fingertip_phase"
    if family == "inverse_dynamics":
        return "inverse_dynamics"
    return "tracking"


def _feature_config(config: dict[str, Any]) -> dict[str, Any]:
    feature_cfg = _as_dict(config.get("features"))
    inverse_cfg = _as_dict(config.get("inverse_dynamics"))
    fingertip_cfg = _as_dict(config.get("fingertip_control"))
    phase_cfg = _as_dict(config.get("phase"))
    return {
        "key_spec": _as_dict(_as_dict(feature_cfg.get("block_kwargs")).get("etude.features.key_blocks:build_key_features", {})).get("spec", {}),
        "inverse_dynamics_spec": inverse_cfg,
        "fingertip_spec": {
            "include_current": bool(fingertip_cfg.get("enabled", True)),
            "include_desired": bool(fingertip_cfg.get("enabled", True)),
            "include_error": bool(fingertip_cfg.get("use_fingertip_error_features", True)),
            "include_weights": bool(fingertip_cfg.get("use_fingertip_weights", True)),
            "include_active_mask": bool(fingertip_cfg.get("use_active_finger_mask", True)),
            "include_inactive_mask": bool(fingertip_cfg.get("use_inactive_finger_mask", True)),
            "flatten": bool(fingertip_cfg.get("flatten", True)),
            "allow_missing": bool(fingertip_cfg.get("allow_missing", True)),
        },
        "phase_spec": phase_cfg,
    }


def _build_tracking_feature_spec(config: dict[str, Any]) -> FeatureSpec:
    tracking_cfg = _as_dict(_as_dict(config.get("features")).get("tracking"))
    return FeatureSpec(
        lookahead_steps=tuple(int(step) for step in tracking_cfg.get("lookahead_steps", (1, 5, 10))),
        include_target_keys=bool(tracking_cfg.get("include_target_keys", True)),
        include_fingertips=bool(tracking_cfg.get("include_fingertips", True)),
    )


def _default_model_module(controller_cfg: dict[str, Any], family: str) -> str:
    residual_model = str(controller_cfg.get("residual_model") or controller_cfg.get("model_type") or "mlp").lower()
    if family in {"pd_residual", "temporal_residual"} and residual_model == "gru":
        return "etude.controllers.residual_gru:ResidualGRU"
    if family in {"pd_residual", "key_aware_residual", "fingertip_residual", "inverse_dynamics", "hierarchical"}:
        return "etude.controllers.residual_mlp:ResidualMLP"
    if family == "temporal_residual":
        return "etude.controllers.residual_mlp:ResidualMLP"
    raise ValueError(f"No default model module for controller family {family}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
