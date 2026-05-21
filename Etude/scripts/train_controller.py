from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from etude.experiments import load_experiment_config
from etude.controllers.factory import normalize_controller_family
from etude.data.feature_builder import FeatureSpec
from etude.data.rp1m_tracking_dataset import RP1MTrackingDataset
from etude.training.bc_trainer import train_bc_epoch
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

    dataset = RP1MTrackingDataset(
        config["data"]["dataset_root"],
        sequence_length=int(config["data"].get("sequence_length", 1)),
        feature_spec=_build_tracking_feature_spec(config),
        feature_mode=_feature_mode(config),
        feature_config=_feature_config(config),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 256)),
        shuffle=True,
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
    best_loss = float("inf")
    for epoch in range(1, int(config["training"].get("epochs", 30)) + 1):
        result = train_bc_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch} train_loss={result.train_loss:.6f}")
        if result.train_loss < best_loss:
            best_loss = result.train_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "input_dim": input_dim,
                    "action_dim": action_dim,
                },
                checkpoint_dir / "best.pt",
            )


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
