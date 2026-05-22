from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nocturne.wandb_logging import finish_run, init_training_run, log_metrics


class ActionMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.05,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ):
        super().__init__()
        mean = np.zeros((int(input_dim),), dtype=np.float32) if feature_mean is None else np.asarray(feature_mean, dtype=np.float32).reshape(-1)
        std = np.ones((int(input_dim),), dtype=np.float32) if feature_std is None else np.asarray(feature_std, dtype=np.float32).reshape(-1)
        if mean.shape != (int(input_dim),) or std.shape != (int(input_dim),):
            raise ValueError("feature_mean/std must match input_dim")
        self.register_buffer("feature_mean", torch.from_numpy(mean))
        self.register_buffer("feature_std", torch.from_numpy(np.maximum(std, 1e-6).astype(np.float32)))
        layers: list[torch.nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(int(num_layers), 1)):
            layers.append(torch.nn.Linear(dim, int(hidden_dim)))
            layers.append(torch.nn.GELU())
            if float(dropout) > 0:
                layers.append(torch.nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        layers.append(torch.nn.Linear(dim, int(action_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.feature_mean) / self.feature_std)


@dataclass(slots=True)
class TrainSummary:
    best_val_loss: float
    best_train_loss: float
    best_checkpoint: str


def train_mlp(
    dataset_npz: str | Path,
    output_root: str | Path,
    *,
    epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.05,
    device: str = "cpu",
    wandb_project: str = "robopianist",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = "Nocturne",
    wandb_notes: str | None = None,
    wandb_mode: str | None = None,
    wandb_tags: list[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    with np.load(Path(dataset_npz), allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
        split = np.asarray(data["split"], dtype=np.int64)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")
    x = torch.from_numpy(features).to(torch_device)
    y = torch.from_numpy(actions).to(torch_device)
    w = torch.from_numpy(weights).to(torch_device)
    train_mask = torch.from_numpy(split == 0).to(torch_device)
    val_mask = torch.from_numpy(split == 1).to(torch_device)
    if not bool(torch.any(val_mask)):
        val_mask = train_mask
    train_features = features[split == 0]
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = train_features.std(axis=0).astype(np.float32)
    model = ActionMLP(
        features.shape[1],
        actions.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(torch_device)
    model_hparams = {"hidden_dim": int(hidden_dim), "num_layers": int(num_layers), "dropout": float(dropout)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    run_config = {
        "module": "Nocturne",
        "task": "stitched_controller_training",
        "dataset_npz": str(Path(dataset_npz)),
        "output_root": str(output),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "requested_device": str(device),
        "resolved_device": str(torch_device),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions.shape[1]),
        "num_frames": int(features.shape[0]),
        "train_frames": int(np.count_nonzero(split == 0)),
        "val_frames": int(np.count_nonzero(split == 1)),
        "normalization": "train_split_feature_mean_std",
        "wandb_required": True,
        "wandb_group": wandb_group,
    }
    wandb_run = init_training_run(
        output_root=output,
        config=run_config,
        project=str(wandb_project),
        entity=wandb_entity,
        name=wandb_name,
        group=wandb_group,
        notes=wandb_notes,
        mode=wandb_mode,
        tags=wandb_tags,
    )
    best_val = float("inf")
    best_train = float("inf")
    history: list[dict[str, float | int]] = []
    try:
        for epoch in range(1, int(epochs) + 1):
            model.train()
            pred = model(x)
            loss = _weighted_loss(pred[train_mask], y[train_mask], w[train_mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                pred_eval = model(x)
                train_loss = float(_weighted_loss(pred_eval[train_mask], y[train_mask], w[train_mask]).detach().cpu())
                val_loss = float(_weighted_loss(pred_eval[val_mask], y[val_mask], w[val_mask]).detach().cpu())
                action_l2 = float(torch.mean(torch.linalg.norm(pred_eval, dim=-1)).detach().cpu())
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            _save_checkpoint(checkpoints / "last.pt", model, features.shape[1], actions.shape[1], epoch, train_loss, val_loss, model_hparams)
            if train_loss < best_train:
                best_train = train_loss
                _save_checkpoint(checkpoints / "best_train.pt", model, features.shape[1], actions.shape[1], epoch, train_loss, val_loss, model_hparams)
            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(checkpoints / "best.pt", model, features.shape[1], actions.shape[1], epoch, train_loss, val_loss, model_hparams)
            log_metrics(
                wandb_run,
                {
                    "epoch": int(epoch),
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "best/train_loss": float(best_train),
                    "best/val_loss": float(best_val),
                    "control/pred_action_l2": action_l2,
                    "optim/lr": float(lr),
                    "optim/grad_norm_clipped": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
                },
                step=epoch,
            )
        _write_history(output / "training_history.csv", history)
        summary = {
            "best_val_loss": best_val,
            "best_train_loss": best_train,
            "best_checkpoint": str(checkpoints / "best.pt"),
            "wandb_run_id": str(getattr(wandb_run, "id", "")),
            "wandb_run_name": str(getattr(wandb_run, "name", "")),
            "wandb_run_url": str(getattr(wandb_run, "url", "")),
            "wandb_project": str(wandb_project),
            "wandb_entity": str(wandb_entity or ""),
            "wandb_group": str(wandb_group or ""),
            "wandb_mode": str(getattr(wandb_run, "mode", wandb_mode or "")),
        }
        summary_path = output / "training_summary.json"
        summary_path.write_text(__import__("json").dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        finish_run(
            wandb_run,
            summary=summary,
            files=[
                summary_path,
                output / "training_history.csv",
                checkpoints / "best.pt",
                checkpoints / "best_train.pt",
                checkpoints / "last.pt",
            ],
        )
        return summary
    except Exception:
        try:
            wandb_run.finish(exit_code=1)
        except Exception:
            pass
        raise


def load_policy(checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu") -> ActionMLP:
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    hparams = dict(checkpoint.get("model_hparams", {}))
    if "feature_mean" in checkpoint and "feature_std" in checkpoint:
        hparams["feature_mean"] = checkpoint["feature_mean"]
        hparams["feature_std"] = checkpoint["feature_std"]
    model = ActionMLP(int(checkpoint["input_dim"]), int(checkpoint["action_dim"]), **hparams)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _weighted_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=-1)
    weighted = torch.mean(mse * weights)
    residual = 1e-4 * torch.mean(pred**2)
    smoothness = torch.tensor(0.0, device=pred.device)
    if pred.shape[0] > 1:
        smoothness = 1e-3 * torch.mean((pred[1:] - pred[:-1]) ** 2)
    return weighted + residual + smoothness


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    input_dim: int,
    action_dim: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    model_hparams: dict[str, float | int],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": int(input_dim),
            "action_dim": int(action_dim),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "model_hparams": dict(model_hparams),
            "feature_mean": model.feature_mean.detach().cpu().numpy(),
            "feature_std": model.feature_std.detach().cpu().numpy(),
        },
        path,
    )


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)
