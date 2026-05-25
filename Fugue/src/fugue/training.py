from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 2048
    epochs: int = 100
    patience: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    press_weight_scale: float = 0.0
    smoothness_weight: float = 0.0
    num_workers: int = 0
    device: str = "cuda"

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "TrainingConfig":
        payload = dict(values or {})
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(name: str = "cuda") -> torch.device:
    device = torch.device(str(name))
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def action_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    press_weight: torch.Tensor | None = None,
    press_weight_scale: float = 0.0,
    smoothness_weight: float = 0.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    squared = torch.square(pred - target).mean(dim=-1)
    if press_weight is not None and float(press_weight_scale) > 0.0:
        weight = 1.0 + float(press_weight_scale) * torch.clamp(press_weight - 1.0, min=0.0) / 2.0
        loss = (squared * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    else:
        loss = squared.mean()
    if float(smoothness_weight) > 0.0 and pred.shape[1] > 1:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        loss = loss + float(smoothness_weight) * torch.square(pred_diff - target_diff).mean()
    return loss


@dataclass
class MetricAccumulator:
    sse: float = 0.0
    sae: float = 0.0
    count: int = 0
    press_sse: float = 0.0
    press_sae: float = 0.0
    press_count: int = 0
    smooth_sse: float = 0.0
    smooth_count: int = 0
    per_dim_sse: np.ndarray | None = None
    per_dim_sae: np.ndarray | None = None
    per_dim_count: int = 0
    offset_sse: np.ndarray | None = None
    offset_sae: np.ndarray | None = None
    offset_count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, press_weight: torch.Tensor | None) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.float64)
        target_np = target.detach().cpu().numpy().astype(np.float64)
        diff = pred_np - target_np
        abs_diff = np.abs(diff)
        self.sse += float(np.square(diff).sum())
        self.sae += float(abs_diff.sum())
        self.count += int(diff.size)
        per_dim_sse = np.square(diff).sum(axis=(0, 1))
        per_dim_sae = abs_diff.sum(axis=(0, 1))
        if self.per_dim_sse is None:
            self.per_dim_sse = np.zeros_like(per_dim_sse, dtype=np.float64)
            self.per_dim_sae = np.zeros_like(per_dim_sae, dtype=np.float64)
        self.per_dim_sse += per_dim_sse
        self.per_dim_sae += per_dim_sae
        self.per_dim_count += int(diff.shape[0] * diff.shape[1])
        offset_sse = np.square(diff).sum(axis=(0, 2))
        offset_sae = abs_diff.sum(axis=(0, 2))
        if self.offset_sse is None:
            self.offset_sse = np.zeros_like(offset_sse, dtype=np.float64)
            self.offset_sae = np.zeros_like(offset_sae, dtype=np.float64)
        self.offset_sse += offset_sse
        self.offset_sae += offset_sae
        self.offset_count += int(diff.shape[0] * diff.shape[2])
        if press_weight is not None:
            mask = press_weight.detach().cpu().numpy() > 1.0
            if mask.any():
                expanded = np.repeat(mask[..., None], diff.shape[-1], axis=-1)
                self.press_sse += float(np.square(diff[expanded]).sum())
                self.press_sae += float(abs_diff[expanded].sum())
                self.press_count += int(expanded.sum())
        if diff.shape[1] > 1:
            pred_diff = pred_np[:, 1:] - pred_np[:, :-1]
            target_diff = target_np[:, 1:] - target_np[:, :-1]
            smooth = pred_diff - target_diff
            self.smooth_sse += float(np.square(smooth).sum())
            self.smooth_count += int(smooth.size)

    def finalize(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "action_mse": self.sse / max(self.count, 1),
            "action_l1": self.sae / max(self.count, 1),
            "press_action_mse": None if self.press_count == 0 else self.press_sse / self.press_count,
            "press_action_l1": None if self.press_count == 0 else self.press_sae / self.press_count,
            "smoothness_mse": None if self.smooth_count == 0 else self.smooth_sse / self.smooth_count,
            "num_action_values": int(self.count),
            "num_press_action_values": int(self.press_count),
        }
        if self.per_dim_sse is not None and self.per_dim_sae is not None:
            metrics["per_dim_mse"] = (self.per_dim_sse / max(self.per_dim_count, 1)).tolist()
            metrics["per_dim_l1"] = (self.per_dim_sae / max(self.per_dim_count, 1)).tolist()
        if self.offset_sse is not None and self.offset_sae is not None:
            offset_mse = self.offset_sse / max(self.offset_count, 1)
            offset_l1 = self.offset_sae / max(self.offset_count, 1)
            metrics["chunk_offset_mse"] = offset_mse.tolist()
            metrics["chunk_offset_l1"] = offset_l1.tolist()
            for offset, value in enumerate(offset_mse):
                metrics[f"chunk_offset_{offset}_mse"] = float(value)
                metrics[f"chunk_offset_{offset}_l1"] = float(offset_l1[offset])
        return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, Any]:
    model.train()
    accumulator = MetricAccumulator()
    losses = []
    for batch in loader:
        features = batch["features"].to(device).float()
        actions = batch["actions"].to(device).float()
        press_weight = batch.get("press_weight")
        if press_weight is not None:
            press_weight = press_weight.to(device).float()
        pred = model(features)
        loss = action_reconstruction_loss(
            pred,
            actions,
            press_weight=press_weight,
            press_weight_scale=config.press_weight_scale,
            smoothness_weight=config.smoothness_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(config.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        accumulator.update(pred, actions, press_weight)
    metrics = accumulator.finalize()
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device).float()
            actions = batch["actions"].to(device).float()
            press_weight = batch.get("press_weight")
            if press_weight is not None:
                press_weight = press_weight.to(device).float()
            pred = model(features)
            accumulator.update(pred, actions, press_weight)
    return accumulator.finalize()


def fit_model(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    output_root: str | Path,
    checkpoint_payload: dict[str, Any],
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if wandb_run is not None:
        wandb_run.summary(
            {
                "status": "running",
                "output_root": str(output_root),
                "train_num_batches": int(len(train_loader)),
                "val_num_batches": int(len(val_loader)),
            }
        )
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config)
        val_metrics = evaluate_model(model, val_loader, device)
        row = {
            "epoch": int(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items() if not isinstance(value, list)},
            **{f"val_{key}": value for key, value in val_metrics.items() if not isinstance(value, list)},
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(_wandb_epoch_payload(row), step=int(epoch))
        current_val = float(val_metrics["action_mse"])
        payload = {
            **checkpoint_payload,
            "epoch": int(epoch),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "model_state_dict": model.state_dict(),
        }
        torch.save(payload, checkpoint_dir / "last.pt")
        if current_val < best_val:
            best_val = current_val
            best_epoch = int(epoch)
            stale_epochs = 0
            torch.save(payload, checkpoint_dir / "best.pt")
            shutil.copy2(checkpoint_dir / "best.pt", checkpoint_dir / "best_val.pt")
            if wandb_run is not None:
                wandb_run.summary(
                    {
                        "best/epoch": int(best_epoch),
                        "best/val_action_mse": float(best_val),
                        "best/checkpoint": str(checkpoint_dir / "best.pt"),
                    }
                )
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch} train_mse={train_metrics['action_mse']:.6f} "
            f"val_mse={val_metrics['action_mse']:.6f} best_val={best_val:.6f}"
        )
        if stale_epochs >= int(config.patience):
            print(f"Early stopping after {epoch} epochs; best_epoch={best_epoch}")
            break
    history_path = output_root / "training_history.csv"
    write_history_csv(history_path, history)
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_action_mse": float(best_val),
        "epochs_ran": int(len(history)),
        "history_csv": str(history_path),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary({"status": "completed", **summary})
    return {"summary": summary, "history": history}


def write_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _wandb_epoch_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, (int, float, str, bool)):
            payload[key.replace("_", "/")] = value
    return payload
