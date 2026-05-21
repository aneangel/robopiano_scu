#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
except ImportError as exc:  # pragma: no cover - depends on training environment.
    torch = None
    F = None
    DataLoader = None
    random_split = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "Bagatelle" / "src", REPO_ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from bagatelle.dataset import AssignmentSequenceDataset  # noqa: E402
from bagatelle.sequence_model import build_cost_bias_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Bagatelle sequential cost-bias assigner.")
    parser.add_argument("--trajectory", action="append", required=True, help="Trajectory NPZ or directory. Can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", choices=("gru", "transformer"), default="gru")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-ik", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    return parser


def collate_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["piano_roll"].shape[0]) for item in batch)
    out: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0]}
    masks: list[torch.Tensor] = []
    for item in batch:
        length = int(item["piano_roll"].shape[0])
        masks.append(torch.arange(max_len) < length)
        for key, value in item.items():
            pad_shape = (max_len - length, *value.shape[1:])
            if max_len == length:
                padded = value
            else:
                fill = -1 if key == "assignments" else 0
                padded = torch.cat([value, torch.full(pad_shape, fill, dtype=value.dtype)], dim=0)
            out[key].append(padded)
    stacked = {key: torch.stack(values, dim=0) for key, values in out.items()}
    stacked["mask"] = torch.stack(masks, dim=0)
    return stacked


def model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_type == "gru":
        return {"hidden_size": int(args.hidden_size), "num_layers": int(args.num_layers)}
    return {"d_model": int(args.d_model), "nhead": int(args.nhead), "num_layers": int(args.num_layers)}


def margin_imitation_loss(
    bias: torch.Tensor,
    piano_roll: torch.Tensor,
    assignments: torch.Tensor,
    ik_weights: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    lambda_ik: float,
    lambda_reg: float,
) -> torch.Tensor:
    active_key_mask = piano_roll[:, :, None, :] > float(threshold)
    logits = -bias.masked_fill(~active_key_mask, 1.0e6)
    valid = (assignments >= 0) & mask[:, :, None]
    if not torch.any(valid):
        return bias.square().mean() * float(lambda_reg)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = assignments.reshape(-1)
    flat_valid = valid.reshape(-1)
    ce = F.cross_entropy(flat_logits[flat_valid], flat_targets[flat_valid], reduction="none")
    weights = (1.0 + float(lambda_ik) * ik_weights[:, :, None].expand_as(assignments).reshape(-1)[flat_valid]).to(ce.dtype)
    imitation = torch.mean(ce * weights)
    return imitation + float(lambda_reg) * bias.square().mean()


def run_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, args: argparse.Namespace) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        piano_roll = batch["piano_roll"].to(args.device)
        assignments = batch["assignments"].to(args.device)
        ik_weights = batch["ik_weights"].to(args.device)
        prev_fingertips = batch["prev_fingertips"].to(args.device)
        mask = batch["mask"].to(args.device)
        with torch.set_grad_enabled(training):
            bias = model(piano_roll, prev_fingertips)
            loss = margin_imitation_loss(
                bias,
                piano_roll,
                assignments,
                ik_weights,
                mask,
                threshold=float(args.threshold),
                lambda_ik=float(args.lambda_ik),
                lambda_reg=float(args.lambda_reg),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def main() -> None:
    args = build_parser().parse_args()
    if torch is None or F is None or DataLoader is None or random_split is None:
        raise SystemExit("train_sequence_assigner.py requires PyTorch") from _TORCH_IMPORT_ERROR
    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = AssignmentSequenceDataset(
        args.trajectory,
        window_size=int(args.window_size),
        stride=int(args.stride),
        threshold=float(args.threshold),
    )
    val_count = int(round(len(dataset) * float(args.val_fraction)))
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(int(args.seed))
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=generator) if val_count else (dataset, None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_windows,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_windows)

    model_config = model_config_from_args(args)
    model = build_cost_bias_model(args.model_type, **model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args)
        val_loss = run_epoch(model, val_loader, None, args) if val_loader is not None else train_loss
        scheduler.step()
        checkpoint = {
            "model_type": args.model_type,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.6g}")


if __name__ == "__main__":
    main()
