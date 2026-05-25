#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fugue.constants import DEFAULT_OUTPUT_ROOT, DEFAULT_RP1M_ROOT, DEFAULT_SONG_KEY  # noqa: E402
from fugue.data import (  # noqa: E402
    FugueActionDataset,
    NormalizationStats,
    SampleConfig,
    fit_normalization_stats,
    validate_demo_split,
    write_dataset_audit,
)
from fugue.models import ModelConfig, build_model  # noqa: E402
from fugue.training import TrainingConfig, action_reconstruction_loss, resolve_device, save_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-validate Fugue train configs on real RP1M data.")
    parser.add_argument("--config", action="append", default=None, help="Config YAML path. Repeatable.")
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--song-key", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "validation"))
    parser.add_argument("--dataset-artifact-root", default=None)
    parser.add_argument("--max-train-demos", type=int, default=1)
    parser.add_argument("--max-val-demos", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_paths = [Path(path) for path in args.config] if args.config else sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    failed = False
    for config_path in config_paths:
        try:
            result = validate_config(config_path=config_path, args=args, output_root=output_root)
            print(
                f"PASS {config_path.name}: feature_shape={result['feature_shape']} "
                f"pred_shape={result['pred_shape']} loss={result['loss']:.6f}"
            )
        except Exception as exc:
            failed = True
            result = {"config": str(config_path), "status": "failed", "error": repr(exc)}
            print(f"FAIL {config_path.name}: {exc}", file=sys.stderr)
        results.append(result)
    report = {"results": results, "num_configs": len(results), "num_failed": sum(row["status"] != "passed" for row in results)}
    save_json(output_root / "validation_report.json", report)
    if failed:
        raise SystemExit(1)


def validate_config(*, config_path: Path, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    dataset_cfg = dict(config.get("dataset", {}))
    song_keys = _dataset_song_keys(dataset_cfg, override=args.song_key)
    song_key = str(song_keys[0])
    artifact_root = Path(args.dataset_artifact_root) if args.dataset_artifact_root else output_root / "dataset"
    _ensure_dataset_artifacts(
        dataset_root=args.rp1m_root,
        artifact_root=artifact_root,
        song_keys=song_keys,
        dataset_cfg=dataset_cfg,
    )
    manifest = pd.read_csv(artifact_root / "manifest.csv")
    validate_demo_split(manifest)
    stats = NormalizationStats.load(artifact_root / "normalization.json")
    sample_config = SampleConfig.from_dict(config.get("sample"))
    training_config = replace(TrainingConfig.from_dict(config.get("training")), batch_size=int(args.batch_size), device=str(args.device))
    dt = float(dataset_cfg.get("dt", stats.dt))
    max_train_demos_per_song = dataset_cfg.get("max_train_demos_per_song")
    max_val_demos_per_song = dataset_cfg.get("max_val_demos_per_song")
    train_dataset = FugueActionDataset(
        dataset_root=args.rp1m_root,
        manifest=manifest,
        stats=stats,
        song_key=song_key,
        split="train",
        sample_config=sample_config,
        dt=dt,
        max_demos=args.max_train_demos,
        max_demos_per_song=None if max_train_demos_per_song is None else int(max_train_demos_per_song),
    )
    val_dataset = FugueActionDataset(
        dataset_root=args.rp1m_root,
        manifest=manifest,
        stats=stats,
        song_key=song_key,
        split="val",
        sample_config=sample_config,
        dt=dt,
        max_demos=args.max_val_demos,
        max_demos_per_song=None if max_val_demos_per_song is None else int(max_val_demos_per_song),
    )
    train_loader = DataLoader(train_dataset, batch_size=int(training_config.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(training_config.batch_size), shuffle=False, num_workers=0)
    model_cfg = ModelConfig.from_dict(config.get("model"))
    model = build_model(
        config=model_cfg,
        input_dim=int(train_dataset.feature_dim),
        action_dim=int(train_dataset.action_dim),
        chunk_horizon=int(sample_config.chunk_horizon),
    )
    device = resolve_device(training_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config.lr), weight_decay=float(training_config.weight_decay))
    batch = next(iter(train_loader))
    features = batch["features"].to(device).float()
    actions = batch["actions"].to(device).float()
    press_weight = batch["press_weight"].to(device).float()
    pred = model(features)
    if pred.shape != actions.shape:
        raise ValueError(f"Prediction shape {tuple(pred.shape)} does not match target {tuple(actions.shape)}")
    loss = action_reconstruction_loss(
        pred,
        actions,
        press_weight=press_weight,
        press_weight_scale=training_config.press_weight_scale,
        smoothness_weight=training_config.smoothness_weight,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if float(training_config.grad_clip) > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config.grad_clip))
    optimizer.step()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        val_pred = model(val_batch["features"].to(device).float())
    if val_pred.shape != val_batch["actions"].shape:
        raise ValueError(f"Validation prediction shape {tuple(val_pred.shape)} does not match target {tuple(val_batch['actions'].shape)}")
    return {
        "config": str(config_path),
        "status": "passed",
        "song_key": song_key,
        "song_keys": [str(value) for value in song_keys],
        "num_songs": int(len(song_keys)),
        "feature_mode": sample_config.feature_mode,
        "model_type": model_cfg.type,
        "chunk_horizon": int(sample_config.chunk_horizon),
        "input_dim": int(train_dataset.feature_dim),
        "train_num_samples": int(len(train_dataset)),
        "val_num_samples": int(len(val_dataset)),
        "feature_shape": list(features.shape),
        "target_shape": list(actions.shape),
        "pred_shape": list(pred.shape),
        "val_pred_shape": list(val_pred.shape),
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss": float(loss.detach().cpu()),
    }


def _ensure_dataset_artifacts(
    *,
    dataset_root: str,
    artifact_root: Path,
    song_keys: list[str],
    dataset_cfg: dict[str, Any],
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "manifest.csv"
    summary_path = artifact_root / "dataset_summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        write_dataset_audit(
            dataset_root=dataset_root,
            output_root=artifact_root,
            song_keys=song_keys,
            train_frac=float(dataset_cfg.get("train_frac", 0.70)),
            val_frac=float(dataset_cfg.get("val_frac", 0.15)),
            test_frac=float(dataset_cfg.get("test_frac", 0.15)),
            seed=int(dataset_cfg.get("split_seed", 7)),
            split_by_song=bool(dataset_cfg.get("split_by_song", False)),
        )
    stats_path = artifact_root / "normalization.json"
    if not stats_path.exists():
        manifest = pd.read_csv(manifest_path)
        stats = fit_normalization_stats(
            dataset_root=dataset_root,
            manifest=manifest,
            song_key=str(song_keys[0]),
            dt=float(dataset_cfg.get("dt", 0.05)),
            max_demos=(
                None
                if dataset_cfg.get("normalization_max_demos") is None
                else int(dataset_cfg.get("normalization_max_demos"))
            ),
            max_demos_per_song=(
                None
                if dataset_cfg.get("normalization_max_demos_per_song", dataset_cfg.get("max_train_demos_per_song"))
                is None
                else int(dataset_cfg.get("normalization_max_demos_per_song", dataset_cfg.get("max_train_demos_per_song")))
            ),
        )
        stats.save(stats_path)


def _dataset_song_keys(dataset_cfg: dict[str, Any], *, override: str | None) -> list[str]:
    if override:
        return [str(override)]
    raw = dataset_cfg.get("song_keys")
    if raw is None:
        return [str(dataset_cfg.get("song_key") or DEFAULT_SONG_KEY)]
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = [str(value).strip() for value in raw if str(value).strip()]
    if not values:
        raise ValueError("dataset.song_keys was provided but empty")
    return values


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return payload


if __name__ == "__main__":
    main()
