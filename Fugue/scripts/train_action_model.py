#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
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
from fugue.plots import plot_per_dim_mse, plot_training_curves  # noqa: E402
from fugue.training import TrainingConfig, fit_model, save_json  # noqa: E402
from fugue.wandb import WandbRun, add_wandb_arguments, apply_wandb_cli_overrides  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Fugue RP1M action reconstruction model.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "model_a_stateless.yaml"))
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--song-key", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-artifact-root", default=None)
    parser.add_argument("--delta", type=int, default=None)
    parser.add_argument("--alignment-sweep", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-demos", type=int, default=None)
    parser.add_argument("--max-val-demos", type=int, default=None)
    add_wandb_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    config = apply_wandb_cli_overrides(config, args)
    dataset_cfg = dict(config.get("dataset", {}))
    song_keys = _dataset_song_keys(dataset_cfg, override=args.song_key)
    song_key = str(song_keys[0])
    run_root = Path(args.output_root) if args.output_root else _default_run_root()
    dataset_artifact_root = Path(args.dataset_artifact_root) if args.dataset_artifact_root else run_root / "dataset"
    dataset_artifact_root.mkdir(parents=True, exist_ok=True)
    _ensure_dataset_artifacts(
        dataset_root=args.rp1m_root,
        artifact_root=dataset_artifact_root,
        song_keys=song_keys,
        dataset_cfg=dataset_cfg,
    )
    if args.alignment_sweep:
        rows = []
        for delta in (-1, 0, 1):
            run_dir = run_root / _delta_label(delta)
            summary = _train_one(
                args=args,
                config=config,
                dataset_root=args.rp1m_root,
                dataset_artifact_root=dataset_artifact_root,
                output_root=run_dir,
                song_key=song_key,
                song_keys=song_keys,
                delta=delta,
            )
            rows.append({"delta": delta, **summary})
        best = min(rows, key=lambda row: float(row["best_val_action_mse"]))
        save_json(run_root / "alignment_summary.json", {"candidates": rows, "best": best})
        print(f"best_delta={best['delta']} best_val_action_mse={best['best_val_action_mse']:.6f}")
    else:
        delta = int(args.delta) if args.delta is not None else int(config.get("sample", {}).get("delta", 0))
        _train_one(
            args=args,
            config=config,
            dataset_root=args.rp1m_root,
            dataset_artifact_root=dataset_artifact_root,
            output_root=run_root,
            song_key=song_key,
            song_keys=song_keys,
            delta=delta,
        )


def _train_one(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_root: str,
    dataset_artifact_root: Path,
    output_root: Path,
    song_key: str,
    song_keys: list[str],
    delta: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(dataset_artifact_root / "manifest.csv")
    validate_demo_split(manifest)
    stats = NormalizationStats.load(dataset_artifact_root / "normalization.json")
    sample_config = replace(SampleConfig.from_dict(config.get("sample")), delta=int(delta))
    training_cfg = TrainingConfig.from_dict(config.get("training"))
    if args.epochs is not None:
        training_cfg = replace(training_cfg, epochs=int(args.epochs))
    if args.batch_size is not None:
        training_cfg = replace(training_cfg, batch_size=int(args.batch_size))
    if args.device is not None:
        training_cfg = replace(training_cfg, device=str(args.device))
    dataset_cfg = dict(config.get("dataset", {}))
    dt = float(dataset_cfg.get("dt", stats.dt))
    max_train_demos = args.max_train_demos if args.max_train_demos is not None else dataset_cfg.get("max_train_demos")
    max_val_demos = args.max_val_demos if args.max_val_demos is not None else dataset_cfg.get("max_val_demos")
    max_train_demos_per_song = dataset_cfg.get("max_train_demos_per_song")
    max_val_demos_per_song = dataset_cfg.get("max_val_demos_per_song")
    augmentation_seed = int(dataset_cfg.get("augmentation_seed", dataset_cfg.get("split_seed", 0)))
    train_dataset = FugueActionDataset(
        dataset_root=dataset_root,
        manifest=manifest,
        stats=stats,
        song_key=song_key,
        split="train",
        sample_config=sample_config,
        dt=dt,
        max_demos=None if max_train_demos is None else int(max_train_demos),
        max_demos_per_song=None if max_train_demos_per_song is None else int(max_train_demos_per_song),
        augment=True,
        augmentation_seed=augmentation_seed,
    )
    val_dataset = FugueActionDataset(
        dataset_root=dataset_root,
        manifest=manifest,
        stats=stats,
        song_key=song_key,
        split="val",
        sample_config=sample_config,
        dt=dt,
        max_demos=None if max_val_demos is None else int(max_val_demos),
        max_demos_per_song=None if max_val_demos_per_song is None else int(max_val_demos_per_song),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.batch_size),
        shuffle=True,
        num_workers=int(training_cfg.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg.batch_size),
        shuffle=False,
        num_workers=int(training_cfg.num_workers),
    )
    model_cfg = ModelConfig.from_dict(config.get("model"))
    model = build_model(
        config=model_cfg,
        input_dim=int(train_dataset.feature_dim),
        action_dim=int(train_dataset.action_dim),
        chunk_horizon=int(sample_config.chunk_horizon),
    )
    checkpoint_payload = {
        "config": config,
        "dataset_root": str(dataset_root),
        "dataset_artifact_root": str(dataset_artifact_root),
        "song_key": str(song_key),
        "song_keys": [str(value) for value in song_keys],
        "num_songs": int(len(song_keys)),
        "max_train_demos": None if max_train_demos is None else int(max_train_demos),
        "max_val_demos": None if max_val_demos is None else int(max_val_demos),
        "max_train_demos_per_song": (
            None if max_train_demos_per_song is None else int(max_train_demos_per_song)
        ),
        "max_val_demos_per_song": None if max_val_demos_per_song is None else int(max_val_demos_per_song),
        "dt": float(dt),
        "sample_config": sample_config.to_dict(),
        "model_config": asdict(model_cfg),
        "training_config": training_cfg.to_dict(),
        "normalization": stats.to_dict(),
        "input_dim": int(train_dataset.feature_dim),
        "action_dim": int(train_dataset.action_dim),
        "train_num_samples": int(len(train_dataset)),
        "val_num_samples": int(len(val_dataset)),
        "oracle_warning": (
            "This run uses true future held-out hand states and is an oracle diagnostic."
            if sample_config.feature_mode == "inverse"
            else None
        ),
        "planner_conditioning_note": (
            "This run trains one-step inverse dynamics from current hand state to a target hand state. "
            "Deployable rollout must supply the current state from the simulator and target states from a planner."
            if sample_config.feature_mode == "planner_next"
            else None
        ),
    }
    save_json(output_root / "run_config.json", {k: v for k, v in checkpoint_payload.items() if k != "normalization"})
    wandb_run = WandbRun(
        config.get("wandb"),
        run_name=str(args.wandb_run_name or _wandb_run_name(output_root, sample_config)),
        config_payload=checkpoint_payload,
        job_type="train",
        tags=["fugue", sample_config.feature_mode, f"delta_{delta}", f"num_songs_{len(song_keys)}"],
    )
    try:
        result = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_cfg,
            output_root=output_root,
            checkpoint_payload=checkpoint_payload,
            wandb_run=wandb_run,
        )
        plot_training_curves(output_root / "training_history.csv", output_root / "plots" / "training_curves.png")
        best_metrics = torch.load(output_root / "checkpoints" / "best.pt", map_location="cpu")["val_metrics"]
        plot_per_dim_mse(best_metrics, output_root / "plots" / "per_dim_mse.png")
        wandb_run.log_artifact_bundle(
            artifact_name=f"{output_root.name}-training-artifacts",
            artifact_type="fugue-training-run",
            entries={
                "run_config": output_root / "run_config.json",
                "training_summary": output_root / "training_summary.json",
                "training_history": output_root / "training_history.csv",
                "best_checkpoint": output_root / "checkpoints" / "best.pt",
                "plots": output_root / "plots",
            },
            aliases=["latest", f"delta-{delta}"],
            metadata={"delta": int(delta), "feature_mode": sample_config.feature_mode},
        )
    finally:
        wandb_run.finish()
    return result["summary"]


def _ensure_dataset_artifacts(
    *,
    dataset_root: str,
    artifact_root: Path,
    song_keys: list[str],
    dataset_cfg: dict[str, Any],
) -> None:
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


def _default_run_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"run_{stamp}"


def _delta_label(delta: int) -> str:
    if delta < 0:
        return f"delta_m{abs(delta)}"
    if delta > 0:
        return f"delta_p{delta}"
    return "delta_0"


def _wandb_run_name(output_root: Path, sample_config: SampleConfig) -> str:
    return f"fugue-{sample_config.feature_mode}-{output_root.parent.name}-{output_root.name}"


if __name__ == "__main__":
    main()
