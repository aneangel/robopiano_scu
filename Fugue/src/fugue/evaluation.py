from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fugue.data import (
    FugueActionDataset,
    NormalizationStats,
    SampleConfig,
    build_feature_vector,
    compute_press_mask,
    load_demo_arrays,
    normalize_episode,
    open_zarr_root,
    unstandardize,
    valid_timesteps,
)
from fugue.models import build_model
from fugue.training import evaluate_model, resolve_device


def load_checkpoint_model(path: str | Path, *, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=resolve_device(device))
    model = build_model(
        config=checkpoint["model_config"],
        input_dim=int(checkpoint["input_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        chunk_horizon=int(checkpoint["sample_config"]["chunk_horizon"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolve_device(device))
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    dataset_artifact_root: str | Path,
    split: str = "test",
    batch_size: int = 2048,
    device: str = "cpu",
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
    artifact_root = Path(dataset_artifact_root)
    manifest = pd.read_csv(artifact_root / "manifest.csv")
    stats = NormalizationStats.load(artifact_root / "normalization.json")
    sample_config = SampleConfig.from_dict(checkpoint["sample_config"])
    dataset = FugueActionDataset(
        dataset_root=dataset_root,
        manifest=manifest,
        stats=stats,
        song_key=str(checkpoint["song_key"]),
        split=split,
        sample_config=sample_config,
        dt=float(checkpoint.get("dt", stats.dt)),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    metrics = evaluate_model(model, loader, resolve_device(device))
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "split": split,
            "num_samples": int(len(dataset)),
            "feature_dim": int(dataset.feature_dim),
        }
    )
    return metrics


def predict_demo_actions(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    dataset_root: str | Path,
    demo_id: int,
    device: str = "cpu",
) -> dict[str, Any]:
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    sample_config = SampleConfig.from_dict(checkpoint["sample_config"])
    root = open_zarr_root(dataset_root)
    group = root[str(checkpoint["song_key"])]
    raw = load_demo_arrays(group, demo_id=int(demo_id), dt=float(checkpoint.get("dt", stats.dt)))
    episode = normalize_episode(raw, stats)
    actions = np.zeros_like(raw["actions"], dtype=np.float32)
    counts = np.zeros((actions.shape[0], 1), dtype=np.float32)
    model.eval()
    model_device = resolve_device(device)
    model.to(model_device)
    with torch.no_grad():
        for t in valid_timesteps(actions.shape[0], sample_config):
            feature = build_feature_vector(episode, t=t, config=sample_config)
            feature_tensor = torch.from_numpy(feature[None]).to(model_device).float()
            pred_norm = model(feature_tensor).detach().cpu().numpy()[0]
            pred = unstandardize(pred_norm.reshape(-1, actions.shape[-1]), stats.action_mean, stats.action_std)
            for offset in range(pred.shape[0]):
                out_t = int(t + sample_config.delta + offset)
                if 0 <= out_t < actions.shape[0]:
                    actions[out_t] += pred[offset]
                    counts[out_t] += 1.0
    valid = counts[:, 0] > 0.0
    actions[valid] = actions[valid] / counts[valid]
    actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
    press_mask = compute_press_mask(raw["goals"], window=sample_config.press_window)
    return {
        "predicted_actions": actions,
        "counts": counts,
        "valid_prediction_mask": valid.astype(np.float32),
        "reference_actions": raw["actions"],
        "hand_joints": raw["q"],
        "qvel": raw["qvel"],
        "goals": raw["goals"],
        "piano_states": raw.get("piano_states"),
        "fingertips": raw.get("fingertips"),
        "press_mask": press_mask,
        "demo_id": int(demo_id),
    }


def save_npz_prediction(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
    scalar_payload = {key: value for key, value in payload.items() if not isinstance(value, np.ndarray)}
    arrays["metadata_json"] = np.asarray(json.dumps(scalar_payload, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return path
