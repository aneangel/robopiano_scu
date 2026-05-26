from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ballade.features import build_feature_vector
from ballade.losses import ResidualLossWeights, residual_controller_loss
from ballade.models import ResidualMLPController
from ballade.replay_buffer import OnlineTeacherReplayBuffer


def transitions_to_training_arrays(buffer: OnlineTeacherReplayBuffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    base_actions = []
    selected_actions = []
    for transition in buffer.transitions:
        feature = build_feature_vector(transition.obs, transition.target, transition.base_action)
        features.append(feature.astype(np.float32))
        base_actions.append(np.asarray(transition.base_action, dtype=np.float32).reshape(-1))
        selected_actions.append(np.asarray(transition.selected_action, dtype=np.float32).reshape(-1))
    if not features:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )
    feature_dim = max(feature.size for feature in features)
    padded = np.zeros((len(features), feature_dim), dtype=np.float32)
    for idx, feature in enumerate(features):
        padded[idx, : feature.size] = feature
    return padded, np.stack(base_actions, axis=0), np.stack(selected_actions, axis=0)


def train_residual_controller(
    *,
    teacher_data: str | Path,
    output_root: str | Path,
    epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    hidden_layers: int = 3,
    residual_scale: float = 0.2,
    device: str = "cpu",
) -> dict[str, Any]:
    buffer = OnlineTeacherReplayBuffer.load_shards(teacher_data)
    features, base_actions, selected_actions = transitions_to_training_arrays(buffer)
    if features.shape[0] == 0:
        raise ValueError(f"No teacher transitions found under {teacher_data}")
    model = ResidualMLPController(
        feature_dim=features.shape[1],
        action_dim=base_actions.shape[1],
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        residual_scale=residual_scale,
    ).to(torch.device(device))
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(base_actions).float(),
        torch.from_numpy(selected_actions).float(),
    )
    loader = DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_weights = ResidualLossWeights()
    history = []
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for feature_batch, base_batch, selected_batch in loader:
            feature_batch = feature_batch.to(torch.device(device))
            base_batch = base_batch.to(torch.device(device))
            selected_batch = selected_batch.to(torch.device(device))
            predicted = model(feature_batch, base_batch)
            loss_dict = residual_controller_loss(
                predicted,
                selected_batch,
                base_batch,
                weights=loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            optimizer.step()
            losses.append(float(loss_dict["total"].detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
    output = Path(output_root)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": int(features.shape[1]),
            "action_dim": int(base_actions.shape[1]),
            "hidden_dim": int(hidden_dim),
            "hidden_layers": int(hidden_layers),
            "residual_scale": float(residual_scale),
            "history": history,
        },
        checkpoint_path,
    )
    summary = {
        "teacher_transitions": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(base_actions.shape[1]),
        "epochs": int(epochs),
        "final_loss": history[-1]["loss"],
        "checkpoint": str(checkpoint_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
