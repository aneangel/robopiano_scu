from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rhapsody.config import RhapsodyConfig
from rhapsody.models import ForwardKinematicsSurrogate, ResidualIKPolicy
from rhapsody.normalization import RhapsodyNormalizer
from rhapsody.trainer import RhapsodyTrainingResult


def save_checkpoint(
    path: str | Path,
    result: RhapsodyTrainingResult,
    config: RhapsodyConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "normalizer": result.normalizer.state_dict(),
            "policy_state_dict": _cpu_state_dict(result.policy),
            "fk_state_dict": _cpu_state_dict(result.fk_model),
            "train_metrics": result.train_metrics.to_dict(),
            "validation_metrics": result.validation_metrics.to_dict(),
            "train_song_names": list(result.train_song_names),
            "validation_song_names": list(result.validation_song_names),
            "history": result.history,
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[RhapsodyConfig, RhapsodyNormalizer, ResidualIKPolicy, ForwardKinematicsSurrogate, dict[str, Any]]:
    state = torch.load(Path(path), map_location=device)
    config_dict = dict(state["config"])
    config = RhapsodyConfig(**config_dict)
    normalizer = RhapsodyNormalizer.from_state_dict(state["normalizer"]).to(device)
    policy = ResidualIKPolicy(
        qpos_dim=config.qpos_dim,
        num_fingers=config.num_fingers,
        hidden_dims=tuple(config.policy_hidden_dims),
        action_scale=config.policy_action_scale,
    ).to(device)
    fk_model = ForwardKinematicsSurrogate(
        qpos_dim=config.qpos_dim,
        num_fingers=config.num_fingers,
        hidden_dims=tuple(config.fk_hidden_dims),
    ).to(device)
    policy.load_state_dict(state["policy_state_dict"])
    fk_model.load_state_dict(state["fk_state_dict"])
    return config, normalizer, policy, fk_model, state


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}
