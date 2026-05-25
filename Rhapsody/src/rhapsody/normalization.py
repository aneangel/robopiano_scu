from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rhapsody.constants import FINGERTIP_COORD_DIM, HAND_STATE_DIM, NUM_FINGERS
from rhapsody.data import RPIKArrays


@dataclass(slots=True)
class RhapsodyNormalizer:
    qpos_mean: torch.Tensor
    qpos_std: torch.Tensor
    fingertip_mean: torch.Tensor
    fingertip_std: torch.Tensor

    @classmethod
    def fit(cls, arrays: RPIKArrays, *, eps: float = 1.0e-6) -> "RhapsodyNormalizer":
        qpos = np.concatenate([arrays.previous_qpos, arrays.expert_qpos], axis=0)
        tips = arrays.target_fingertips.reshape(len(arrays), NUM_FINGERS * FINGERTIP_COORD_DIM)
        qpos_mean = torch.from_numpy(qpos.mean(axis=0).astype(np.float32))
        qpos_std = torch.from_numpy(np.maximum(qpos.std(axis=0), eps).astype(np.float32))
        tip_mean = torch.from_numpy(tips.mean(axis=0).astype(np.float32))
        tip_std = torch.from_numpy(np.maximum(tips.std(axis=0), eps).astype(np.float32))
        return cls(
            qpos_mean=qpos_mean,
            qpos_std=qpos_std,
            fingertip_mean=tip_mean,
            fingertip_std=tip_std,
        )

    def to(self, device: torch.device | str) -> "RhapsodyNormalizer":
        return RhapsodyNormalizer(
            qpos_mean=self.qpos_mean.to(device),
            qpos_std=self.qpos_std.to(device),
            fingertip_mean=self.fingertip_mean.to(device),
            fingertip_std=self.fingertip_std.to(device),
        )

    def normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        return (qpos - self.qpos_mean) / self.qpos_std

    def denormalize_qpos(self, qpos_norm: torch.Tensor) -> torch.Tensor:
        return qpos_norm * self.qpos_std + self.qpos_mean

    def normalize_fingertips(self, fingertips: torch.Tensor) -> torch.Tensor:
        flat = fingertips.reshape(fingertips.shape[0], NUM_FINGERS * FINGERTIP_COORD_DIM)
        return (flat - self.fingertip_mean) / self.fingertip_std

    def denormalize_fingertips(self, fingertips_norm: torch.Tensor) -> torch.Tensor:
        flat_in = fingertips_norm.reshape(
            fingertips_norm.shape[0], NUM_FINGERS * FINGERTIP_COORD_DIM
        )
        flat = flat_in * self.fingertip_std + self.fingertip_mean
        return flat.reshape(flat.shape[0], NUM_FINGERS, FINGERTIP_COORD_DIM)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "qpos_mean": self.qpos_mean.detach().cpu(),
            "qpos_std": self.qpos_std.detach().cpu(),
            "fingertip_mean": self.fingertip_mean.detach().cpu(),
            "fingertip_std": self.fingertip_std.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "RhapsodyNormalizer":
        return cls(
            qpos_mean=torch.as_tensor(state["qpos_mean"], dtype=torch.float32),
            qpos_std=torch.as_tensor(state["qpos_std"], dtype=torch.float32),
            fingertip_mean=torch.as_tensor(state["fingertip_mean"], dtype=torch.float32),
            fingertip_std=torch.as_tensor(state["fingertip_std"], dtype=torch.float32),
        )


def validate_normalizer_shape(normalizer: RhapsodyNormalizer) -> None:
    if tuple(normalizer.qpos_mean.shape) != (HAND_STATE_DIM,):
        raise ValueError(f"qpos_mean must have shape [{HAND_STATE_DIM}]")
    if tuple(normalizer.qpos_std.shape) != (HAND_STATE_DIM,):
        raise ValueError(f"qpos_std must have shape [{HAND_STATE_DIM}]")
    if tuple(normalizer.fingertip_mean.shape) != (NUM_FINGERS * FINGERTIP_COORD_DIM,):
        raise ValueError("fingertip_mean has wrong shape")
    if tuple(normalizer.fingertip_std.shape) != (NUM_FINGERS * FINGERTIP_COORD_DIM,):
        raise ValueError("fingertip_std has wrong shape")
