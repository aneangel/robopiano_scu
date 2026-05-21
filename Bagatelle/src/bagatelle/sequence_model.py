from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from bagatelle.assignment import FingerAssignmentResult, assign_fingers_previous_pose

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only in non-ML envs.
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


NUM_KEYS = 88
NUM_FINGERS = 10


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("bagatelle.sequence_model requires PyTorch") from _TORCH_IMPORT_ERROR


class SinusoidalPositionalEncoding(nn.Module if nn is not None else object):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        _require_torch()
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class TransformerCostBias(nn.Module if nn is not None else object):
    """Bidirectional offline planner that predicts [B, W, 10, 88] cost biases."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        _require_torch()
        super().__init__()
        self.input = nn.Linear(NUM_KEYS, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, NUM_FINGERS * NUM_KEYS)

    def forward(self, piano_roll: torch.Tensor, prev_fingertips: torch.Tensor | None = None) -> torch.Tensor:
        del prev_fingertips
        x = self.position(self.input(piano_roll.float()))
        x = self.encoder(x)
        return self.output(x).reshape(piano_roll.shape[0], piano_roll.shape[1], NUM_FINGERS, NUM_KEYS)


class GRUCostBias(nn.Module if nn is not None else object):
    """Streaming-compatible sequence model for [B, W, 10, 88] cost biases."""

    def __init__(
        self,
        *,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        input_prev_fingertips: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        self.input_prev_fingertips = bool(input_prev_fingertips)
        input_size = NUM_KEYS + (NUM_FINGERS * 3 if self.input_prev_fingertips else 0)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, NUM_FINGERS * NUM_KEYS)

    def forward(self, piano_roll: torch.Tensor, prev_fingertips: torch.Tensor | None = None) -> torch.Tensor:
        inputs = [piano_roll.float()]
        if self.input_prev_fingertips:
            if prev_fingertips is None:
                prev_fingertips = torch.zeros(
                    piano_roll.shape[0],
                    piano_roll.shape[1],
                    NUM_FINGERS,
                    3,
                    dtype=piano_roll.dtype,
                    device=piano_roll.device,
                )
            inputs.append(prev_fingertips.reshape(piano_roll.shape[0], piano_roll.shape[1], NUM_FINGERS * 3).float())
        x = torch.cat(inputs, dim=-1)
        y, _ = self.gru(x)
        return self.output(y).reshape(piano_roll.shape[0], piano_roll.shape[1], NUM_FINGERS, NUM_KEYS)


def build_cost_bias_model(model_type: str, **kwargs: Any) -> Any:
    _require_torch()
    normalized = str(model_type).lower()
    if normalized in {"transformer", "transformer_encoder"}:
        return TransformerCostBias(**kwargs)
    if normalized == "gru":
        return GRUCostBias(**kwargs)
    raise ValueError(f"Unsupported sequence cost-bias model type: {model_type}")


@dataclass
class CostBiasAssigner:
    model: Any
    alpha: float = 1.0
    device: str = "cpu"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        model_type: str | None = None,
        alpha: float = 1.0,
        device: str = "cpu",
    ) -> "CostBiasAssigner":
        _require_torch()
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        cfg = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
        checkpoint_model_type = checkpoint.get("model_type") if isinstance(checkpoint, dict) else None
        resolved_type = model_type or checkpoint_model_type
        if not resolved_type:
            raise ValueError("model_type must be provided when the checkpoint does not contain it")
        model = build_cost_bias_model(str(resolved_type), **cfg)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return cls(model=model, alpha=float(alpha), device=device)

    def predict_biases(
        self,
        piano_roll: np.ndarray,
        prev_fingertips: np.ndarray | None = None,
    ) -> np.ndarray:
        _require_torch()
        roll = np.asarray(piano_roll, dtype=np.float32)
        if roll.ndim != 2 or roll.shape[1] < NUM_KEYS:
            raise ValueError(f"piano_roll must have shape [W, 88+], got {roll.shape}")
        roll_tensor = torch.from_numpy(roll[:, :NUM_KEYS]).unsqueeze(0).to(self.device)
        fingertip_tensor = None
        if prev_fingertips is not None:
            tips = np.asarray(prev_fingertips, dtype=np.float32)
            if tips.ndim != 3 or tips.shape[1:] != (NUM_FINGERS, 3):
                raise ValueError(f"prev_fingertips must have shape [W, 10, 3], got {tips.shape}")
            fingertip_tensor = torch.from_numpy(tips).unsqueeze(0).to(self.device)
        with torch.no_grad():
            bias = self.model(roll_tensor, fingertip_tensor).squeeze(0).detach().cpu().numpy()
        return bias.astype(np.float32)

    def assign_all(
        self,
        waypoint_target_keys: np.ndarray,
        contact_targets_all: np.ndarray,
        previous_fingertips_sequence: np.ndarray | None = None,
    ) -> np.ndarray:
        del contact_targets_all
        return self.predict_biases(waypoint_target_keys, previous_fingertips_sequence)

    def assign_one(
        self,
        active_keys: np.ndarray,
        previous_fingertips: np.ndarray,
        key_targets: np.ndarray,
        cost_bias: np.ndarray,
        config: object | None = None,
    ) -> FingerAssignmentResult:
        return assign_fingers_previous_pose(
            active_keys,
            previous_fingertips,
            key_targets,
            config,
            cost_bias=cost_bias,
            cost_bias_alpha=self.alpha,
        )
