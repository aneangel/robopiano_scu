from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ModelConfig:
    type: str = "mlp"
    hidden_dim: int = 256
    hidden_layers: int = 3
    dropout: float = 0.05
    layer_norm: bool = True
    d_model: int = 128
    nhead: int = 4
    transformer_layers: int = 3
    tcn_layers: int = 4
    kernel_size: int = 3

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "ModelConfig":
        payload = dict(values or {})
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


class MLPActionModel(nn.Module):
    """Feed-forward action predictor returning `[B, C, action_dim]`."""

    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        chunk_horizon: int = 1,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        dropout: float = 0.05,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        layers: list[nn.Module] = []
        dim = self.input_dim
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            if bool(layer_norm):
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.SiLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, self.action_dim * self.chunk_horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"MLPActionModel expects [B, D], got {tuple(features.shape)}")
        out = self.net(features)
        return out.reshape(features.shape[0], self.chunk_horizon, self.action_dim)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class CausalTransformerActionModel(nn.Module):
    """Causal transformer for `[B, L, token_dim]` inputs."""

    def __init__(
        self,
        *,
        token_dim: int,
        action_dim: int,
        chunk_horizon: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.input = nn.Linear(int(token_dim), int(d_model))
        self.pos = SinusoidalPositionalEncoding(int(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(d_model) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), self.action_dim * self.chunk_horizon),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Transformer expects [B, L, D], got {tuple(tokens.shape)}")
        length = int(tokens.shape[1])
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        hidden = self.pos(self.input(tokens))
        encoded = self.encoder(hidden, mask=mask)
        out = self.head(encoded[:, -1])
        return out.reshape(tokens.shape[0], self.chunk_horizon, self.action_dim)


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(channels, channels, kernel_size=int(kernel_size), dilation=int(dilation))
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.conv(y)
        y = self.norm(y)
        y = nn.functional.silu(y)
        y = self.dropout(y)
        return residual + y


class TemporalConvActionModel(nn.Module):
    """Causal TCN baseline for `[B, L, token_dim]` inputs."""

    def __init__(
        self,
        *,
        token_dim: int,
        action_dim: int,
        chunk_horizon: int = 1,
        d_model: int = 128,
        layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.input = nn.Linear(int(token_dim), int(d_model))
        self.blocks = nn.Sequential(
            *[
                CausalConv1d(
                    int(d_model),
                    kernel_size=int(kernel_size),
                    dilation=2**layer_idx,
                    dropout=float(dropout),
                )
                for layer_idx in range(int(layers))
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), self.action_dim * self.chunk_horizon),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"TCN expects [B, L, D], got {tuple(tokens.shape)}")
        hidden = self.input(tokens).transpose(1, 2)
        encoded = self.blocks(hidden).transpose(1, 2)
        out = self.head(encoded[:, -1])
        return out.reshape(tokens.shape[0], self.chunk_horizon, self.action_dim)


def build_model(
    *,
    config: ModelConfig | dict[str, Any] | None,
    input_dim: int,
    action_dim: int,
    chunk_horizon: int = 1,
) -> nn.Module:
    cfg = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
    model_type = str(cfg.type).lower()
    if model_type == "mlp":
        return MLPActionModel(
            input_dim=input_dim,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )
    if model_type in {"transformer", "action_chunk_transformer"}:
        return CausalTransformerActionModel(
            token_dim=input_dim,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.transformer_layers,
            dropout=cfg.dropout,
        )
    if model_type in {"tcn", "temporal_conv"}:
        return TemporalConvActionModel(
            token_dim=input_dim,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
            d_model=cfg.d_model,
            layers=cfg.tcn_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unsupported model type: {cfg.type}")
