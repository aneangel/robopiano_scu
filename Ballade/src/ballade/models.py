from __future__ import annotations

import torch
from torch import nn


class ResidualMLPController(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        residual_scale: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.residual_scale = float(residual_scale)
        layers: list[nn.Module] = []
        dim = self.feature_dim
        for _idx in range(int(hidden_layers)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.SiLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be [batch, feature_dim], got {tuple(features.shape)}")
        if base_action.ndim != 2:
            raise ValueError(f"base_action must be [batch, action_dim], got {tuple(base_action.shape)}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"feature width {features.shape[-1]} does not match {self.feature_dim}")
        if base_action.shape[-1] != self.action_dim:
            raise ValueError(f"action width {base_action.shape[-1]} does not match {self.action_dim}")
        residual = self.residual_scale * torch.tanh(self.net(features))
        return torch.clamp(base_action + residual, -1.0, 1.0)
