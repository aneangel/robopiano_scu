from __future__ import annotations

import torch
from torch import nn

from rhapsody.constants import FINGERTIP_COORD_DIM, HAND_STATE_DIM, NUM_FINGERS


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    *,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden)))
        layers.append(activation())
        last_dim = int(hidden)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class ResidualIKPolicy(nn.Module):
    """Policy mapping target fingertips and previous hand state to next hand state.

    The policy works in normalized joint space. It predicts a bounded residual
    from the previous hand state, which makes it usable for online rollout while
    still allowing large enough corrections for chord transitions.
    """

    def __init__(
        self,
        *,
        qpos_dim: int = HAND_STATE_DIM,
        num_fingers: int = NUM_FINGERS,
        hidden_dims: tuple[int, ...] = (256, 256),
        action_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.qpos_dim = int(qpos_dim)
        self.num_fingers = int(num_fingers)
        self.action_scale = float(action_scale)
        input_dim = self.num_fingers * FINGERTIP_COORD_DIM + self.num_fingers + self.qpos_dim
        self.linear = nn.Linear(input_dim, self.qpos_dim)
        self.residual = build_mlp(input_dim, self.qpos_dim, hidden_dims)
        self._zero_residual_output()

    def forward(
        self,
        target_fingertips_norm: torch.Tensor,
        active_mask: torch.Tensor,
        previous_qpos_norm: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(target_fingertips_norm, active_mask, previous_qpos_norm)
        residual_logits = self.linear(features) + self.residual(features)
        residual = torch.tanh(residual_logits) * self.action_scale
        return previous_qpos_norm + residual

    def features(
        self,
        target_fingertips_norm: torch.Tensor,
        active_mask: torch.Tensor,
        previous_qpos_norm: torch.Tensor,
    ) -> torch.Tensor:
        target = target_fingertips_norm.reshape(target_fingertips_norm.shape[0], -1)
        mask = active_mask.to(dtype=target.dtype)
        target = target * mask.repeat_interleave(FINGERTIP_COORD_DIM, dim=1)
        return torch.cat([target, mask, previous_qpos_norm], dim=1)

    def _zero_residual_output(self) -> None:
        for module in reversed(self.residual):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                return


class ForwardKinematicsSurrogate(nn.Module):
    """Differentiable RP1M-trained FK surrogate: normalized qpos to fingertips."""

    def __init__(
        self,
        *,
        qpos_dim: int = HAND_STATE_DIM,
        num_fingers: int = NUM_FINGERS,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.qpos_dim = int(qpos_dim)
        self.num_fingers = int(num_fingers)
        self.linear = nn.Linear(
            self.qpos_dim,
            self.num_fingers * FINGERTIP_COORD_DIM,
        )
        self.residual = build_mlp(
            self.qpos_dim,
            self.num_fingers * FINGERTIP_COORD_DIM,
            hidden_dims,
        )
        self._zero_residual_output()

    def forward(self, qpos_norm: torch.Tensor) -> torch.Tensor:
        flat = self.linear(qpos_norm) + self.residual(qpos_norm)
        return flat.reshape(qpos_norm.shape[0], self.num_fingers, FINGERTIP_COORD_DIM)

    def _zero_residual_output(self) -> None:
        for module in reversed(self.residual):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                return
