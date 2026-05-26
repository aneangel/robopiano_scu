from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ResidualLossWeights:
    action_imitation: float = 1.0
    residual_size: float = 0.01
    action_smoothness: float = 0.02
    action_saturation: float = 0.005


def residual_controller_loss(
    predicted_action: torch.Tensor,
    selected_action: torch.Tensor,
    base_action: torch.Tensor,
    previous_action: torch.Tensor | None = None,
    *,
    weights: ResidualLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or ResidualLossWeights()
    imitation = torch.mean((predicted_action - selected_action) ** 2)
    residual_size = torch.mean((predicted_action - base_action) ** 2)
    if previous_action is None:
        smoothness = torch.zeros((), dtype=predicted_action.dtype, device=predicted_action.device)
    else:
        smoothness = torch.mean((predicted_action - previous_action) ** 2)
    saturation = torch.mean(torch.relu(torch.abs(predicted_action) - 0.95) ** 2)
    total = (
        float(weights.action_imitation) * imitation
        + float(weights.residual_size) * residual_size
        + float(weights.action_smoothness) * smoothness
        + float(weights.action_saturation) * saturation
    )
    return {
        "total": total,
        "imitation": imitation.detach(),
        "residual_size": residual_size.detach(),
        "smoothness": smoothness.detach(),
        "saturation": saturation.detach(),
    }
