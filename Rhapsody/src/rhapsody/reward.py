from __future__ import annotations

import torch


def masked_fingertip_distances(
    predicted_fingertips: torch.Tensor,
    target_fingertips: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    distances = torch.linalg.norm(predicted_fingertips - target_fingertips, dim=-1)
    return distances * active_mask.to(distances.dtype)


def active_mean_error(
    predicted_fingertips: torch.Tensor,
    target_fingertips: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    distances = masked_fingertip_distances(predicted_fingertips, target_fingertips, active_mask)
    counts = active_mask.sum(dim=-1).clamp_min(1.0)
    return distances.sum(dim=-1) / counts


def active_max_error(
    predicted_fingertips: torch.Tensor,
    target_fingertips: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    distances = masked_fingertip_distances(predicted_fingertips, target_fingertips, active_mask)
    inactive_floor = torch.full_like(distances, -1.0)
    return torch.where(active_mask > 0, distances, inactive_floor).max(dim=-1).values.clamp_min(0.0)


def fingertip_reward(
    *,
    predicted_fingertips: torch.Tensor,
    target_fingertips: torch.Tensor,
    active_mask: torch.Tensor,
    predicted_qpos_norm: torch.Tensor,
    previous_qpos_norm: torch.Tensor,
    active_weight: float,
    max_error_weight: float,
    smoothness_weight: float,
) -> torch.Tensor:
    mean_error = active_mean_error(predicted_fingertips, target_fingertips, active_mask)
    max_error = active_max_error(predicted_fingertips, target_fingertips, active_mask)
    smoothness = torch.mean((predicted_qpos_norm - previous_qpos_norm) ** 2, dim=-1)
    return -(
        float(active_weight) * mean_error
        + float(max_error_weight) * max_error
        + float(smoothness_weight) * smoothness
    )
