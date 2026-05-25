from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from rhapsody.config import RhapsodyConfig
from rhapsody.data import RPIKArrays, RPIKDataset
from rhapsody.models import ForwardKinematicsSurrogate, ResidualIKPolicy
from rhapsody.normalization import RhapsodyNormalizer
from rhapsody.reward import active_max_error, active_mean_error


@dataclass(slots=True)
class EvaluationMetrics:
    fingertip_mean_error_m: float
    fingertip_max_error_m: float
    qpos_rmse: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fingertip_mean_error_m": float(self.fingertip_mean_error_m),
            "fingertip_max_error_m": float(self.fingertip_max_error_m),
            "qpos_rmse": float(self.qpos_rmse),
            "samples": int(self.samples),
        }


def evaluate_policy(
    policy: ResidualIKPolicy,
    fk_model: ForwardKinematicsSurrogate,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    config: RhapsodyConfig,
    *,
    device: torch.device | str = "cpu",
    refinement_steps: int = 0,
    refinement_lr: float = 0.05,
) -> EvaluationMetrics:
    device = torch.device(device)
    policy.eval()
    fk_model.eval()
    norm = normalizer.to(device)
    loader = DataLoader(
        RPIKDataset(arrays),
        batch_size=max(1, int(config.batch_size)),
        shuffle=False,
        drop_last=False,
    )
    mean_errors = []
    max_errors = []
    qpos_sse = 0.0
    qpos_count = 0
    sample_count = 0
    for batch in loader:
        target = batch["target_fingertips"].to(device)
        mask = batch["active_mask"].to(device)
        prev = batch["previous_qpos"].to(device)
        expert = batch["expert_qpos"].to(device)
        with torch.no_grad():
            target_norm = norm.normalize_fingertips(target).reshape(target.shape)
            prev_norm = norm.normalize_qpos(prev)
            pred_norm = policy(target_norm, mask, prev_norm)
        pred_norm = _refine_batch(
            fk_model,
            norm,
            pred_norm,
            target,
            mask,
            refinement_steps=int(refinement_steps),
            refinement_lr=float(refinement_lr),
        )
        with torch.no_grad():
            pred_tips = norm.denormalize_fingertips(fk_model(pred_norm))
            pred_qpos = norm.denormalize_qpos(pred_norm)
            mean_errors.append(active_mean_error(pred_tips, target, mask).detach().cpu())
            max_errors.append(active_max_error(pred_tips, target, mask).detach().cpu())
            qpos_sse += float(torch.sum((pred_qpos - expert) ** 2).detach().cpu())
        qpos_count += int(expert.numel())
        sample_count += int(expert.shape[0])
    return EvaluationMetrics(
        fingertip_mean_error_m=float(torch.cat(mean_errors).mean().item()),
        fingertip_max_error_m=float(torch.cat(max_errors).mean().item()),
        qpos_rmse=float((qpos_sse / max(1, qpos_count)) ** 0.5),
        samples=sample_count,
    )


def _refine_batch(
    fk_model: ForwardKinematicsSurrogate,
    normalizer: RhapsodyNormalizer,
    qpos_norm: torch.Tensor,
    target_fingertips: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    refinement_steps: int,
    refinement_lr: float,
) -> torch.Tensor:
    if refinement_steps <= 0:
        return qpos_norm.detach()
    anchor = qpos_norm.detach()
    qpos_var = anchor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([qpos_var], lr=refinement_lr)
    for _ in range(refinement_steps):
        predicted = normalizer.denormalize_fingertips(fk_model(qpos_var))
        tip_loss = active_mean_error(predicted, target_fingertips, active_mask).mean()
        anchor_loss = torch.mean((qpos_var - anchor) ** 2)
        loss = tip_loss + 0.01 * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return qpos_var.detach()


@torch.no_grad()
def evaluate_forward_model(
    fk_model: ForwardKinematicsSurrogate,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    config: RhapsodyConfig,
    *,
    device: torch.device | str = "cpu",
) -> EvaluationMetrics:
    device = torch.device(device)
    fk_model.eval()
    norm = normalizer.to(device)
    loader = DataLoader(
        RPIKDataset(arrays),
        batch_size=max(1, int(config.batch_size)),
        shuffle=False,
        drop_last=False,
    )
    mean_errors = []
    max_errors = []
    sample_count = 0
    for batch in loader:
        target = batch["target_fingertips"].to(device)
        mask = batch["active_mask"].to(device)
        expert = batch["expert_qpos"].to(device)
        pred_tips = norm.denormalize_fingertips(fk_model(norm.normalize_qpos(expert)))
        mean_errors.append(active_mean_error(pred_tips, target, mask).detach().cpu())
        max_errors.append(active_max_error(pred_tips, target, mask).detach().cpu())
        sample_count += int(expert.shape[0])
    return EvaluationMetrics(
        fingertip_mean_error_m=float(torch.cat(mean_errors).mean().item()),
        fingertip_max_error_m=float(torch.cat(max_errors).mean().item()),
        qpos_rmse=0.0,
        samples=sample_count,
    )
