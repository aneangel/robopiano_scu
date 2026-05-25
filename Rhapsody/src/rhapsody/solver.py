from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rhapsody.checkpoint import load_checkpoint
from rhapsody.constants import FINGERTIP_COORD_DIM, HAND_STATE_DIM, NUM_FINGERS
from rhapsody.data import active_mask_from_targets
from rhapsody.models import ForwardKinematicsSurrogate, ResidualIKPolicy
from rhapsody.normalization import RhapsodyNormalizer
from rhapsody.reward import active_max_error, active_mean_error


@dataclass(slots=True)
class IKSolution:
    qpos: np.ndarray
    predicted_fingertips: np.ndarray
    active_mask: np.ndarray
    mean_error_m: float
    max_error_m: float
    refinement_steps: int

    def to_dict(self) -> dict[str, object]:
        return {
            "qpos": self.qpos.astype(float).tolist(),
            "predicted_fingertips": self.predicted_fingertips.astype(float).tolist(),
            "active_mask": self.active_mask.astype(float).tolist(),
            "mean_error_m": float(self.mean_error_m),
            "max_error_m": float(self.max_error_m),
            "refinement_steps": int(self.refinement_steps),
        }


class RhapsodyIKSolver:
    """Online IK solver backed by the reward-trained Rhapsody policy."""

    def __init__(
        self,
        *,
        normalizer: RhapsodyNormalizer,
        policy: ResidualIKPolicy,
        fk_model: ForwardKinematicsSurrogate,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.normalizer = normalizer.to(self.device)
        self.policy = policy.to(self.device).eval()
        self.fk_model = fk_model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "RhapsodyIKSolver":
        _config, normalizer, policy, fk_model, _state = load_checkpoint(path, device=device)
        return cls(normalizer=normalizer, policy=policy, fk_model=fk_model, device=device)

    def solve(
        self,
        target_fingertips: np.ndarray,
        *,
        active_mask: np.ndarray | None = None,
        previous_qpos: np.ndarray | None = None,
        refinement_steps: int = 0,
        refinement_lr: float = 0.05,
    ) -> IKSolution:
        target = np.asarray(target_fingertips, dtype=np.float32)
        if target.shape == (NUM_FINGERS * FINGERTIP_COORD_DIM,):
            target = target.reshape(NUM_FINGERS, FINGERTIP_COORD_DIM)
        if target.shape != (NUM_FINGERS, FINGERTIP_COORD_DIM):
            raise ValueError(f"target_fingertips must have shape [10, 3], got {target.shape}")
        mask = (
            active_mask_from_targets(target[None, ...])[0]
            if active_mask is None
            else np.asarray(active_mask, dtype=np.float32)
        )
        if mask.shape != (NUM_FINGERS,):
            raise ValueError(f"active_mask must have shape [10], got {mask.shape}")
        qprev = (
            self.normalizer.qpos_mean.detach().cpu().numpy()
            if previous_qpos is None
            else np.asarray(previous_qpos, dtype=np.float32)
        )
        if qprev.shape != (HAND_STATE_DIM,):
            raise ValueError(f"previous_qpos must have shape [{HAND_STATE_DIM}], got {qprev.shape}")

        target_filled = np.where(
            np.isfinite(target),
            target,
            self.normalizer.fingertip_mean.detach().cpu().numpy().reshape(
                NUM_FINGERS, FINGERTIP_COORD_DIM
            ),
        ).astype(np.float32)

        target_t = torch.from_numpy(target_filled[None, ...]).to(self.device)
        mask_t = torch.from_numpy(mask[None, ...]).to(self.device)
        previous_t = torch.from_numpy(qprev[None, ...]).to(self.device)
        target_norm = self.normalizer.normalize_fingertips(target_t).reshape(target_t.shape)
        prev_norm = self.normalizer.normalize_qpos(previous_t)

        with torch.no_grad():
            qpos_norm = self.policy(target_norm, mask_t, prev_norm)

        qpos_norm = self._refine(
            qpos_norm,
            target_t,
            mask_t,
            anchor_qpos_norm=qpos_norm.detach(),
            steps=int(refinement_steps),
            lr=float(refinement_lr),
        )
        with torch.no_grad():
            predicted_tips = self.normalizer.denormalize_fingertips(self.fk_model(qpos_norm))
            qpos = self.normalizer.denormalize_qpos(qpos_norm)
            mean_error = active_mean_error(predicted_tips, target_t, mask_t)
            max_error = active_max_error(predicted_tips, target_t, mask_t)
        return IKSolution(
            qpos=qpos.detach().cpu().numpy()[0].astype(np.float32, copy=False),
            predicted_fingertips=predicted_tips.detach().cpu().numpy()[0].astype(np.float32, copy=False),
            active_mask=mask.astype(np.float32, copy=False),
            mean_error_m=float(mean_error.detach().cpu().item()),
            max_error_m=float(max_error.detach().cpu().item()),
            refinement_steps=int(refinement_steps),
        )

    def _refine(
        self,
        qpos_norm: torch.Tensor,
        target_fingertips: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        anchor_qpos_norm: torch.Tensor,
        steps: int,
        lr: float,
    ) -> torch.Tensor:
        if steps <= 0:
            return qpos_norm.detach()
        qpos_var = qpos_norm.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([qpos_var], lr=lr)
        for _ in range(steps):
            pred = self.normalizer.denormalize_fingertips(self.fk_model(qpos_var))
            tip_loss = active_mean_error(pred, target_fingertips, active_mask).mean()
            anchor_loss = torch.mean((qpos_var - anchor_qpos_norm) ** 2)
            loss = tip_loss + 0.01 * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return qpos_var.detach()
