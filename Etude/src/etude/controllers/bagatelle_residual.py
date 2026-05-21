from __future__ import annotations

from typing import Any

import numpy as np
import torch

from etude.controllers.base import TrajectoryFollower
from etude.features.fingertip_phase_blocks import (
    FingertipFeatureSpec,
    PhaseFeatureSpec,
    build_fingertip_phase_features,
    resolve_phase_state,
)
from etude.robopianist.state_mapping import StateMapping


class BagatelleResidualController(TrajectoryFollower):
    """Wrap an existing controller with a Bagatelle-conditioned additive residual head."""

    def __init__(
        self,
        mapping: StateMapping,
        base_controller: TrajectoryFollower,
        correction_model: torch.nn.Module,
        *,
        fingertip_spec: FingertipFeatureSpec | None = None,
        phase_spec: PhaseFeatureSpec | None = None,
        device: str | torch.device = "cpu",
        residual_limit: float | None = None,
        residual_scale: float = 1.0,
    ) -> None:
        self.mapping = mapping
        self.base_controller = base_controller
        self.correction_model = correction_model.to(device)
        self.correction_model.eval()
        self.fingertip_spec = fingertip_spec or FingertipFeatureSpec(allow_missing=False)
        self.phase_spec = phase_spec or PhaseFeatureSpec(allow_missing=True)
        self.device = torch.device(device)
        self.residual_limit = None if residual_limit is None else float(residual_limit)
        self.residual_scale = float(residual_scale)
        self.metadata: dict[str, Any] = {}
        self._last_diagnostics: dict[str, float] = {}

    def reset(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = metadata or {}
        self.base_controller.reset(q_ref, qdot_ref, metadata=self.metadata)
        self._last_diagnostics = {}

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        base_action = self.base_controller.act(obs, t)
        features = self.build_features(obs, t)
        residual = self.predict_residual(features)
        action = self.mapping.clip_action(base_action + residual)
        phase_state = resolve_phase_state(
            t=t,
            metadata=self.metadata,
            plan_bundle=self.metadata.get("plan_bundle"),
            target_keys=obs.get("target_keys"),
            phase_names=self.phase_spec.phase_names,
            allow_missing=self.phase_spec.allow_missing,
            fill_value=self.phase_spec.missing_fill_value,
        )
        self._last_diagnostics = {
            "bagatelle_residual/l2": float(np.linalg.norm(residual)),
            "bagatelle_residual/phase_mask": float(phase_state["mask"]),
        }
        return action

    def build_features(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        return build_fingertip_phase_features(
            t=t,
            metadata=self.metadata,
            plan_bundle=self.metadata.get("plan_bundle"),
            target_keys=obs.get("target_keys"),
            current_fingertips=obs.get("fingertips"),
            desired_fingertips=_metadata_timestep(self.metadata.get("desired_fingertips"), t),
            fingertip_weights=_metadata_timestep(self.metadata.get("fingertip_weights"), t),
            active_finger_mask=_metadata_timestep(self.metadata.get("active_finger_mask"), t),
            inactive_finger_mask=_metadata_timestep(self.metadata.get("inactive_finger_mask"), t),
            fingertip_spec=self.fingertip_spec,
            phase_spec=self.phase_spec,
        )

    def predict_residual(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            residual = (
                self.correction_model(torch.from_numpy(features).to(self.device).float().unsqueeze(0))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        residual = self.residual_scale * np.asarray(residual, dtype=np.float32).reshape(-1)
        if self.residual_limit is not None:
            residual = np.clip(residual, -self.residual_limit, self.residual_limit)
        return residual.astype(np.float32)

    def diagnostics(self) -> dict[str, float]:
        diagnostics_fn = getattr(self.base_controller, "diagnostics", None)
        base = diagnostics_fn() if callable(diagnostics_fn) else {}
        return {**base, **self._last_diagnostics}


def _metadata_timestep(value: Any, t: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim <= 1:
        return array.astype(np.float32)
    if array.ndim == 2 and array.shape == (10, 3):
        return array.astype(np.float32)
    index = int(np.clip(t, 0, array.shape[0] - 1))
    return array[index].astype(np.float32)
