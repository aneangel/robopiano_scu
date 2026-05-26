from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ballade.config import JacobianTrackerConfig
from ballade.features import as_float_array, get_field, pad_or_trim


def _state_vector_from_obs(obs: Any) -> np.ndarray:
    parts = [
        as_float_array(get_field(obs, "q", "hand_qpos")),
        as_float_array(get_field(obs, "qvel", "hand_qvel")),
        as_float_array(get_field(obs, "piano_activation", "piano_state")),
    ]
    nonempty = [part for part in parts if part.size]
    if not nonempty:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(nonempty, axis=0).astype(np.float32)


def _target_state_vector(target: Any, obs_widths: tuple[int, int, int]) -> np.ndarray:
    q_width, qvel_width, piano_width = obs_widths
    q = as_float_array(get_field(target, "target_q_micro", "q"), default_width=q_width)
    qvel = as_float_array(get_field(target, "target_qvel_micro", "qvel"), default_width=qvel_width)
    goal = as_float_array(get_field(target, "goal_key_mask", "goals"), default_width=piano_width)
    return np.concatenate(
        [pad_or_trim(q, q_width), pad_or_trim(qvel, qvel_width), pad_or_trim(goal, piano_width)],
        axis=0,
    ).astype(np.float32)


def _obs_widths(obs: Any) -> tuple[int, int, int]:
    return (
        as_float_array(get_field(obs, "q", "hand_qpos")).size,
        as_float_array(get_field(obs, "qvel", "hand_qvel")).size,
        as_float_array(get_field(obs, "piano_activation", "piano_state")).size,
    )


@dataclass(slots=True)
class TrackerDiagnostics:
    sample_count: int
    used_regression: bool
    action_delta_norm: float
    desired_state_delta_norm: float


class OnlineJacobianTracker:
    """Online ridge tracker for local action-to-state changes."""

    def __init__(
        self,
        action_dim: int,
        config: JacobianTrackerConfig | None = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.config = config or JacobianTrackerConfig()
        self._action_deltas: list[np.ndarray] = []
        self._state_deltas: list[np.ndarray] = []
        self._previous_action: np.ndarray | None = None
        self._transition_model: np.ndarray | None = None
        self.last_diagnostics = TrackerDiagnostics(0, False, 0.0, 0.0)

    def update(self, obs_t: Any, action_t: np.ndarray, obs_tp1: Any) -> None:
        action = np.asarray(action_t, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"action size {action.size} does not match action_dim={self.action_dim}")
        state_t = _state_vector_from_obs(obs_t)
        state_tp1 = _state_vector_from_obs(obs_tp1)
        width = min(state_t.size, state_tp1.size)
        if width <= 0:
            self._previous_action = action.copy()
            return
        if self._previous_action is None:
            action_delta = action.copy()
        else:
            action_delta = action - self._previous_action
        self._previous_action = action.copy()
        self._action_deltas.append(action_delta.astype(np.float32))
        self._state_deltas.append((state_tp1[:width] - state_t[:width]).astype(np.float32))
        overflow = len(self._action_deltas) - int(self.config.history_size)
        if overflow > 0:
            del self._action_deltas[:overflow]
            del self._state_deltas[:overflow]
        self._fit()

    def _fit(self) -> None:
        if len(self._action_deltas) < max(int(self.config.min_samples), 1):
            return
        x = np.stack(self._action_deltas, axis=0).astype(np.float64)
        y = np.stack(self._state_deltas, axis=0).astype(np.float64)
        width = min(y.shape[1], min(delta.size for delta in self._state_deltas))
        y = y[:, :width]
        ridge = float(self.config.ridge)
        lhs = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
        rhs = x.T @ y
        self._transition_model = np.linalg.solve(lhs, rhs).astype(np.float32)

    def propose_action(
        self,
        obs: Any,
        target: Any,
        previous_action: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
        if previous.size != self.action_dim:
            raise ValueError(f"previous_action size {previous.size} does not match action_dim={self.action_dim}")
        state = _state_vector_from_obs(obs)
        target_state = _target_state_vector(target, _obs_widths(obs))
        width = min(state.size, target_state.size)
        desired = (target_state[:width] - state[:width]).astype(np.float32)
        used_regression = self._transition_model is not None and self._transition_model.shape[1] >= width
        if used_regression:
            model = np.asarray(self._transition_model[:, :width], dtype=np.float64)
            design = model.T
            lhs = design.T @ design + float(self.config.damping) * np.eye(self.action_dim, dtype=np.float64)
            rhs = design.T @ desired.astype(np.float64)
            action_delta = np.linalg.solve(lhs, rhs).astype(np.float32)
        else:
            action_delta = np.zeros((self.action_dim,), dtype=np.float32)
            target_q = as_float_array(get_field(target, "target_q_micro", "q"))
            obs_q = as_float_array(get_field(obs, "q", "hand_qpos"))
            q_width = min(target_q.size, obs_q.size)
            q_error = target_q[:q_width] - obs_q[:q_width]
            q_width = min(action_delta.size, q_error.size)
            if q_width > 0:
                action_delta[:q_width] = float(self.config.proportional_gain) * q_error[:q_width]
        if active_mask is not None:
            mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if mask.size != self.action_dim:
                raise ValueError(f"active_mask size {mask.size} does not match action_dim={self.action_dim}")
            action_delta = np.where(mask, action_delta, 0.0)
        max_delta = float(self.config.max_action_delta)
        if max_delta > 0.0:
            action_delta = np.clip(action_delta, -max_delta, max_delta)
        action = np.clip(
            previous + action_delta,
            float(self.config.action_low),
            float(self.config.action_high),
        ).astype(np.float32)
        self.last_diagnostics = TrackerDiagnostics(
            sample_count=len(self._action_deltas),
            used_regression=bool(used_regression),
            action_delta_norm=float(np.linalg.norm(action_delta)),
            desired_state_delta_norm=float(np.linalg.norm(desired)),
        )
        return action
