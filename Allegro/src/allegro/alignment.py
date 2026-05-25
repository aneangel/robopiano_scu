from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class HighFrequencyAlignmentConfig:
    """Configuration for a 200 Hz residual alignment controller.

    The controller treats a 20 Hz Fugue action as the action prior and adds a
    clipped high-frequency residual. The residual is supervised online by the
    closed-loop PD feedback command generated from planner tracking error.
    """

    source_hz: float = 20.0
    control_hz: float = 200.0
    kp: float = 0.35
    kd: float = 0.015
    feedback_residual_scale: float = 1.0
    learned_residual_scale: float = 0.5
    residual_clip_norm: float | None = 0.75
    residual_clip_per_dim: float | np.ndarray | None = 0.12
    smoothing_alpha: float = 0.25
    max_action_delta_per_substep: float | None = 0.18
    action_low: float = -1.0
    action_high: float = 1.0
    use_shadow_action_jacobian: bool = False
    zero_sustain_residual: bool = True
    inverse_model_enabled: bool = False
    inverse_model_capacity: int = 512
    inverse_model_warmup: int = 24
    inverse_model_ridge: float = 1e-3
    inverse_model_damping: float = 2e-2
    inverse_model_refresh_every: int = 1
    inverse_tracking_gain: float = 1.0
    inverse_residual_scale: float = 1.0
    online_learning: bool = True
    replay_capacity: int = 4096
    warmup_samples: int = 32
    update_every_substeps: int = 5
    batch_size: int = 64
    train_steps_per_update: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    hidden_layers: int = 2
    device: str = "cpu"
    seed: int = 7
    max_updates_per_rollout: int | None = None

    @property
    def source_dt(self) -> float:
        return 1.0 / float(self.source_hz)

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @property
    def substeps_per_source_step(self) -> int:
        self.validate()
        return int(round(float(self.control_hz) / float(self.source_hz)))

    def validate(self) -> None:
        if float(self.source_hz) <= 0.0 or float(self.control_hz) <= 0.0:
            raise ValueError("source_hz and control_hz must be positive")
        if float(self.control_hz) < float(self.source_hz):
            raise ValueError("control_hz must be >= source_hz")
        ratio = float(self.control_hz) / float(self.source_hz)
        rounded = round(ratio)
        if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError("control_hz must be an integer multiple of source_hz")
        if float(self.smoothing_alpha) < 0.0 or float(self.smoothing_alpha) >= 1.0:
            raise ValueError("smoothing_alpha must be in [0, 1)")
        if int(self.replay_capacity) <= 0:
            raise ValueError("replay_capacity must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.update_every_substeps) <= 0:
            raise ValueError("update_every_substeps must be positive")
        if int(self.train_steps_per_update) <= 0:
            raise ValueError("train_steps_per_update must be positive")
        if int(self.inverse_model_capacity) <= 0:
            raise ValueError("inverse_model_capacity must be positive")
        if int(self.inverse_model_warmup) < 0:
            raise ValueError("inverse_model_warmup must be nonnegative")
        if float(self.inverse_model_ridge) < 0.0:
            raise ValueError("inverse_model_ridge must be nonnegative")
        if float(self.inverse_model_damping) < 0.0:
            raise ValueError("inverse_model_damping must be nonnegative")
        if int(self.inverse_model_refresh_every) <= 0:
            raise ValueError("inverse_model_refresh_every must be positive")


@dataclass(slots=True)
class AlignmentStepResult:
    action: np.ndarray
    residual: np.ndarray
    feedback_residual: np.ndarray
    learned_residual: np.ndarray
    feature: np.ndarray
    diagnostics: dict[str, float | int | bool]


class ResidualMLP(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(feature_dim)
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(dim, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.SiLU()])
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"ResidualMLP expects [B, D], got {tuple(features.shape)}")
        return self.net(features)


class OnlineResidualReplayBuffer:
    def __init__(self, *, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._features: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._weights: list[float] = []
        self.feature_dim: int | None = None
        self.action_dim: int | None = None

    def append(self, feature: np.ndarray, target: np.ndarray, *, weight: float = 1.0) -> None:
        feature_arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
        if self.feature_dim is None:
            self.feature_dim = int(feature_arr.size)
            self.action_dim = int(target_arr.size)
        if feature_arr.size != self.feature_dim:
            raise ValueError(f"feature dim {feature_arr.size} does not match {self.feature_dim}")
        if target_arr.size != self.action_dim:
            raise ValueError(f"target dim {target_arr.size} does not match {self.action_dim}")
        self._features.append(feature_arr.copy())
        self._targets.append(target_arr.copy())
        self._weights.append(float(weight))
        overflow = len(self._features) - self.capacity
        if overflow > 0:
            del self._features[:overflow]
            del self._targets[:overflow]
            del self._weights[:overflow]

    def sample(self, batch_size: int, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._features:
            raise ValueError("Cannot sample an empty replay buffer")
        count = min(int(batch_size), len(self._features))
        indices = rng.integers(0, len(self._features), size=count)
        features = np.stack([self._features[int(idx)] for idx in indices], axis=0).astype(np.float32)
        targets = np.stack([self._targets[int(idx)] for idx in indices], axis=0).astype(np.float32)
        weights = np.asarray([self._weights[int(idx)] for idx in indices], dtype=np.float32)
        return features, targets, weights

    def as_arrays(self) -> dict[str, np.ndarray]:
        if not self._features:
            return {
                "features": np.zeros((0, 0), dtype=np.float32),
                "targets": np.zeros((0, 0), dtype=np.float32),
                "weights": np.zeros((0,), dtype=np.float32),
            }
        return {
            "features": np.stack(self._features, axis=0).astype(np.float32),
            "targets": np.stack(self._targets, axis=0).astype(np.float32),
            "weights": np.asarray(self._weights, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._features)


class OnlineResidualTrainer:
    """Small online supervised learner for feedback-error residual targets."""

    def __init__(self, *, config: HighFrequencyAlignmentConfig, action_dim: int) -> None:
        self.config = config
        self.action_dim = int(action_dim)
        self.replay = OnlineResidualReplayBuffer(capacity=int(config.replay_capacity))
        self.rng = np.random.default_rng(int(config.seed))
        self.model: ResidualMLP | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.updates = 0
        self.examples_seen = 0
        self.last_loss: float | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.updates > 0

    def _ensure_model(self, feature_dim: int) -> None:
        if self.model is not None:
            return
        torch.manual_seed(int(self.config.seed))
        self.model = ResidualMLP(
            feature_dim=int(feature_dim),
            action_dim=int(self.action_dim),
            hidden_dim=int(self.config.hidden_dim),
            hidden_layers=int(self.config.hidden_layers),
        ).to(torch.device(str(self.config.device)))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

    def predict(self, feature: np.ndarray) -> np.ndarray:
        feature_arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not self.ready:
            return np.zeros((self.action_dim,), dtype=np.float32)
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(feature_arr[None]).to(torch.device(str(self.config.device))).float()
            out = self.model(tensor).detach().cpu().numpy()[0]
        return np.asarray(out, dtype=np.float32)

    def observe(self, feature: np.ndarray, target: np.ndarray, *, weight: float = 1.0) -> None:
        feature_arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
        if target_arr.size != self.action_dim:
            raise ValueError(f"target dim {target_arr.size} does not match action_dim={self.action_dim}")
        self._ensure_model(feature_arr.size)
        self.replay.append(feature_arr, target_arr, weight=float(weight))
        self.examples_seen += 1

    def maybe_update(self, *, substep_index: int) -> dict[str, float | int | bool]:
        if not self.config.online_learning:
            return {"online_update": False, "online_update_reason": 0}
        if self.config.max_updates_per_rollout is not None and self.updates >= int(self.config.max_updates_per_rollout):
            return {"online_update": False, "online_update_reason": 1}
        if len(self.replay) < int(self.config.warmup_samples):
            return {"online_update": False, "online_update_reason": 2}
        if (int(substep_index) + 1) % int(self.config.update_every_substeps) != 0:
            return {"online_update": False, "online_update_reason": 3}
        loss = self.update(train_steps=int(self.config.train_steps_per_update))
        return {"online_update": True, "online_loss": loss, "online_updates": int(self.updates)}

    def update(self, *, train_steps: int = 1) -> float:
        if self.model is None or self.optimizer is None:
            if self.replay.feature_dim is None:
                raise ValueError("Cannot update before observing at least one feature")
            self._ensure_model(self.replay.feature_dim)
        assert self.model is not None
        assert self.optimizer is not None
        device = torch.device(str(self.config.device))
        self.model.train()
        loss_value = 0.0
        for _ in range(int(train_steps)):
            features, targets, weights = self.replay.sample(int(self.config.batch_size), rng=self.rng)
            feature_tensor = torch.from_numpy(features).to(device).float()
            target_tensor = torch.from_numpy(targets).to(device).float()
            weight_tensor = torch.from_numpy(weights).to(device).float().clamp_min(1e-6)
            pred = self.model(feature_tensor)
            mse = torch.mean(torch.square(pred - target_tensor), dim=-1)
            loss = torch.sum(mse * weight_tensor) / torch.sum(weight_tensor)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            self.updates += 1
        self.last_loss = loss_value
        return loss_value

    def summary(self) -> dict[str, Any]:
        return {
            "examples_seen": int(self.examples_seen),
            "replay_size": int(len(self.replay)),
            "updates": int(self.updates),
            "ready": bool(self.ready),
            "last_loss": self.last_loss,
        }


class OnlineAffineActionModel:
    """Online affine model: hand_delta ~= action @ weights + intercept."""

    def __init__(
        self,
        *,
        action_dim: int,
        q_dim: int,
        capacity: int,
        ridge: float,
        refresh_every: int = 1,
    ) -> None:
        self.action_dim = int(action_dim)
        self.q_dim = int(q_dim)
        self.capacity = int(capacity)
        self.ridge = float(ridge)
        self.refresh_every = int(refresh_every)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        if self.refresh_every <= 0:
            raise ValueError("refresh_every must be positive")
        self._actions: list[np.ndarray] = []
        self._deltas: list[np.ndarray] = []
        self.weights: np.ndarray | None = None
        self.intercept: np.ndarray | None = None
        self.fit_count = 0
        self.last_fit_samples = 0
        self.last_fit_rmse: float | None = None

    @property
    def sample_count(self) -> int:
        return len(self._actions)

    @property
    def ready(self) -> bool:
        return self.weights is not None and self.intercept is not None

    def observe(self, action: np.ndarray, q_before: np.ndarray, q_after: np.ndarray) -> None:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        before = np.asarray(q_before, dtype=np.float32).reshape(-1)
        after = np.asarray(q_after, dtype=np.float32).reshape(-1)
        if action_arr.size != self.action_dim:
            raise ValueError(f"action dim {action_arr.size} does not match {self.action_dim}")
        if before.size != self.q_dim or after.size != self.q_dim:
            raise ValueError("q_before and q_after must match q_dim")
        self._actions.append(action_arr.copy())
        self._deltas.append((after - before).astype(np.float32))
        overflow = len(self._actions) - self.capacity
        if overflow > 0:
            del self._actions[:overflow]
            del self._deltas[:overflow]
        if len(self._actions) % self.refresh_every == 0:
            self.fit()

    def fit(self) -> None:
        if not self._actions:
            return
        actions = np.stack(self._actions, axis=0).astype(np.float64)
        deltas = np.stack(self._deltas, axis=0).astype(np.float64)
        ones = np.ones((actions.shape[0], 1), dtype=np.float64)
        design = np.concatenate([actions, ones], axis=1)
        gram = design.T @ design
        if self.ridge > 0.0:
            penalty = np.eye(gram.shape[0], dtype=np.float64) * self.ridge
            penalty[-1, -1] = 0.0
            gram = gram + penalty
        rhs = design.T @ deltas
        try:
            coeff = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coeff = np.linalg.pinv(gram) @ rhs
        pred = design @ coeff
        self.weights = coeff[:-1].astype(np.float32)
        self.intercept = coeff[-1].astype(np.float32)
        self.fit_count += 1
        self.last_fit_samples = int(actions.shape[0])
        self.last_fit_rmse = float(np.sqrt(np.mean(np.square(pred - deltas))))

    def inverse_action(
        self,
        *,
        desired_delta: np.ndarray,
        base_action: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        if not self.ready:
            return np.asarray(base_action, dtype=np.float32).reshape(-1).copy()
        desired = np.asarray(desired_delta, dtype=np.float32).reshape(-1)
        base = np.asarray(base_action, dtype=np.float32).reshape(-1)
        if desired.size != self.q_dim or base.size != self.action_dim:
            raise ValueError("desired_delta and base_action dimensions do not match inverse model")
        assert self.weights is not None
        assert self.intercept is not None
        # weights maps action -> delta as [action_dim, q_dim].  Solve
        # min_a ||weights.T @ a + intercept - desired||^2 + damping ||a-base||^2.
        b = self.weights.T.astype(np.float64)
        target = (desired - self.intercept).astype(np.float64)
        base64 = base.astype(np.float64)
        damp = max(float(damping), 0.0)
        lhs = b.T @ b + damp * np.eye(self.action_dim, dtype=np.float64)
        rhs = b.T @ target + damp * base64
        try:
            action = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            action = np.linalg.pinv(lhs) @ rhs
        return action.astype(np.float32)

    def predict_delta(self, action: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.zeros((self.q_dim,), dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        assert self.weights is not None
        assert self.intercept is not None
        return (action_arr @ self.weights + self.intercept).astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "ready": bool(self.ready),
            "fit_count": int(self.fit_count),
            "last_fit_samples": int(self.last_fit_samples),
            "last_fit_rmse": self.last_fit_rmse,
        }


def build_shadow_hand_action_jacobian(*, action_dim: int, q_dim: int) -> np.ndarray | None:
    """Return the RoboPianist Shadow Hand action-to-qpos feedback map.

    The RP1M hand state exposes 23 qpos entries per hand in joint order, while
    the reduced RoboPianist action space exposes 19 actuators per hand in
    actuator order: wrist, thumb, fingers, forearm.  Middle/distal finger joints
    are tendon-coupled, so their feedback is averaged into one actuator.
    """

    if int(action_dim) != 39 or int(q_dim) < 46:
        return None
    jac = np.zeros((int(action_dim), int(q_dim)), dtype=np.float32)

    def add(row: int, columns: int | tuple[int, ...], weight: float = 1.0) -> None:
        if isinstance(columns, int):
            jac[row, columns] = float(weight)
        else:
            for column in columns:
                jac[row, column] = float(weight)

    # Right hand action rows 0..18 map to qpos rows 0..22.
    add(0, 0)
    add(1, 1)
    add(2, 18)
    add(3, 19)
    add(4, 20)
    add(5, 2)
    add(6, 3)
    add(7, (4, 5), 0.5)
    add(8, 6)
    add(9, 7)
    add(10, (8, 9), 0.5)
    add(11, 10)
    add(12, 11)
    add(13, (12, 13), 0.5)
    add(14, 14)
    add(15, 15)
    add(16, (16, 17), 0.5)
    add(17, 21)
    add(18, 22)

    # Left hand action rows 19..37 map to qpos rows 23..45.
    offset = 19
    qoff = 23
    add(offset + 0, qoff + 0)
    add(offset + 1, qoff + 1)
    add(offset + 2, qoff + 18)
    add(offset + 3, qoff + 19)
    add(offset + 4, qoff + 20)
    add(offset + 5, qoff + 2)
    add(offset + 6, qoff + 3)
    add(offset + 7, (qoff + 4, qoff + 5), 0.5)
    add(offset + 8, qoff + 6)
    add(offset + 9, qoff + 7)
    add(offset + 10, (qoff + 8, qoff + 9), 0.5)
    add(offset + 11, qoff + 10)
    add(offset + 12, qoff + 11)
    add(offset + 13, (qoff + 12, qoff + 13), 0.5)
    add(offset + 14, qoff + 14)
    add(offset + 15, qoff + 15)
    add(offset + 16, (qoff + 16, qoff + 17), 0.5)
    add(offset + 17, qoff + 21)
    add(offset + 18, qoff + 22)
    return jac


@dataclass(slots=True)
class ActionSpaceProjector:
    action_dim: int
    q_dim: int
    action_jacobian: np.ndarray | None = None
    zero_sustain_residual: bool = True

    def project(self, q_error: np.ndarray, qvel_error: np.ndarray, *, kp: float, kd: float) -> np.ndarray:
        q_err = np.asarray(q_error, dtype=np.float32).reshape(-1)
        qvel_err = np.asarray(qvel_error, dtype=np.float32).reshape(-1)
        if q_err.size != int(self.q_dim) or qvel_err.size != int(self.q_dim):
            raise ValueError("q_error and qvel_error must match projector q_dim")
        joint_feedback = float(kp) * q_err + float(kd) * qvel_err
        if self.action_jacobian is not None:
            jac = np.asarray(self.action_jacobian, dtype=np.float32)
            if jac.shape != (int(self.action_dim), int(self.q_dim)):
                raise ValueError(
                    f"action_jacobian shape {jac.shape} must be {(int(self.action_dim), int(self.q_dim))}"
                )
            residual = jac @ joint_feedback
        else:
            residual = np.zeros((int(self.action_dim),), dtype=np.float32)
            width = min(int(self.action_dim), int(self.q_dim))
            if bool(self.zero_sustain_residual) and int(self.action_dim) == 39:
                width = min(width, 38)
            residual[:width] = joint_feedback[:width]
        if bool(self.zero_sustain_residual) and int(self.action_dim) == 39:
            residual[-1] = 0.0
        return residual.astype(np.float32)


class ResidualSafetyProcessor:
    def __init__(self, *, config: HighFrequencyAlignmentConfig, action_dim: int) -> None:
        self.config = config
        self.action_dim = int(action_dim)
        self.previous_residual: np.ndarray | None = None
        self.previous_action: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_residual = None
        self.previous_action = None

    def process(self, base_action: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        base = np.asarray(base_action, dtype=np.float32).reshape(-1)
        raw = np.asarray(residual, dtype=np.float32).reshape(-1)
        if base.size != self.action_dim or raw.size != self.action_dim:
            raise ValueError("base_action and residual must match action_dim")
        per_dim = clip_residual_per_dim(raw, self.config.residual_clip_per_dim)
        clipped = clip_residual_norm(per_dim, self.config.residual_clip_norm)
        alpha = float(self.config.smoothing_alpha)
        if self.previous_residual is not None and alpha > 0.0:
            clipped = (alpha * self.previous_residual + (1.0 - alpha) * clipped).astype(np.float32)
        if bool(self.config.zero_sustain_residual) and self.action_dim == 39:
            clipped[-1] = 0.0
        action_unclipped = base + clipped
        if self.previous_action is not None and self.config.max_action_delta_per_substep is not None:
            limit = abs(float(self.config.max_action_delta_per_substep))
            action_unclipped = self.previous_action + np.clip(action_unclipped - self.previous_action, -limit, limit)
        action = np.clip(action_unclipped, float(self.config.action_low), float(self.config.action_high)).astype(np.float32)
        diagnostics = {
            "residual/raw_norm": float(np.linalg.norm(raw)),
            "residual/clipped_norm": float(np.linalg.norm(clipped)),
            "residual/clip_fraction": float(np.mean(np.not_equal(raw, clipped).astype(np.float32))),
            "action/clip_fraction": float(np.mean(np.not_equal(action_unclipped, action).astype(np.float32))),
        }
        self.previous_residual = clipped.astype(np.float32)
        self.previous_action = action.astype(np.float32)
        return action, clipped.astype(np.float32), diagnostics


class AllegroAligner:
    """High-frequency residual controller for Fugue planner-following rollouts."""

    def __init__(
        self,
        *,
        config: HighFrequencyAlignmentConfig | None,
        q_dim: int,
        action_dim: int,
        action_jacobian: np.ndarray | None = None,
    ) -> None:
        self.config = config or HighFrequencyAlignmentConfig()
        self.config.validate()
        self.q_dim = int(q_dim)
        self.action_dim = int(action_dim)
        if action_jacobian is None:
            action_jacobian = (
                build_shadow_hand_action_jacobian(action_dim=self.action_dim, q_dim=self.q_dim)
                if bool(self.config.use_shadow_action_jacobian)
                else None
            )
        self.projector = ActionSpaceProjector(
            action_dim=self.action_dim,
            q_dim=self.q_dim,
            action_jacobian=action_jacobian,
            zero_sustain_residual=bool(self.config.zero_sustain_residual),
        )
        self.trainer = OnlineResidualTrainer(config=self.config, action_dim=self.action_dim)
        self.inverse_model = OnlineAffineActionModel(
            action_dim=self.action_dim,
            q_dim=self.q_dim,
            capacity=int(self.config.inverse_model_capacity),
            ridge=float(self.config.inverse_model_ridge),
            refresh_every=int(self.config.inverse_model_refresh_every),
        )
        self.safety = ResidualSafetyProcessor(config=self.config, action_dim=self.action_dim)
        self.previous_base_action: np.ndarray | None = None
        self.substep_counter = 0

    def reset(self) -> None:
        self.safety.reset()
        self.previous_base_action = None
        self.substep_counter = 0

    def act(
        self,
        *,
        base_action: np.ndarray,
        current_q: np.ndarray,
        current_qvel: np.ndarray,
        target_q: np.ndarray,
        target_qvel: np.ndarray,
        source_step: int,
        substep: int,
        feedback_weight: float = 1.0,
    ) -> AlignmentStepResult:
        base = np.asarray(base_action, dtype=np.float32).reshape(-1)
        current = np.asarray(current_q, dtype=np.float32).reshape(-1)
        current_vel = np.asarray(current_qvel, dtype=np.float32).reshape(-1)
        target = np.asarray(target_q, dtype=np.float32).reshape(-1)
        target_vel = np.asarray(target_qvel, dtype=np.float32).reshape(-1)
        if base.size != self.action_dim:
            raise ValueError(f"base_action dim {base.size} does not match action_dim={self.action_dim}")
        q_error = target - current
        qvel_error = target_vel - current_vel
        phase = float(int(substep) + 1) / float(self.config.substeps_per_source_step)
        feature = build_alignment_feature(
            q_error=q_error,
            qvel_error=qvel_error,
            base_action=base,
            previous_base_action=self.previous_base_action,
            previous_residual=self.safety.previous_residual,
            source_phase=phase,
        )
        feedback = self.projector.project(q_error, qvel_error, kp=float(self.config.kp), kd=float(self.config.kd))
        learned = self.trainer.predict(feature)
        inverse_residual = np.zeros((self.action_dim,), dtype=np.float32)
        inverse_ready = bool(self.config.inverse_model_enabled and self.inverse_model.ready)
        if inverse_ready and self.inverse_model.sample_count >= int(self.config.inverse_model_warmup):
            desired_delta = float(self.config.inverse_tracking_gain) * q_error
            inverse_action = self.inverse_model.inverse_action(
                desired_delta=desired_delta,
                base_action=base,
                damping=float(self.config.inverse_model_damping),
            )
            inverse_residual = inverse_action - base
        raw_residual = (
            float(self.config.feedback_residual_scale) * feedback
            + float(self.config.learned_residual_scale) * learned
            + float(self.config.inverse_residual_scale) * inverse_residual
        )
        action, residual, diagnostics = self.safety.process(base, raw_residual)
        self.trainer.observe(feature, feedback, weight=float(feedback_weight))
        update_diag = self.trainer.maybe_update(substep_index=self.substep_counter)
        diagnostics.update(
            {
                "alignment/source_step": int(source_step),
                "alignment/substep": int(substep),
                "alignment/phase": float(phase),
                "alignment/q_error_norm": float(np.linalg.norm(q_error)),
                "alignment/qvel_error_norm": float(np.linalg.norm(qvel_error)),
                "alignment/feedback_norm": float(np.linalg.norm(feedback)),
                "alignment/learned_norm": float(np.linalg.norm(learned)),
                "alignment/inverse_residual_norm": float(np.linalg.norm(inverse_residual)),
                "alignment/inverse_ready": bool(inverse_ready),
                "alignment/inverse_samples": int(self.inverse_model.sample_count),
                "alignment/online_ready": bool(self.trainer.ready),
            }
        )
        diagnostics.update(update_diag)
        self.previous_base_action = base.copy()
        self.substep_counter += 1
        return AlignmentStepResult(
            action=action,
            residual=residual,
            feedback_residual=feedback.astype(np.float32),
            learned_residual=learned.astype(np.float32),
            feature=feature.astype(np.float32),
            diagnostics=diagnostics,
        )

    def observe_transition(self, *, action: np.ndarray, q_before: np.ndarray, q_after: np.ndarray) -> None:
        if not self.config.inverse_model_enabled:
            return
        self.inverse_model.observe(action=action, q_before=q_before, q_after=q_after)

    def summary(self) -> dict[str, Any]:
        return {
            "module": "allegro",
            "method": "200hz_residual_feedback_error_alignment",
            "config": asdict(self.config),
            "trainer": self.trainer.summary(),
            "inverse_model": self.inverse_model.summary(),
        }


def build_alignment_feature(
    *,
    q_error: np.ndarray,
    qvel_error: np.ndarray,
    base_action: np.ndarray,
    previous_base_action: np.ndarray | None,
    previous_residual: np.ndarray | None,
    source_phase: float,
) -> np.ndarray:
    q_err = np.asarray(q_error, dtype=np.float32).reshape(-1)
    qvel_err = np.asarray(qvel_error, dtype=np.float32).reshape(-1)
    base = np.asarray(base_action, dtype=np.float32).reshape(-1)
    prev_base = np.zeros_like(base, dtype=np.float32) if previous_base_action is None else np.asarray(previous_base_action, dtype=np.float32).reshape(-1)
    prev_residual = np.zeros_like(base, dtype=np.float32) if previous_residual is None else np.asarray(previous_residual, dtype=np.float32).reshape(-1)
    if prev_base.size != base.size or prev_residual.size != base.size:
        raise ValueError("previous action and residual must match base_action shape")
    phase = float(np.clip(source_phase, 0.0, 1.0))
    phase_features = np.asarray(
        [phase, math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
        dtype=np.float32,
    )
    return np.concatenate([q_err, qvel_err, base, prev_base, prev_residual, phase_features], axis=0).astype(np.float32)


def interpolate_source_target(
    values: np.ndarray,
    *,
    source_step: int,
    substep: int,
    substeps_per_source_step: int,
    phase_power: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D] values, got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("Cannot interpolate an empty trajectory")
    start = min(max(int(source_step), 0), int(arr.shape[0]) - 1)
    end = min(start + 1, int(arr.shape[0]) - 1)
    alpha = float(int(substep) + 1) / float(max(int(substeps_per_source_step), 1))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    power = float(phase_power)
    if power <= 0.0:
        raise ValueError("phase_power must be positive")
    alpha = float(alpha**power)
    return ((1.0 - alpha) * arr[start] + alpha * arr[end]).astype(np.float32)


def finite_difference_target(
    values: np.ndarray,
    *,
    source_step: int,
    source_dt: float,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D] values, got {arr.shape}")
    start = min(max(int(source_step), 0), int(arr.shape[0]) - 1)
    end = min(start + 1, int(arr.shape[0]) - 1)
    if end == start:
        return np.zeros((arr.shape[-1],), dtype=np.float32)
    return ((arr[end] - arr[start]) / float(source_dt)).astype(np.float32)


def clip_residual_per_dim(residual: np.ndarray, clip_per_dim: float | np.ndarray | None) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float32).reshape(-1)
    if clip_per_dim is None:
        return values.copy()
    limit = np.asarray(clip_per_dim, dtype=np.float32)
    if limit.ndim == 0:
        limit = np.full(values.shape, abs(float(limit)), dtype=np.float32)
    if limit.shape != values.shape:
        raise ValueError(f"clip_per_dim shape {limit.shape} does not match residual shape {values.shape}")
    return np.clip(values, -np.abs(limit), np.abs(limit)).astype(np.float32)


def clip_residual_norm(residual: np.ndarray, clip_norm: float | None) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float32).reshape(-1)
    if clip_norm is None:
        return values.copy()
    limit = float(clip_norm)
    if limit <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm <= limit or norm == 0.0:
        return values.copy()
    return (values * np.float32(limit / norm)).astype(np.float32)
