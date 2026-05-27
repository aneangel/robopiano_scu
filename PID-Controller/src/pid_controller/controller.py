from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from pid_controller.mapping import (
    REDUCED_ACTION_DIM,
    ActionJointMapEntry,
    ProjectionMode,
    action_signals_from_hand_state,
    build_reduced_action_mapping,
    mapping_to_jsonable,
)


ControllerKind = Literal["p", "pd", "pid"]
SetpointPolicy = Literal["next", "linear", "minimum_jerk"]


@dataclass(slots=True)
class PIDGains:
    kp: float = 1.0
    kd: float = 0.0
    ki: float = 0.0
    integral_limit: float = 0.25


@dataclass(slots=True)
class PIDControllerConfig:
    gains: PIDGains
    setpoint_policy: SetpointPolicy = "minimum_jerk"
    use_target_velocity: bool = False
    target_velocity_scale: float = 0.0
    feedforward_scale: float = 0.0
    lookahead_substeps: int = 10
    sustain_value: float = 0.0
    action_dim: int = REDUCED_ACTION_DIM
    projection: ProjectionMode = "weighted_least_squares"


def make_controller_config(
    kind: ControllerKind,
    *,
    kp: float | None = None,
    kd: float | None = None,
    ki: float | None = None,
    integral_limit: float = 0.25,
    setpoint_policy: SetpointPolicy = "minimum_jerk",
    use_target_velocity: bool = False,
    target_velocity_scale: float = 0.0,
    feedforward_scale: float = 0.0,
    lookahead_substeps: int = 10,
    sustain_value: float = 0.0,
    projection: ProjectionMode = "weighted_least_squares",
) -> PIDControllerConfig:
    mode = str(kind).lower()
    if mode == "p":
        gains = PIDGains(kp=1.0 if kp is None else float(kp), kd=0.0, ki=0.0)
    elif mode == "pd":
        gains = PIDGains(
            kp=1.0 if kp is None else float(kp),
            kd=0.02 if kd is None else float(kd),
            ki=0.0,
        )
    elif mode == "pid":
        gains = PIDGains(
            kp=1.0 if kp is None else float(kp),
            kd=0.02 if kd is None else float(kd),
            ki=0.05 if ki is None else float(ki),
            integral_limit=float(integral_limit),
        )
    else:
        raise ValueError(f"Unknown controller kind: {kind!r}")
    if ki is not None and mode != "pid":
        gains.ki = float(ki)
    if kd is not None and mode == "p":
        gains.kd = float(kd)
    gains.integral_limit = float(integral_limit)
    return PIDControllerConfig(
        gains=gains,
        setpoint_policy=setpoint_policy,
        use_target_velocity=bool(use_target_velocity),
        target_velocity_scale=float(target_velocity_scale),
        feedforward_scale=float(feedforward_scale),
        lookahead_substeps=max(int(lookahead_substeps), 1),
        sustain_value=float(sustain_value),
        projection=projection,
    )


class HandPIDController:
    """Position-target P/PD/PID controller for reduced RoboPianist actions.

    RoboPianist hand actions are position-actuator targets, not torques. This
    controller therefore emits actuator-unit position targets at 200 Hz. For a
    direct joint actuator the controlled signal is the joint qpos. For a fixed
    tendon actuator such as FFJ0, the controlled signal is the weighted
    least-squares projection of its coupled middle/distal joint qpos values.
    """

    def __init__(self, config: PIDControllerConfig | None = None) -> None:
        self.config = config or make_controller_config("pd")
        self.mapping: tuple[ActionJointMapEntry, ...] = build_reduced_action_mapping(
            action_dim=self.config.action_dim
        )
        self.minimum = np.full((self.config.action_dim,), -np.inf, dtype=np.float32)
        self.maximum = np.full((self.config.action_dim,), np.inf, dtype=np.float32)
        self.integral = np.zeros((self.config.action_dim,), dtype=np.float32)
        self.previous_error: np.ndarray | None = None
        self.previous_control: np.ndarray | None = None
        self.trajectory_hand_qpos: np.ndarray | None = None

    def reset(
        self,
        *,
        action_spec: Any | None = None,
        hand_joint_names: Sequence[str] | None = None,
        actuator_names: Sequence[str] | None = None,
        initial_hand_qpos: np.ndarray | None = None,
        trajectory_hand_qpos: np.ndarray | None = None,
        **_kwargs: Any,
    ) -> None:
        shape = getattr(action_spec, "shape", (self.config.action_dim,))
        action_dim = int(shape[0])
        self.mapping = build_reduced_action_mapping(
            action_dim=action_dim,
            hand_joint_names=hand_joint_names,
            actuator_names=actuator_names,
        )
        if action_spec is not None:
            self.minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1).copy()
            self.maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1).copy()
        else:
            self.minimum = np.full((action_dim,), -np.inf, dtype=np.float32)
            self.maximum = np.full((action_dim,), np.inf, dtype=np.float32)
        self.integral = np.zeros((action_dim,), dtype=np.float32)
        self.previous_error = None
        if initial_hand_qpos is not None:
            self.previous_control = self.hand_state_to_action(initial_hand_qpos)
        else:
            self.previous_control = None
        if trajectory_hand_qpos is not None:
            trajectory = np.asarray(trajectory_hand_qpos, dtype=np.float32)
            if trajectory.ndim == 2 and trajectory.shape[1] == 46:
                self.trajectory_hand_qpos = trajectory.copy()
            else:
                self.trajectory_hand_qpos = None
        else:
            self.trajectory_hand_qpos = None

    def hand_state_to_action(self, hand_qpos: np.ndarray) -> np.ndarray:
        action = action_signals_from_hand_state(
            hand_qpos,
            self.mapping,
            projection=self.config.projection,
        )
        action[-1] = np.float32(self.config.sustain_value)
        return np.clip(action, self.minimum, self.maximum).astype(np.float32)

    @staticmethod
    def _minimum_jerk(alpha: float) -> float:
        x = float(np.clip(alpha, 0.0, 1.0))
        return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5

    @staticmethod
    def _minimum_jerk_derivative(alpha: float) -> float:
        x = float(np.clip(alpha, 0.0, 1.0))
        return 30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4

    def _preview_target_signal(
        self,
        *,
        source_t: int | None,
        fallback_target_signal: np.ndarray,
        substeps: int,
    ) -> np.ndarray:
        if self.trajectory_hand_qpos is None or source_t is None:
            return fallback_target_signal
        trajectory = self.trajectory_hand_qpos
        if trajectory.shape[0] == 0:
            return fallback_target_signal
        horizon = max(int(self.config.lookahead_substeps), 1)
        source_index = int(np.clip(int(source_t), 0, trajectory.shape[0] - 1))
        source_position = float(source_index) + float(horizon) / max(float(substeps), 1.0)
        lower = int(np.floor(source_position))
        upper = min(lower + 1, trajectory.shape[0] - 1)
        lower = int(np.clip(lower, 0, trajectory.shape[0] - 1))
        if upper == lower:
            target_qpos = trajectory[lower]
        else:
            frac = float(np.clip(source_position - lower, 0.0, 1.0))
            target_qpos = ((1.0 - frac) * trajectory[lower] + frac * trajectory[upper]).astype(np.float32)
        return action_signals_from_hand_state(
            target_qpos,
            self.mapping,
            projection=self.config.projection,
        )

    def _setpoint_signal(
        self,
        *,
        source_signal: np.ndarray,
        target_signal: np.ndarray,
        substep: int,
        substeps: int,
    ) -> np.ndarray:
        source = np.asarray(source_signal, dtype=np.float32).reshape(-1)
        target = np.asarray(target_signal, dtype=np.float32).reshape(-1)
        if self.config.setpoint_policy == "next":
            return target
        if self.config.setpoint_policy in {"linear", "minimum_jerk"}:
            denominator = max(int(substeps) - 1, 1)
            raw_alpha = float(np.clip(int(substep) / denominator, 0.0, 1.0))
            alpha = self._minimum_jerk(raw_alpha) if self.config.setpoint_policy == "minimum_jerk" else raw_alpha
            return (source + alpha * (target - source)).astype(np.float32)
        raise ValueError(f"Unknown setpoint policy: {self.config.setpoint_policy}")

    def _reference_signal_and_velocity(
        self,
        *,
        source_signal: np.ndarray,
        target_signal: np.ndarray,
        substep: int,
        simulation_timestep: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = np.asarray(source_signal, dtype=np.float32).reshape(-1)
        target = np.asarray(target_signal, dtype=np.float32).reshape(-1)
        delta = target - source
        horizon = max(int(self.config.lookahead_substeps), 1)
        denominator = max(horizon - 1, 1)
        raw_alpha = float(np.clip(int(substep) / denominator, 0.0, 1.0))
        duration_s = max(float(denominator) * float(simulation_timestep), 1e-8)
        if self.config.setpoint_policy == "next":
            return target.astype(np.float32), np.zeros_like(delta, dtype=np.float32)
        if self.config.setpoint_policy == "linear":
            return (
                (source + raw_alpha * delta).astype(np.float32),
                (delta / duration_s).astype(np.float32),
            )
        if self.config.setpoint_policy == "minimum_jerk":
            alpha = self._minimum_jerk(raw_alpha)
            alpha_dot = self._minimum_jerk_derivative(raw_alpha) / duration_s
            return (
                (source + alpha * delta).astype(np.float32),
                (alpha_dot * delta).astype(np.float32),
            )
        raise ValueError(f"Unknown setpoint policy: {self.config.setpoint_policy}")

    def compute_control(
        self,
        *,
        source_t: int | None = None,
        current_hand_qpos: np.ndarray,
        current_hand_qvel: np.ndarray | None = None,
        source_hand_qpos: np.ndarray,
        target_hand_qpos: np.ndarray,
        substep: int,
        substeps: int,
        simulation_timestep: float,
        dataset_timestep: float,
        **_kwargs: Any,
    ) -> np.ndarray:
        current_signal = action_signals_from_hand_state(
            current_hand_qpos,
            self.mapping,
            projection=self.config.projection,
        )
        source_signal = action_signals_from_hand_state(
            source_hand_qpos,
            self.mapping,
            projection=self.config.projection,
        )
        final_target_signal = action_signals_from_hand_state(
            target_hand_qpos,
            self.mapping,
            projection=self.config.projection,
        )
        final_target_signal = self._preview_target_signal(
            source_t=source_t,
            fallback_target_signal=final_target_signal,
            substeps=substeps,
        )
        desired_signal, reference_velocity = self._reference_signal_and_velocity(
            source_signal=source_signal,
            target_signal=final_target_signal,
            substep=substep,
            simulation_timestep=simulation_timestep,
        )
        error = desired_signal - current_signal
        error[-1] = 0.0

        dt = max(float(simulation_timestep), 1e-8)
        target_velocity = np.zeros_like(error)
        if self.config.use_target_velocity:
            target_velocity = reference_velocity.copy()
            target_velocity *= float(self.config.target_velocity_scale)

        if current_hand_qvel is not None:
            current_velocity = action_signals_from_hand_state(
                current_hand_qvel,
                self.mapping,
                projection=self.config.projection,
            )
            derivative_error = target_velocity - current_velocity
        elif self.previous_error is not None:
            derivative_error = (error - self.previous_error) / dt
        else:
            derivative_error = target_velocity
        derivative_error[-1] = 0.0

        self.integral = self.integral + error * dt
        limit = abs(float(self.config.gains.integral_limit))
        if limit > 0.0:
            self.integral = np.clip(self.integral, -limit, limit)
        self.integral[-1] = 0.0

        gains = self.config.gains
        feedforward_scale = float(self.config.feedforward_scale)
        feedforward_base = (
            feedforward_scale * desired_signal
            + (1.0 - feedforward_scale) * current_signal
        )
        control = (
            feedforward_base
            + float(gains.kp) * error
            + float(gains.kd) * derivative_error
            + float(gains.ki) * self.integral
        )
        control[-1] = np.float32(self.config.sustain_value)
        control = np.clip(control, self.minimum, self.maximum).astype(np.float32)
        self.previous_error = error.astype(np.float32)
        self.previous_control = control.copy()
        return control

    def metadata(self) -> dict[str, object]:
        return {
            "controller": "HandPIDController",
            "config": asdict(self.config),
            "mapping": mapping_to_jsonable(self.mapping),
            "action_units": "actuator_position_targets",
        }
