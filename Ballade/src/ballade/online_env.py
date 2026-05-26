from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ballade.constants import CONTROL_DT, SOURCE_DT


@dataclass(frozen=True, slots=True)
class BalladeObservation:
    q: np.ndarray
    qvel: np.ndarray
    fingertips: np.ndarray | None
    piano_activation: np.ndarray
    reward: float = 0.0
    step_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "qvel": self.qvel,
            "fingertips": self.fingertips,
            "piano_activation": self.piano_activation,
            "reward": self.reward,
            "step_index": self.step_index,
        }


@dataclass(slots=True)
class BalladeOnlineEnvConfig:
    source_dt: float = SOURCE_DT
    control_dt: float = CONTROL_DT
    seed: int = 0
    threshold: float = 0.5
    reduced_action_space: bool = True
    hand_anchor_y_offset: float | None = None
    auto_hand_anchor_y_offset: bool = True
    restore_initial_hand: bool = True
    set_initial_qvel: bool = True
    initial_qvel_scale: float = 0.5
    gravity_compensation: bool = False
    primitive_fingertip_collisions: bool = False
    disable_hand_collisions: bool = False


class BalladeOnlineEnv:
    """Thin action-only wrapper around rp1m_simulator/RoboPianist."""

    def __init__(
        self,
        *,
        trajectory: Any,
        output_dir: str | Path,
        config: BalladeOnlineEnvConfig | None = None,
        environment_name: str | None = None,
    ) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from dataclasses import replace

        from rp1m_simulator import simulator as rp1m

        self.rp1m = rp1m
        self.trajectory = trajectory
        self.config = config or BalladeOnlineEnvConfig()
        self.output_dir = rp1m.ensure_dir(output_dir)
        self.environment_name = environment_name or trajectory.environment_name
        self._step_index = 0
        self._last_reward = 0.0
        self._closed = False

        goals = np.asarray(trajectory.goals, dtype=np.float32)
        substeps = int(round(float(self.config.source_dt) / float(self.config.control_dt)))
        dense_goals = np.repeat(goals[:, :88], substeps, axis=0)
        self.midi_proto_path = rp1m.write_goals_proto(
            dense_goals,
            self.output_dir / "ballade_target_goals.proto",
            dt=float(self.config.control_dt),
            title=f"Ballade {trajectory.song_key} demo {trajectory.demo_id}",
        )
        base_rollout_config = rp1m.RolloutConfig(
            mode="action",
            dataset_timestep=float(self.config.source_dt),
            simulation_timestep=float(self.config.control_dt),
            reduced_action_space=bool(self.config.reduced_action_space),
            action_source_scale="normalized_minus_one_to_one",
            hand_anchor_y_offset=self.config.hand_anchor_y_offset,
            auto_hand_anchor_y_offset=bool(self.config.auto_hand_anchor_y_offset),
            restore_initial_hand=bool(self.config.restore_initial_hand),
            set_hand_qvel=bool(self.config.set_initial_qvel),
            initial_hand_qvel_scale=float(self.config.initial_qvel_scale),
            gravity_compensation=bool(self.config.gravity_compensation),
            primitive_fingertip_collisions=bool(self.config.primitive_fingertip_collisions),
            disable_hand_collisions=bool(self.config.disable_hand_collisions),
            seed=int(self.config.seed),
            threshold=float(self.config.threshold),
            render_mp4=False,
            render_audio=False,
        )
        calibration = rp1m._calibrate_hand_anchor_y_offset(
            trajectory,
            base_rollout_config,
            self.midi_proto_path,
            float(self.config.control_dt),
        )
        self.effective_rollout_config = replace(
            base_rollout_config,
            hand_anchor_y_offset=calibration.get("effective_hand_anchor_y_offset"),
            auto_hand_anchor_y_offset=False,
        )
        self.hand_anchor_calibration = calibration
        self.env, self.load_info = rp1m._load_env(
            self.effective_rollout_config,
            self.midi_proto_path,
            self.environment_name,
            float(self.config.control_dt),
        )
        self.task = None
        self.physics = None
        self.piano = None
        self.action_spec_value = None

    @classmethod
    def from_rp1m(
        cls,
        *,
        rp1m_root: str | Path,
        song_key: str,
        demo_id: int,
        output_dir: str | Path,
        config: BalladeOnlineEnvConfig | None = None,
    ) -> "BalladeOnlineEnv":
        from rp1m_simulator import load_rp1m_trajectory

        trajectory = load_rp1m_trajectory(rp1m_root, song_key, int(demo_id), include_reference_piano_states=True)
        return cls(trajectory=trajectory, output_dir=output_dir, config=config)

    def reset(self) -> BalladeObservation:
        self.env.reset()
        self.task, self.physics, self.piano = self.rp1m._locate_task_physics_piano(self.env)
        if self.effective_rollout_config.hand_anchor_application == "post_reset":
            self.rp1m._apply_post_reset_hand_anchor_y_offset(
                self.task,
                self.physics,
                self.effective_rollout_config.hand_anchor_y_offset,
            )
        if self.config.restore_initial_hand:
            hand = np.asarray(self.trajectory.hand_joints, dtype=np.float32)
            self.rp1m._set_hand_qpos(self.task, self.physics, hand[0])
            if self.config.set_initial_qvel and hand.shape[0] > 1:
                qvel = (hand[1] - hand[0]) / float(self.config.source_dt)
                self.rp1m._set_hand_qvel(
                    self.task,
                    self.physics,
                    float(self.config.initial_qvel_scale) * qvel,
                )
            if hasattr(self.physics, "forward"):
                self.physics.forward()
        self.action_spec_value = self.env.action_spec()
        self._step_index = 0
        self._last_reward = 0.0
        return self.observe()

    def action_spec(self) -> Any:
        if self.action_spec_value is None:
            self.action_spec_value = self.env.action_spec()
        return self.action_spec_value

    @property
    def action_dim(self) -> int:
        return int(self.action_spec().shape[0])

    def normalize_action(self, actuator_action: np.ndarray) -> np.ndarray:
        spec = self.action_spec()
        low = np.asarray(spec.minimum, dtype=np.float32).reshape(-1)
        high = np.asarray(spec.maximum, dtype=np.float32).reshape(-1)
        action = np.asarray(actuator_action, dtype=np.float32).reshape(-1)
        width = min(action.size, low.size)
        out = np.zeros_like(low, dtype=np.float32)
        out[:width] = 2.0 * (action[:width] - low[:width]) / np.maximum(high[:width] - low[:width], 1e-6) - 1.0
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def actuator_action(self, normalized_action: np.ndarray) -> np.ndarray:
        return self.rp1m._prepare_control(
            np.asarray(normalized_action, dtype=np.float32),
            self.action_spec(),
            self.effective_rollout_config,
        )

    def _hand_qvel(self) -> np.ndarray:
        if self.task is None or self.physics is None:
            return np.zeros((0,), dtype=np.float32)
        values: list[float] = []
        for joint in self.rp1m._hand_joint_handles(self.task):
            try:
                qvel = np.asarray(self.physics.bind(joint).qvel, dtype=np.float32).reshape(-1)
                values.append(float(qvel[0]) if qvel.size else 0.0)
            except Exception:
                values.append(0.0)
        return np.asarray(values, dtype=np.float32)

    def observe(self) -> BalladeObservation:
        if self.task is None or self.physics is None:
            self.task, self.physics, self.piano = self.rp1m._locate_task_physics_piano(self.env)
        q = self.rp1m._capture_hand_qpos(self.task, self.physics)
        fingertips = self.rp1m._capture_fingertips(self.task, self.physics)
        piano = self.rp1m._capture_piano_activation(self.env)
        return BalladeObservation(
            q=np.zeros((0,), dtype=np.float32) if q is None else np.asarray(q, dtype=np.float32),
            qvel=self._hand_qvel(),
            fingertips=None if fingertips is None else np.asarray(fingertips, dtype=np.float32),
            piano_activation=(
                np.zeros((88,), dtype=np.float32) if piano is None else np.asarray(piano, dtype=np.float32)[:88]
            ),
            reward=float(self._last_reward),
            step_index=int(self._step_index),
        )

    def step_normalized(self, action: np.ndarray) -> tuple[BalladeObservation, float, bool, dict[str, Any]]:
        control = self.actuator_action(action)
        timestep = self.env.step(control)
        reward = float(timestep.reward or 0.0)
        self._last_reward = reward
        self._step_index += 1
        done = bool(timestep.last())
        return self.observe(), reward, done, {"control": control}

    def snapshot(self) -> dict[str, Any]:
        if self.physics is None:
            self.task, self.physics, self.piano = self.rp1m._locate_task_physics_piano(self.env)
        physics = self.physics
        if hasattr(physics, "get_state"):
            state = np.asarray(physics.get_state(), dtype=np.float64).copy()
        else:
            state = {
                "qpos": np.asarray(physics.data.qpos, dtype=np.float64).copy(),
                "qvel": np.asarray(physics.data.qvel, dtype=np.float64).copy(),
                "ctrl": np.asarray(physics.data.ctrl, dtype=np.float64).copy(),
            }
        time_value = float(getattr(getattr(physics, "data", None), "time", 0.0))
        return {
            "physics_state": state,
            "physics_time": time_value,
            "step_index": int(self._step_index),
            "last_reward": float(self._last_reward),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if self.physics is None:
            self.task, self.physics, self.piano = self.rp1m._locate_task_physics_piano(self.env)
        state = snapshot["physics_state"]
        if hasattr(self.physics, "set_state") and not isinstance(state, dict):
            self.physics.set_state(state)
        else:
            self.physics.data.qpos[:] = state["qpos"]
            self.physics.data.qvel[:] = state["qvel"]
            self.physics.data.ctrl[:] = state["ctrl"]
        if hasattr(self.physics.data, "time"):
            self.physics.data.time = float(snapshot.get("physics_time", self.physics.data.time))
        if hasattr(self.physics, "forward"):
            self.physics.forward()
        self._step_index = int(snapshot.get("step_index", self._step_index))
        self._last_reward = float(snapshot.get("last_reward", self._last_reward))

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self._closed = True
