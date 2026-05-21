from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from etude.evaluation.metrics import align_reference, error_profile_metrics
from etude.evaluation.rollout import rollout_controller
from etude.robopianist.state_mapping import StateMapping


@dataclass
class _FakeTimeStep:
    observation: dict[str, np.ndarray]
    terminal: bool = False

    def last(self) -> bool:
        return self.terminal


class _FakeEnv:
    def __init__(self, observations: list[dict[str, np.ndarray]], dt: float) -> None:
        self._observations = observations
        self._dt = float(dt)
        self._index = 0

    def reset(self) -> _FakeTimeStep:
        self._index = 0
        return _FakeTimeStep(self._observations[self._index], terminal=False)

    def step(self, action: np.ndarray) -> _FakeTimeStep:
        del action
        self._index += 1
        terminal = self._index >= len(self._observations) - 1
        return _FakeTimeStep(self._observations[min(self._index, len(self._observations) - 1)], terminal=terminal)

    def control_timestep(self) -> float:
        return self._dt


class _SpyController:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, np.ndarray]]] = []

    def reset(self, q_ref: np.ndarray, qdot_ref: np.ndarray | None = None, metadata: dict[str, np.ndarray] | None = None) -> None:
        del q_ref, qdot_ref, metadata
        self.calls.clear()

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        self.calls.append((t, dict(obs)))
        return np.zeros(46, dtype=np.float32)


def _mapping() -> StateMapping:
    return StateMapping(
        qpos_indices_46=list(range(46)),
        qvel_indices_46=list(range(46)),
        action_indices=None,
        fingertip_indices=None,
        key_state_indices=list(range(88)),
        action_low=-np.ones(46, dtype=np.float32),
        action_high=np.ones(46, dtype=np.float32),
    )


def test_rollout_repeats_reference_indices_for_slower_plan_dt() -> None:
    observations = []
    for step in range(8):
        observations.append(
            {
                "q": np.full(46, step, dtype=np.float32),
                "qdot": np.zeros(46, dtype=np.float32),
                "key_state": np.full(88, step % 2, dtype=np.float32),
            }
        )
    env = _FakeEnv(observations, dt=0.01)
    controller = _SpyController()
    q_ref = np.zeros((3, 46), dtype=np.float32)

    rollout = rollout_controller(
        env,
        controller,
        _mapping(),
        q_ref,
        metadata={"dt": 0.02},
    )

    assert rollout["reference_indices"].tolist() == [0, 0, 1, 1, 2, 2]
    assert rollout["key_state"].shape == (6, 88)
    assert [t for t, _ in controller.calls] == [0, 0, 1, 1, 2, 2]
    assert all("key_state" not in obs for _, obs in controller.calls)


def test_align_reference_and_error_profile_metrics_capture_drift() -> None:
    reference = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    aligned = align_reference(reference, np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32))
    current = np.asarray([[0.0], [0.2], [1.0], [1.4], [2.2], [2.8]], dtype=np.float32)

    metrics = error_profile_metrics(current, aligned, prefix="tracking/joint", dt=0.01)

    assert aligned.shape == current.shape
    assert metrics["tracking/joint_error_final"] > metrics["tracking/joint_error_initial"]
    assert metrics["tracking/joint_error_drift"] > 0.0
    assert metrics["tracking/joint_error_slope_per_s"] > 0.0
