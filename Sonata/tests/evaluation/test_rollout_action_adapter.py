from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.causal_rollout_contract import CausalRolloutConfig  # noqa: E402
from sonata.evaluation.primitive_online_eval import PrimitiveInstance, _execute_action_rollout  # noqa: E402
from sonata.evaluation.task_config import (  # noqa: E402
    adapt_action_to_spec,
    build_rollout_task_kwargs,
    validate_rollout_action_dim,
)


class DummyActionSpec:
    def __init__(self, minimum: list[float], maximum: list[float]):
        self.minimum = np.asarray(minimum, dtype=np.float32)
        self.maximum = np.asarray(maximum, dtype=np.float32)
        self.shape = self.minimum.shape
        self.dtype = np.float32


class DummyTimeStep:
    reward = 0.0

    def last(self) -> bool:
        return False


class DummyEnv:
    def __init__(self, spec: DummyActionSpec):
        self._spec = spec
        self.stepped: list[np.ndarray] = []

    def reset(self) -> DummyTimeStep:
        return DummyTimeStep()

    def action_spec(self) -> DummyActionSpec:
        return self._spec

    def step(self, control: np.ndarray) -> DummyTimeStep:
        self.stepped.append(np.asarray(control, dtype=np.float32).copy())
        return DummyTimeStep()


def test_build_rollout_task_kwargs_uses_reduced_action_space_for_39d_actions() -> None:
    kwargs = build_rollout_task_kwargs(control_timestep=0.05, expected_action_dim=39)

    assert kwargs["control_timestep"] == 0.05
    assert kwargs["n_steps_lookahead"] == 1
    assert kwargs["reduced_action_space"] is True


def test_adapt_action_to_spec_maps_normalized_actions_to_bounds() -> None:
    spec = DummyActionSpec(minimum=[0.0, 10.0, -2.0], maximum=[2.0, 20.0, 2.0])

    control = adapt_action_to_spec(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32), spec)

    np.testing.assert_allclose(control, np.asarray([0.0, 15.0, 2.0], dtype=np.float32))


def test_validate_rollout_action_dim_requires_exact_match_by_default() -> None:
    validate_rollout_action_dim(actual_action_dim=39, expected_action_dim=39, environment_name="env")

    with pytest.raises(ValueError, match="action_dim=40"):
        validate_rollout_action_dim(actual_action_dim=40, expected_action_dim=39, environment_name="env")
    with pytest.raises(ValueError, match="action_dim=38"):
        validate_rollout_action_dim(actual_action_dim=38, expected_action_dim=39, environment_name="env")


def test_primitive_execute_action_rollout_scales_normalized_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    import sonata.evaluation.primitive_online_eval as primitive_eval

    spec = DummyActionSpec(minimum=[0.0, 10.0, -2.0], maximum=[2.0, 20.0, 2.0])
    env = DummyEnv(spec)
    instance = _dummy_instance()
    monkeypatch.setattr(primitive_eval, "_capture_piano_state", lambda env: np.zeros((89,), dtype=np.float32))
    monkeypatch.setattr(primitive_eval, "_capture_hand_joint_state", lambda env: np.zeros((3,), dtype=np.float32))
    monkeypatch.setattr(primitive_eval, "_capture_fingertips", lambda env: np.zeros((15,), dtype=np.float32))
    monkeypatch.setattr(
        primitive_eval,
        "collect_fingertip_key_contacts",
        lambda *args, **kwargs: SimpleNamespace(contact_roll=None, contact_method="unavailable", notes=[]),
    )

    _execute_action_rollout(
        env=env,
        instance=instance,
        actions=np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32),
        action_dim=3,
        restore_mode="unsafe_legacy",
        causal_config=CausalRolloutConfig.from_mapping({"enabled": False, "restore_mode": "unsafe_legacy"}),
        rollout_config={"action_source_scale": "normalized_minus_one_to_one"},
        render_video=False,
    )

    assert len(env.stepped) == 1
    np.testing.assert_allclose(env.stepped[0], np.asarray([0.0, 15.0, 2.0], dtype=np.float32))


def test_primitive_zero_action_path_uses_same_normalized_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import sonata.evaluation.primitive_online_eval as primitive_eval

    spec = DummyActionSpec(minimum=[0.0, 10.0, -2.0], maximum=[2.0, 20.0, 2.0])
    env = DummyEnv(spec)
    instance = _dummy_instance()
    monkeypatch.setattr(primitive_eval, "_capture_piano_state", lambda env: np.zeros((89,), dtype=np.float32))
    monkeypatch.setattr(primitive_eval, "_capture_hand_joint_state", lambda env: np.zeros((3,), dtype=np.float32))
    monkeypatch.setattr(primitive_eval, "_capture_fingertips", lambda env: np.zeros((15,), dtype=np.float32))
    monkeypatch.setattr(
        primitive_eval,
        "collect_fingertip_key_contacts",
        lambda *args, **kwargs: SimpleNamespace(contact_roll=None, contact_method="unavailable", notes=[]),
    )

    _execute_action_rollout(
        env=env,
        instance=instance,
        actions=np.zeros((1, 3), dtype=np.float32),
        action_dim=3,
        restore_mode="unsafe_legacy",
        causal_config=CausalRolloutConfig.from_mapping({"enabled": False, "restore_mode": "unsafe_legacy"}),
        rollout_config={"action_source_scale": "normalized_minus_one_to_one"},
        render_video=False,
    )

    np.testing.assert_allclose(env.stepped[0], np.asarray([1.0, 15.0, 0.0], dtype=np.float32))


def _dummy_instance() -> PrimitiveInstance:
    return PrimitiveInstance(
        segment_id="seg0",
        primitive_id="primitive_000",
        song_id="song",
        demo_id="episode",
        episode_id="episode",
        split="train",
        start_frame=0,
        end_frame=1,
        duration_steps=1,
        control_timestep=0.05,
        hand=None,
        start_joint_state=np.zeros((3,), dtype=np.float32),
        start_joint_velocity=None,
        start_fingertip_state=None,
        start_piano_state=np.zeros((89,), dtype=np.float32),
        intended_keys=(),
        realized_keys_gt=(),
        onset_frames_gt=(),
        release_frames_gt=(),
        conditioning_features=np.zeros((1,), dtype=np.float32),
        chunk_path="",
        chunk_index=0,
        raw_chunk_path=None,
        raw_chunk_index=None,
        gmr_target_name="actions",
        primitive_prior_path=None,
    )
