from __future__ import annotations

import numpy as np
import torch

from rhapsody.config import RhapsodyConfig
from rhapsody.constants import FINGERTIP_COORD_DIM, HAND_STATE_DIM, NUM_FINGERS
from rhapsody.data import RPIKArrays, active_mask_from_targets
from rhapsody.models import ResidualIKPolicy
from rhapsody.normalization import RhapsodyNormalizer
from rhapsody.reward import active_mean_error, fingertip_reward
from rhapsody.solver import RhapsodyIKSolver
from rhapsody.trainer import train_rhapsody


def test_active_mask_marks_finite_targets() -> None:
    target = np.zeros((2, NUM_FINGERS, FINGERTIP_COORD_DIM), dtype=np.float32)
    target[1, 3, 0] = np.nan
    mask = active_mask_from_targets(target)
    assert mask.shape == (2, NUM_FINGERS)
    assert float(mask[0].sum()) == NUM_FINGERS
    assert mask[1, 3] == 0.0


def test_normalizer_roundtrip() -> None:
    arrays = _synthetic_arrays(20)
    normalizer = RhapsodyNormalizer.fit(arrays)
    qpos = torch.from_numpy(arrays.expert_qpos[:4])
    tips = torch.from_numpy(arrays.target_fingertips[:4])
    assert torch.allclose(normalizer.denormalize_qpos(normalizer.normalize_qpos(qpos)), qpos)
    assert torch.allclose(
        normalizer.denormalize_fingertips(normalizer.normalize_fingertips(tips)),
        tips,
    )


def test_policy_output_shape_and_bound() -> None:
    policy = ResidualIKPolicy(hidden_dims=(32,), action_scale=1.25)
    target = torch.randn(5, NUM_FINGERS, FINGERTIP_COORD_DIM)
    mask = torch.ones(5, NUM_FINGERS)
    previous = torch.randn(5, HAND_STATE_DIM)
    predicted = policy(target, mask, previous)
    assert predicted.shape == (5, HAND_STATE_DIM)
    assert torch.max(torch.abs(predicted - previous)).item() <= 1.2501


def test_reward_prefers_closer_fingertips() -> None:
    target = torch.zeros(2, NUM_FINGERS, FINGERTIP_COORD_DIM)
    mask = torch.ones(2, NUM_FINGERS)
    close = torch.full_like(target, 0.01)
    far = torch.full_like(target, 0.10)
    qpos = torch.zeros(2, HAND_STATE_DIM)
    reward_close = fingertip_reward(
        predicted_fingertips=close,
        target_fingertips=target,
        active_mask=mask,
        predicted_qpos_norm=qpos,
        previous_qpos_norm=qpos,
        active_weight=1.0,
        max_error_weight=0.25,
        smoothness_weight=0.0,
    )
    reward_far = fingertip_reward(
        predicted_fingertips=far,
        target_fingertips=target,
        active_mask=mask,
        predicted_qpos_norm=qpos,
        previous_qpos_norm=qpos,
        active_weight=1.0,
        max_error_weight=0.25,
        smoothness_weight=0.0,
    )
    assert torch.all(reward_close > reward_far)
    assert torch.all(active_mean_error(close, target, mask) < active_mean_error(far, target, mask))


def test_training_and_solver_smoke_on_synthetic_pairs() -> None:
    arrays = _synthetic_arrays(96)
    config = RhapsodyConfig(
        policy_hidden_dims=(64,),
        fk_hidden_dims=(64,),
        batch_size=32,
        policy_samples_per_state=2,
        policy_exploration_std=0.20,
        imitation_weight=0.20,
        fk_learning_rate=1.0e-3,
        policy_learning_rate=1.0e-3,
        validation_fraction=0.25,
        seed=11,
    )
    result = train_rhapsody(arrays, config, fk_epochs=2, bc_epochs=1, policy_epochs=2, device="cpu")
    assert np.isfinite(result.validation_metrics.fingertip_mean_error_m)
    assert result.validation_metrics.samples > 0
    solver = RhapsodyIKSolver(
        normalizer=result.normalizer,
        policy=result.policy,
        fk_model=result.fk_model,
        device="cpu",
    )
    solution = solver.solve(
        arrays.target_fingertips[0],
        active_mask=arrays.active_mask[0],
        previous_qpos=arrays.previous_qpos[0],
        refinement_steps=1,
    )
    assert solution.qpos.shape == (HAND_STATE_DIM,)
    assert solution.predicted_fingertips.shape == (NUM_FINGERS, FINGERTIP_COORD_DIM)
    assert np.isfinite(solution.mean_error_m)


def _synthetic_arrays(count: int) -> RPIKArrays:
    rng = np.random.default_rng(1234)
    qpos = rng.normal(0.0, 0.35, size=(count, HAND_STATE_DIM)).astype(np.float32)
    previous = qpos + rng.normal(0.0, 0.03, size=qpos.shape).astype(np.float32)
    matrix = rng.normal(
        0.0,
        0.02,
        size=(NUM_FINGERS * FINGERTIP_COORD_DIM, HAND_STATE_DIM),
    ).astype(np.float32)
    bias = rng.normal(0.0, 0.04, size=(NUM_FINGERS * FINGERTIP_COORD_DIM,)).astype(np.float32)
    fingertips = (qpos @ matrix.T + bias).reshape(count, NUM_FINGERS, FINGERTIP_COORD_DIM)
    return RPIKArrays(
        target_fingertips=fingertips,
        active_mask=np.ones((count, NUM_FINGERS), dtype=np.float32),
        previous_qpos=previous,
        expert_qpos=qpos,
        song_names=("synthetic",),
    )
