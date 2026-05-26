from __future__ import annotations

import numpy as np
import torch

from ballade.costs import hand_q_tracking_cost, keypress_cost, total_tracking_cost
from ballade.models import ResidualMLPController


def test_costs_are_finite_and_improve_with_better_q() -> None:
    target = {
        "target_q_micro": np.asarray([1.0, 2.0], dtype=np.float32),
        "target_qvel_micro": np.asarray([0.0, 0.0], dtype=np.float32),
        "goal_key_mask": np.zeros((88,), dtype=np.float32),
        "microstep_phase": 1.0,
    }
    obs_bad = {"q": np.asarray([0.0, 0.0]), "qvel": np.zeros(2), "piano_activation": np.zeros(88)}
    obs_good = {"q": np.asarray([1.0, 2.0]), "qvel": np.zeros(2), "piano_activation": np.zeros(88)}
    assert hand_q_tracking_cost(obs_good, target) < hand_q_tracking_cost(obs_bad, target)
    cost = total_tracking_cost(obs_bad, target, np.zeros(2), np.zeros(2), None)
    assert np.isfinite(cost)


def test_keypress_cost_rewards_target_activation_and_penalizes_mispress() -> None:
    target = {"goal_key_mask": np.zeros((88,), dtype=np.float32)}
    target["goal_key_mask"][40] = 1.0
    obs_hit = {"piano_activation": np.zeros((88,), dtype=np.float32)}
    obs_hit["piano_activation"][40] = 1.0
    obs_miss = {"piano_activation": np.zeros((88,), dtype=np.float32)}
    obs_false = {"piano_activation": np.zeros((88,), dtype=np.float32)}
    obs_false["piano_activation"][41] = 1.0
    assert keypress_cost(obs_hit, target) < keypress_cost(obs_miss, target)
    assert keypress_cost(obs_false, target) > keypress_cost(obs_hit, target)


def test_residual_model_forward_shape_and_clamp() -> None:
    model = ResidualMLPController(feature_dim=5, action_dim=3, hidden_dim=8, hidden_layers=1, residual_scale=10.0)
    features = torch.randn(4, 5)
    base = torch.zeros(4, 3)
    out = model(features, base)
    assert tuple(out.shape) == (4, 3)
    assert torch.all(out <= 1.0)
    assert torch.all(out >= -1.0)
