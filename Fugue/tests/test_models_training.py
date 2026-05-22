from __future__ import annotations

import torch

from fugue.models import CausalTransformerActionModel, MLPActionModel, TemporalConvActionModel
from fugue.training import action_reconstruction_loss


def test_mlp_action_model_outputs_chunk_shape() -> None:
    model = MLPActionModel(input_dim=92, action_dim=39, chunk_horizon=4, hidden_dim=32, hidden_layers=2)
    out = model(torch.zeros(5, 92))
    assert out.shape == (5, 4, 39)


def test_temporal_models_output_chunk_shape() -> None:
    transformer = CausalTransformerActionModel(
        token_dim=16,
        action_dim=39,
        chunk_horizon=2,
        d_model=32,
        nhead=4,
        num_layers=1,
    )
    tcn = TemporalConvActionModel(token_dim=16, action_dim=39, chunk_horizon=2, d_model=32, layers=2)
    tokens = torch.zeros(3, 8, 16)
    assert transformer(tokens).shape == (3, 2, 39)
    assert tcn(tokens).shape == (3, 2, 39)


def test_weighted_loss_matches_plain_loss_when_unweighted() -> None:
    pred = torch.ones(2, 1, 3)
    target = torch.zeros(2, 1, 3)
    press_weight = torch.ones(2, 1) * 3.0
    plain = action_reconstruction_loss(pred, target)
    weighted_off = action_reconstruction_loss(pred, target, press_weight=press_weight, press_weight_scale=0.0)
    assert torch.allclose(plain, weighted_off)
