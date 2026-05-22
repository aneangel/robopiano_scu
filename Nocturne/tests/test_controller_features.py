from __future__ import annotations

import numpy as np

from nocturne.controller_dataset import build_features_for_trajectory


def test_controller_feature_names_exclude_privileged_realized_outcomes() -> None:
    q = np.zeros((6, 46), dtype=np.float32)
    qdot = np.zeros_like(q)
    actions = np.zeros((6, 39), dtype=np.float32)
    goals = np.zeros((6, 89), dtype=np.float32)

    features, names = build_features_for_trajectory(q=q, qdot=qdot, actions=actions, goals=goals)

    assert features.shape[0] == 6
    forbidden = ("piano_state", "source_demo", "segment_id")
    assert not any(any(token in name for token in forbidden) for name in names)
