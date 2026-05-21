from __future__ import annotations

import numpy as np

from etude.data.target_schema import (
    compute_time_to_next_press,
    metadata_at_timestep,
    standardize_controller_metadata,
)


def test_standardize_controller_metadata_derives_goal_fields() -> None:
    target_keys = np.zeros((5, 88), dtype=np.float32)
    target_keys[2, 40] = 1.0
    metadata = standardize_controller_metadata(
        {"target_keys": target_keys, "dt": 0.02},
        q_ref=np.zeros((5, 46), dtype=np.float32),
    )

    assert metadata["target_keys"].shape == (5, 88)
    assert metadata["target_key_lookahead"].shape == (5, 3, 88)
    assert np.isclose(metadata["time_to_next_press"][0], 0.04)
    assert metadata["phase_schedule"].tolist() == [
        "recovery",
        "approach",
        "contact",
        "release",
        "recovery",
    ]


def test_metadata_at_timestep_exposes_now_and_sequence_fields() -> None:
    target_keys = np.zeros((3, 88), dtype=np.float32)
    target_keys[1, 12] = 1.0
    metadata = standardize_controller_metadata({"target_keys": target_keys, "dt": 0.01})

    step = metadata_at_timestep(metadata, 1)

    assert step["target_keys"].shape == (3, 88)
    assert np.flatnonzero(step["target_keys_now"]).tolist() == [12]
    assert step["phase"] == "contact"


def test_time_to_next_press_returns_zero_after_last_press() -> None:
    target_keys = np.zeros((4, 88), dtype=np.float32)
    target_keys[1, 0] = 1.0

    values = compute_time_to_next_press(target_keys, dt=0.5)

    assert values.tolist() == [0.5, 0.0, 0.0, 0.0]
