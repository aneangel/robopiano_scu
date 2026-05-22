from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fugue.data import (
    FugueActionDataset,
    SampleConfig,
    audit_song,
    build_planner_next_feature,
    build_demo_manifest,
    compute_press_mask,
    finite_difference,
    fit_normalization_stats,
    validate_demo_split,
)


SONG_KEY = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0_0"


def test_finite_difference_uses_centered_interior() -> None:
    q = np.asarray([[0.0], [1.0], [3.0], [6.0]], dtype=np.float32)
    qvel = finite_difference(q, dt=0.5)
    np.testing.assert_allclose(qvel[:, 0], [2.0, 3.0, 5.0, 6.0])


def test_demo_split_rejects_overlap() -> None:
    manifest = pd.DataFrame({"demo_id": [0, 0], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="Demo split leakage"):
        validate_demo_split(manifest)


def test_audit_manifest_and_dataset_shapes(tmp_path: Path) -> None:
    root = _write_tiny_zarr(tmp_path)
    summary = audit_song(root, SONG_KEY)
    manifest = build_demo_manifest(summary=summary, seed=3)
    counts = validate_demo_split(manifest)
    assert set(counts) == {"train", "val", "test"}
    stats = fit_normalization_stats(dataset_root=root, manifest=manifest, song_key=SONG_KEY, dt=0.05)

    stateless = FugueActionDataset(
        dataset_root=root,
        manifest=manifest,
        stats=stats,
        song_key=SONG_KEY,
        split="train",
        sample_config=SampleConfig(feature_mode="stateless", chunk_horizon=1),
    )
    sample = stateless[0]
    assert sample["features"].shape == (92,)
    assert sample["actions"].shape == (1, 39)
    assert sample["press_weight"].shape == (1,)

    history_cfg = SampleConfig(
        feature_mode="history",
        history=3,
        goal_horizon=4,
        include_action_history=True,
        include_goals=True,
        chunk_horizon=2,
    )
    history = FugueActionDataset(
        dataset_root=root,
        manifest=manifest,
        stats=stats,
        song_key=SONG_KEY,
        split="train",
        sample_config=history_cfg,
    )
    history_sample = history[0]
    expected_dim = 3 * 92 + 3 * 39 + 4 * 88
    assert history_sample["features"].shape == (expected_dim,)
    assert history_sample["actions"].shape == (2, 39)

    sequence_cfg = SampleConfig(
        feature_mode="sequence",
        history=3,
        goal_horizon=4,
        include_action_history=True,
        include_goals=True,
        chunk_horizon=2,
    )
    sequence = FugueActionDataset(
        dataset_root=root,
        manifest=manifest,
        stats=stats,
        song_key=SONG_KEY,
        split="train",
        sample_config=sequence_cfg,
    )
    sequence_sample = sequence[0]
    expected_token_dim = 92 + 39 + 88 + 2
    assert sequence.feature_dim == expected_token_dim
    assert sequence_sample["features"].shape == (7, expected_token_dim)
    assert sequence_sample["actions"].shape == (2, 39)

    planner_cfg = SampleConfig(feature_mode="planner_next", lookahead=1, include_qpos=True, include_qvel=True)
    planner = FugueActionDataset(
        dataset_root=root,
        manifest=manifest,
        stats=stats,
        song_key=SONG_KEY,
        split="train",
        sample_config=planner_cfg,
    )
    planner_sample = planner[0]
    assert planner.feature_dim == 184
    assert planner_sample["features"].shape == (184,)
    assert planner_sample["actions"].shape == (1, 39)


def test_planner_next_feature_builder_matches_expected_shape() -> None:
    cfg = SampleConfig(feature_mode="planner_next", include_qpos=True, include_qvel=True)
    feature = build_planner_next_feature(
        current_q_norm=np.zeros((46,), dtype=np.float32),
        current_qvel_norm=np.ones((46,), dtype=np.float32),
        target_q_norm=np.ones((46,), dtype=np.float32) * 2.0,
        config=cfg,
    )
    assert feature.shape == (184,)
    np.testing.assert_allclose(feature[:46], 0.0)
    np.testing.assert_allclose(feature[46:92], 1.0)
    np.testing.assert_allclose(feature[92:138], 2.0)
    np.testing.assert_allclose(feature[138:184], 2.0)


def test_planner_next_feature_builder_uses_full_target_window() -> None:
    cfg = SampleConfig(
        feature_mode="planner_next",
        goal_horizon=3,
        include_qpos=True,
        include_qvel=True,
        include_future_qvel=True,
    )
    target = np.arange(3 * 46, dtype=np.float32).reshape(3, 46)
    feature = build_planner_next_feature(
        current_q_norm=np.ones((46,), dtype=np.float32),
        current_qvel_norm=np.zeros((46,), dtype=np.float32),
        target_q_norm=target,
        target_qvel_norm=np.ones((3, 46), dtype=np.float32) * 3.0,
        config=cfg,
    )
    assert feature.shape == (92 + 3 * 46 + 3 * 46 + 3 * 46,)
    np.testing.assert_allclose(feature[92 : 92 + 3 * 46], target.reshape(-1))
    np.testing.assert_allclose(feature[92 + 3 * 46 : 92 + 6 * 46], (target - 1.0).reshape(-1))
    np.testing.assert_allclose(feature[-3 * 46 :], 3.0)


def test_press_mask_dilates_active_and_onset_frames() -> None:
    goals = np.zeros((8, 88), dtype=np.float32)
    goals[4, 10] = 1.0
    mask = compute_press_mask(goals, window=1)
    assert np.flatnonzero(mask).tolist() == [3, 4, 5]


def _write_tiny_zarr(tmp_path: Path) -> Path:
    import zarr

    root_path = tmp_path / "rp1m_repertoire.zarr"
    root = zarr.open(str(root_path), mode="w")
    group = root.create_group(SONG_KEY)
    rng = np.random.default_rng(0)
    demos, steps = 6, 10
    q = rng.normal(size=(demos, steps, 46)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(demos, steps, 39)).astype(np.float32)
    goals = np.zeros((demos, steps, 89), dtype=np.float32)
    goals[:, 4, 10] = 1.0
    fingertips = rng.normal(size=(demos, steps, 30)).astype(np.float32)
    group.create_dataset("hand_joints", data=q, chunks=(1, steps, 46))
    group.create_dataset("actions", data=actions, chunks=(1, steps, 39))
    group.create_dataset("goals", data=goals, chunks=(1, steps, 89))
    group.create_dataset("hand_fingertips", data=fingertips, chunks=(1, steps, 30))
    return root_path
