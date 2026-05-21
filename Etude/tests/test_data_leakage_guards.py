from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.data.rp1m_tracking_dataset import RP1MTrackingDataset, validate_split_integrity
from etude.features.inverse_dynamics_blocks import InverseDynamicsFeatureSpec
from etude.features.key_blocks import KeyFeatureSpec
from train_controller import _feature_config


def test_split_integrity_rejects_train_val_overlap() -> None:
    manifest = pd.DataFrame(
        {
            "episode_id": [1, 2],
            "path": ["episodes/a.npz", "episodes/b.npz"],
        }
    )
    splits = pd.DataFrame(
        {
            "episode_id": [1, 1, 2],
            "path": ["episodes/a.npz", "episodes/a.npz", "episodes/b.npz"],
            "split": ["train", "val", "test"],
        }
    )

    with pytest.raises(ValueError, match="Split leakage"):
        validate_split_integrity(manifest, splits)


def test_key_aware_dataset_does_not_use_target_keys_as_current_state_by_default(tmp_path: Path) -> None:
    _write_key_episode_dataset(tmp_path)

    dataset = RP1MTrackingDataset(
        tmp_path,
        feature_mode="key_aware",
        feature_config={"key_spec": {"lookahead_steps": (), "include_key_error": False}},
        split="train",
    )
    features = dataset[0]["features"].numpy().reshape(-1)

    state_block_start = 88
    current_state = features[state_block_start : state_block_start + 88]
    assert np.count_nonzero(current_state) == 0


def test_key_aware_teacher_key_state_requires_explicit_config(tmp_path: Path) -> None:
    _write_key_episode_dataset(tmp_path)

    dataset = RP1MTrackingDataset(
        tmp_path,
        feature_mode="key_aware",
        feature_config={
            "teacher_key_state": True,
            "key_spec": {"lookahead_steps": (), "include_key_error": False},
        },
        split="train",
    )
    features = dataset[0]["features"].numpy().reshape(-1)

    state_block_start = 88
    current_state = features[state_block_start : state_block_start + 88]
    assert np.flatnonzero(current_state).tolist() == [7]


def test_feature_config_defaults_teacher_key_state_off() -> None:
    assert _feature_config({})["teacher_key_state"] is False
    assert _feature_config({"features": {"teacher_key_state": True}})["teacher_key_state"] is True


def test_tracking_lookahead_uses_only_configured_steps() -> None:
    q_ref = np.zeros((5, 2), dtype=np.float32)
    q_ref[1] = 1.0
    q_ref[3] = 3.0

    features = build_tracking_features(
        q=np.zeros(2, dtype=np.float32),
        qdot=np.zeros(2, dtype=np.float32),
        q_ref=q_ref,
        qdot_ref=np.zeros_like(q_ref),
        t=0,
        previous_action=np.zeros(1, dtype=np.float32),
        spec=FeatureSpec(lookahead_steps=(1,), include_target_keys=False, include_fingertips=False),
    )

    assert 1.0 in features
    assert 3.0 not in features


def test_negative_future_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="lookahead_steps"):
        KeyFeatureSpec(lookahead_steps=(-1,))
    with pytest.raises(ValueError, match="future_steps"):
        InverseDynamicsFeatureSpec(future_steps=(-1,))


def test_desired_fingertips_are_not_used_as_observed_fingertips(tmp_path: Path) -> None:
    _write_fingertip_dataset(tmp_path)
    dataset = RP1MTrackingDataset(
        tmp_path,
        feature_mode="fingertip_phase",
        feature_config={
            "fingertip_spec": {
                "include_current": True,
                "include_desired": True,
                "include_error": False,
                "include_weights": False,
                "include_active_mask": False,
                "include_inactive_mask": False,
                "allow_missing": True,
            },
            "phase_spec": {"allow_missing": True},
        },
        split="train",
    )

    features = dataset[0]["features"].numpy().reshape(-1)
    current_block = features[:30]
    desired_block = features[30:60]
    assert np.all(current_block == 0.0)
    assert np.all(desired_block == 2.0)


def _write_key_episode_dataset(root: Path) -> None:
    episodes = root / "episodes"
    episodes.mkdir()
    q = np.zeros((2, 2), dtype=np.float32)
    target_keys = np.zeros((2, 88), dtype=np.float32)
    target_keys[0, 7] = 1.0
    np.savez_compressed(
        episodes / "episode_000000.npz",
        q=q,
        qdot=q,
        q_ref=q,
        qdot_ref=q,
        actions=np.zeros((2, 1), dtype=np.float32),
        target_keys=target_keys,
    )
    _write_manifest_and_splits(root)


def _write_fingertip_dataset(root: Path) -> None:
    episodes = root / "episodes"
    episodes.mkdir()
    q = np.zeros((2, 2), dtype=np.float32)
    np.savez_compressed(
        episodes / "episode_000000.npz",
        q=q,
        qdot=q,
        q_ref=q,
        qdot_ref=q,
        actions=np.zeros((2, 1), dtype=np.float32),
        desired_fingertips=np.full((2, 10, 3), 2.0, dtype=np.float32),
    )
    _write_manifest_and_splits(root)


def _write_manifest_and_splits(root: Path) -> None:
    row = {"episode_id": 0, "path": "episodes/episode_000000.npz", "source": "unit", "timesteps": 2}
    pd.DataFrame([row]).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame([{**row, "split": "train"}]).to_csv(root / "splits.csv", index=False)
