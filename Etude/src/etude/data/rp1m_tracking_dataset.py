from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.features.fingertip_phase_blocks import (
    FingertipFeatureSpec,
    PhaseFeatureSpec,
    build_fingertip_phase_features,
)
from etude.features.inverse_dynamics_blocks import (
    InverseDynamicsFeatureSpec,
    build_inverse_dynamics_features,
)
from etude.features.key_blocks import KeyFeatureSpec, build_key_features


class RP1MTrackingDataset(Dataset):
    """Dataset over Etude episode `.npz` files listed in a manifest."""

    def __init__(
        self,
        dataset_root: str | Path,
        sequence_length: int = 1,
        feature_spec: FeatureSpec | None = None,
        feature_mode: str = "tracking",
        feature_config: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.feature_spec = feature_spec or FeatureSpec()
        self.feature_mode = str(feature_mode)
        self.feature_config = dict(feature_config or {})
        manifest_path = self.dataset_root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = pd.read_csv(manifest_path)
        self._episodes: list[dict[str, np.ndarray]] = []
        self._index: list[tuple[int, int]] = []
        for episode_idx, row in self.manifest.iterrows():
            path = self.dataset_root / str(row["path"])
            with np.load(path, allow_pickle=False) as npz:
                episode = {key: np.asarray(npz[key]) for key in npz.files}
            length = int(episode["q"].shape[0])
            for t in range(max(0, length - self.sequence_length + 1)):
                self._index.append((episode_idx, t))
            self._episodes.append(episode)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, start = self._index[idx]
        episode = self._episodes[episode_idx]
        features = []
        actions = []
        previous_action = np.zeros(episode["actions"].shape[1], dtype=np.float32)
        if start > 0:
            previous_action = episode["actions"][start - 1].astype(np.float32)
        for offset in range(self.sequence_length):
            t = start + offset
            feat = self._build_features(episode, t, previous_action)
            features.append(feat)
            action = episode["actions"][t].astype(np.float32)
            actions.append(action)
            previous_action = action
        return {
            "features": torch.from_numpy(np.stack(features)),
            "actions": torch.from_numpy(np.stack(actions)),
        }

    def _build_features(
        self,
        episode: dict[str, np.ndarray],
        t: int,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        if self.feature_mode == "tracking":
            return build_tracking_features(
                q=episode["q"][t],
                qdot=episode["qdot"][t],
                q_ref=episode["q_ref"],
                qdot_ref=episode["qdot_ref"],
                t=t,
                previous_action=previous_action,
                target_keys=episode.get("target_keys"),
                fingertips=episode.get("fingertips"),
                spec=self.feature_spec,
            )

        if self.feature_mode == "key_aware":
            block_cfg = dict(self.feature_config.get("key_spec", {}))
            return build_key_features(
                t=t,
                target_keys=episode.get("target_keys"),
                key_state=episode.get("target_keys"),
                metadata={"dt": float(episode.get("dt", 0.005))},
                spec=KeyFeatureSpec(**block_cfg) if block_cfg else None,
            )

        if self.feature_mode == "fingertip_phase":
            fingertip_cfg = dict(self.feature_config.get("fingertip_spec", {}))
            phase_cfg = dict(self.feature_config.get("phase_spec", {}))
            fingertips = _reshape_fingertips(episode.get("fingertips"))
            current = fingertips[t] if fingertips is not None else None
            return build_fingertip_phase_features(
                t=t,
                metadata={"dt": float(episode.get("dt", 0.005))},
                target_keys=episode.get("target_keys"),
                current_fingertips=current,
                desired_fingertips=current,
                fingertip_weights=np.ones((10,), dtype=np.float32) if current is not None and current.shape[0] == 10 else None,
                fingertip_spec=FingertipFeatureSpec(**fingertip_cfg) if fingertip_cfg else None,
                phase_spec=PhaseFeatureSpec(**phase_cfg) if phase_cfg else None,
            )

        if self.feature_mode == "inverse_dynamics":
            inverse_cfg = dict(self.feature_config.get("inverse_dynamics_spec", {}))
            return build_inverse_dynamics_features(
                q=episode["q"][t],
                qdot=episode["qdot"][t],
                q_ref=episode["q_ref"],
                t=t,
                fingertips=episode.get("fingertips"),
                fingertip_ref=episode.get("fingertips"),
                target_keys=episode.get("target_keys"),
                previous_action=previous_action,
                spec=InverseDynamicsFeatureSpec(**inverse_cfg) if inverse_cfg else None,
            )

        raise ValueError(f"Unsupported RP1MTrackingDataset feature_mode: {self.feature_mode}")


def _reshape_fingertips(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        return array
    if array.ndim == 2 and array.shape[1] % 3 == 0:
        return array.reshape(array.shape[0], array.shape[1] // 3, 3).astype(np.float32)
    return None
