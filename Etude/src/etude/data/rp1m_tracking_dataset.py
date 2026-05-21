from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.data.target_schema import standardize_controller_metadata
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
        split: str | None = None,
        splits_file: str | Path | None = None,
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
        if split is not None:
            split_path = Path(splits_file) if splits_file is not None else self.dataset_root / "splits.csv"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Requested split={split!r}, but split file does not exist: {split_path}"
                )
            splits = pd.read_csv(split_path)
            validate_split_integrity(self.manifest, splits, split_path=split_path)
            if "split" not in splits.columns:
                raise ValueError(f"Split file must contain a 'split' column: {split_path}")
            if "episode_id" in self.manifest.columns and "episode_id" in splits.columns:
                split_rows = splits[splits["split"].astype(str) == str(split)]
                keep = set(split_rows["episode_id"].astype(str))
                self.manifest = self.manifest[
                    self.manifest["episode_id"].astype(str).isin(keep)
                ].reset_index(drop=True)
            elif "path" in splits.columns:
                split_rows = splits[splits["split"].astype(str) == str(split)]
                keep = set(split_rows["path"].astype(str))
                self.manifest = self.manifest[
                    self.manifest["path"].astype(str).isin(keep)
                ].reset_index(drop=True)
            else:
                raise ValueError(
                    f"Split file must contain episode_id or path for filtering: {split_path}"
                )
            if self.manifest.empty:
                raise ValueError(f"Requested split={split!r} is empty in {split_path}")
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
            metadata = standardize_controller_metadata(
                episode,
                q_ref=episode["q_ref"],
                qdot_ref=episode["qdot_ref"],
            )
            return build_tracking_features(
                q=episode["q"][t],
                qdot=episode["qdot"][t],
                q_ref=episode["q_ref"],
                qdot_ref=episode["qdot_ref"],
                t=t,
                previous_action=previous_action,
                target_keys=episode.get("target_keys"),
                fingertips=episode.get("fingertips"),
                metadata=metadata,
                spec=self.feature_spec,
            )

        if self.feature_mode == "key_aware":
            block_cfg = dict(self.feature_config.get("key_spec", {}))
            return build_key_features(
                t=t,
                target_keys=episode.get("target_keys"),
                key_state=episode.get("target_keys")
                if bool(self.feature_config.get("teacher_key_state", False))
                else episode.get("key_state"),
                metadata=standardize_controller_metadata(
                    episode,
                    horizon=episode["q_ref"].shape[0],
                    dt=float(episode.get("dt", 0.005)),
                ),
                spec=KeyFeatureSpec(**block_cfg) if block_cfg else None,
            )

        if self.feature_mode == "fingertip_phase":
            fingertip_cfg = dict(self.feature_config.get("fingertip_spec", {}))
            phase_cfg = dict(self.feature_config.get("phase_spec", {}))
            fingertips = _reshape_fingertips(episode.get("fingertips"))
            current = fingertips[t] if fingertips is not None else None
            desired = _reshape_fingertips(episode.get("desired_fingertips"))
            desired_t = desired[t] if desired is not None else current
            weights = episode.get("fingertip_weights")
            active = episode.get("active_finger_mask")
            inactive = episode.get("inactive_finger_mask")
            return build_fingertip_phase_features(
                t=t,
                metadata=standardize_controller_metadata(
                    episode,
                    horizon=episode["q_ref"].shape[0],
                    dt=float(episode.get("dt", 0.005)),
                ),
                target_keys=episode.get("target_keys"),
                current_fingertips=current,
                desired_fingertips=desired_t,
                fingertip_weights=_timestep_vector(weights, t),
                active_finger_mask=_timestep_vector(active, t),
                inactive_finger_mask=_timestep_vector(inactive, t),
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


def validate_split_integrity(
    manifest: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    split_path: str | Path | None = None,
) -> dict[str, int]:
    """Validate that split membership is explicit and mutually exclusive."""
    label = str(split_path) if split_path is not None else "splits.csv"
    if "split" not in splits.columns:
        raise ValueError(f"Split file must contain a 'split' column: {label}")
    identifier = _split_identifier_column(manifest, splits, label=label)
    split_rows = splits[[identifier, "split"]].copy()
    split_rows[identifier] = split_rows[identifier].astype(str)
    split_rows["split"] = split_rows["split"].astype(str)
    duplicated_rows = split_rows.duplicated(subset=[identifier, "split"], keep=False)
    if bool(duplicated_rows.any()):
        values = sorted(split_rows.loc[duplicated_rows, identifier].unique().tolist())[:5]
        raise ValueError(f"Duplicate split rows in {label} for {identifier}: {values}")
    memberships = split_rows.groupby(identifier)["split"].nunique()
    overlaps = sorted(memberships[memberships > 1].index.astype(str).tolist())
    if overlaps:
        raise ValueError(f"Split leakage in {label}; {identifier} appears in multiple splits: {overlaps[:5]}")
    manifest_ids = set(manifest[identifier].astype(str))
    split_ids = set(split_rows[identifier])
    missing = sorted(manifest_ids - split_ids)
    if missing:
        raise ValueError(f"Split file {label} is missing manifest {identifier} values: {missing[:5]}")
    return {str(key): int(value) for key, value in split_rows["split"].value_counts().to_dict().items()}


def _split_identifier_column(manifest: pd.DataFrame, splits: pd.DataFrame, *, label: str) -> str:
    if "episode_id" in manifest.columns and "episode_id" in splits.columns:
        return "episode_id"
    if "path" in manifest.columns and "path" in splits.columns:
        return "path"
    raise ValueError(f"Split file must contain episode_id or path for filtering: {label}")


def _reshape_fingertips(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        return array
    if array.ndim == 2 and array.shape[1] % 3 == 0:
        return array.reshape(array.shape[0], array.shape[1] // 3, 3).astype(np.float32)
    return None


def _timestep_vector(value: np.ndarray | None, t: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array.astype(np.float32)
    if array.ndim == 2 and array.shape[0] > 0:
        return array[min(int(t), array.shape[0] - 1)].astype(np.float32)
    return None
