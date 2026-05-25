from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from rhapsody.constants import FINGERTIP_COORD_DIM, HAND_STATE_DIM, NUM_FINGERS


@dataclass(slots=True)
class RPIKArrays:
    """Flat RP1M examples for fingertip-to-hand-state IK."""

    target_fingertips: np.ndarray
    active_mask: np.ndarray
    previous_qpos: np.ndarray
    expert_qpos: np.ndarray
    song_names: tuple[str, ...] = ()
    source_song_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.target_fingertips = _as_float32(self.target_fingertips)
        self.active_mask = _as_float32(self.active_mask)
        self.previous_qpos = _as_float32(self.previous_qpos)
        self.expert_qpos = _as_float32(self.expert_qpos)
        if self.source_song_ids is None:
            self.source_song_ids = np.zeros((self.expert_qpos.shape[0],), dtype=np.int64)
        else:
            self.source_song_ids = np.asarray(self.source_song_ids, dtype=np.int64)
        if self.target_fingertips.ndim == 2:
            self.target_fingertips = self.target_fingertips.reshape(
                -1, NUM_FINGERS, FINGERTIP_COORD_DIM
            )
        if self.target_fingertips.shape[1:] != (NUM_FINGERS, FINGERTIP_COORD_DIM):
            raise ValueError(
                "target_fingertips must have shape [N, 10, 3], "
                f"got {self.target_fingertips.shape}"
            )
        if self.active_mask.shape != (len(self), NUM_FINGERS):
            raise ValueError(f"active_mask must have shape [N, 10], got {self.active_mask.shape}")
        if self.previous_qpos.shape != (len(self), HAND_STATE_DIM):
            raise ValueError(
                f"previous_qpos must have shape [N, {HAND_STATE_DIM}], "
                f"got {self.previous_qpos.shape}"
            )
        if self.expert_qpos.shape != (len(self), HAND_STATE_DIM):
            raise ValueError(
                f"expert_qpos must have shape [N, {HAND_STATE_DIM}], got {self.expert_qpos.shape}"
            )
        if self.source_song_ids.shape != (len(self),):
            raise ValueError(f"source_song_ids must have shape [N], got {self.source_song_ids.shape}")

    def __len__(self) -> int:
        return int(self.expert_qpos.shape[0])

    def split(self, validation_fraction: float, *, seed: int) -> tuple["RPIKArrays", "RPIKArrays"]:
        if not 0.0 < float(validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self))
        val_count = max(1, int(round(len(self) * float(validation_fraction))))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        if train_indices.size == 0:
            raise ValueError("Not enough examples to create a non-empty train split")
        return self.take(train_indices), self.take(val_indices)

    def split_by_song(
        self,
        validation_song_count: int,
        *,
        seed: int,
    ) -> tuple["RPIKArrays", "RPIKArrays"]:
        unique_song_ids = np.unique(self.source_song_ids)
        holdout_count = int(validation_song_count)
        if holdout_count <= 0:
            raise ValueError("validation_song_count must be positive")
        if holdout_count >= unique_song_ids.size:
            raise ValueError(
                "validation_song_count must leave at least one training song; "
                f"got {holdout_count} for {unique_song_ids.size} songs"
            )
        rng = np.random.default_rng(seed)
        holdout_ids = rng.choice(unique_song_ids, size=holdout_count, replace=False)
        val_mask = np.isin(self.source_song_ids, holdout_ids)
        train_indices = np.flatnonzero(~val_mask)
        val_indices = np.flatnonzero(val_mask)
        if train_indices.size == 0 or val_indices.size == 0:
            raise ValueError("Song split produced an empty train or validation split")
        return self.take(train_indices), self.take(val_indices)

    def take(self, indices: np.ndarray) -> "RPIKArrays":
        idx = np.asarray(indices, dtype=np.int64)
        return RPIKArrays(
            target_fingertips=self.target_fingertips[idx],
            active_mask=self.active_mask[idx],
            previous_qpos=self.previous_qpos[idx],
            expert_qpos=self.expert_qpos[idx],
            song_names=self.song_names,
            source_song_ids=self.source_song_ids[idx],
        )

    def active_song_names(self) -> tuple[str, ...]:
        names = []
        for song_id in sorted(int(v) for v in np.unique(self.source_song_ids).tolist()):
            if 0 <= song_id < len(self.song_names):
                names.append(self.song_names[song_id])
            else:
                names.append(str(song_id))
        return tuple(names)


class RPIKDataset(Dataset):
    def __init__(self, arrays: RPIKArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "target_fingertips": torch.from_numpy(self.arrays.target_fingertips[index]),
            "active_mask": torch.from_numpy(self.arrays.active_mask[index]),
            "previous_qpos": torch.from_numpy(self.arrays.previous_qpos[index]),
            "expert_qpos": torch.from_numpy(self.arrays.expert_qpos[index]),
        }


def load_rp1m_ik_pairs(
    rp1m_root: str | Path,
    *,
    song_names: Iterable[str] | None = None,
    max_songs: int | None = None,
    random_songs: bool = False,
    num_demos: int = 2,
    frame_stride: int = 10,
    max_pairs: int | None = None,
    previous_mode: str = "adjacent",
    seed: int = 524,
) -> RPIKArrays:
    """Load RP1M fingertip and hand-state pairs for IK training.

    RP1M stores all fingertip sites for every frame, so the default active mask
    trains all ten fingertips. Online callers can pass a sparse mask when only a
    subset of fingertips should be constrained to key centers.
    """

    try:
        import zarr
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise ModuleNotFoundError("zarr is required to read RP1M.") from exc

    root = zarr.open_group(str(Path(rp1m_root)), mode="r")
    rng = np.random.default_rng(seed)
    available = sorted([name for name in root.keys() if hasattr(root[name], "keys")])
    if song_names is not None:
        selected = list(song_names)
    elif max_songs is not None and bool(random_songs):
        count = min(int(max_songs), len(available))
        selected = sorted(rng.choice(np.asarray(available, dtype=object), size=count, replace=False).tolist())
    else:
        selected = available
    if max_songs is not None and song_names is not None:
        selected = selected[: int(max_songs)]
    if not selected:
        raise ValueError(f"No RP1M songs selected from {rp1m_root}")
    if previous_mode not in {"adjacent", "random", "mean", "zero"}:
        raise ValueError(f"Unsupported previous_mode: {previous_mode}")

    targets: list[np.ndarray] = []
    prev_qpos: list[np.ndarray] = []
    expert_qpos: list[np.ndarray] = []
    source_song_ids: list[np.ndarray] = []
    song_to_id = {song_name: index for index, song_name in enumerate(selected)}

    for song_name in selected:
        if song_name not in root:
            raise KeyError(f"Song not found in RP1M: {song_name}")
        group = root[song_name]
        for required in ("hand_joints", "hand_fingertips"):
            if required not in group:
                raise KeyError(f"Song {song_name} missing RP1M array {required!r}")
        demo_count = int(group["hand_joints"].shape[0])
        demo_ids = np.arange(demo_count)
        if demo_count > int(num_demos):
            demo_ids = rng.choice(demo_ids, size=int(num_demos), replace=False)
        for demo_id in sorted(int(v) for v in demo_ids.tolist()):
            qpos = np.asarray(group["hand_joints"][demo_id], dtype=np.float32)
            fingertips = np.asarray(group["hand_fingertips"][demo_id], dtype=np.float32)
            if qpos.ndim != 2 or qpos.shape[1] != HAND_STATE_DIM:
                raise ValueError(f"Unexpected hand_joints shape for {song_name}: {qpos.shape}")
            if fingertips.ndim != 2 or fingertips.shape[1] != NUM_FINGERS * FINGERTIP_COORD_DIM:
                raise ValueError(
                    f"Unexpected hand_fingertips shape for {song_name}: {fingertips.shape}"
                )
            frame_ids = np.arange(0, min(qpos.shape[0], fingertips.shape[0]), int(frame_stride))
            qpos_frames = qpos[frame_ids]
            tip_frames = fingertips[frame_ids].reshape(-1, NUM_FINGERS, FINGERTIP_COORD_DIM)
            previous = qpos[np.maximum(frame_ids - 1, 0)]
            finite = (
                np.isfinite(qpos_frames).all(axis=1)
                & np.isfinite(previous).all(axis=1)
                & np.isfinite(tip_frames.reshape(tip_frames.shape[0], -1)).all(axis=1)
            )
            if bool(finite.any()):
                targets.append(tip_frames[finite])
                prev_qpos.append(previous[finite])
                expert_qpos.append(qpos_frames[finite])
                source_song_ids.append(
                    np.full((int(finite.sum()),), song_to_id[song_name], dtype=np.int64)
                )

    if not targets:
        raise ValueError("No finite RP1M IK pairs were loaded")

    target_arr = np.concatenate(targets, axis=0).astype(np.float32, copy=False)
    previous_arr = np.concatenate(prev_qpos, axis=0).astype(np.float32, copy=False)
    expert_arr = np.concatenate(expert_qpos, axis=0).astype(np.float32, copy=False)
    source_song_arr = np.concatenate(source_song_ids, axis=0).astype(np.int64, copy=False)

    if max_pairs is not None and target_arr.shape[0] > int(max_pairs):
        keep = rng.choice(np.arange(target_arr.shape[0]), size=int(max_pairs), replace=False)
        target_arr = target_arr[keep]
        previous_arr = previous_arr[keep]
        expert_arr = expert_arr[keep]
        source_song_arr = source_song_arr[keep]

    if previous_mode == "random":
        previous_arr = expert_arr[rng.permutation(expert_arr.shape[0])].copy()
    elif previous_mode == "mean":
        previous_arr = np.broadcast_to(expert_arr.mean(axis=0, keepdims=True), expert_arr.shape).copy()
    elif previous_mode == "zero":
        previous_arr = np.zeros_like(expert_arr, dtype=np.float32)

    active = np.ones((target_arr.shape[0], NUM_FINGERS), dtype=np.float32)
    return RPIKArrays(
        target_fingertips=target_arr,
        active_mask=active,
        previous_qpos=previous_arr,
        expert_qpos=expert_arr,
        song_names=tuple(selected),
        source_song_ids=source_song_arr,
    )


def active_mask_from_targets(target_fingertips: np.ndarray) -> np.ndarray:
    target = np.asarray(target_fingertips, dtype=np.float32)
    if target.ndim == 2:
        target = target.reshape(-1, NUM_FINGERS, FINGERTIP_COORD_DIM)
    return np.isfinite(target).all(axis=-1).astype(np.float32)


def _as_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)
