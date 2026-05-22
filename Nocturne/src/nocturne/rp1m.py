from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REQUIRED_ARRAYS = ("actions", "goals", "piano_states", "hand_joints", "hand_fingertips")


@dataclass(slots=True)
class SongDemos:
    song_name: str
    demo_ids: np.ndarray
    actions: np.ndarray
    goals: np.ndarray
    piano_states: np.ndarray
    hand_joints: np.ndarray
    hand_fingertips: np.ndarray

    @property
    def num_demos(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.actions.shape[1])

    def array(self, name: str) -> np.ndarray:
        return getattr(self, name)


def open_rp1m_root(path: str | Path):
    try:
        import zarr
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise ModuleNotFoundError("zarr is required to read RP1M.") from exc
    root_path = Path(path)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    return zarr.open(str(root_path), mode="r")


def list_songs(rp1m_root: str | Path, limit: int | None = None) -> list[str]:
    root = open_rp1m_root(rp1m_root)
    songs = sorted([name for name in root.keys() if hasattr(root[name], "keys")])
    return songs[:limit] if limit is not None else songs


def inspect_song(rp1m_root: str | Path, song_name: str) -> dict[str, object]:
    root = open_rp1m_root(rp1m_root)
    if song_name not in root:
        raise KeyError(f"Song not found in RP1M: {song_name}")
    group = root[song_name]
    arrays = {}
    for name in sorted(group.keys()):
        item = group[name]
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            arrays[name] = {"shape": list(item.shape), "dtype": str(item.dtype), "chunks": list(getattr(item, "chunks", ()))}
    return {"song_name": song_name, "arrays": arrays}


def load_song_demos(
    rp1m_root: str | Path,
    song_name: str,
    *,
    num_demos: int = 50,
    demo_ids: Iterable[int] | None = None,
    arrays: Iterable[str] = REQUIRED_ARRAYS,
) -> SongDemos:
    root = open_rp1m_root(rp1m_root)
    if song_name not in root:
        raise KeyError(f"Song not found in RP1M: {song_name}")
    group = root[song_name]
    requested = tuple(arrays)
    missing = [name for name in requested if name not in group]
    if missing:
        raise KeyError(f"Song {song_name} missing required arrays: {missing}")
    total = int(group["actions"].shape[0])
    ids = np.asarray(list(demo_ids) if demo_ids is not None else list(range(min(int(num_demos), total))), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("At least one demo is required.")
    if int(ids.min()) < 0 or int(ids.max()) >= total:
        raise IndexError(f"Demo ids out of range 0..{total - 1}: {ids.tolist()}")
    loaded = {name: _read_demo_stack(group[name], ids) for name in requested}
    _validate_loaded_shapes(loaded)
    return SongDemos(
        song_name=song_name,
        demo_ids=ids,
        actions=loaded["actions"].astype(np.float32, copy=False),
        goals=loaded["goals"].astype(np.float32, copy=False),
        piano_states=loaded["piano_states"].astype(np.float32, copy=False),
        hand_joints=loaded["hand_joints"].astype(np.float32, copy=False),
        hand_fingertips=loaded["hand_fingertips"].astype(np.float32, copy=False),
    )


def _read_demo_stack(array, demo_ids: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(array[int(index)]) for index in demo_ids.tolist()], axis=0)


def _validate_loaded_shapes(arrays: dict[str, np.ndarray]) -> None:
    frames = {name: int(value.shape[1]) for name, value in arrays.items() if value.ndim >= 2}
    if len(set(frames.values())) != 1:
        raise ValueError(f"Loaded arrays disagree on frame count: {frames}")
    demos = {name: int(value.shape[0]) for name, value in arrays.items() if value.ndim >= 1}
    if len(set(demos.values())) != 1:
        raise ValueError(f"Loaded arrays disagree on demo count: {demos}")
