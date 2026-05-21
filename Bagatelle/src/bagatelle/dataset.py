from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised only in non-ML envs.
    torch = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


IK_MAX_RESIDUAL_COLUMN = 5
NUM_KEYS = 88
NUM_FINGERS = 10


def _require_torch() -> None:
    if torch is None:
        raise ImportError("bagatelle.dataset requires PyTorch") from _TORCH_IMPORT_ERROR


def _trajectory_paths(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        path = Path(item).expanduser()
        if path.is_dir():
            out.extend(sorted(path.rglob("*.npz")))
        else:
            out.append(path)
    return out


class AssignmentSequenceDataset(Dataset):
    """Windowed Bagatelle assignment examples from saved trajectory NPZs.

    Each item contains piano rolls, dense finger-to-key labels, IK-derived weights,
    and best-available fingertip/contact arrays. Existing Bagatelle trajectories do
    not store all 88 key contact positions, so `contact_targets` falls back to the
    dense assigned fingertip target tensor when a fuller array is absent.
    """

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        window_size: int = 64,
        stride: int | None = None,
        threshold: float = 0.5,
        normalize_ik_weights: bool = True,
    ) -> None:
        _require_torch()
        self.paths = _trajectory_paths(paths)
        if not self.paths:
            raise ValueError("AssignmentSequenceDataset needs at least one NPZ path or directory")
        self.window_size = int(window_size)
        self.stride = int(stride or window_size)
        self.threshold = float(threshold)
        self.normalize_ik_weights = bool(normalize_ik_weights)
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self._records: list[dict[str, np.ndarray]] = []
        self._index: list[tuple[int, int, int]] = []
        for path in self.paths:
            record = self._load_record(path)
            record_index = len(self._records)
            self._records.append(record)
            length = int(record["piano_roll"].shape[0])
            if length == 0:
                continue
            for start in range(0, max(length - self.window_size, 0) + 1, self.stride):
                end = min(start + self.window_size, length)
                self._index.append((record_index, start, end))
            if self._index and self._index[-1][0] == record_index and self._index[-1][2] < length:
                self._index.append((record_index, max(0, length - self.window_size), length))
        if not self._index:
            raise ValueError("No non-empty trajectory windows found")

    def _load_record(self, path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=True) as data:
            if "waypoint_target_keys" not in data or "assignments" not in data:
                raise ValueError(f"{path} is missing waypoint_target_keys or assignments")
            piano_roll = np.asarray(data["waypoint_target_keys"], dtype=np.float32)[:, :NUM_KEYS]
            assignments = np.asarray(data["assignments"], dtype=np.int64)
            if assignments.ndim != 2 or assignments.shape[1] != NUM_FINGERS:
                raise ValueError(f"{path} assignments must have shape [W, 10], got {assignments.shape}")
            if "key_contact_targets" in data:
                contact_targets = np.asarray(data["key_contact_targets"], dtype=np.float32)
            elif "contact_targets" in data:
                contact_targets = np.asarray(data["contact_targets"], dtype=np.float32)
            elif "fingertip_targets" in data:
                contact_targets = np.asarray(data["fingertip_targets"], dtype=np.float32)
            else:
                contact_targets = np.zeros((piano_roll.shape[0], NUM_FINGERS, 3), dtype=np.float32)
            if "waypoint_fingertips" in data:
                fingertips = np.asarray(data["waypoint_fingertips"], dtype=np.float32)
            else:
                fingertips = np.zeros((piano_roll.shape[0], NUM_FINGERS, 3), dtype=np.float32)
            if "ik_weights" in data:
                ik_weights = np.asarray(data["ik_weights"], dtype=np.float32)
            elif "ik_metrics" in data:
                metrics = np.asarray(data["ik_metrics"], dtype=np.float32)
                residual = (
                    metrics[:, IK_MAX_RESIDUAL_COLUMN]
                    if metrics.ndim == 2 and metrics.shape[1] > IK_MAX_RESIDUAL_COLUMN
                    else np.zeros((piano_roll.shape[0],), dtype=np.float32)
                )
                ik_weights = 1.0 + residual.astype(np.float32)
            else:
                ik_weights = np.ones((piano_roll.shape[0],), dtype=np.float32)
        if self.normalize_ik_weights and ik_weights.size:
            mean = float(np.mean(ik_weights))
            if mean > 0.0:
                ik_weights = ik_weights / mean
        return {
            "piano_roll": piano_roll,
            "contact_targets": contact_targets,
            "assignments": assignments,
            "ik_weights": ik_weights.astype(np.float32),
            "prev_fingertips": fingertips,
        }

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, start, end = self._index[index]
        record = self._records[record_index]
        item = {name: value[start:end] for name, value in record.items()}
        return {
            "piano_roll": torch.from_numpy(item["piano_roll"].astype(np.float32)),
            "contact_targets": torch.from_numpy(item["contact_targets"].astype(np.float32)),
            "assignments": torch.from_numpy(item["assignments"].astype(np.int64)),
            "ik_weights": torch.from_numpy(item["ik_weights"].astype(np.float32)),
            "prev_fingertips": torch.from_numpy(item["prev_fingertips"].astype(np.float32)),
        }
