from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fugue.constants import (
    ACTION_DIM,
    DEFAULT_DT,
    DEFAULT_RP1M_ROOT,
    DEFAULT_SONG_KEY,
    FINGERTIP_DIM,
    GOAL_DIM,
    HAND_QPOS_DIM,
)


ARRAY_NAMES = (
    "hand_joints",
    "actions",
    "goals",
    "piano_states",
    "hand_fingertips",
    "joint_velocities",
)


@dataclass(frozen=True, slots=True)
class SampleConfig:
    feature_mode: str = "stateless"
    history: int = 1
    goal_horizon: int = 1
    chunk_horizon: int = 1
    delta: int = 0
    include_qpos: bool = True
    include_qvel: bool = True
    include_action_history: bool = False
    include_goals: bool = False
    include_fingertips: bool = False
    include_future_qvel: bool = False
    oracle_future_hand_state: bool = False
    press_window: int = 2

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "SampleConfig":
        payload = dict(values or {})
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.feature_mode not in {"stateless", "history", "inverse", "sequence"}:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")
        if self.history < 1:
            raise ValueError("history must be >= 1")
        if self.goal_horizon < 1:
            raise ValueError("goal_horizon must be >= 1")
        if self.chunk_horizon < 1:
            raise ValueError("chunk_horizon must be >= 1")
        if self.press_window < 0:
            raise ValueError("press_window must be >= 0")

    def feature_dim(
        self,
        *,
        q_dim: int = HAND_QPOS_DIM,
        action_dim: int = ACTION_DIM,
        goal_dim: int = GOAL_DIM,
        fingertip_dim: int = FINGERTIP_DIM,
    ) -> int:
        self.validate()
        current_dim = 0
        if self.include_qpos:
            current_dim += q_dim
        if self.include_qvel:
            current_dim += q_dim
        if self.include_fingertips:
            current_dim += fingertip_dim
        if self.feature_mode == "stateless":
            return current_dim
        if self.feature_mode == "history":
            dim = current_dim * self.history
            if self.include_action_history:
                dim += action_dim * self.history
            if self.include_goals:
                dim += goal_dim * self.goal_horizon
            return dim
        if self.feature_mode == "sequence":
            token_dim = current_dim + 2
            if self.include_action_history:
                token_dim += action_dim
            if self.include_goals:
                token_dim += goal_dim
            return token_dim
        if self.feature_mode == "inverse":
            dim = current_dim + q_dim * self.goal_horizon
            if self.include_future_qvel:
                dim += q_dim * self.goal_horizon
            if self.include_goals:
                dim += goal_dim * self.goal_horizon
            return dim
        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")


@dataclass(slots=True)
class NormalizationStats:
    q_mean: list[float]
    q_std: list[float]
    qvel_mean: list[float]
    qvel_std: list[float]
    action_mean: list[float]
    action_std: list[float]
    goal_mean: list[float]
    goal_std: list[float]
    fingertip_mean: list[float] | None = None
    fingertip_std: list[float] | None = None
    source_split: str = "train"
    dt: float = DEFAULT_DT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationStats":
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class _StatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None

    def update(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[0] == 0:
            return
        if self.sum is None:
            self.sum = np.zeros((arr.shape[-1],), dtype=np.float64)
            self.sumsq = np.zeros((arr.shape[-1],), dtype=np.float64)
        if arr.shape[-1] != self.sum.shape[0]:
            raise ValueError(f"Feature dim mismatch: got {arr.shape[-1]}, expected {self.sum.shape[0]}")
        self.count += int(arr.shape[0])
        self.sum += arr.sum(axis=0)
        self.sumsq += np.square(arr).sum(axis=0)

    def finalize(self, *, min_std: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0 or self.sum is None or self.sumsq is None:
            raise ValueError("Cannot finalize empty accumulator")
        mean = self.sum / float(self.count)
        var = np.maximum(self.sumsq / float(self.count) - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(var), float(min_std))
        return mean.astype(np.float32), std.astype(np.float32)


def finite_difference(q: np.ndarray, dt: float = DEFAULT_DT) -> np.ndarray:
    values = np.asarray(q, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected q with shape [T, D], got {values.shape}")
    if values.shape[0] < 2:
        return np.zeros_like(values, dtype=np.float32)
    step = float(dt)
    if step <= 0.0:
        raise ValueError("dt must be positive")
    qvel = np.zeros_like(values, dtype=np.float32)
    qvel[0] = (values[1] - values[0]) / step
    qvel[-1] = (values[-1] - values[-2]) / step
    if values.shape[0] > 2:
        qvel[1:-1] = (values[2:] - values[:-2]) / (2.0 * step)
    return qvel.astype(np.float32)


def open_zarr_root(dataset_root: str | Path = DEFAULT_RP1M_ROOT):
    try:
        import zarr
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("zarr is required to read RP1M archives") from exc
    return zarr.open(str(Path(dataset_root)), mode="r")


def audit_song(dataset_root: str | Path, song_key: str = DEFAULT_SONG_KEY) -> dict[str, Any]:
    root = open_zarr_root(dataset_root)
    if song_key not in root:
        matches = [name for name in root.keys() if song_key.lower() in str(name).lower()]
        raise KeyError(f"Song key {song_key!r} not found. partial_matches={matches[:10]}")
    group = root[song_key]
    arrays: dict[str, dict[str, Any]] = {}
    for name in ARRAY_NAMES:
        if name in group:
            array = group[name]
            arrays[name] = {"shape": [int(value) for value in array.shape], "dtype": str(array.dtype)}
        else:
            arrays[name] = {"shape": None, "dtype": None}
    missing = [name for name in ("hand_joints", "actions", "goals") if arrays[name]["shape"] is None]
    if missing:
        raise ValueError(f"Song {song_key} is missing required arrays: {missing}")
    hand_shape = arrays["hand_joints"]["shape"]
    action_shape = arrays["actions"]["shape"]
    goal_shape = arrays["goals"]["shape"]
    if hand_shape[:2] != action_shape[:2] or hand_shape[:2] != goal_shape[:2]:
        raise ValueError(
            "hand_joints/actions/goals must share [num_demos, timesteps], got "
            f"{hand_shape}, {action_shape}, {goal_shape}"
        )
    return {
        "dataset_root": str(Path(dataset_root)),
        "song_key": song_key,
        "num_demos": int(hand_shape[0]),
        "timesteps": int(hand_shape[1]),
        "hand_joint_dim": int(hand_shape[2]),
        "action_dim": int(action_shape[2]),
        "goal_dim_raw": int(goal_shape[2]),
        "goal_dim_used": min(int(goal_shape[2]), GOAL_DIM),
        "has_joint_velocities": bool(arrays["joint_velocities"]["shape"] is not None),
        "has_hand_fingertips": bool(arrays["hand_fingertips"]["shape"] is not None),
        "arrays": arrays,
    }


def build_demo_manifest(
    *,
    summary: dict[str, Any],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 7,
) -> pd.DataFrame:
    _validate_split_fractions(train_frac, val_frac, test_frac)
    num_demos = int(summary["num_demos"])
    labels = _assign_demo_splits(num_demos, train_frac=train_frac, val_frac=val_frac, seed=seed)
    rows = []
    for demo_id, split in enumerate(labels):
        rows.append(
            {
                "song_key": str(summary["song_key"]),
                "demo_id": int(demo_id),
                "split": str(split),
                "timesteps": int(summary["timesteps"]),
                "hand_joint_dim": int(summary["hand_joint_dim"]),
                "action_dim": int(summary["action_dim"]),
                "goal_dim_raw": int(summary["goal_dim_raw"]),
                "goal_dim_used": int(summary["goal_dim_used"]),
                "has_joint_velocities": bool(summary["has_joint_velocities"]),
                "has_hand_fingertips": bool(summary["has_hand_fingertips"]),
            }
        )
    manifest = pd.DataFrame(rows)
    validate_demo_split(manifest)
    return manifest


def validate_demo_split(manifest: pd.DataFrame) -> dict[str, int]:
    required = {"demo_id", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    duplicate_membership = manifest.groupby("demo_id")["split"].nunique()
    leaked = duplicate_membership[duplicate_membership > 1]
    if not leaked.empty:
        raise ValueError(f"Demo split leakage: demo ids appear in multiple splits: {list(leaked.index[:5])}")
    duplicated_rows = manifest.duplicated(subset=["demo_id"], keep=False)
    if bool(duplicated_rows.any()):
        values = manifest.loc[duplicated_rows, "demo_id"].tolist()[:5]
        raise ValueError(f"Duplicate demo_id rows in manifest: {values}")
    return {str(key): int(value) for key, value in manifest["split"].value_counts().to_dict().items()}


def write_song_audit(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    song_key: str = DEFAULT_SONG_KEY,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 7,
) -> dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = audit_song(dataset_root, song_key=song_key)
    manifest = build_demo_manifest(
        summary=summary,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    split_counts = validate_demo_split(manifest)
    summary = {
        **summary,
        "split_seed": int(seed),
        "split_fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "split_counts": split_counts,
    }
    manifest_path = output_root / "manifest.csv"
    splits_path = output_root / "splits.csv"
    summary_path = output_root / "dataset_summary.json"
    manifest.to_csv(manifest_path, index=False)
    manifest[["demo_id", "split"]].to_csv(splits_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest_path, "splits": splits_path, "summary": summary_path}


def fit_normalization_stats(
    *,
    dataset_root: str | Path,
    manifest: pd.DataFrame,
    song_key: str = DEFAULT_SONG_KEY,
    dt: float = DEFAULT_DT,
    split: str = "train",
) -> NormalizationStats:
    validate_demo_split(manifest)
    rows = manifest[manifest["split"].astype(str) == split]
    if rows.empty:
        raise ValueError(f"No rows found for split={split!r}")
    root = open_zarr_root(dataset_root)
    group = root[song_key]
    q_acc = _StatsAccumulator()
    qvel_acc = _StatsAccumulator()
    action_acc = _StatsAccumulator()
    goal_acc = _StatsAccumulator()
    fingertip_acc = _StatsAccumulator() if "hand_fingertips" in group else None
    for demo_id in rows["demo_id"].astype(int).tolist():
        episode = load_demo_arrays(group, demo_id=demo_id, dt=dt)
        q_acc.update(episode["q"])
        qvel_acc.update(episode["qvel"])
        action_acc.update(episode["actions"])
        goal_acc.update(episode["goals"])
        if fingertip_acc is not None and episode.get("fingertips") is not None:
            fingertip_acc.update(episode["fingertips"])
    q_mean, q_std = q_acc.finalize()
    qvel_mean, qvel_std = qvel_acc.finalize()
    action_mean, action_std = action_acc.finalize()
    goal_mean, _ = goal_acc.finalize()
    goal_var = np.maximum(goal_mean * (1.0 - goal_mean), 0.0)
    goal_std = np.sqrt(goal_var + 1e-6).astype(np.float32)
    fingertip_mean = None
    fingertip_std = None
    if fingertip_acc is not None and fingertip_acc.count > 0:
        ft_mean, ft_std = fingertip_acc.finalize()
        fingertip_mean = ft_mean.tolist()
        fingertip_std = ft_std.tolist()
    return NormalizationStats(
        q_mean=q_mean.tolist(),
        q_std=q_std.tolist(),
        qvel_mean=qvel_mean.tolist(),
        qvel_std=qvel_std.tolist(),
        action_mean=action_mean.tolist(),
        action_std=action_std.tolist(),
        goal_mean=goal_mean.astype(np.float32).tolist(),
        goal_std=goal_std.tolist(),
        fingertip_mean=fingertip_mean,
        fingertip_std=fingertip_std,
        source_split=split,
        dt=float(dt),
    )


def load_demo_arrays(group: Any, *, demo_id: int, dt: float = DEFAULT_DT) -> dict[str, np.ndarray | None]:
    q = np.asarray(group["hand_joints"][int(demo_id)], dtype=np.float32)
    actions = np.asarray(group["actions"][int(demo_id)], dtype=np.float32)
    goals = np.asarray(group["goals"][int(demo_id)], dtype=np.float32)[..., :GOAL_DIM]
    if "joint_velocities" in group:
        qvel = np.asarray(group["joint_velocities"][int(demo_id)], dtype=np.float32)
    else:
        qvel = finite_difference(q, dt=dt)
    fingertips = None
    if "hand_fingertips" in group:
        fingertips = np.asarray(group["hand_fingertips"][int(demo_id)], dtype=np.float32)
    piano_states = None
    if "piano_states" in group:
        piano_states = np.asarray(group["piano_states"][int(demo_id)], dtype=np.float32)
    steps = min(int(q.shape[0]), int(qvel.shape[0]), int(actions.shape[0]), int(goals.shape[0]))
    return {
        "q": q[:steps].astype(np.float32),
        "qvel": qvel[:steps].astype(np.float32),
        "actions": actions[:steps].astype(np.float32),
        "goals": goals[:steps].astype(np.float32),
        "fingertips": None if fingertips is None else fingertips[:steps].astype(np.float32),
        "piano_states": None if piano_states is None else piano_states[:steps].astype(np.float32),
    }


def normalize_episode(episode: dict[str, np.ndarray | None], stats: NormalizationStats) -> dict[str, np.ndarray | None]:
    normalized = dict(episode)
    normalized["q"] = standardize(episode["q"], stats.q_mean, stats.q_std)
    normalized["qvel"] = standardize(episode["qvel"], stats.qvel_mean, stats.qvel_std)
    normalized["actions"] = standardize(episode["actions"], stats.action_mean, stats.action_std)
    normalized["goals"] = standardize(episode["goals"], stats.goal_mean, stats.goal_std)
    if episode.get("fingertips") is not None and stats.fingertip_mean is not None:
        normalized["fingertips"] = standardize(episode["fingertips"], stats.fingertip_mean, stats.fingertip_std)
    return normalized


def standardize(value: np.ndarray | None, mean: list[float] | np.ndarray, std: list[float] | np.ndarray | None) -> np.ndarray:
    if value is None:
        raise ValueError("Cannot standardize None")
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1) if std is not None else np.ones_like(mean_arr)
    return ((np.asarray(value, dtype=np.float32) - mean_arr) / np.maximum(std_arr, 1e-6)).astype(np.float32)


def unstandardize(value: np.ndarray, mean: list[float] | np.ndarray, std: list[float] | np.ndarray) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1)
    return (np.asarray(value, dtype=np.float32) * np.maximum(std_arr, 1e-6) + mean_arr).astype(np.float32)


def compute_press_mask(goals: np.ndarray, *, window: int = 2, threshold: float = 0.5) -> np.ndarray:
    goal = np.asarray(goals, dtype=np.float32)[..., :GOAL_DIM]
    if goal.ndim != 2:
        raise ValueError(f"Expected goals [T, 88], got {goal.shape}")
    active = np.any(goal > float(threshold), axis=1)
    prev = np.concatenate([np.zeros((1,), dtype=bool), active[:-1]])
    onset = active & ~prev
    base = active | onset
    mask = np.zeros_like(base, dtype=bool)
    radius = int(window)
    for offset in range(-radius, radius + 1):
        if offset < 0:
            mask[:offset] |= base[-offset:]
        elif offset > 0:
            mask[offset:] |= base[:-offset]
        else:
            mask |= base
    return mask.astype(np.float32)


class FugueActionDataset(Dataset):
    """Demo-split RP1M action reconstruction dataset."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        manifest: pd.DataFrame | str | Path,
        stats: NormalizationStats,
        song_key: str = DEFAULT_SONG_KEY,
        split: str = "train",
        sample_config: SampleConfig | None = None,
        dt: float = DEFAULT_DT,
        max_demos: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.song_key = str(song_key)
        self.stats = stats
        self.sample_config = sample_config or SampleConfig()
        self.sample_config.validate()
        self.dt = float(dt)
        self.manifest = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        validate_demo_split(self.manifest)
        rows = self.manifest[self.manifest["split"].astype(str) == str(split)].copy()
        if max_demos is not None:
            rows = rows.sort_values("demo_id").head(int(max_demos))
        if rows.empty:
            raise ValueError(f"No demos found for split={split!r}")
        self.demo_ids = rows["demo_id"].astype(int).tolist()
        root = open_zarr_root(self.dataset_root)
        group = root[self.song_key]
        self.episodes: dict[int, dict[str, np.ndarray | None]] = {}
        self.raw_episodes: dict[int, dict[str, np.ndarray | None]] = {}
        self.press_masks: dict[int, np.ndarray] = {}
        self.index: list[tuple[int, int]] = []
        for demo_id in self.demo_ids:
            raw = load_demo_arrays(group, demo_id=demo_id, dt=self.dt)
            norm = normalize_episode(raw, self.stats)
            self.raw_episodes[demo_id] = raw
            self.episodes[demo_id] = norm
            self.press_masks[demo_id] = compute_press_mask(raw["goals"], window=self.sample_config.press_window)
            for t in valid_timesteps(norm["actions"].shape[0], self.sample_config):
                self.index.append((demo_id, t))
        if not self.index:
            raise ValueError(f"No valid samples for split={split!r} with config={self.sample_config}")
        sample_episode = self.episodes[self.demo_ids[0]]
        self.q_dim = int(sample_episode["q"].shape[-1])
        self.action_dim = int(sample_episode["actions"].shape[-1])
        self.goal_dim = int(sample_episode["goals"].shape[-1])
        ft = sample_episode.get("fingertips")
        self.fingertip_dim = FINGERTIP_DIM if ft is None else int(ft.shape[-1])
        self.feature_dim = self.sample_config.feature_dim(
            q_dim=self.q_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
            fingertip_dim=self.fingertip_dim,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_id, t = self.index[idx]
        episode = self.episodes[demo_id]
        target_start = int(t + self.sample_config.delta)
        target_end = target_start + int(self.sample_config.chunk_horizon)
        feature = build_feature_vector(episode, t=t, config=self.sample_config)
        target = np.asarray(episode["actions"][target_start:target_end], dtype=np.float32)
        press = np.asarray(self.press_masks[demo_id][target_start:target_end], dtype=np.float32)
        return {
            "features": torch.from_numpy(feature),
            "actions": torch.from_numpy(target),
            "press_weight": torch.from_numpy(1.0 + 2.0 * press),
            "demo_id": torch.tensor(demo_id, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }


def valid_timesteps(length: int, config: SampleConfig) -> range:
    start = 0
    if config.feature_mode in {"history", "sequence"}:
        start = max(start, int(config.history) - 1)
    start = max(start, -int(config.delta))
    stop = int(length) - int(config.delta) - int(config.chunk_horizon) + 1
    stop = min(stop, int(length))
    if stop <= start:
        return range(0, 0)
    return range(start, stop)


def build_feature_vector(episode: dict[str, np.ndarray | None], *, t: int, config: SampleConfig) -> np.ndarray:
    parts: list[np.ndarray] = []
    if config.feature_mode == "stateless":
        _append_current(parts, episode, t=t, config=config)
    elif config.feature_mode == "history":
        state_history = []
        for idx in _clamped_indices(t - config.history + 1, config.history, len(episode["q"])):
            current_parts: list[np.ndarray] = []
            _append_current(current_parts, episode, t=int(idx), config=config)
            state_history.append(np.concatenate(current_parts, axis=0))
        parts.append(np.concatenate(state_history, axis=0))
        if config.include_action_history:
            parts.append(_previous_action_history(episode["actions"], t=t, history=config.history))
        if config.include_goals:
            parts.append(_window(episode["goals"], start=t, length=config.goal_horizon).reshape(-1))
    elif config.feature_mode == "inverse":
        _append_current(parts, episode, t=t, config=config)
        parts.append(_window(episode["q"], start=t + 1, length=config.goal_horizon).reshape(-1))
        if config.include_future_qvel:
            parts.append(_window(episode["qvel"], start=t + 1, length=config.goal_horizon).reshape(-1))
        if config.include_goals:
            parts.append(_window(episode["goals"], start=t, length=config.goal_horizon).reshape(-1))
    elif config.feature_mode == "sequence":
        return _sequence_feature_tokens(episode, t=t, config=config)
    else:
        raise ValueError(f"Unsupported feature_mode: {config.feature_mode}")
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]).astype(np.float32)


def _append_current(parts: list[np.ndarray], episode: dict[str, np.ndarray | None], *, t: int, config: SampleConfig) -> None:
    if config.include_qpos:
        parts.append(np.asarray(episode["q"][t], dtype=np.float32).reshape(-1))
    if config.include_qvel:
        parts.append(np.asarray(episode["qvel"][t], dtype=np.float32).reshape(-1))
    if config.include_fingertips:
        fingertips = episode.get("fingertips")
        if fingertips is None:
            parts.append(np.zeros((FINGERTIP_DIM,), dtype=np.float32))
        else:
            parts.append(np.asarray(fingertips[t], dtype=np.float32).reshape(-1))


def _window(array: np.ndarray, *, start: int, length: int) -> np.ndarray:
    indices = _clamped_indices(start, length, len(array))
    return np.asarray(array[indices], dtype=np.float32)


def _sequence_feature_tokens(
    episode: dict[str, np.ndarray | None],
    *,
    t: int,
    config: SampleConfig,
) -> np.ndarray:
    tokens = []
    total = len(episode["q"])
    for idx in _clamped_indices(t - config.history + 1, config.history, total):
        tokens.append(_sequence_token(episode, t=int(idx), config=config, token_type=(1.0, 0.0)))
    if config.include_goals:
        for idx in _clamped_indices(t, config.goal_horizon, total):
            tokens.append(_sequence_token(episode, t=int(idx), config=config, token_type=(0.0, 1.0), future_goal=True))
    return np.stack(tokens, axis=0).astype(np.float32)


def _sequence_token(
    episode: dict[str, np.ndarray | None],
    *,
    t: int,
    config: SampleConfig,
    token_type: tuple[float, float],
    future_goal: bool = False,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    q_dim = int(episode["q"].shape[-1])
    action_dim = int(episode["actions"].shape[-1])
    fingertip = episode.get("fingertips")
    fingertip_dim = FINGERTIP_DIM if fingertip is None else int(fingertip.shape[-1])
    if future_goal:
        if config.include_qpos:
            parts.append(np.zeros((q_dim,), dtype=np.float32))
        if config.include_qvel:
            parts.append(np.zeros((q_dim,), dtype=np.float32))
        if config.include_fingertips:
            parts.append(np.zeros((fingertip_dim,), dtype=np.float32))
        if config.include_action_history:
            parts.append(np.zeros((action_dim,), dtype=np.float32))
    else:
        _append_current(parts, episode, t=t, config=config)
        if config.include_action_history:
            if t <= 0:
                parts.append(np.zeros((action_dim,), dtype=np.float32))
            else:
                parts.append(np.asarray(episode["actions"][t - 1], dtype=np.float32).reshape(-1))
    if config.include_goals:
        parts.append(np.asarray(episode["goals"][t], dtype=np.float32).reshape(-1))
    parts.append(np.asarray(token_type, dtype=np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _clamped_indices(start: int, length: int, total: int) -> np.ndarray:
    raw = np.arange(int(start), int(start) + int(length), dtype=np.int64)
    return np.clip(raw, 0, max(int(total) - 1, 0))


def _previous_action_history(actions: np.ndarray, *, t: int, history: int) -> np.ndarray:
    action_dim = int(actions.shape[-1])
    rows = []
    for idx in range(int(t) - int(history), int(t)):
        if idx < 0:
            rows.append(np.zeros((action_dim,), dtype=np.float32))
        else:
            rows.append(np.asarray(actions[min(idx, len(actions) - 1)], dtype=np.float32))
    return np.concatenate(rows, axis=0).astype(np.float32)


def _validate_split_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    values = (float(train_frac), float(val_frac), float(test_frac))
    if any(value < 0.0 for value in values):
        raise ValueError("split fractions must be non-negative")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1.0")
    if train_frac <= 0.0:
        raise ValueError("train fraction must be positive")


def _assign_demo_splits(count: int, *, train_frac: float, val_frac: float, seed: int) -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(count))
    labels = np.empty((count,), dtype=object)
    train_count = int(np.floor(count * float(train_frac)))
    val_count = int(np.floor(count * float(val_frac)))
    if count >= 3:
        train_count = max(1, min(train_count, count - 2))
        val_count = max(1, min(val_count, count - train_count - 1))
    elif count == 2:
        train_count = 1
        val_count = 0
    else:
        train_count = 1
        val_count = 0
    test_start = train_count + val_count
    labels[order[:train_count]] = "train"
    labels[order[train_count:test_start]] = "val"
    labels[order[test_start:]] = "test"
    return [str(value) for value in labels.tolist()]
