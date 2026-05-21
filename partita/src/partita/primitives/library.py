from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from partita.primitives.features import FEATURE_MODE, resample_array

LIBRARY_VERSION = 2


def build_primitive_library(
    data: dict[str, np.ndarray],
    segments: pd.DataFrame,
    assignments: pd.DataFrame,
    transformed_features: np.ndarray,
    scaler,
    pca,
    clusterer,
    feature_names: list[str],
    resample_len: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    primitives = {}
    merged = segments.merge(assignments[["global_segment_id", "primitive_id"]], on="global_segment_id", how="left")
    for primitive_id, group in merged.groupby("primitive_id"):
        pid = int(primitive_id)
        action_segments = []
        member_ids = []
        member_trajs = []
        for _, row in group.iterrows():
            traj_idx = int(row["trajectory_index"])
            start = int(row["start_t"])
            end = int(row["end_t"])
            member_ids.append(int(row["global_segment_id"]))
            member_trajs.append(int(row["trajectory_id"]))
            action_segments.append(resample_array(data["actions"][traj_idx, start:end], resample_len))
        primitives[pid] = {
            "primitive_id": pid,
            "member_segment_ids": member_ids,
            "member_trajectory_ids": sorted(set(member_trajs)),
            "mean_duration": float(group["duration"].mean()),
            "mean_action_trajectory": np.mean(np.stack(action_segments, axis=0), axis=0).astype(np.float32),
            "feature_center": clusterer.cluster_centers_[pid].astype(np.float32),
        }
    return {
        "version": LIBRARY_VERSION,
        "feature_mode": FEATURE_MODE,
        "feature_names": feature_names,
        "scaler": scaler,
        "pca": pca,
        "clusterer": clusterer,
        "primitives": primitives,
        "resample_len": int(resample_len),
        "config": config,
    }


def save_library(path: str | Path, library: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(library, f)


def load_library(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        library = pickle.load(f)
    if int(library.get("version", 0)) < LIBRARY_VERSION or library.get("feature_mode") != FEATURE_MODE:
        raise RuntimeError(
            f"Primitive library {path} is incompatible with goal-conditioned Partita. "
            "Rerun train_primitives.py to regenerate it."
        )
    return library


def primitive_summary(segments: pd.DataFrame, assignments: pd.DataFrame, num_training_trajectories: int) -> pd.DataFrame:
    merged = segments.merge(assignments[["global_segment_id", "primitive_id"]], on="global_segment_id", how="left")
    rows = []
    for pid, group in merged.groupby("primitive_id"):
        rows.append({
            "primitive_id": int(pid),
            "count": int(len(group)),
            "num_trajectories_used_in": int(group["trajectory_id"].nunique()),
            "trajectory_coverage_fraction": float(group["trajectory_id"].nunique() / max(num_training_trajectories, 1)),
            "mean_duration": float(group["duration"].mean()),
            "mean_num_goal_keys": float(group["num_goal_keys"].mean()),
        })
    return pd.DataFrame(rows).sort_values("primitive_id").reset_index(drop=True)


def primitive_usage_by_trajectory(assignments: pd.DataFrame) -> pd.DataFrame:
    table = assignments.pivot_table(index="trajectory_id", columns="primitive_id", values="global_segment_id", aggfunc="count", fill_value=0)
    table = table.sort_index(axis=0).sort_index(axis=1)
    table.columns = [f"primitive_{int(c)}" for c in table.columns]
    return table.reset_index()
