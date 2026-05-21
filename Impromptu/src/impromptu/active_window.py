from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActiveWindowCrop:
    target_keys: np.ndarray
    start_frame: int
    end_frame: int
    metadata: dict[str, Any]


def crop_active_window(
    target_keys: np.ndarray,
    *,
    dt: float,
    threshold: float,
    active_window_last_s: float | None,
    active_window_preroll_s: float,
    active_window_postroll_s: float,
) -> ActiveWindowCrop:
    keys = np.asarray(target_keys, dtype=np.float32)
    if keys.ndim != 2 or keys.shape[1] < 88:
        raise ValueError(f"target_keys must have shape [T, >=88], got {keys.shape}")

    original_steps = int(keys.shape[0])
    active = np.flatnonzero(np.any(keys[:, :88] > float(threshold), axis=1))
    if active.size == 0:
        meta = {
            "original_steps": original_steps,
            "cropped_steps": original_steps,
            "crop_start_frame": 0,
            "crop_end_frame": original_steps,
            "crop_start_s": 0.0,
            "crop_end_s": float(original_steps) * float(dt),
            "active_window_last_s": active_window_last_s,
            "active_window_preroll_s": float(active_window_preroll_s),
            "active_window_postroll_s": float(active_window_postroll_s),
            "active_frames_found": False,
        }
        return ActiveWindowCrop(target_keys=keys.astype(np.float32, copy=True), start_frame=0, end_frame=original_steps, metadata=meta)

    safe_dt = max(float(dt), 1e-8)
    preroll = max(int(np.ceil(float(active_window_preroll_s) / safe_dt)), 0)
    postroll = max(int(np.ceil(float(active_window_postroll_s) / safe_dt)), 0)
    first_active = int(active[0])
    last_active = int(active[-1])
    start = max(0, first_active - preroll)
    end = min(original_steps, last_active + postroll + 1)

    if active_window_last_s is not None and float(active_window_last_s) > 0.0:
        last_frames = max(int(np.ceil(float(active_window_last_s) / safe_dt)), 1)
        start = max(start, last_active + 1 - last_frames)

    cropped = keys[start:end, :88].astype(np.float32, copy=True)
    meta = {
        "original_steps": original_steps,
        "cropped_steps": int(cropped.shape[0]),
        "crop_start_frame": int(start),
        "crop_end_frame": int(end),
        "crop_start_s": float(start) * safe_dt,
        "crop_end_s": float(end) * safe_dt,
        "active_window_last_s": active_window_last_s,
        "active_window_preroll_s": float(active_window_preroll_s),
        "active_window_postroll_s": float(active_window_postroll_s),
        "first_active_frame": first_active,
        "last_active_frame": last_active,
        "active_frames_found": True,
    }
    return ActiveWindowCrop(target_keys=cropped, start_frame=int(start), end_frame=int(end), metadata=meta)
