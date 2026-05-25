#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, REPO_ROOT / "Bagatelle" / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402


def swap_hands_qpos(qpos: np.ndarray) -> np.ndarray:
    row = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if row.shape[0] != 46:
        raise ValueError(row.shape)
    return np.concatenate([row[23:], row[:23]], axis=0).astype(np.float32)


def swap_hands_tips(tips: np.ndarray) -> np.ndarray:
    arr = np.asarray(tips, dtype=np.float32).reshape(10, 3)
    return np.concatenate([arr[5:], arr[:5]], axis=0).astype(np.float32)


def error_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32), axis=-1)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(np.max(err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RP1M hand-state/fingertip alignment against Bagatelle kinematics.")
    parser.add_argument("--rp1m-root", default="/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--song", default="")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument("--seed", type=int, default=524)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    args = parser.parse_args()

    try:
        import zarr
    except Exception as exc:
        raise ModuleNotFoundError("zarr is required") from exc

    root = zarr.open_group(str(Path(args.rp1m_root)), mode="r")
    songs = sorted([name for name in root.keys() if hasattr(root[name], "keys")])
    song = str(args.song) if args.song else songs[int(np.random.default_rng(int(args.seed)).integers(0, len(songs)))]
    group = root[song]
    qpos_all = np.asarray(group["hand_joints"][0], dtype=np.float32)
    tips_all = np.asarray(group["hand_fingertips"][0], dtype=np.float32).reshape(qpos_all.shape[0], 10, 3)
    frame_ids = np.arange(0, min(qpos_all.shape[0], tips_all.shape[0]), max(int(args.frame_stride), 1), dtype=np.int64)
    if frame_ids.size > int(args.samples):
        rng = np.random.default_rng(int(args.seed))
        frame_ids = np.sort(rng.choice(frame_ids, size=int(args.samples), replace=False)).astype(np.int64)

    zeros = np.zeros((1, 88), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    combos: dict[str, list[np.ndarray]] = {
        "qpos_as_is__tips_as_is": [],
        "qpos_as_is__tips_swapped": [],
        "qpos_swapped__tips_as_is": [],
        "qpos_swapped__tips_swapped": [],
    }
    offsets: dict[str, list[np.ndarray]] = {key: [] for key in combos}
    with BagatelleKinematics(
        config=BagatelleConfig(environment_name=str(args.environment_name), seed=int(args.seed)),
        target_keys=zeros,
    ) as kin:
        for frame in frame_ids:
            qpos = qpos_all[int(frame)]
            target_tips = tips_all[int(frame)]
            for qpos_name, qrow in (("qpos_as_is", qpos), ("qpos_swapped", swap_hands_qpos(qpos))):
                pred = kin.fingertip_positions_for_qpos(qrow)
                for tip_name, tref in (("tips_as_is", target_tips), ("tips_swapped", swap_hands_tips(target_tips))):
                    key = f"{qpos_name}__{tip_name}"
                    diffs = pred - tref
                    combos[key].append(np.linalg.norm(diffs, axis=-1))
                    offsets[key].append(diffs.reshape(-1, 3))
        for key in sorted(combos):
            err = np.concatenate(combos[key], axis=0)
            off = np.concatenate(offsets[key], axis=0)
            rows.append(
                {
                    "combo": key,
                    "mean_error_m": float(np.mean(err)),
                    "median_error_m": float(np.median(err)),
                    "p90_error_m": float(np.percentile(err, 90)),
                    "max_error_m": float(np.max(err)),
                    "median_offset_xyz": np.median(off, axis=0).astype(float).tolist(),
                    "mean_offset_xyz": np.mean(off, axis=0).astype(float).tolist(),
                }
            )
    result = {
        "song": song,
        "frames": frame_ids.astype(int).tolist(),
        "samples": int(frame_ids.size),
        "rows": rows,
        "best_combo": min(rows, key=lambda row: row["median_error_m"])["combo"] if rows else None,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
