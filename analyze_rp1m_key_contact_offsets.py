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


def rp1m_tips_to_bagatelle(tips: np.ndarray, y_offset: float) -> np.ndarray:
    arr = np.asarray(tips, dtype=np.float32).reshape(10, 3)
    swapped = np.concatenate([arr[5:], arr[:5]], axis=0).astype(np.float32)
    swapped[:, 1] += np.float32(float(y_offset))
    return swapped


def summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate key contact target offsets from RP1M fingertip/key contacts.")
    parser.add_argument("--rp1m-root", default="/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--songs", type=int, default=4)
    parser.add_argument("--demos-per-song", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-contacts", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--y-offset", type=float, default=-0.062666565)
    parser.add_argument("--seed", type=int, default=524)
    args = parser.parse_args()

    try:
        import zarr
    except Exception as exc:
        raise ModuleNotFoundError("zarr is required") from exc

    root = zarr.open_group(str(Path(args.rp1m_root)), mode="r")
    rng = np.random.default_rng(int(args.seed))
    names = sorted([name for name in root.keys() if hasattr(root[name], "keys")])
    chosen = sorted(rng.choice(np.asarray(names, dtype=object), size=min(int(args.songs), len(names)), replace=False))

    zeros = np.zeros((1, 88), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    with BagatelleKinematics(
        config=BagatelleConfig(environment_name=str(args.environment_name), seed=int(args.seed)),
        target_keys=zeros,
    ) as kin:
        key_pos = []
        key_size = []
        for key in range(88):
            geom = kin.piano.keys[key].geom[0]
            bind = kin.physics.bind(geom)
            key_pos.append(np.asarray(bind.xpos, dtype=np.float32).copy())
            key_size.append(np.asarray(bind.size, dtype=np.float32).copy())
        key_pos_arr = np.stack(key_pos, axis=0).astype(np.float32)
        key_size_arr = np.stack(key_size, axis=0).astype(np.float32)

        for song in chosen:
            group = root[song]
            demos = np.arange(int(group["goals"].shape[0]))
            if demos.size > int(args.demos_per_song):
                demos = rng.choice(demos, size=int(args.demos_per_song), replace=False)
            for demo in sorted(int(v) for v in demos.tolist()):
                goals = np.asarray(group["goals"][demo], dtype=np.float32)[:, :88]
                piano = np.asarray(group["piano_states"][demo], dtype=np.float32)[:, :88]
                tips = np.asarray(group["hand_fingertips"][demo], dtype=np.float32).reshape(-1, 10, 3)
                frames = np.arange(0, min(goals.shape[0], tips.shape[0]), max(int(args.frame_stride), 1))
                for frame in frames:
                    active = np.flatnonzero((goals[frame] > float(args.threshold)) | (piano[frame] > float(args.threshold)))
                    if active.size == 0:
                        continue
                    bag_tips = rp1m_tips_to_bagatelle(tips[frame], y_offset=float(args.y_offset))
                    for key in active.astype(int).tolist():
                        xy_dist = np.linalg.norm(bag_tips[:, :2] - key_pos_arr[key, :2], axis=1)
                        finger = int(np.argmin(xy_dist))
                        tip = bag_tips[finger]
                        pos = key_pos_arr[key]
                        size = key_size_arr[key]
                        rows.append(
                            {
                                "song": song,
                                "demo": int(demo),
                                "frame": int(frame),
                                "key": int(key),
                                "finger": int(finger),
                                "xy_dist_m": float(xy_dist[finger]),
                                "front_offset_norm": float((tip[0] - pos[0]) / max(float(size[0]), 1e-6)),
                                "width_offset_norm": float((tip[1] - pos[1]) / max(float(size[1]), 1e-6)),
                                "top_minus_depth_norm": float((tip[2] - pos[2]) / max(float(size[2]), 1e-6)),
                                "z_delta_m": float(tip[2] - pos[2]),
                            }
                        )
                        if len(rows) >= int(args.max_contacts):
                            break
                    if len(rows) >= int(args.max_contacts):
                        break
                if len(rows) >= int(args.max_contacts):
                    break
            if len(rows) >= int(args.max_contacts):
                break

    close_rows = [row for row in rows if row["xy_dist_m"] <= 0.035]
    result = {
        "songs": chosen,
        "contacts": int(len(rows)),
        "close_contacts": int(len(close_rows)),
        "threshold": float(args.threshold),
        "y_offset": float(args.y_offset),
        "all": {
            "xy_dist_m": summarize(np.asarray([row["xy_dist_m"] for row in rows], dtype=np.float32)),
            "front_offset_norm": summarize(np.asarray([row["front_offset_norm"] for row in rows], dtype=np.float32)),
            "width_offset_norm": summarize(np.asarray([row["width_offset_norm"] for row in rows], dtype=np.float32)),
            "top_minus_depth_norm": summarize(np.asarray([row["top_minus_depth_norm"] for row in rows], dtype=np.float32)),
            "z_delta_m": summarize(np.asarray([row["z_delta_m"] for row in rows], dtype=np.float32)),
        },
        "close_xy": {
            "front_offset_norm": summarize(np.asarray([row["front_offset_norm"] for row in close_rows], dtype=np.float32)),
            "width_offset_norm": summarize(np.asarray([row["width_offset_norm"] for row in close_rows], dtype=np.float32)),
            "top_minus_depth_norm": summarize(np.asarray([row["top_minus_depth_norm"] for row in close_rows], dtype=np.float32)),
            "z_delta_m": summarize(np.asarray([row["z_delta_m"] for row in close_rows], dtype=np.float32)),
        },
        "sample_rows": rows[:50],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
