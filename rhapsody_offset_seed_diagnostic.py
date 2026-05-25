#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, REPO_ROOT / "Bagatelle" / "src", REPO_ROOT / "Rhapsody" / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402


def _metrics(errors: np.ndarray) -> dict[str, float]:
    arr = np.asarray(errors, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real Bagatelle errors for Rhapsody IK seeds under y offsets.")
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--offsets", default="0.0,0.062666565,0.08289646")
    parser.add_argument("--max-waypoints", type=int, default=64)
    parser.add_argument("--refinement-steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=524)
    args = parser.parse_args()

    offsets = [float(v) for v in str(args.offsets).split(",") if v.strip()]
    with np.load(args.trajectory_npz, allow_pickle=False) as data:
        target_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
        waypoint_targets = np.asarray(data["fingertip_targets"], dtype=np.float32)
        waypoint_qpos = np.asarray(data["waypoint_hand_joints"], dtype=np.float32)
    total = int(waypoint_targets.shape[0])
    if total == 0:
        raise ValueError("trajectory contains no waypoints")
    rng = np.random.default_rng(int(args.seed))
    indices = np.arange(total, dtype=np.int64)
    if indices.size > int(args.max_waypoints):
        indices = np.sort(rng.choice(indices, size=int(args.max_waypoints), replace=False)).astype(np.int64)

    rows: list[dict[str, Any]] = []
    with BagatelleKinematics(
        config=BagatelleConfig(environment_name=str(args.environment_name), seed=int(args.seed)),
        target_keys=target_keys,
    ) as kin:
        baseline_errors: list[float] = []
        for idx in indices:
            targets = waypoint_targets[int(idx)]
            mask = np.isfinite(targets).all(axis=1)
            if not bool(np.any(mask)):
                continue
            tips = kin.fingertip_positions_for_qpos(waypoint_qpos[int(idx)])
            baseline_errors.extend(np.linalg.norm(tips[mask] - targets[mask], axis=1).astype(float).tolist())
        rows.append(
            {
                "variant": "existing_waypoint_qpos",
                "offset": None,
                "waypoints": int(indices.size),
                "active_errors_m": _metrics(np.asarray(baseline_errors, dtype=np.float32)),
            }
        )
        for offset in offsets:
            cfg = BagatelleConfig(
                environment_name=str(args.environment_name),
                seed=int(args.seed),
                rhapsody_ik_enabled=True,
                rhapsody_ik_checkpoint=str(args.checkpoint),
                rhapsody_ik_refinement_steps=int(args.refinement_steps),
                rhapsody_ik_device=str(args.device),
                rhapsody_ik_y_offset=float(offset),
                rhapsody_ik_coordinate_transform="bagatelle_to_rp1m",
                rhapsody_ik_fill_inactive_from_previous=True,
            )
            active_errors: list[float] = []
            static_hits = static_wrong = static_missed = static_played = 0
            failures = 0
            for idx in indices:
                targets = waypoint_targets[int(idx)].copy()
                mask = np.isfinite(targets).all(axis=1).astype(np.float32)
                if not bool(np.any(mask > 0.0)):
                    continue
                previous = waypoint_qpos[max(int(idx) - 1, 0)]
                try:
                    qpos = kin.rhapsody_seed_for_fingertips(
                        targets,
                        mask,
                        previous,
                        config=cfg,
                    )
                    tips = kin.fingertip_positions_for_qpos(qpos)
                    active = mask > 0.0
                    active_errors.extend(np.linalg.norm(tips[active] - targets[active], axis=1).astype(float).tolist())
                    assigned_keys = np.flatnonzero(target_keys[min(int(idx), target_keys.shape[0] - 1)] > 0.5)
                    metrics = kin.static_contact_metrics(
                        qpos,
                        assigned_keys.astype(np.int32),
                        threshold=0.5,
                        settle_steps=1,
                    )
                    static_hits += int(metrics["target_hit_count"])
                    static_wrong += int(metrics["wrong_key_count"])
                    static_missed += int(metrics["missed_key_count"])
                    static_played += int(metrics["played_key_count"])
                except Exception:
                    failures += 1
            rows.append(
                {
                    "variant": "rhapsody_seed",
                    "offset": float(offset),
                    "waypoints": int(indices.size),
                    "failures": int(failures),
                    "active_errors_m": _metrics(np.asarray(active_errors, dtype=np.float32)),
                    "static_contact": {
                        "hits": int(static_hits),
                        "wrong": int(static_wrong),
                        "missed": int(static_missed),
                        "played": int(static_played),
                    },
                }
            )

    result = {
        "trajectory_npz": str(args.trajectory_npz),
        "checkpoint": str(args.checkpoint),
        "indices": indices.astype(int).tolist(),
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
