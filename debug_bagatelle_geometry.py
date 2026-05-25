#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[0]
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Variations" / "src",
    REPO_ROOT / "Variations",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _config_from_metadata(metadata_path: Path) -> BagatelleConfig:
    if not metadata_path.is_file():
        return BagatelleConfig()
    metadata = json.loads(metadata_path.read_text())
    source = metadata.get("bagatelle_config") or metadata.get("config") or {}
    names = {field.name for field in fields(BagatelleConfig)}
    kwargs = {name: source[name] for name in names if name in source}
    return BagatelleConfig(**kwargs)


def _direct_reward_key_targets(kin: BagatelleKinematics, keys: np.ndarray, *, press_depth: float) -> np.ndarray:
    out = []
    for key in np.asarray(keys, dtype=np.int32).reshape(-1):
        key_geom = kin.piano.keys[int(key)].geom[0]
        bind = kin.physics.bind(key_geom)
        pos = np.asarray(bind.xpos, dtype=np.float32).copy()
        size = np.asarray(bind.size, dtype=np.float32)
        pos[-1] += 0.5 * float(size[2])
        pos[0] += 0.35 * float(size[0])
        pos[-1] -= float(press_depth)
        out.append(pos.astype(np.float32))
    return np.stack(out, axis=0).astype(np.float32) if out else np.zeros((0, 3), dtype=np.float32)


def _sorted_counts(values: np.ndarray) -> list[dict[str, int]]:
    flat = np.asarray(values, dtype=np.int32).reshape(-1)
    flat = flat[flat >= 0]
    if flat.size == 0:
        return []
    keys, counts = np.unique(flat, return_counts=True)
    rows = [{"key": int(key), "count": int(count)} for key, count in zip(keys, counts)]
    rows.sort(key=lambda row: (-row["count"], row["key"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--max-waypoints", type=int, default=500)
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory_npz).expanduser().resolve()
    metadata_path = Path(args.metadata_json).expanduser().resolve() if args.metadata_json else trajectory_path.parent / "metadata.json"
    data = np.load(trajectory_path, allow_pickle=False)
    target_keys = np.asarray(data["target_keys"], dtype=np.float32)
    waypoint_frames = np.asarray(data["waypoint_frames"], dtype=np.int64)
    assignments = np.asarray(data["assignments"], dtype=np.int32)
    fingertip_targets = np.asarray(data["fingertip_targets"], dtype=np.float32)
    waypoint_fingertips = np.asarray(data["waypoint_fingertips"], dtype=np.float32)

    config = _config_from_metadata(metadata_path)
    if args.environment_name:
        config = BagatelleConfig(**{**config.to_dict(), "environment_name": str(args.environment_name)})

    report: dict[str, Any] = {
        "trajectory_npz": str(trajectory_path),
        "metadata_json": str(metadata_path),
        "config": config.to_dict(),
    }
    with BagatelleKinematics(config, target_keys=target_keys) as kin:
        all_keys = np.arange(88, dtype=np.int32)
        contact = kin.key_contact_targets(all_keys)
        direct_contact = _direct_reward_key_targets(kin, all_keys, press_depth=0.0)
        press = kin.key_press_targets(all_keys)
        direct_press = _direct_reward_key_targets(kin, all_keys, press_depth=float(config.key_press_depth))
        diffs = np.diff(contact, axis=0)
        report.update(
            environment_name=str(kin.environment_name),
            finger_order=str(kin.order_metadata.get("finger_order")),
            joint_order=str(kin.order_metadata.get("joint_order")),
            fingertip_site_names=list(kin.fingertip_site_names),
            joint_names=list(kin.joint_names),
            neutral_fingertips=np.asarray(kin.neutral_fingertips, dtype=np.float32),
            key_contact_formula_max_abs_diff=float(np.max(np.abs(contact - direct_contact))),
            key_press_formula_max_abs_diff=float(np.max(np.abs(press - direct_press))),
            key_contact_position_summary={
                "min": np.min(contact, axis=0),
                "max": np.max(contact, axis=0),
                "mean_delta": np.mean(diffs, axis=0),
                "min_delta": np.min(diffs, axis=0),
                "max_delta": np.max(diffs, axis=0),
                "sample_keys": {
                    str(key): {
                        "contact": contact[key],
                        "press": press[key],
                        "direct_press": direct_press[key],
                    }
                    for key in (0, 1, 20, 40, 44, 60, 87)
                },
            },
        )

        row_reports = []
        mismatch_count = 0
        target_diff_max = 0.0
        residuals = []
        hand_key_rows = []
        for row, frame in enumerate(waypoint_frames[: max(int(args.max_waypoints), 0)]):
            active = np.flatnonzero(target_keys[int(frame), :88] > float(config.threshold)).astype(np.int32)
            assigned_pairs = [
                (int(finger), int(key))
                for finger, key in enumerate(assignments[row].astype(np.int32).tolist())
                if int(key) >= 0
            ]
            assigned = np.asarray([key for _finger, key in assigned_pairs], dtype=np.int32)
            missing = sorted(set(active.tolist()) - set(assigned.tolist()))
            extra = sorted(set(assigned.tolist()) - set(active.tolist()))
            if missing or extra:
                mismatch_count += 1
            diffs_for_row = []
            residuals_for_row = []
            for finger, key in assigned_pairs:
                expected = kin.key_press_targets(np.asarray([key], dtype=np.int32))[0]
                stored = fingertip_targets[row, finger]
                diff = float(np.linalg.norm(stored - expected))
                diffs_for_row.append(diff)
                residual = float(np.linalg.norm(waypoint_fingertips[row, finger] - stored))
                residuals_for_row.append(residual)
                hand_key_rows.append({"finger": finger, "hand": "left" if finger < 5 else "right", "key": key})
            if diffs_for_row:
                target_diff_max = max(target_diff_max, max(diffs_for_row))
                residuals.extend(residuals_for_row)
            if row < 30 or missing or extra:
                row_reports.append(
                    {
                        "waypoint_row": int(row),
                        "frame": int(frame),
                        "active_keys": active,
                        "assigned_pairs": assigned_pairs,
                        "missing_active_keys": missing,
                        "extra_assigned_keys": extra,
                        "stored_vs_live_target_max_distance": max(diffs_for_row) if diffs_for_row else 0.0,
                        "waypoint_fingertip_residual_max": max(residuals_for_row) if residuals_for_row else 0.0,
                    }
                )

        report.update(
            waypoint_count=int(waypoint_frames.size),
            assignment_active_key_mismatch_count=int(mismatch_count),
            stored_vs_live_key_target_max_distance=float(target_diff_max),
            waypoint_fingertip_residual_summary={
                "count": int(len(residuals)),
                "p50": float(np.percentile(residuals, 50)) if residuals else 0.0,
                "p90": float(np.percentile(residuals, 90)) if residuals else 0.0,
                "max": float(np.max(residuals)) if residuals else 0.0,
            },
            assigned_key_histogram=_sorted_counts(assignments),
            assigned_hand_key_examples=hand_key_rows[:100],
            waypoint_reports=row_reports,
        )

    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable({
        "output_json": str(output),
        "key_contact_formula_max_abs_diff": report["key_contact_formula_max_abs_diff"],
        "key_press_formula_max_abs_diff": report["key_press_formula_max_abs_diff"],
        "assignment_active_key_mismatch_count": report["assignment_active_key_mismatch_count"],
        "stored_vs_live_key_target_max_distance": report["stored_vs_live_key_target_max_distance"],
        "waypoint_fingertip_residual_summary": report["waypoint_fingertip_residual_summary"],
        "finger_order": report["finger_order"],
        "fingertip_site_names": report["fingertip_site_names"],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
