#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Variations" / "src",
    REPO_ROOT / "Variations",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from bagatelle.assignment import FingerAssignmentResult, NUM_FINGERS  # noqa: E402
from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from intermezzo.constants import FINGER_JOINT_INDICES, FOREARM_TY_INDICES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Bagatelle fingertip and reduced-joint ordering.")
    parser.add_argument("--output-root", required=True, help="Directory where finger_order_audit.json and .md are written.")
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--diagnostic-key", type=int, default=40, help="Known key used for one diagnostic pose per finger.")
    parser.add_argument("--ik-max-nfev", type=int, default=60)
    parser.add_argument("--skip-poses", action="store_true", help="Only audit names/order; do not solve diagnostic poses.")
    return parser


def _single_finger_assignment(kin: BagatelleKinematics, finger_index: int, key: int) -> FingerAssignmentResult:
    active_keys = np.asarray([key], dtype=np.int32)
    target = kin.key_press_targets(active_keys).astype(np.float32)
    cost_matrix = np.full((NUM_FINGERS, 1), np.nan, dtype=np.float32)
    cost_matrix[int(finger_index), 0] = 0.0
    return FingerAssignmentResult(
        active_keys=active_keys,
        assigned_finger_indices=np.asarray([finger_index], dtype=np.int32),
        assigned_keys=active_keys.copy(),
        assigned_key_positions=np.asarray([0], dtype=np.int32),
        target_positions=target,
        unassigned_keys=np.zeros((0,), dtype=np.int32),
        cost_matrix=cost_matrix,
        total_cost=0.0,
        mean_cost=0.0,
        strategy="finger_order_audit",
    )


def _diagnostic_poses(kin: BagatelleKinematics, config: BagatelleConfig, key: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = kin.neutral_qpos.copy()
    for finger_index, site_name in enumerate(kin.fingertip_site_names):
        assignment = _single_finger_assignment(kin, finger_index, key)
        result = kin.solve_press_pose(assignment, previous, neutral_qpos=kin.neutral_qpos, config=config)
        rows.append(
            {
                "finger_index": int(finger_index),
                "site_name": str(site_name),
                "key": int(key),
                "success": bool(result.success),
                "optimizer_success": bool(result.optimizer_success),
                "max_residual": float(result.max_residual),
                "residual_norm": float(result.residual_norm),
                "nfev": int(result.nfev),
                "pose": result.pose.astype(float).tolist(),
                "fingertip_position": result.fingertip_positions[int(finger_index)].astype(float).tolist(),
            }
        )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bagatelle Finger Order Audit",
        "",
        f"Environment: `{report['environment_name']}`",
        f"Finger order: `{report['finger_order']}`",
        f"Joint order: `{report['joint_order']}`",
        "",
        "## Assignment Finger To Fingertip Site",
        "",
        "| Finger index | Hand | Finger | Fingertip site |",
        "| ---: | --- | --- | --- |",
    ]
    hands = report.get("finger_hands", [])
    finger_names = report.get("finger_names", [])
    for row in report["assignment_finger_to_site"]:
        index = int(row["finger_index"])
        hand = hands[index] if index < len(hands) else ""
        finger = finger_names[index] if index < len(finger_names) else ""
        lines.append(f"| {index} | {hand} | {finger} | `{row['site_name']}` |")
    lines.extend(["", "## Reduced Joint Index Ranges", "", "| Hand | Start | End exclusive |", "| --- | ---: | ---: |"])
    for hand, bounds in report["joint_index_ranges_by_hand"].items():
        lines.append(f"| {hand} | {bounds[0]} | {bounds[1]} |")
    lines.extend(["", "## Intermezzo Comparison", ""])
    lines.append(f"Intermezzo forearm TY indices: `{report['intermezzo']['forearm_ty_indices']}`")
    lines.append(f"Intermezzo finger joint indices are right-hand first: `{report['intermezzo']['finger_joint_indices']}`")
    if report["diagnostic_poses"]:
        lines.extend(["", "## Diagnostic Single-Key Poses", "", "| Finger | Site | Key | Success | Max residual | nfev |", "| ---: | --- | ---: | --- | ---: | ---: |"])
        for row in report["diagnostic_poses"]:
            lines.append(
                f"| {row['finger_index']} | `{row['site_name']}` | {row['key']} | {row['success']} | {row['max_residual']:.6f} | {row['nfev']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    config = BagatelleConfig(environment_name=str(args.environment_name), ik_max_nfev=int(args.ik_max_nfev))
    target_keys = np.zeros((1, 88), dtype=np.float32)
    target_keys[0, int(args.diagnostic_key)] = 1.0
    with BagatelleKinematics(config, target_keys=target_keys, output_dir=output_root) as kin:
        order_metadata = kin.order_metadata
        report: dict[str, Any] = {
            **order_metadata,
            "environment_name": str(kin.environment_name),
            "midi_proto_path": str(kin.midi_proto_path),
            "load_info": kin.load_info,
            "joint_bounds": {
                "lower": kin.joint_lower.astype(float).tolist(),
                "upper": kin.joint_upper.astype(float).tolist(),
            },
            "intermezzo": {
                "finger_joint_indices": [list(indices) for indices in FINGER_JOINT_INDICES],
                "forearm_ty_indices": list(FOREARM_TY_INDICES),
                "note": "Intermezzo FINGER_JOINT_INDICES are right hand first, then left hand; Bagatelle assignment indices are left fingertip sites first, then right fingertip sites.",
            },
            "diagnostic_poses": [] if args.skip_poses else _diagnostic_poses(kin, config, int(args.diagnostic_key)),
        }
    json_path = output_root / "finger_order_audit.json"
    md_path = output_root / "finger_order_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
