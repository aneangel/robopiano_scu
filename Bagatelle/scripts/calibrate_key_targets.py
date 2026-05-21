#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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


HAND_FINGERS = {"left": 1, "right": 6}


def _float_list(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _int_list(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic Bagatelle key target offset/depth sweep.")
    parser.add_argument("--output-root", required=True, help="Directory where key_target_calibration.* files are written.")
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--keys", default="28,29,40,41,52,53", help="Comma-separated piano key indices to sweep.")
    parser.add_argument("--hands", default="left,right", help="Comma-separated hands to force: left,right.")
    parser.add_argument("--front-offsets", default="0.25,0.35,0.45")
    parser.add_argument("--top-offsets", default="0.5")
    parser.add_argument("--press-depths", default="0.006,0.008,0.012")
    parser.add_argument("--ik-max-nfev", type=int, default=60)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap for quick smoke runs; 0 means no cap.")
    return parser


def _assignment(kin: BagatelleKinematics, finger_index: int, key: int) -> FingerAssignmentResult:
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
        strategy="key_target_calibration_forced_finger",
    )


def _zero_qvel(physics: Any) -> None:
    qvel = getattr(getattr(physics, "data", None), "qvel", None)
    if qvel is not None:
        qvel[:] = 0.0


def _set_qpos(kin: BagatelleKinematics, pose: np.ndarray) -> None:
    for joint, value in zip(kin.joint_handles, np.asarray(pose, dtype=np.float32)):
        kin.physics.bind(joint).qpos = float(value)


def _activation_after_replay(kin: BagatelleKinematics, pose: np.ndarray, settle_steps: int) -> np.ndarray:
    _set_qpos(kin, pose)
    _zero_qvel(kin.physics)
    kin.physics.forward()
    for _ in range(max(int(settle_steps), 0)):
        kin.physics.step()
    update = getattr(kin.piano, "_update_key_state", None)
    if callable(update):
        update(kin.physics)
    return np.asarray(kin.piano.activation, dtype=bool).copy()


def _case_row(
    *,
    kin: BagatelleKinematics,
    config: BagatelleConfig,
    key: int,
    hand: str,
    settle_steps: int,
) -> dict[str, Any]:
    finger_index = HAND_FINGERS[str(hand)]
    previous = kin.neutral_qpos.copy()
    first = kin.solve_press_pose(_assignment(kin, finger_index, key), previous, neutral_qpos=kin.neutral_qpos, config=config)
    second = kin.solve_press_pose(_assignment(kin, finger_index, key), first.pose, neutral_qpos=kin.neutral_qpos, config=config)
    activation = _activation_after_replay(kin, first.pose, settle_steps)
    active_keys = np.flatnonzero(activation).astype(np.int32)
    wrong_keys = active_keys[active_keys != int(key)]
    neighbor_keys = [candidate for candidate in (int(key) - 1, int(key) + 1) if 0 <= candidate < kin.piano.n_keys]
    return {
        "diagnostic_pose_replay": True,
        "key": int(key),
        "is_black_key": bool(kin.piano.is_key_black(int(key))),
        "hand": str(hand),
        "finger_index": int(finger_index),
        "front_offset": float(config.key_target_front_offset),
        "top_offset": float(config.key_target_top_offset),
        "press_depth": float(config.key_press_depth),
        "target_activated": bool(activation[int(key)]),
        "wrong_key_count": int(wrong_keys.size),
        "wrong_keys": wrong_keys.astype(int).tolist(),
        "neighbor_wrong_keys": [int(k) for k in neighbor_keys if bool(activation[int(k)])],
        "active_keys_after_replay": active_keys.astype(int).tolist(),
        "max_residual_m": float(first.max_residual),
        "residual_norm_m": float(first.residual_norm),
        "optimizer_success": bool(first.optimizer_success),
        "ik_success": bool(first.success),
        "nfev": int(first.nfev),
        "qpos_movement_norm": float(np.linalg.norm(first.pose.astype(np.float32) - previous.astype(np.float32))),
        "repeat_hold_qpos_delta_norm": float(np.linalg.norm(second.pose.astype(np.float32) - first.pose.astype(np.float32))),
        "repeat_hold_max_residual_m": float(second.max_residual),
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["front_offset"]), float(row["top_offset"]), float(row["press_depth"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (front, top, depth), group in grouped.items():
        residuals = np.asarray([float(row["max_residual_m"]) for row in group], dtype=np.float64)
        movements = np.asarray([float(row["qpos_movement_norm"]) for row in group], dtype=np.float64)
        summaries.append(
            {
                "front_offset": front,
                "top_offset": top,
                "press_depth": depth,
                "cases": int(len(group)),
                "target_activation_rate": float(np.mean([bool(row["target_activated"]) for row in group])),
                "wrong_key_rate": float(np.mean([int(row["wrong_key_count"]) > 0 for row in group])),
                "mean_max_residual_m": float(np.mean(residuals)) if residuals.size else None,
                "p95_max_residual_m": float(np.percentile(residuals, 95)) if residuals.size else None,
                "mean_qpos_movement_norm": float(np.mean(movements)) if movements.size else None,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -float(row["target_activation_rate"]),
            float(row["wrong_key_rate"]),
            float(row["mean_max_residual_m"] or 1e9),
            float(row["mean_qpos_movement_norm"] or 1e9),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bagatelle Key Target Calibration",
        "",
        "This is a diagnostic direct-pose replay sweep, not a rollout policy evaluation.",
        "",
        f"Recommended offsets/depth: `{report['recommended']}`",
        f"Best scored offsets/depth: `{report.get('best_scored')}`",
        f"Recommendation note: {report.get('recommendation_note')}",
        "",
        "## Summary By Target Parameters",
        "",
        "| Front | Top | Depth | Cases | Target activation | Wrong-key rate | Mean max residual |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summaries"]:
        lines.append(
            f"| {row['front_offset']:.3f} | {row['top_offset']:.3f} | {row['press_depth']:.3f} | {row['cases']} | "
            f"{row['target_activation_rate']:.3f} | {row['wrong_key_rate']:.3f} | {row['mean_max_residual_m']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    keys = _int_list(args.keys)
    hands = [hand.strip() for hand in str(args.hands).split(",") if hand.strip()]
    for hand in hands:
        if hand not in HAND_FINGERS:
            raise ValueError(f"Unknown hand {hand!r}; expected one of {sorted(HAND_FINGERS)}")
    base_config = BagatelleConfig(environment_name=str(args.environment_name), ik_max_nfev=int(args.ik_max_nfev))
    target_keys = np.zeros((1, 88), dtype=np.float32)
    for key in keys:
        target_keys[0, int(key)] = 1.0

    rows: list[dict[str, Any]] = []
    with BagatelleKinematics(base_config, target_keys=target_keys, output_dir=output_root) as kin:
        case_count = 0
        for front in _float_list(args.front_offsets):
            for top in _float_list(args.top_offsets):
                for depth in _float_list(args.press_depths):
                    config = replace(
                        base_config,
                        key_target_front_offset=float(front),
                        key_target_top_offset=float(top),
                        key_press_depth=float(depth),
                    )
                    kin.config = config
                    for key in keys:
                        for hand in hands:
                            kin.env.reset()
                            rows.append(_case_row(kin=kin, config=config, key=int(key), hand=hand, settle_steps=int(args.settle_steps)))
                            case_count += 1
                            if int(args.max_cases) > 0 and case_count >= int(args.max_cases):
                                break
                        if int(args.max_cases) > 0 and case_count >= int(args.max_cases):
                            break
                    if int(args.max_cases) > 0 and case_count >= int(args.max_cases):
                        break
                if int(args.max_cases) > 0 and case_count >= int(args.max_cases):
                    break
            if int(args.max_cases) > 0 and case_count >= int(args.max_cases):
                break

    summaries = _summaries(rows)
    best_scored = summaries[0] if summaries else None
    recommended = best_scored if best_scored and float(best_scored["target_activation_rate"]) > 0.0 else None
    report = {
        "diagnostic_pose_replay": True,
        "keys": keys,
        "hands": hands,
        "settle_steps": int(args.settle_steps),
        "cases": rows,
        "summaries": summaries,
        "best_scored": best_scored,
        "recommended": recommended,
        "recommendation_note": (
            "No target offset/depth is recommended from this run because no swept case actuated the target key."
            if recommended is None else
            "Recommended by highest target activation rate, lowest wrong-key rate, residual, and qpos movement."
        ),
    }
    json_path = output_root / "key_target_calibration.json"
    md_path = output_root / "key_target_calibration.md"
    csv_path = output_root / "key_target_calibration_cases.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    _write_csv(csv_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    if report["recommended"] is not None:
        print(f"Recommended: {report['recommended']}")
    elif report["best_scored"] is not None:
        print(f"No recommended target from this run. Best scored: {report['best_scored']}")


if __name__ == "__main__":
    main()
