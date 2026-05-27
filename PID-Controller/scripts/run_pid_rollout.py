#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_SRC = Path(__file__).resolve().parents[1] / "src"
for path in (
    MODULE_SRC,
    REPO_ROOT,
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "partita" / "src",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from pid_controller.rollout import run_impromptu_pid_rollout  # noqa: E402


def _infer_environment_name(trajectory_npz: Path, fallback: str) -> str:
    metadata_path = trajectory_npz.parent / "metadata.json"
    if not metadata_path.exists():
        return fallback
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    return str(metadata.get("environment_name") or fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a P/PD/PID action-only rollout from Impromptu hand targets.")
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--controller", choices=["p", "pd", "pid"], default="pd")
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--ki", type=float, default=None)
    parser.add_argument("--integral-limit", type=float, default=0.25)
    parser.add_argument("--setpoint-policy", choices=["next", "linear", "minimum_jerk"], default="minimum_jerk")
    parser.add_argument("--target-velocity-scale", type=float, default=0.0)
    parser.add_argument("--no-target-velocity", dest="use_target_velocity", action="store_false")
    parser.add_argument("--use-target-velocity", dest="use_target_velocity", action="store_true")
    parser.set_defaults(use_target_velocity=False)
    parser.add_argument("--feedforward-scale", type=float, default=0.0)
    parser.add_argument("--lookahead-substeps", type=int, default=10)
    parser.add_argument("--sustain-value", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-source-steps", type=int, default=20)
    parser.add_argument("--full-song", action="store_true")
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--hand-anchor-y-offset", type=float, default=None)
    parser.add_argument("--disable-hand-collisions", action="store_true")
    args = parser.parse_args()

    trajectory_npz = Path(args.trajectory_npz)
    environment_name = args.environment_name or _infer_environment_name(
        trajectory_npz,
        "RoboPianist-debug-NocturneRousseau-v0",
    )
    result = run_impromptu_pid_rollout(
        trajectory_npz=trajectory_npz,
        output_dir=Path(args.output_dir),
        environment_name=environment_name,
        controller_kind=args.controller,
        kp=args.kp,
        kd=args.kd,
        ki=args.ki,
        integral_limit=float(args.integral_limit),
        setpoint_policy=args.setpoint_policy,
        use_target_velocity=bool(args.use_target_velocity),
        target_velocity_scale=float(args.target_velocity_scale),
        feedforward_scale=float(args.feedforward_scale),
        lookahead_substeps=int(args.lookahead_substeps),
        sustain_value=float(args.sustain_value),
        threshold=float(args.threshold),
        seed=int(args.seed),
        max_source_steps=None if args.full_song else int(args.max_source_steps),
        render_mp4=bool(args.render_mp4),
        hand_anchor_y_offset=args.hand_anchor_y_offset,
        disable_hand_collisions=bool(args.disable_hand_collisions),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
