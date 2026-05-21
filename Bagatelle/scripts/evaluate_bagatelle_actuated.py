#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Etude" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Variations" / "src",
    REPO_ROOT / "Variations",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.evaluation import DEFAULT_OUTPUT_ROOT, evaluate_bagatelle_actuated_trajectory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Bagatelle playback through RoboPianist actuators.")
    parser.add_argument("--trajectory-npz", required=True, help="Path to Bagatelle trajectory.npz.")
    parser.add_argument("--controller-family", default="pd", choices=("pd", "scheduled_pd", "pd_scheduled"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for unique evaluation run dirs.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--label", default="bagatelle")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--timing-tolerance-s", type=float, default=0.15)
    parser.add_argument("--residual-success-threshold", type=float, default=0.02)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--kp", type=float, default=12.0)
    parser.add_argument("--kd", type=float, default=0.6)
    parser.add_argument("--lookahead-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional env-step cap for smoke tests.")
    parser.add_argument(
        "--skip-pose-replay-diagnostic",
        action="store_true",
        help="Only run actuated playback. By default the summary includes diagnostic pose replay side by side.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BagatelleConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        seed=int(args.seed),
        environment_name=str(args.environment_name),
        residual_success_threshold=float(args.residual_success_threshold),
        settle_steps=int(args.settle_steps),
    )
    summary = evaluate_bagatelle_actuated_trajectory(
        args.trajectory_npz,
        output_root=args.output_root,
        run_name=args.run_name,
        label=str(args.label),
        config=config,
        controller_family=str(args.controller_family),
        kp=float(args.kp),
        kd=float(args.kd),
        lookahead_steps=int(args.lookahead_steps),
        timing_tolerance_s=float(args.timing_tolerance_s),
        max_steps=args.max_steps,
        include_pose_replay_diagnostic=not bool(args.skip_pose_replay_diagnostic),
    )
    actuated = summary["actuated_rollout_metrics"]
    score = actuated["score"]
    validity = actuated["metric_validity"]
    print(f"Wrote Bagatelle actuated evaluation: {summary['run_dir']}")
    print(
        f"actuated_actions={actuated['actions_executed']} "
        f"action_dim={actuated['action_dim']} "
        f"matched={score['matched_press_events']}/{score['target_press_events']} "
        f"missed={score['missed_key_presses']} mispresses={score['mispresses']} "
        f"leakage_checks={summary['metric_validity']['actuated_passes_leakage_checks']}"
    )
    if not summary["metric_validity"]["actuated_passes_leakage_checks"]:
        failed = [key for key, value in validity.items() if key != "controller_input_uses_colorization" and not value]
        raise SystemExit(f"Actuated metric validity checks failed: {failed}")


if __name__ == "__main__":
    main()
