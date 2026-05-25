#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from fugue.constants import DEFAULT_RP1M_ROOT  # noqa: E402
from rp1m_simulator.simulator import (  # noqa: E402
    ACTION_MAPPINGS,
    ACTION_SOURCE_SCALES,
    DEFAULT_HAND_ANCHOR_Y_OFFSET,
    RolloutConfig,
    load_rp1m_trajectory,
    save_json,
    simulate_rp1m_rollout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an RP1M Fugue demo through the shared rp1m_simulator package."
    )
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--song-key", required=True)
    parser.add_argument("--demo-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--mode", choices=["hand_state", "action"], default="hand_state")
    parser.add_argument("--dataset-dt", "--dataset-timestep", dest="dataset_timestep", type=float, default=0.05)
    parser.add_argument("--sim-dt", "--simulation-timestep", dest="simulation_timestep", type=float, default=0.005)
    parser.add_argument("--max-frames", "--max-source-steps", dest="max_source_steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every-source-frame", "--render-every-source-step", dest="render_every_source_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--action-source-scale", default="normalized_minus_one_to_one", choices=list(ACTION_SOURCE_SCALES))
    parser.add_argument("--action-mapping", default="as_is", choices=list(ACTION_MAPPINGS))
    parser.add_argument("--full-action-space", dest="reduced_action_space", action="store_false")
    parser.set_defaults(reduced_action_space=True)
    parser.add_argument("--no-set-hand-qvel", dest="set_hand_qvel", action="store_false")
    parser.set_defaults(set_hand_qvel=True)
    parser.add_argument("--disable-hand-collisions", action="store_true")
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--primitive-fingertip-collisions", action="store_true")
    parser.add_argument("--hand-anchor-y-offset", type=float, default=DEFAULT_HAND_ANCHOR_Y_OFFSET)
    parser.add_argument("--no-hand-anchor-y-offset", dest="hand_anchor_y_offset", action="store_const", const=None)
    parser.add_argument("--auto-hand-anchor-y-offset", action="store_true")
    parser.add_argument("--no-audio", dest="render_audio", action="store_false")
    parser.set_defaults(render_audio=True)
    parser.add_argument(
        "--restore-initial-piano",
        action="store_true",
        help="Accepted for compatibility; piano states are never restored.",
    )
    parser.add_argument("--no-restore-initial-piano", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-fingering-reward", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-forearm-reward", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectory = load_rp1m_trajectory(args.rp1m_root, args.song_key, int(args.demo_id), include_reference_piano_states=True)
    if args.environment_name:
        trajectory.environment_name = str(args.environment_name)
    config = RolloutConfig(
        mode=str(args.mode),  # type: ignore[arg-type]
        dataset_timestep=float(args.dataset_timestep),
        simulation_timestep=float(args.simulation_timestep),
        hand_anchor_y_offset=args.hand_anchor_y_offset,
        auto_hand_anchor_y_offset=bool(args.auto_hand_anchor_y_offset),
        reduced_action_space=bool(args.reduced_action_space),
        action_source_scale=str(args.action_source_scale),  # type: ignore[arg-type]
        action_mapping=str(args.action_mapping),  # type: ignore[arg-type]
        hand_state_action_source="recorded",
        set_hand_qvel=bool(args.set_hand_qvel),
        gravity_compensation=bool(args.gravity_compensation),
        primitive_fingertip_collisions=bool(args.primitive_fingertip_collisions),
        disable_hand_collisions=bool(args.disable_hand_collisions),
        seed=int(args.seed),
        threshold=float(args.threshold),
        max_source_steps=args.max_source_steps,
        render_mp4=True,
        render_audio=bool(args.render_audio),
        render_every_source_step=max(int(args.render_every_source_step), 1),
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
    )
    output_dir = Path(args.output_dir)
    summary = simulate_rp1m_rollout(trajectory, config, output_dir)
    summary_path = save_json(output_dir / "interpolated_playback_summary.json", summary)
    print(f"summary_path={summary_path}")
    print(f"rollout_npz={summary.get('rollout_npz')}")
    print(f"video_path={summary.get('video_path')}")
    print(f"recorded_goal_f1={(summary.get('recorded_reference_against_goals') or {}).get('key_f1')}")
    print(f"rollout_goal_f1={(summary.get('against_goals') or {}).get('key_f1')}")
    print(f"rollout_reference_f1={(summary.get('against_reference_piano_states') or {}).get('key_f1')}")
    print(f"piano_state_policy={summary.get('piano_state_policy')}")


if __name__ == "__main__":
    main()
