from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PARTITA_ROOT.parent
for import_root in [REPO_ROOT, PARTITA_ROOT / "src"]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rp1m_simulator.simulator import (  # noqa: E402
    ACTION_MAPPINGS,
    ACTION_SOURCE_SCALES,
    DEFAULT_HAND_ANCHOR_Y_OFFSET,
    DEFAULT_RP1M_ROOT,
    RolloutConfig,
    load_rp1m_trajectory,
    save_json,
    simulate_rp1m_rollout,
)


DEFAULT_SONG_KEY = "RoboPianist-GP-AlberyDorothyPreludeInDFlatMajor-v0_0"
DEFAULT_DEMO_ID = 206
DEFAULT_RUN_ROOT = "/WAVE/datasets/ccoelho_lab-jlanders/Fugue/runs"


def _default_output_dir(song_key: str, demo_id: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_song = song_key.replace("/", "_")
    return Path(DEFAULT_RUN_ROOT) / f"partita_rp1m_simulator_{safe_song}_demo{demo_id}_{stamp}"


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        return str(args.mode)
    if args.playback_mode == "action":
        return "action"
    return "hand_state"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RP1M playback through the shared rp1m_simulator package."
    )
    parser.add_argument("--rp1m-root", default=DEFAULT_RP1M_ROOT)
    parser.add_argument("--song-key", default=DEFAULT_SONG_KEY)
    parser.add_argument("--demo-id", "--trajectory-id", dest="demo_id", type=int, default=DEFAULT_DEMO_ID)
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["hand_state", "action"], default=None)
    parser.add_argument(
        "--playback-mode",
        choices=["recorded_state", "physical_action", "hand_state", "action"],
        default="hand_state",
        help="Deprecated alias. recorded_state/physical_action map to hand_state without piano-state restore.",
    )
    parser.add_argument("--dataset-timestep", type=float, default=0.05)
    parser.add_argument("--simulation-timestep", type=float, default=0.005)
    parser.add_argument("--max-source-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every-source-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--action-source-scale", choices=list(ACTION_SOURCE_SCALES), default="normalized_minus_one_to_one")
    parser.add_argument("--action-mapping", choices=list(ACTION_MAPPINGS), default="as_is")
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
        "--no-restore-initial-piano",
        action="store_true",
        help="Accepted for compatibility; piano states are never restored by this script.",
    )
    parser.add_argument("--enable-fingering-reward", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable-forearm-reward", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prefer-canonical-midi", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", default="rp1m_demo_interpolated")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.song_key, args.demo_id)
    trajectory = load_rp1m_trajectory(args.rp1m_root, args.song_key, int(args.demo_id), include_reference_piano_states=True)
    if args.environment_name:
        trajectory.environment_name = str(args.environment_name)

    config = RolloutConfig(
        mode=_resolve_mode(args),  # type: ignore[arg-type]
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
    summary = simulate_rp1m_rollout(trajectory, config, output_dir)
    summary["label"] = str(args.label)
    summary["requested_playback_mode"] = str(args.playback_mode)
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
