from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from rp1m_simulator.simulator import (
    DEFAULT_HAND_ANCHOR_Y_OFFSET,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RP1M_ROOT,
    RolloutConfig,
    find_high_f1_examples,
    load_rp1m_trajectory,
    parse_example,
    row_from_summary,
    save_json,
    simulate_rp1m_rollout,
    write_validation_csv,
)


def _add_rollout_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["hand_state", "action"], default="hand_state")
    parser.add_argument("--dataset-timestep", type=float, default=0.05)
    parser.add_argument("--simulation-timestep", type=float, default=0.005)
    parser.add_argument("--hand-anchor-y-offset", type=float, default=DEFAULT_HAND_ANCHOR_Y_OFFSET)
    parser.add_argument("--no-hand-anchor-y-offset", dest="hand_anchor_y_offset", action="store_const", const=None)
    parser.add_argument("--auto-hand-anchor-y-offset", dest="auto_hand_anchor_y_offset", action="store_true")
    parser.add_argument("--no-auto-hand-anchor-y-offset", dest="auto_hand_anchor_y_offset", action="store_false")
    parser.set_defaults(auto_hand_anchor_y_offset=True)
    parser.add_argument("--hand-anchor-application", choices=["compile_time", "post_reset"], default="post_reset")
    parser.add_argument("--hand-state-action-source", choices=["recorded", "zero"], default="recorded")
    parser.add_argument("--action-source-scale", choices=["normalized_minus_one_to_one", "actuator_units"], default="normalized_minus_one_to_one")
    parser.add_argument("--action-mapping", choices=["as_is", "swap_hands", "zero_sustain", "invert_sustain", "swap_hands_zero_sustain"], default="as_is")
    parser.add_argument("--action-substep-policy", choices=["zero_pad_hold", "zero_control", "zero_source", "repeat"], default="zero_pad_hold")
    parser.add_argument("--wrist-action-policy", choices=["hold_initial", "recorded"], default="hold_initial")
    parser.add_argument("--full-action-space", dest="reduced_action_space", action="store_false")
    parser.set_defaults(reduced_action_space=True)
    parser.add_argument("--no-restore-initial-hand", dest="restore_initial_hand", action="store_false")
    parser.set_defaults(restore_initial_hand=True)
    parser.add_argument("--no-set-hand-qvel", dest="set_hand_qvel", action="store_false")
    parser.set_defaults(set_hand_qvel=True)
    parser.add_argument("--initial-hand-qvel-scale", type=float, default=0.5)
    parser.add_argument("--hand-resync-interval", type=int, default=None)
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--primitive-fingertip-collisions", action="store_true")
    parser.add_argument("--disable-hand-collisions", action="store_true")
    parser.add_argument("--mujoco-integrator", type=int, default=None)
    parser.add_argument("--mujoco-solver", type=int, default=None)
    parser.add_argument("--mujoco-cone", type=int, default=None)
    parser.add_argument("--mujoco-jacobian", type=int, default=None)
    parser.add_argument("--mujoco-iterations", type=int, default=None)
    parser.add_argument("--mujoco-ls-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-source-steps", type=int, default=None)
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--no-audio", dest="render_audio", action="store_false")
    parser.set_defaults(render_audio=True)
    parser.add_argument("--render-every-source-step", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--fps", type=int, default=20)


def _config_from_args(args: argparse.Namespace) -> RolloutConfig:
    return RolloutConfig(
        mode=args.mode,
        dataset_timestep=args.dataset_timestep,
        simulation_timestep=args.simulation_timestep,
        hand_anchor_y_offset=args.hand_anchor_y_offset,
        auto_hand_anchor_y_offset=args.auto_hand_anchor_y_offset,
        hand_anchor_application=args.hand_anchor_application,
        reduced_action_space=args.reduced_action_space,
        action_source_scale=args.action_source_scale,
        action_mapping=args.action_mapping,
        action_substep_policy=args.action_substep_policy,
        wrist_action_policy=args.wrist_action_policy,
        hand_state_action_source=args.hand_state_action_source,
        restore_initial_hand=args.restore_initial_hand,
        set_hand_qvel=args.set_hand_qvel,
        initial_hand_qvel_scale=args.initial_hand_qvel_scale,
        hand_resync_interval=args.hand_resync_interval,
        gravity_compensation=args.gravity_compensation,
        primitive_fingertip_collisions=args.primitive_fingertip_collisions,
        disable_hand_collisions=args.disable_hand_collisions,
        mujoco_integrator=args.mujoco_integrator,
        mujoco_solver=args.mujoco_solver,
        mujoco_cone=args.mujoco_cone,
        mujoco_jacobian=args.mujoco_jacobian,
        mujoco_iterations=args.mujoco_iterations,
        mujoco_ls_iterations=args.mujoco_ls_iterations,
        seed=args.seed,
        threshold=args.threshold,
        max_source_steps=args.max_source_steps,
        render_mp4=args.render_mp4,
        render_audio=args.render_audio,
        render_every_source_step=args.render_every_source_step,
        width=args.width,
        height=args.height,
        camera_id=args.camera_id,
        fps=args.fps,
    )


def _run_rollout(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    trajectory = load_rp1m_trajectory(args.rp1m_root, args.song_key, args.demo_id, include_reference_piano_states=True)
    output_dir = Path(args.output_dir) if args.output_dir else Path(DEFAULT_OUTPUT_ROOT) / f"{trajectory.song_key}_demo{trajectory.demo_id}_{config.mode}"
    summary = simulate_rp1m_rollout(trajectory, config, output_dir)
    print(f"summary_path={output_dir / 'summary.json'}")
    print(f"rollout_npz={summary.get('rollout_npz')}")
    print(f"video_path={summary.get('video_path')}")
    print(f"recorded_goal_f1={(summary.get('recorded_reference_against_goals') or {}).get('key_f1')}")
    print(f"rollout_goal_f1={(summary.get('against_goals') or {}).get('key_f1')}")
    print(f"rollout_reference_f1={(summary.get('against_reference_piano_states') or {}).get('key_f1')}")
    print(f"piano_state_policy={summary.get('piano_state_policy')}")


def _resolve_examples(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.example:
        return list(args.example)
    selected = find_high_f1_examples(
        args.rp1m_root,
        max_songs=args.scan_songs,
        examples=args.examples,
        min_recorded_f1=args.min_recorded_f1,
        threshold=args.threshold,
    )
    return [(song_key, demo_id) for song_key, demo_id, _score in selected]


def _run_validate(args: argparse.Namespace) -> None:
    base_config = _config_from_args(args)
    examples = _resolve_examples(args)
    if not examples:
        raise RuntimeError("No validation examples selected.")
    modes = [base_config.mode]
    if args.also_action and "action" not in modes:
        modes.append("action")
    output_root = Path(args.output_dir or DEFAULT_OUTPUT_ROOT) / "validation"
    rows = []
    summaries = []
    for index, (song_key, demo_id) in enumerate(examples):
        trajectory = load_rp1m_trajectory(args.rp1m_root, song_key, demo_id, include_reference_piano_states=True)
        for mode in modes:
            render_this = bool(args.render_mp4 and index < args.render_first)
            config = replace(base_config, mode=mode, render_mp4=render_this)
            safe_song = song_key.replace("/", "_")
            run_dir = output_root / f"{safe_song}_demo{demo_id}_{mode}"
            summary = simulate_rp1m_rollout(trajectory, config, run_dir)
            summary_path = run_dir / "summary.json"
            rows.append(row_from_summary(summary, summary_path))
            summaries.append(summary)
            print(
                f"{song_key}:{demo_id} {mode} "
                f"recorded={(summary.get('recorded_reference_against_goals') or {}).get('key_f1')} "
                f"goal={(summary.get('against_goals') or {}).get('key_f1')} "
                f"reference={(summary.get('against_reference_piano_states') or {}).get('key_f1')} "
                f"summary={summary_path}"
            )
    csv_path = write_validation_csv(output_root / "validation_summary.csv", rows)
    save_json(output_root / "validation_summary.json", {"rows": rows, "summaries": summaries})
    print(f"validation_csv={csv_path}")
    print(f"validation_json={output_root / 'validation_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RP1M simulator rollouts through RoboPianist without loading piano states.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout = subparsers.add_parser("rollout", help="Run one RP1M trajectory.")
    rollout.add_argument("--rp1m-root", default=DEFAULT_RP1M_ROOT)
    rollout.add_argument("--song-key", required=True)
    rollout.add_argument("--demo-id", type=int, required=True)
    rollout.add_argument("--output-dir", default=None)
    _add_rollout_config_args(rollout)
    rollout.set_defaults(func=_run_rollout)

    validate = subparsers.add_parser("validate", help="Run several RP1M examples and write aggregate metrics.")
    validate.add_argument("--rp1m-root", default=DEFAULT_RP1M_ROOT)
    validate.add_argument("--example", type=parse_example, action="append", default=[])
    validate.add_argument("--examples", type=int, default=4)
    validate.add_argument("--scan-songs", type=int, default=8)
    validate.add_argument("--min-recorded-f1", type=float, default=0.85)
    validate.add_argument("--output-dir", default=None)
    validate.add_argument("--also-action", action="store_true")
    validate.add_argument("--render-first", type=int, default=1)
    _add_rollout_config_args(validate)
    validate.set_defaults(func=_run_validate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
