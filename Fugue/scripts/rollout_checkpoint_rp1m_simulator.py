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
from fugue.rp1m_closed_loop import (  # noqa: E402
    ClosedLoopParams,
    OnlineAdaptationConfig,
    load_policy,
    load_raw_episode,
    make_rollout_config,
    resolve_manifest_examples,
    rollout_loaded_policy_with_rp1m_simulator,
)
from rp1m_simulator.simulator import ACTION_MAPPINGS, ACTION_SOURCE_SCALES, DEFAULT_HAND_ANCHOR_Y_OFFSET  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Fugue checkpoint in closed loop through the shared rp1m_simulator backend."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--dataset-artifact-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--demo-id", type=int, default=None)
    parser.add_argument("--demo-song-key", default=None)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--chunk-execution", choices=["first", "temporal_aggregate"], default="first")
    parser.add_argument("--temporal-agg-decay", type=float, default=0.7)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--right-gain", type=float, default=1.0)
    parser.add_argument("--left-gain", type=float, default=1.0)
    parser.add_argument("--sustain-gain", type=float, default=1.0)
    parser.add_argument("--action-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--max-action-delta", type=float, default=None)
    parser.add_argument("--target-lead", type=int, default=0)

    parser.add_argument("--online-adaptation", action="store_true")
    parser.add_argument("--adapt-window", type=int, default=20)
    parser.add_argument("--adapt-interval", type=int, default=10)
    parser.add_argument("--adapt-target-recall", type=float, default=0.35)
    parser.add_argument("--adapt-max-mispress-rate", type=float, default=0.25)
    parser.add_argument("--adapt-min-precision-for-gain", type=float, default=0.45)
    parser.add_argument("--adapt-gain-step", type=float, default=0.05)
    parser.add_argument("--adapt-smoothing-step", type=float, default=0.05)
    parser.add_argument("--adapt-decay-step", type=float, default=0.05)
    parser.add_argument("--adapt-min-action-gain", type=float, default=0.4)
    parser.add_argument("--adapt-max-action-gain", type=float, default=1.6)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--action-source-scale", choices=list(ACTION_SOURCE_SCALES), default="normalized_minus_one_to_one")
    parser.add_argument("--action-mapping", choices=list(ACTION_MAPPINGS), default="as_is")
    parser.add_argument("--full-action-space", dest="reduced_action_space", action="store_false")
    parser.set_defaults(reduced_action_space=True)
    parser.add_argument("--hand-anchor-y-offset", type=float, default=DEFAULT_HAND_ANCHOR_Y_OFFSET)
    parser.add_argument("--no-hand-anchor-y-offset", dest="hand_anchor_y_offset", action="store_const", const=None)
    parser.add_argument("--auto-hand-anchor-y-offset", action="store_true")
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--primitive-fingertip-collisions", action="store_true")
    parser.add_argument("--disable-hand-collisions", action="store_true")

    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--render-audio", action="store_true")
    parser.add_argument("--render-every-source-step", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    return parser


def _params_from_args(args: argparse.Namespace) -> ClosedLoopParams:
    return ClosedLoopParams(
        chunk_execution=str(args.chunk_execution),
        temporal_agg_decay=float(args.temporal_agg_decay),
        action_gain=float(args.action_gain),
        right_gain=float(args.right_gain),
        left_gain=float(args.left_gain),
        sustain_gain=float(args.sustain_gain),
        action_smoothing_alpha=float(args.action_smoothing_alpha),
        max_action_delta=None if args.max_action_delta is None else float(args.max_action_delta),
        target_lead=int(args.target_lead),
    )


def _adaptation_from_args(args: argparse.Namespace) -> OnlineAdaptationConfig:
    return OnlineAdaptationConfig(
        enabled=bool(args.online_adaptation),
        window=int(args.adapt_window),
        interval=int(args.adapt_interval),
        target_recall=float(args.adapt_target_recall),
        max_mispress_rate=float(args.adapt_max_mispress_rate),
        min_precision_for_gain=float(args.adapt_min_precision_for_gain),
        gain_step=float(args.adapt_gain_step),
        smoothing_step=float(args.adapt_smoothing_step),
        decay_step=float(args.adapt_decay_step),
        min_action_gain=float(args.adapt_min_action_gain),
        max_action_gain=float(args.adapt_max_action_gain),
    )


def main() -> None:
    args = build_parser().parse_args()
    policy = load_policy(args.checkpoint, device=str(args.device))
    dt = float(policy.checkpoint.get("dt", policy.stats.dt))
    examples = resolve_manifest_examples(
        dataset_artifact_root=args.dataset_artifact_root,
        split=str(args.split),
        demo_id=args.demo_id,
        demo_song_key=args.demo_song_key,
        demo_index=int(args.demo_index),
        limit=1,
    )
    example = examples[0]
    raw_episode = load_raw_episode(
        rp1m_root=args.rp1m_root,
        song_key=str(example["song_key"]),
        demo_id=int(example["demo_id"]),
        dt=dt,
    )
    rollout_config = make_rollout_config(
        dataset_timestep=dt,
        seed=int(args.seed),
        threshold=float(args.threshold),
        render_mp4=bool(args.render_mp4),
        render_audio=bool(args.render_audio),
        render_every_source_step=max(int(args.render_every_source_step), 1),
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        action_source_scale=str(args.action_source_scale),
        action_mapping=str(args.action_mapping),
        reduced_action_space=bool(args.reduced_action_space),
        hand_anchor_y_offset=args.hand_anchor_y_offset,
        auto_hand_anchor_y_offset=bool(args.auto_hand_anchor_y_offset),
        gravity_compensation=bool(args.gravity_compensation),
        primitive_fingertip_collisions=bool(args.primitive_fingertip_collisions),
        disable_hand_collisions=bool(args.disable_hand_collisions),
    )
    summary = rollout_loaded_policy_with_rp1m_simulator(
        policy=policy,
        raw_episode=raw_episode,
        song_key=str(example["song_key"]),
        demo_id=int(example["demo_id"]),
        output_dir=Path(args.output_dir),
        params=_params_from_args(args),
        adaptation=_adaptation_from_args(args),
        rollout_config=rollout_config,
        environment_name=args.environment_name,
        max_steps=args.max_steps,
    )
    goals = summary.get("against_goals") or {}
    print(f"summary_path={summary.get('summary_path')}")
    print(f"rollout_npz={summary.get('rollout_npz')}")
    print(f"video_path={summary.get('video_path')}")
    print(f"closed_loop_score={summary.get('closed_loop_score')}")
    print(f"goal_f1={goals.get('key_f1')} precision={goals.get('key_precision')} recall={goals.get('key_recall')}")
    print(f"mispress_rate={goals.get('mispress_rate')}")
    print(f"params_final={summary.get('params_final')}")
    print(f"rollout_backend={summary.get('rollout_backend')}")


if __name__ == "__main__":
    main()
