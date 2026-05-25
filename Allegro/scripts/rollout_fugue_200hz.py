#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", REPO_ROOT / "Fugue" / "src", REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from allegro.alignment import HighFrequencyAlignmentConfig  # noqa: E402
from allegro.fugue_rollout import (  # noqa: E402
    make_allegro_rollout_config,
    rollout_loaded_policy_with_allegro,
)
from fugue.constants import DEFAULT_RP1M_ROOT  # noqa: E402
from fugue.rp1m_closed_loop import (  # noqa: E402
    ClosedLoopParams,
    load_policy,
    load_raw_episode,
    resolve_manifest_examples,
)
from rp1m_simulator.simulator import ACTION_MAPPINGS, ACTION_SOURCE_SCALES, DEFAULT_HAND_ANCHOR_Y_OFFSET  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Fugue planner-following checkpoint with Allegro 200 Hz online residual alignment."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--dataset-artifact-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--demo-id", type=int, default=None)
    parser.add_argument("--demo-song-key", default=None)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--alignment-target-mode", choices=["linear", "endpoint"], default="linear")
    parser.add_argument("--alignment-target-power", type=float, default=1.0)
    parser.add_argument(
        "--base-action-source",
        choices=["fugue", "reference", "zero", "target_qpos_sum", "target_qpos_mean"],
        default="fugue",
    )

    parser.add_argument("--chunk-execution", choices=["first", "temporal_aggregate"], default="first")
    parser.add_argument("--temporal-agg-decay", type=float, default=0.7)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--right-gain", type=float, default=1.0)
    parser.add_argument("--left-gain", type=float, default=1.0)
    parser.add_argument("--sustain-gain", type=float, default=1.0)
    parser.add_argument("--action-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--max-action-delta", type=float, default=None)
    parser.add_argument("--target-lead", type=int, default=0)

    parser.add_argument("--control-hz", type=float, default=200.0)
    parser.add_argument("--source-hz", type=float, default=20.0)
    parser.add_argument("--kp", type=float, default=0.35)
    parser.add_argument("--kd", type=float, default=0.015)
    parser.add_argument("--feedback-residual-scale", type=float, default=1.0)
    parser.add_argument("--learned-residual-scale", type=float, default=0.5)
    parser.add_argument("--residual-clip-norm", type=float, default=0.75)
    parser.add_argument("--residual-clip-per-dim", type=float, default=0.12)
    parser.add_argument("--residual-smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--max-action-delta-per-substep", type=float, default=0.18)
    parser.add_argument("--use-shadow-action-jacobian", action="store_true")
    parser.add_argument("--allow-sustain-residual", dest="zero_sustain_residual", action="store_false")
    parser.set_defaults(zero_sustain_residual=True)
    parser.add_argument("--inverse-model", action="store_true")
    parser.add_argument("--inverse-model-capacity", type=int, default=512)
    parser.add_argument("--inverse-model-warmup", type=int, default=24)
    parser.add_argument("--inverse-model-ridge", type=float, default=1e-3)
    parser.add_argument("--inverse-model-damping", type=float, default=2e-2)
    parser.add_argument("--inverse-model-refresh-every", type=int, default=1)
    parser.add_argument("--inverse-tracking-gain", type=float, default=1.0)
    parser.add_argument("--inverse-residual-scale", type=float, default=1.0)

    parser.add_argument("--disable-online-learning", dest="online_learning", action="store_false")
    parser.set_defaults(online_learning=True)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--warmup-samples", type=int, default=32)
    parser.add_argument("--update-every-substeps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps-per-update", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--max-updates-per-rollout", type=int, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--action-source-scale", choices=list(ACTION_SOURCE_SCALES), default="normalized_minus_one_to_one")
    parser.add_argument("--action-mapping", choices=list(ACTION_MAPPINGS), default="as_is")
    parser.add_argument("--full-action-space", dest="reduced_action_space", action="store_false")
    parser.set_defaults(reduced_action_space=True)
    parser.add_argument("--hand-anchor-y-offset", type=float, default=DEFAULT_HAND_ANCHOR_Y_OFFSET)
    parser.add_argument("--no-hand-anchor-y-offset", dest="hand_anchor_y_offset", action="store_const", const=None)
    parser.add_argument("--auto-hand-anchor-y-offset", dest="auto_hand_anchor_y_offset", action="store_true")
    parser.add_argument("--no-auto-hand-anchor-y-offset", dest="auto_hand_anchor_y_offset", action="store_false")
    parser.set_defaults(auto_hand_anchor_y_offset=True)
    parser.add_argument("--wrist-action-policy", choices=["hold_initial", "recorded"], default="hold_initial")
    parser.add_argument("--initial-hand-qvel-scale", type=float, default=0.5)
    parser.add_argument("--hand-resync-interval", type=int, default=None)
    parser.add_argument("--state-correction-gain", type=float, default=0.0)
    parser.add_argument("--state-correction-qvel-scale", type=float, default=1.0)
    parser.add_argument("--post-step-state-correction", action="store_true")
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


def _fugue_params_from_args(args: argparse.Namespace) -> ClosedLoopParams:
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


def _alignment_from_args(args: argparse.Namespace) -> HighFrequencyAlignmentConfig:
    return HighFrequencyAlignmentConfig(
        source_hz=float(args.source_hz),
        control_hz=float(args.control_hz),
        kp=float(args.kp),
        kd=float(args.kd),
        feedback_residual_scale=float(args.feedback_residual_scale),
        learned_residual_scale=float(args.learned_residual_scale),
        residual_clip_norm=None if args.residual_clip_norm is None else float(args.residual_clip_norm),
        residual_clip_per_dim=None if args.residual_clip_per_dim is None else float(args.residual_clip_per_dim),
        smoothing_alpha=float(args.residual_smoothing_alpha),
        max_action_delta_per_substep=(
            None if args.max_action_delta_per_substep is None else float(args.max_action_delta_per_substep)
        ),
        use_shadow_action_jacobian=bool(args.use_shadow_action_jacobian),
        zero_sustain_residual=bool(args.zero_sustain_residual),
        inverse_model_enabled=bool(args.inverse_model),
        inverse_model_capacity=int(args.inverse_model_capacity),
        inverse_model_warmup=int(args.inverse_model_warmup),
        inverse_model_ridge=float(args.inverse_model_ridge),
        inverse_model_damping=float(args.inverse_model_damping),
        inverse_model_refresh_every=int(args.inverse_model_refresh_every),
        inverse_tracking_gain=float(args.inverse_tracking_gain),
        inverse_residual_scale=float(args.inverse_residual_scale),
        online_learning=bool(args.online_learning),
        replay_capacity=int(args.replay_capacity),
        warmup_samples=int(args.warmup_samples),
        update_every_substeps=int(args.update_every_substeps),
        batch_size=int(args.batch_size),
        train_steps_per_update=int(args.train_steps_per_update),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        device=str(args.device),
        seed=int(args.seed),
        max_updates_per_rollout=args.max_updates_per_rollout,
    )


def main() -> None:
    args = build_parser().parse_args()
    alignment = _alignment_from_args(args)
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
    rollout_config = make_allegro_rollout_config(
        dataset_timestep=dt,
        alignment=alignment,
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
        wrist_action_policy=str(args.wrist_action_policy),
        initial_hand_qvel_scale=float(args.initial_hand_qvel_scale),
        hand_resync_interval=args.hand_resync_interval,
        gravity_compensation=bool(args.gravity_compensation),
        primitive_fingertip_collisions=bool(args.primitive_fingertip_collisions),
        disable_hand_collisions=bool(args.disable_hand_collisions),
    )
    summary = rollout_loaded_policy_with_allegro(
        policy=policy,
        raw_episode=raw_episode,
        song_key=str(example["song_key"]),
        demo_id=int(example["demo_id"]),
        output_dir=Path(args.output_dir),
        alignment=alignment,
        params=_fugue_params_from_args(args),
        rollout_config=rollout_config,
        environment_name=args.environment_name,
        max_steps=args.max_steps,
        alignment_target_mode=str(args.alignment_target_mode),
        base_action_source=str(args.base_action_source),
        alignment_target_power=float(args.alignment_target_power),
        state_correction_gain=float(args.state_correction_gain),
        state_correction_qvel_scale=float(args.state_correction_qvel_scale),
        post_step_state_correction=bool(args.post_step_state_correction),
    )
    goals = summary.get("against_goals") or {}
    trainer = (summary.get("alignment") or {}).get("trainer") or {}
    print(f"summary_path={summary.get('summary_path')}")
    print(f"rollout_npz={summary.get('rollout_npz')}")
    print(f"video_path={summary.get('video_path')}")
    print(f"closed_loop_score={summary.get('closed_loop_score')}")
    print(f"goal_f1={goals.get('key_f1')} precision={goals.get('key_precision')} recall={goals.get('key_recall')}")
    print(f"mispress_rate={goals.get('mispress_rate')}")
    print(f"allegro_updates={trainer.get('updates')} replay_size={trainer.get('replay_size')}")
    print(f"control_hz={summary.get('control_hz')} substeps={summary.get('substeps_per_source_step')}")


if __name__ == "__main__":
    main()
