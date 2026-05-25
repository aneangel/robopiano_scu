#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import random
from pathlib import Path
import sys
from typing import Any

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
    row_from_summary,
    write_summary_rows_csv,
)
from rp1m_simulator.simulator import ACTION_MAPPINGS, ACTION_SOURCE_SCALES, DEFAULT_HAND_ANCHOR_Y_OFFSET, save_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Fugue closed-loop rollout parameters using rp1m_simulator rollouts."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--dataset-artifact-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--demo-id", type=int, action="append", default=[])
    parser.add_argument("--demo-song-key", default=None)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--num-demos", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)

    parser.add_argument("--chunk-execution", action="append", choices=["first", "temporal_aggregate"], default=[])
    parser.add_argument("--temporal-agg-decay", type=float, action="append", default=[])
    parser.add_argument("--action-gain", type=float, action="append", default=[])
    parser.add_argument("--right-gain", type=float, action="append", default=[])
    parser.add_argument("--left-gain", type=float, action="append", default=[])
    parser.add_argument("--sustain-gain", type=float, action="append", default=[])
    parser.add_argument("--action-smoothing-alpha", type=float, action="append", default=[])
    parser.add_argument("--max-action-delta", type=float, action="append", default=[])
    parser.add_argument("--include-no-action-delta", action="store_true")
    parser.add_argument("--target-lead", type=int, action="append", default=[])

    parser.add_argument("--online-adaptation", action="store_true")
    parser.add_argument("--adapt-window", type=int, default=20)
    parser.add_argument("--adapt-interval", type=int, default=10)
    parser.add_argument("--adapt-target-recall", type=float, default=0.35)
    parser.add_argument("--adapt-max-mispress-rate", type=float, default=0.25)
    parser.add_argument("--adapt-min-precision-for-gain", type=float, default=0.45)
    parser.add_argument("--adapt-gain-step", type=float, default=0.05)
    parser.add_argument("--adapt-smoothing-step", type=float, default=0.05)
    parser.add_argument("--adapt-decay-step", type=float, default=0.05)

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
    parser.add_argument("--render-every-source-step", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    return parser


def _values(values: list[Any], default: Any) -> list[Any]:
    return values if values else [default]


def _trial_params(args: argparse.Namespace) -> list[ClosedLoopParams]:
    max_action_delta_values: list[float | None] = list(args.max_action_delta)
    if args.include_no_action_delta or not max_action_delta_values:
        max_action_delta_values = [None, *max_action_delta_values]
    combos = itertools.product(
        _values(args.chunk_execution, "first"),
        _values(args.temporal_agg_decay, 0.7),
        _values(args.action_gain, 1.0),
        _values(args.right_gain, 1.0),
        _values(args.left_gain, 1.0),
        _values(args.sustain_gain, 1.0),
        _values(args.action_smoothing_alpha, 0.0),
        max_action_delta_values,
        _values(args.target_lead, 0),
    )
    trials = [
        ClosedLoopParams(
            chunk_execution=str(chunk_execution),
            temporal_agg_decay=float(decay),
            action_gain=float(action_gain),
            right_gain=float(right_gain),
            left_gain=float(left_gain),
            sustain_gain=float(sustain_gain),
            action_smoothing_alpha=float(smoothing),
            max_action_delta=None if max_delta is None else float(max_delta),
            target_lead=int(target_lead),
        )
        for (
            chunk_execution,
            decay,
            action_gain,
            right_gain,
            left_gain,
            sustain_gain,
            smoothing,
            max_delta,
            target_lead,
        ) in combos
    ]
    if args.max_trials is not None and len(trials) > int(args.max_trials):
        rng = random.Random(int(args.random_seed))
        trials = rng.sample(trials, int(args.max_trials))
    return trials


def _examples_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.demo_id:
        examples: list[dict[str, Any]] = []
        for demo_id in args.demo_id:
            examples.extend(
                resolve_manifest_examples(
                    dataset_artifact_root=args.dataset_artifact_root,
                    split=str(args.split),
                    demo_id=int(demo_id),
                    demo_song_key=args.demo_song_key,
                    demo_index=0,
                    limit=1,
                )
            )
        return examples
    return resolve_manifest_examples(
        dataset_artifact_root=args.dataset_artifact_root,
        split=str(args.split),
        demo_id=None,
        demo_song_key=args.demo_song_key,
        demo_index=int(args.demo_index),
        limit=int(args.num_demos),
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
    )


def _aggregate_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trial: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(int(row["trial_id"]), []).append(row)
    out = []
    for trial_id, trial_rows in sorted(by_trial.items()):
        first = trial_rows[0]
        scores = [_as_float(row.get("score")) for row in trial_rows]
        f1s = [_as_float(row.get("goal_f1")) for row in trial_rows]
        mispress = [_as_float(row.get("mispress_rate")) for row in trial_rows]
        out.append(
            {
                "trial_id": trial_id,
                "mean_score": sum(scores) / max(len(scores), 1),
                "mean_goal_f1": sum(f1s) / max(len(f1s), 1),
                "mean_mispress_rate": sum(mispress) / max(len(mispress), 1),
                "num_rollouts": len(trial_rows),
                "chunk_execution": first.get("chunk_execution"),
                "temporal_agg_decay": first.get("temporal_agg_decay"),
                "action_gain": first.get("action_gain"),
                "right_gain": first.get("right_gain"),
                "left_gain": first.get("left_gain"),
                "sustain_gain": first.get("sustain_gain"),
                "action_smoothing_alpha": first.get("action_smoothing_alpha"),
                "max_action_delta": first.get("max_action_delta"),
                "target_lead": first.get("target_lead"),
            }
        )
    out.sort(key=lambda row: float(row["mean_score"]), reverse=True)
    return out


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.checkpoint, device=str(args.device))
    dt = float(policy.checkpoint.get("dt", policy.stats.dt))
    examples = _examples_from_args(args)
    trial_params = _trial_params(args)
    adaptation = _adaptation_from_args(args)
    base_rollout_config = make_rollout_config(
        dataset_timestep=dt,
        seed=int(args.seed),
        threshold=float(args.threshold),
        render_mp4=False,
        render_audio=False,
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
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    print(f"trials={len(trial_params)} demos={len(examples)} backend=rp1m_simulator")
    for trial_id, params in enumerate(trial_params):
        for example in examples:
            cache_key = (str(example["song_key"]), int(example["demo_id"]))
            if cache_key not in raw_cache:
                raw_cache[cache_key] = load_raw_episode(
                    rp1m_root=args.rp1m_root,
                    song_key=cache_key[0],
                    demo_id=cache_key[1],
                    dt=dt,
                )
            run_dir = output_root / f"trial_{trial_id:04d}" / f"demo_{cache_key[1]}"
            summary = rollout_loaded_policy_with_rp1m_simulator(
                policy=policy,
                raw_episode=raw_cache[cache_key],
                song_key=cache_key[0],
                demo_id=cache_key[1],
                output_dir=run_dir,
                params=params,
                adaptation=adaptation,
                rollout_config=base_rollout_config,
                environment_name=args.environment_name,
                max_steps=args.max_steps,
            )
            row = {"trial_id": trial_id, **row_from_summary(summary)}
            rows.append(row)
            summaries.append(summary)
            print(
                f"trial={trial_id:04d} demo={cache_key[1]} "
                f"score={row.get('score')} f1={row.get('goal_f1')} mispress={row.get('mispress_rate')} "
                f"summary={row.get('summary_path')}"
            )
    rows_csv = write_summary_rows_csv(output_root / "closed_loop_sweep_rows.csv", rows)
    aggregate_rows = _aggregate_trial_rows(rows)
    aggregate_csv = write_summary_rows_csv(output_root / "closed_loop_sweep_trials.csv", aggregate_rows)
    payload = {
        "checkpoint": str(args.checkpoint),
        "backend": "rp1m_simulator",
        "examples": examples,
        "adaptation": asdict(adaptation),
        "trial_params": [asdict(params) for params in trial_params],
        "rows_csv": str(rows_csv),
        "aggregate_csv": str(aggregate_csv),
        "best_trial": aggregate_rows[0] if aggregate_rows else None,
        "summaries": summaries,
    }
    json_path = save_json(output_root / "closed_loop_sweep_summary.json", payload)
    print(f"rows_csv={rows_csv}")
    print(f"trials_csv={aggregate_csv}")
    print(f"summary_json={json_path}")
    if aggregate_rows:
        print(f"best_trial={aggregate_rows[0]}")


if __name__ == "__main__":
    main()
