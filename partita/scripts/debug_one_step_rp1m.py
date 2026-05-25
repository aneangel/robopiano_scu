#!/usr/bin/env python3
"""One-step RP1M consistency: restore state[t], apply action[t], compare to state[t+1].

Does not accumulate rollout error across time; each step starts from a fresh env.reset()
and RP1M poses at index t. See ``diagnose_rp1m_one_step_consistency`` in rollout.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PARTITA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PARTITA_ROOT.parent
SRC_ROOT = PARTITA_ROOT / "src"
for import_root in [REPO_ROOT, SRC_ROOT]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

from partita.evaluation.rollout import ACTION_MAPPINGS, ACTION_SOURCE_SCALES, diagnose_rp1m_one_step_consistency
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import load_json


_KNOWN_TASK_KWARGS: tuple[str, ...] = (
    "n_steps_lookahead",
    "n_seconds_lookahead",
    "trim_silence",
    "wrong_press_termination",
    "initial_buffer_time",
    "disable_fingering_reward",
    "disable_forearm_reward",
    "disable_colorization",
    "disable_hand_collisions",
    "energy_penalty_coef",
    "randomize_hand_positions",
    "gravity_compensation",
    "primitive_fingertip_collisions",
    "attachment_yaw",
    "change_color_on_activation",
    "hand_anchor_y_offset",
)

_KNOWN_SUITE_LOAD_KWARGS: tuple[str, ...] = (
    "stretch",
    "shift",
    "recompile_physics",
    "legacy_step",
)


def _filter_kwargs(raw: dict[str, Any] | None, allowed: tuple[str, ...]) -> dict[str, Any]:
    if not raw:
        return {}
    return {key: value for key, value in raw.items() if key in allowed and value is not None}


def _load_target_npz(data_dir: Path) -> dict[str, np.ndarray]:
    target = np.load(data_dir / "target_trajectory.npz")
    if "goals" not in target.files:
        raise RuntimeError("target_trajectory.npz must contain goals.")
    return {name: target[name] for name in target.files}


def _load_calibration(rollout_dir: Path) -> dict[str, Any] | None:
    path = rollout_dir / "calibration.json"
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RP1M one-step dynamics consistency diagnostic.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap number of (t,t+1) pairs (default: full trajectory).")
    parser.add_argument("--start-t", type=int, default=0, help="First timestep index t.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ignore-calibration",
        action="store_true",
        help="Use rollout YAML only for action_source_scale/action_mapping.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exp = experiment_name(config)
    root = output_root(config)
    data_dir = root / "data" / exp
    rollout_dir = root / "rollout" / exp
    selection = load_json(data_dir / "selection.json")
    target = _load_target_npz(data_dir)

    if "hand_joints" not in target:
        raise RuntimeError("target_trajectory.npz must include hand_joints.")

    actions = np.asarray(target["actions"], dtype=np.float32)
    goals = np.asarray(target["goals"], dtype=np.float32)
    hand_joints = np.asarray(target["hand_joints"], dtype=np.float32)
    piano_states = np.asarray(target["piano_states"], dtype=np.float32) if "piano_states" in target else None
    song_name = selection.get("target_song_name", selection.get("song_name"))
    threshold = float(selection.get("key_threshold", config.get("selection", {}).get("key_threshold", 0.5)))
    control_timestep = float(config.get("control_timestep", 0.05))
    rollout_cfg = config.get("rollout", {}) or {}

    reduced_action_space = bool(rollout_cfg.get("reduced_action_space", True))
    action_source_scale = str(rollout_cfg.get("action_source_scale", "normalized_minus_one_to_one"))
    action_mapping = str(rollout_cfg.get("action_mapping", "as_is"))
    prefer_canonical_midi = bool(rollout_cfg.get("prefer_canonical_midi", False))
    extra_task_kwargs = _filter_kwargs(rollout_cfg.get("task_kwargs"), _KNOWN_TASK_KWARGS)
    suite_load_kwargs = _filter_kwargs(rollout_cfg.get("suite_load_kwargs"), _KNOWN_SUITE_LOAD_KWARGS)

    if not args.ignore_calibration:
        calib = _load_calibration(rollout_dir)
        if calib and calib.get("best"):
            best = calib["best"]
            action_source_scale = str(best.get("action_source_scale", action_source_scale))
            action_mapping = str(best.get("action_mapping", action_mapping))
            print(f"Using calibration.json: scale={action_source_scale} mapping={action_mapping}")

    if action_source_scale not in ACTION_SOURCE_SCALES:
        raise ValueError(f"Invalid action_source_scale: {action_source_scale}")
    if action_mapping not in ACTION_MAPPINGS:
        raise ValueError(f"Invalid action_mapping: {action_mapping}")

    rollout_dir.mkdir(parents=True, exist_ok=True)
    summary = diagnose_rp1m_one_step_consistency(
        actions=actions,
        hand_joints=hand_joints,
        piano_states=piano_states,
        goals=goals,
        song_name=song_name,
        output_dir=rollout_dir,
        control_timestep=control_timestep,
        seed=args.seed,
        reduced_action_space=reduced_action_space,
        action_source_scale=action_source_scale,
        action_mapping=action_mapping,
        extra_task_kwargs=extra_task_kwargs,
        suite_load_kwargs=suite_load_kwargs,
        prefer_canonical_midi=prefer_canonical_midi,
        key_threshold=threshold,
        max_pairs=args.max_pairs,
        start_t=args.start_t,
    )

    h = summary.get("hand_qpos_l2_vs_rp1m_next") or {}
    p = summary.get("piano_next_step_key_f1_vs_rp1m") or {}
    print(
        f"Wrote {summary.get('csv_path')} and {summary.get('summary_json_path')}\n"
        f"  steps_diagnosed={summary.get('steps_diagnosed')}\n"
        f"  hand_qpos_l2 vs RP1M next: mean={h.get('mean')} median={h.get('median')} p95={h.get('p95')} max={h.get('max')}\n"
        f"  piano key F1 vs RP1M next: mean={p.get('mean')} median={p.get('median')} min={p.get('min')}"
    )


if __name__ == "__main__":
    main()
