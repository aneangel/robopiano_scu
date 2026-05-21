#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/WAVE/projects/ECEN-524-Wi26/robopiano")
for _path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from impromptu.active_window import crop_active_window  # noqa: E402
from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.evaluation import evaluate_trajectory_payload  # noqa: E402
from impromptu.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
from render_hand_state_playback import render_dense_playback  # noqa: E402
from sweep_common import (  # noqa: E402
    ASSIGNMENT_DEFAULTS,
    DEFAULT_CODE_ROOT,
    DEFAULT_FIXED_ARGS,
    DEFAULT_SWEEP_ROOT,
    MAGNET_DEFAULTS,
    PARAMETER_GRID,
    SWEEP_KEYS,
)


DEFAULT_MAESTRO_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0")
DEFAULT_MAESTRO_CSV = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0.csv")
DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/maestro_refinement")

PARAMETER_BOUNDS = {key: (min(values), max(values)) for key, values in PARAMETER_GRID.items()}
PARAMETER_SPAN = {key: PARAMETER_BOUNDS[key][1] - PARAMETER_BOUNDS[key][0] for key in PARAMETER_GRID}


@dataclass
class SongSpec:
    composer: str
    title: str
    split: str
    year: int
    midi_path: Path
    duration_s: float

    @property
    def slug(self) -> str:
        stem = f"{self.composer}_{self.title}_{self.year}_{self.midi_path.stem}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")[:180]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a precision-first adaptive Impromptu refinement sweep on MAESTRO.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--maestro-root", default=str(DEFAULT_MAESTRO_ROOT))
    parser.add_argument("--maestro-csv", default=str(DEFAULT_MAESTRO_CSV))
    parser.add_argument("--song-count", type=int, default=10)
    parser.add_argument("--song-seed", type=int, default=52426)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--top-seed-count", type=int, default=3)
    parser.add_argument("--sampler-seed", type=int, default=1518)
    parser.add_argument("--environment-name", default=DEFAULT_FIXED_ARGS["environment_name"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _coerce_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _historical_seed_objective(row: dict[str, Any]) -> float:
    precision = _coerce_float(row.get("frame_precision"))
    f1 = _coerce_float(row.get("frame_f1"))
    recall = _coerce_float(row.get("frame_recall"))
    fp = _coerce_float(row.get("FP"))
    tp = _coerce_float(row.get("TP"))
    matched = _coerce_float(row.get("matched_press_events"))
    timing_p95 = _coerce_float(row.get("timing_abs_error_p95_s"))
    score = 0.0
    score += 9.0 * precision
    score += 3.0 * f1
    score += 1.5 * recall
    score += 0.0015 * tp
    score += 0.05 * matched
    score -= 0.0010 * fp
    score -= 0.4 * timing_p95
    if precision < 0.58:
        score -= 4.0 * (0.58 - precision)
    return float(score)


def _maestro_song_manifest_row(song: SongSpec) -> dict[str, Any]:
    return {
        "composer": song.composer,
        "title": song.title,
        "split": song.split,
        "year": song.year,
        "duration_s": song.duration_s,
        "midi_path": str(song.midi_path),
        "slug": song.slug,
    }


def _load_seed_configs(base_root: Path, top_seed_count: int, code_root: Path) -> list[dict[str, Any]]:
    csv_path = base_root / "sweep_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing prior sweep results: {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cooked: dict[str, Any] = dict(row)
            for key, value in list(cooked.items()):
                try:
                    cooked[key] = float(value)
                except Exception:
                    cooked[key] = value
            cooked["historical_seed_objective"] = _historical_seed_objective(cooked)
            rows.append(cooked)
    rows.sort(key=lambda row: float(row["historical_seed_objective"]), reverse=True)
    seeds: list[dict[str, Any]] = []
    for index, row in enumerate(rows[: max(top_seed_count, 1)]):
        config = {
            **DEFAULT_FIXED_ARGS,
            **MAGNET_DEFAULTS,
            **ASSIGNMENT_DEFAULTS,
            "code_root": str(code_root),
            "environment_name": DEFAULT_FIXED_ARGS["environment_name"],
        }
        for key in SWEEP_KEYS:
            config[key] = float(row[key])
        config["seed_source_run_name"] = str(row["run_name"])
        config["seed_source_rank"] = index
        config["seed_source_precision"] = _coerce_float(row.get("frame_precision"))
        config["seed_source_f1"] = _coerce_float(row.get("frame_f1"))
        config["seed_source_recall"] = _coerce_float(row.get("frame_recall"))
        seeds.append(config)
    return seeds


def _sample_maestro_songs(maestro_root: Path, maestro_csv: Path, song_count: int, song_seed: int) -> list[SongSpec]:
    rows: list[SongSpec] = []
    with maestro_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            midi_path = maestro_root / row["midi_filename"]
            if not midi_path.is_file():
                continue
            rows.append(
                SongSpec(
                    composer=str(row["canonical_composer"]),
                    title=str(row["canonical_title"]),
                    split=str(row["split"]),
                    year=int(row["year"]),
                    midi_path=midi_path.resolve(),
                    duration_s=float(row["duration"]),
                )
            )
    if len(rows) < song_count:
        raise RuntimeError(f"Requested {song_count} songs but only found {len(rows)} MAESTRO entries")
    rng = random.Random(song_seed)
    selected = rng.sample(rows, k=song_count)
    selected.sort(key=lambda song: (song.year, song.composer, song.title, song.midi_path.name))
    return selected


def _config_signature(config: dict[str, Any]) -> str:
    payload = json.dumps({key: round(float(config[key]), 8) for key in SWEEP_KEYS}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _named_config(config: dict[str, Any], round_index: int, config_index: int) -> dict[str, Any]:
    named = dict(config)
    signature = _config_signature(named)[:10]
    named["run_name"] = f"round{round_index:02d}_cfg{config_index:02d}_{signature}"
    named["round_index"] = round_index
    named["config_index"] = config_index
    named["config_signature"] = signature
    return named


def _plan_config_from_payload(payload: dict[str, Any], run_root: Path, environment_name: str) -> ImpromptuConfig:
    return ImpromptuConfig(
        control_timestep=float(payload["control_timestep"]),
        threshold=float(payload["threshold"]),
        seed=int(payload["seed"]),
        environment_name=environment_name,
        interpolation_substeps=int(payload["interpolation_substeps"]),
        clearance_height=float(payload["clearance_height"]),
        key_press_depth=float(payload["key_press_depth"]),
        inactive_clearance_weight=float(payload["inactive_clearance_weight"]),
        hover_weight=float(payload["hover_weight"]),
        release_end_weight=float(payload["release_end_weight"]),
        travel_weight=float(payload["travel_weight"]),
        press_weight=float(payload["active_press_weight"]),
        press_lead_s=float(payload["press_lead_s"]),
        wrong_key_avoid_weight=float(payload["wrong_key_avoid_weight"]),
        anchor_stride=int(payload["anchor_stride"]),
        solve_contact_window_only=False,
        include_midpoint_anchors=True,
        ik_max_nfev=int(payload["ik_max_nfev"]),
        output_root=str(run_root.parent),
        assignment_distance_weight=float(payload["distance_weight"]),
        same_finger_bonus=float(payload["same_finger_bonus"]),
        reassignment_penalty=float(payload["reassignment_penalty"]),
        finger_crossing_penalty=float(payload["finger_crossing_penalty"]),
        wrong_hand_penalty=float(payload["wrong_hand_penalty"]),
        large_jump_penalty=float(payload["large_jump_penalty"]),
        same_key_same_finger_bonus=float(payload["same_key_same_finger_bonus"]),
    )


def _evaluate_config_on_song(
    *,
    config: dict[str, Any],
    song: SongSpec,
    run_root: Path,
    environment_name: str,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    render_dir = run_root / "render_dense"
    payload = dict(config)
    payload["midi_path"] = str(song.midi_path)
    payload["environment_name"] = environment_name
    payload["song_slug"] = song.slug
    payload["song_title"] = song.title
    payload["song_composer"] = song.composer
    atomic_save_json(run_root / "config.json", payload)
    try:
        target_keys, midi_meta = load_target_keys_from_midi(
            str(song.midi_path),
            control_timestep=float(payload["control_timestep"]),
        )
        crop = crop_active_window(
            target_keys,
            dt=float(payload["control_timestep"]),
            threshold=float(payload["threshold"]),
            active_window_last_s=float(payload["active_window_last_s"]),
            active_window_preroll_s=float(payload["active_window_preroll_s"]),
            active_window_postroll_s=float(payload["active_window_postroll_s"]),
        )
        plan = plan_target_keys(
            crop.target_keys,
            config=_plan_config_from_payload(payload, run_root=run_root, environment_name=environment_name),
        )
        npz_payload = plan.npz_payload()
        npz_payload.update(
            active_window_crop_start_frame=crop.start_frame,
            active_window_crop_end_frame=crop.end_frame,
            active_window_original_steps=crop.metadata["original_steps"],
            active_window_cropped_steps=crop.metadata["cropped_steps"],
        )
        trajectory_path = run_root / "trajectory.npz"
        atomic_save_npz(trajectory_path, **npz_payload)
        metadata = {
            "run_dir": str(run_root),
            "trajectory_npz": str(trajectory_path),
            "source_type": "midi",
            "midi_path": str(song.midi_path),
            "midi": midi_meta,
            "song": _maestro_song_manifest_row(song),
            "active_window": crop.metadata,
            **plan.metadata,
        }
        atomic_save_json(run_root / "metadata.json", metadata)
        atomic_save_json(run_root / "active_window_summary.json", crop.metadata)

        metrics = evaluate_trajectory_payload(npz_payload)
        atomic_save_json(run_root / "metrics.json", metrics)
        render_summary = render_dense_playback(
            trajectory_npz=trajectory_path,
            output_dir=render_dir,
            environment_name=environment_name,
            control_timestep=float(payload["control_timestep"]),
            interpolation_substeps=int(payload["interpolation_substeps"]),
            fps=int(payload["fps"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            seed=int(payload["seed"]),
            threshold=float(payload["threshold"]),
            active_window_last_s=None,
        )
        for name in (
            "render_summary.json",
            "score.json",
            "lag_sweep.json",
            "threshold_sweep.json",
            "fp_by_key.csv",
            "fn_by_key.csv",
        ):
            src = render_dir / name
            if src.is_file():
                shutil.copy2(src, run_root / name)
        score = dict(render_summary["score"])
        result = {
            "success": True,
            "song": _maestro_song_manifest_row(song),
            "render_dir": str(render_dir),
            "metrics": metrics,
            "score": score,
            "run_dir": str(run_root),
        }
        atomic_save_json(run_root / "song_result.json", result)
        return result
    except Exception as exc:
        failure = {
            "success": False,
            "song": _maestro_song_manifest_row(song),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_root),
        }
        atomic_save_json(run_root / "failure.json", failure)
        return failure


def _bounded_score(value: float, good_max: float) -> float:
    if good_max <= 0.0:
        return 0.0
    return float(np.clip(1.0 - (value / good_max), 0.0, 1.0))


def _planner_native_objective(summary: dict[str, Any]) -> float:
    mean_assignment = _coerce_float(summary.get("mean_assignment_rate"))
    min_assignment = _coerce_float(summary.get("min_assignment_rate"))
    mean_waypoint_success = _coerce_float(summary.get("mean_waypoint_success_rate_010m"))
    min_waypoint_success = _coerce_float(summary.get("min_waypoint_success_rate_010m"))
    mean_dense_activity = _coerce_float(summary.get("mean_waypoint_dense_activity_rate"))
    min_dense_activity = _coerce_float(summary.get("min_waypoint_dense_activity_rate"))
    mean_exact_anchor = _coerce_float(summary.get("mean_waypoint_has_exact_anchor_rate"))
    min_exact_anchor = _coerce_float(summary.get("min_waypoint_has_exact_anchor_rate"))
    mean_anchor_success = _coerce_float(summary.get("mean_ik_anchor_success_rate"))
    mean_waypoint_error = _coerce_float(summary.get("mean_waypoint_fingertip_error_p95"))
    mean_anchor_error = _coerce_float(summary.get("mean_ik_anchor_fingertip_distance_p95"))
    mean_joint_velocity = _coerce_float(summary.get("mean_joint_velocity_p95"))
    mean_joint_acceleration = _coerce_float(summary.get("mean_joint_acceleration_p95"))
    mean_joint_jerk = _coerce_float(summary.get("mean_joint_jerk_p95"))
    success_rate = _coerce_float(summary.get("song_success_rate"))

    waypoint_error_score = _bounded_score(mean_waypoint_error, 0.10)
    anchor_error_score = _bounded_score(mean_anchor_error, 0.10)
    joint_velocity_score = _bounded_score(mean_joint_velocity, 5.0)
    joint_acceleration_score = _bounded_score(mean_joint_acceleration, 100.0)
    joint_jerk_score = _bounded_score(mean_joint_jerk, 2000.0)

    score = 0.0
    score += 8.0 * mean_assignment
    score += 2.0 * min_assignment
    score += 9.0 * mean_waypoint_success
    score += 3.0 * min_waypoint_success
    score += 6.0 * mean_dense_activity
    score += 2.0 * min_dense_activity
    score += 6.0 * mean_exact_anchor
    score += 2.0 * min_exact_anchor
    score += 4.0 * mean_anchor_success
    score += 5.0 * waypoint_error_score
    score += 3.0 * anchor_error_score
    score += 1.5 * joint_velocity_score
    score += 1.0 * joint_acceleration_score
    score += 0.5 * joint_jerk_score
    score -= 5.0 * (1.0 - success_rate)
    if mean_assignment < 0.95:
        score -= 8.0 * (0.95 - mean_assignment)
    if mean_waypoint_success < 0.85:
        score -= 10.0 * (0.85 - mean_waypoint_success)
    if mean_dense_activity < 0.95:
        score -= 6.0 * (0.95 - mean_dense_activity)
    if mean_exact_anchor < 0.75:
        score -= 5.0 * (0.75 - mean_exact_anchor)
    return float(score)


def _aggregate_song_results(config: dict[str, Any], song_results: list[dict[str, Any]]) -> dict[str, Any]:
    success_results = [row for row in song_results if row.get("success")]
    aggregate: dict[str, Any] = {
        "run_name": config["run_name"],
        "round_index": int(config["round_index"]),
        "config_index": int(config["config_index"]),
        "config_signature": config["config_signature"],
        "song_count": len(song_results),
        "song_success_count": len(success_results),
        "song_success_rate": float(len(success_results) / max(len(song_results), 1)),
    }
    score_metric_map = {
        "frame_precision": "mean_frame_precision",
        "frame_recall": "mean_frame_recall",
        "frame_f1": "mean_frame_f1",
        "frame_false_positives": "mean_FP",
        "frame_true_positives": "mean_TP",
        "frame_false_negatives": "mean_FN",
        "matched_press_events": "mean_matched_press_events",
        "mispresses": "mean_mispresses",
        "missed_key_presses": "mean_missed_key_presses",
        "timing_abs_error_p95_s": "mean_timing_abs_error_p95_s",
    }
    planner_metric_map = {
        "assignment_rate": "mean_assignment_rate",
        "waypoint_fingertip_error_p95": "mean_waypoint_fingertip_error_p95",
        "waypoint_success_rate_010m": "mean_waypoint_success_rate_010m",
        "waypoint_dense_activity_rate": "mean_waypoint_dense_activity_rate",
        "waypoint_has_exact_anchor_rate": "mean_waypoint_has_exact_anchor_rate",
        "ik_anchor_success_rate": "mean_ik_anchor_success_rate",
        "ik_anchor_fingertip_distance_p95": "mean_ik_anchor_fingertip_distance_p95",
        "joint_velocity_p95": "mean_joint_velocity_p95",
        "joint_acceleration_p95": "mean_joint_acceleration_p95",
        "joint_jerk_p95": "mean_joint_jerk_p95",
    }
    raw_values: dict[str, list[float]] = {
        **{target: [] for target in score_metric_map.values()},
        **{target: [] for target in planner_metric_map.values()},
    }
    raw_precisions: list[float] = []
    raw_assignment_rates: list[float] = []
    raw_waypoint_success_rates: list[float] = []
    raw_dense_activity_rates: list[float] = []
    raw_exact_anchor_rates: list[float] = []
    for row in success_results:
        score = row["score"]
        metrics = row["metrics"]
        for source_key, target_key in score_metric_map.items():
            value = score.get(source_key)
            if value is not None:
                raw_values[target_key].append(float(value))
        for source_key, target_key in planner_metric_map.items():
            value = metrics.get(source_key)
            if value is not None:
                raw_values[target_key].append(float(value))
        precision = score.get("frame_precision")
        if precision is not None:
            raw_precisions.append(float(precision))
        assignment_rate = metrics.get("assignment_rate")
        if assignment_rate is not None:
            raw_assignment_rates.append(float(assignment_rate))
        waypoint_success_rate = metrics.get("waypoint_success_rate_010m")
        if waypoint_success_rate is not None:
            raw_waypoint_success_rates.append(float(waypoint_success_rate))
        dense_activity_rate = metrics.get("waypoint_dense_activity_rate")
        if dense_activity_rate is not None:
            raw_dense_activity_rates.append(float(dense_activity_rate))
        exact_anchor_rate = metrics.get("waypoint_has_exact_anchor_rate")
        if exact_anchor_rate is not None:
            raw_exact_anchor_rates.append(float(exact_anchor_rate))
    for target_key, values in raw_values.items():
        aggregate[target_key] = float(np.mean(values)) if values else 0.0
    aggregate["min_frame_precision"] = float(min(raw_precisions)) if raw_precisions else 0.0
    aggregate["min_assignment_rate"] = float(min(raw_assignment_rates)) if raw_assignment_rates else 0.0
    aggregate["min_waypoint_success_rate_010m"] = float(min(raw_waypoint_success_rates)) if raw_waypoint_success_rates else 0.0
    aggregate["min_waypoint_dense_activity_rate"] = float(min(raw_dense_activity_rates)) if raw_dense_activity_rates else 0.0
    aggregate["min_waypoint_has_exact_anchor_rate"] = float(min(raw_exact_anchor_rates)) if raw_exact_anchor_rates else 0.0
    for key in SWEEP_KEYS:
        aggregate[key] = float(config[key])
    aggregate["planner_native_objective"] = _planner_native_objective(aggregate)
    aggregate["song_results"] = song_results
    return aggregate


def _write_round_csv(round_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = round_dir / "round_summary.csv"
    fieldnames = [
        "run_name",
        "round_index",
        "config_index",
        "config_signature",
        "song_count",
        "song_success_count",
        "song_success_rate",
        "mean_assignment_rate",
        "min_assignment_rate",
        "mean_waypoint_fingertip_error_p95",
        "mean_waypoint_success_rate_010m",
        "min_waypoint_success_rate_010m",
        "mean_waypoint_dense_activity_rate",
        "min_waypoint_dense_activity_rate",
        "mean_waypoint_has_exact_anchor_rate",
        "min_waypoint_has_exact_anchor_rate",
        "mean_ik_anchor_success_rate",
        "mean_ik_anchor_fingertip_distance_p95",
        "mean_joint_velocity_p95",
        "mean_joint_acceleration_p95",
        "mean_joint_jerk_p95",
        "mean_frame_precision",
        "min_frame_precision",
        "mean_frame_recall",
        "mean_frame_f1",
        "mean_FP",
        "mean_TP",
        "mean_FN",
        "mean_matched_press_events",
        "mean_mispresses",
        "mean_missed_key_presses",
        "mean_timing_abs_error_p95_s",
        "planner_native_objective",
        *SWEEP_KEYS,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def _initial_sampler(seeds: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in SWEEP_KEYS:
        values = np.asarray([float(seed[key]) for seed in seeds], dtype=np.float64)
        mean[key] = float(np.mean(values))
        observed_std = float(np.std(values)) if values.size > 1 else 0.0
        std[key] = max(0.10 * PARAMETER_SPAN[key], observed_std, 1e-6)
    return mean, std


def _sample_config(mean: dict[str, float], std: dict[str, float], rng: random.Random) -> dict[str, Any]:
    config = {**DEFAULT_FIXED_ARGS, **MAGNET_DEFAULTS, **ASSIGNMENT_DEFAULTS}
    for key in SWEEP_KEYS:
        lower, upper = PARAMETER_BOUNDS[key]
        sample = rng.gauss(mean[key], std[key])
        config[key] = float(min(max(sample, lower), upper))
    return config


def _global_elites(history: list[dict[str, Any]], elite_count: int) -> list[dict[str, Any]]:
    ordered = sorted(history, key=lambda row: float(row["planner_native_objective"]), reverse=True)
    return ordered[: max(elite_count, 1)]


def _update_sampler(
    *,
    history: list[dict[str, Any]],
    elite_count: int,
    previous_best: float,
    current_best: float,
) -> tuple[dict[str, float], dict[str, float]]:
    elites = _global_elites(history, elite_count)
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    improved = current_best > previous_best + 1e-6
    spread_scale = 0.90 if improved else 1.20
    for key in SWEEP_KEYS:
        values = np.asarray([float(row[key]) for row in elites], dtype=np.float64)
        mean[key] = float(np.mean(values))
        elite_std = float(np.std(values)) if values.size > 1 else 0.0
        min_std = 0.03 * PARAMETER_SPAN[key]
        max_std = 0.35 * PARAMETER_SPAN[key]
        candidate = max(elite_std * spread_scale, min_std)
        std[key] = float(min(max(candidate, min_std), max_std))
    return mean, std


def _build_round_population(
    *,
    round_index: int,
    seeds: list[dict[str, Any]],
    history: list[dict[str, Any]],
    population_size: int,
    sampler_mean: dict[str, float],
    sampler_std: dict[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(config: dict[str, Any]) -> None:
        digest = _config_signature(config)
        if digest in seen:
            return
        seen.add(digest)
        configs.append(config)

    if round_index == 0:
        for seed in seeds[:population_size]:
            push(seed)
    else:
        best_global = max(history, key=lambda row: float(row["planner_native_objective"]))
        carry = {key: float(best_global[key]) for key in SWEEP_KEYS}
        carry.update(DEFAULT_FIXED_ARGS)
        carry.update(MAGNET_DEFAULTS)
        carry.update(ASSIGNMENT_DEFAULTS)
        for key in SWEEP_KEYS:
            carry[key] = float(best_global[key])
        push(carry)

    attempts = 0
    while len(configs) < population_size:
        candidate = _sample_config(sampler_mean, sampler_std, rng)
        push(candidate)
        attempts += 1
        if attempts > population_size * 50:
            raise RuntimeError("Could not sample a unique round population")
    return [_named_config(config, round_index=round_index, config_index=index) for index, config in enumerate(configs)]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    base_root = Path(args.base_root).expanduser().resolve()
    code_root = Path(args.code_root).expanduser().resolve()
    maestro_root = Path(args.maestro_root).expanduser().resolve()
    maestro_csv = Path(args.maestro_csv).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = _load_seed_configs(base_root=base_root, top_seed_count=int(args.top_seed_count), code_root=code_root)
    songs = _sample_maestro_songs(
        maestro_root=maestro_root,
        maestro_csv=maestro_csv,
        song_count=int(args.song_count),
        song_seed=int(args.song_seed),
    )
    manifest = {
        "output_root": str(output_root),
        "base_root": str(base_root),
        "code_root": str(code_root),
        "maestro_root": str(maestro_root),
        "maestro_csv": str(maestro_csv),
        "environment_name": str(args.environment_name),
        "song_seed": int(args.song_seed),
        "sampler_seed": int(args.sampler_seed),
        "rounds": int(args.rounds),
        "population_size": int(args.population_size),
        "elite_count": int(args.elite_count),
        "top_seed_count": int(args.top_seed_count),
        "songs": [_maestro_song_manifest_row(song) for song in songs],
        "seed_configs": [
            {
                "seed_source_run_name": seed.get("seed_source_run_name"),
                "seed_source_rank": seed.get("seed_source_rank"),
                "seed_source_precision": seed.get("seed_source_precision"),
                "seed_source_f1": seed.get("seed_source_f1"),
                "seed_source_recall": seed.get("seed_source_recall"),
                **{key: seed[key] for key in SWEEP_KEYS},
            }
            for seed in seeds
        ],
    }
    atomic_save_json(output_root / "manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    rng = random.Random(int(args.sampler_seed))
    sampler_mean, sampler_std = _initial_sampler(seeds)
    history: list[dict[str, Any]] = []
    best_objective = -math.inf

    for round_index in range(int(args.rounds)):
        round_dir = output_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        population = _build_round_population(
            round_index=round_index,
            seeds=seeds,
            history=history,
            population_size=int(args.population_size),
            sampler_mean=sampler_mean,
            sampler_std=sampler_std,
            rng=rng,
        )
        atomic_save_json(round_dir / "proposed_configs.json", {"configs": population})

        round_rows: list[dict[str, Any]] = []
        for config in population:
            config_dir = round_dir / config["run_name"]
            song_results: list[dict[str, Any]] = []
            for song in songs:
                song_dir = config_dir / "songs" / song.slug
                song_result = _evaluate_config_on_song(
                    config=config,
                    song=song,
                    run_root=song_dir,
                    environment_name=str(args.environment_name),
                )
                song_results.append(song_result)
            aggregate = _aggregate_song_results(config, song_results)
            round_rows.append(aggregate)
            atomic_save_json(config_dir / "aggregate_result.json", aggregate)

        round_rows.sort(key=lambda row: float(row["planner_native_objective"]), reverse=True)
        _write_round_csv(round_dir, round_rows)
        atomic_save_json(round_dir / "round_summary.json", {"results": round_rows})
        history.extend(round_rows)
        current_best = float(max(history, key=lambda row: float(row["planner_native_objective"]))["planner_native_objective"])
        sampler_mean, sampler_std = _update_sampler(
            history=history,
            elite_count=int(args.elite_count),
            previous_best=best_objective,
            current_best=current_best,
        )
        best_objective = current_best
        atomic_save_json(
            round_dir / "sampler_state.json",
            {
                "sampler_mean": sampler_mean,
                "sampler_std": sampler_std,
                "best_objective": best_objective,
            },
        )

    history.sort(key=lambda row: float(row["planner_native_objective"]), reverse=True)
    atomic_save_json(output_root / "final_summary.json", {"results": history})
    with (output_root / "final_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "run_name",
            "round_index",
            "config_index",
            "song_count",
            "song_success_count",
            "song_success_rate",
            "mean_assignment_rate",
            "min_assignment_rate",
            "mean_waypoint_fingertip_error_p95",
            "mean_waypoint_success_rate_010m",
            "min_waypoint_success_rate_010m",
            "mean_waypoint_dense_activity_rate",
            "min_waypoint_dense_activity_rate",
            "mean_waypoint_has_exact_anchor_rate",
            "min_waypoint_has_exact_anchor_rate",
            "mean_ik_anchor_success_rate",
            "mean_ik_anchor_fingertip_distance_p95",
            "mean_joint_velocity_p95",
            "mean_joint_acceleration_p95",
            "mean_joint_jerk_p95",
            "mean_frame_precision",
            "min_frame_precision",
            "mean_frame_recall",
            "mean_frame_f1",
            "mean_FP",
            "mean_TP",
            "mean_FN",
            "mean_matched_press_events",
            "mean_mispresses",
            "mean_missed_key_presses",
            "mean_timing_abs_error_p95_s",
            "planner_native_objective",
            *SWEEP_KEYS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in history])
    print(json.dumps({"output_root": str(output_root), "best_result": history[0]}, indent=2))


if __name__ == "__main__":
    main()
