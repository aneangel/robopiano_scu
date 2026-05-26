#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
for path in (
    REPO,
    REPO / "Intermezzo" / "src",
    REPO / "Impromptu" / "src",
    REPO / "Bagatelle" / "src",
    REPO / "Rhapsody" / "src",
    REPO / "partita" / "src",
    REPO / "Variations",
    REPO / "robopianist",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.assignment import assign_fingers_previous_pose  # noqa: E402
from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import HAND_STATE_DIM, BagatelleKinematics  # noqa: E402
from intermezzo.io import atomic_save_json  # noqa: E402
from intermezzo.keys import extract_waypoint_frames  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402


MAESTRO_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0")
DEFAULT_OUTPUT = Path("/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAccuracyRecovery")
DEFAULT_RHAPSODY = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/"
    "random10_heldout2_prev_random_scale6_refine16_20260522/rhapsody_rpik.pt"
)
ENV_NAME = "RoboPianist-debug-NocturneRousseau-v0"
CONTROL_TIMESTEP = 0.05
NUM_KEYS = 88
NUM_FINGERS = 10
PRODUCTION_TRAJECTORY_MODE = "joint_space_straighten"
HARD_WINDOW_TRAJECTORY_MODE = "dense_fingertip_ik"
RECOVERY_VARIANTS = (
    "impromptu_accuracy_baseline",
    "maestroso_bagatelle_assign_cpu_verify",
    "maestroso_bagatelle_assign_gpu_seed_cpu_verify",
    "maestroso_bagatelle_assign_gpu_seed_multistart",
    "maestroso_bagatelle_assign_gpu_seed_topk",
    "maestroso_hard_window_dense_repair",
)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_json(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", score.get("matched", 0.0)) or 0.0)
    played = float(score.get("played_press_events", score.get("played", 0.0)) or 0.0)
    target = float(score.get("target_press_events", score.get("target", 0.0)) or 0.0)
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = float(tp) / float(tp + fp) if tp + fp else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def read_maestro(maestro_root: Path, limit: int, sample_manifest: Path | None = None) -> list[dict[str, Any]]:
    if sample_manifest is not None and sample_manifest.exists():
        if sample_manifest.suffix.lower() == ".json":
            raw_rows = json.loads(sample_manifest.read_text(encoding="utf-8"))
        else:
            with sample_manifest.open("r", encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
    else:
        with (maestro_root / "maestro-v3.0.0.csv").open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    songs: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, start=1):
        midi_value = str(row.get("midi_path") or row.get("midi_filename") or "")
        midi_path = Path(midi_value)
        if not midi_path.is_absolute():
            midi_path = maestro_root / midi_value
        if not midi_path.is_file():
            continue
        songs.append(
            {
                **row,
                "run_id": str(row.get("run_id") or f"maestro_{index:04d}"),
                "manifest_index": index,
                "midi_path": str(midi_path),
            }
        )
        if len(songs) >= int(limit):
            break
    return songs


def recovery_bagatelle_config(*, ik_max_nfev: int, multistart: bool, topk: bool = False) -> BagatelleConfig:
    return BagatelleConfig(
        control_timestep=CONTROL_TIMESTEP,
        threshold=0.5,
        environment_name=ENV_NAME,
        key_press_depth=0.006,
        wrong_hand_penalty=4.0,
        wrong_hand_split_key=48,
        assignment_dynamic_hand_split=True,
        assignment_dynamic_hand_split_min_span=12,
        assignment_dynamic_hand_split_min_keys=3,
        same_key_same_finger_bonus=0.25,
        assignment_strategy="ik_aware_topk" if bool(topk) else "legacy_previous_pose",
        assignment_top_k=4 if bool(topk) else 1,
        assignment_fail_if_unassigned=True,
        ik_unassigned_fingertip_strategy="avoid_mispresses",
        ik_unassigned_fingertip_avoidance_weight=0.5,
        ik_unassigned_fingertip_avoidance_radius=0.03,
        ik_max_nfev=int(ik_max_nfev),
        residual_success_threshold=0.02,
        ik_multistart_on_failure=bool(multistart),
        ik_multistart_seed_count=4 if bool(multistart) else 2,
        ik_static_contact_validation=True,
        ik_static_contact_settle_steps=1,
        ik_static_contact_wrong_key_weight=1.0,
        ik_static_contact_missed_key_weight=4.0,
        ik_static_contact_residual_weight=10.0,
        ik_static_contact_failure_weight=25.0,
    )


def build_bagatelle_waypoint_targets(
    *,
    kin: BagatelleKinematics,
    waypoint_target_keys: np.ndarray,
    key_targets: np.ndarray,
    config: BagatelleConfig,
) -> dict[str, Any]:
    """Build proposal targets with Bagatelle's sequential previous-pose assignment.

    This replaces split-even assignment for production proposal generation.
    It intentionally walks waypoints in order, updating the previous qpos and
    previous fingertip positions after each exact Bagatelle solve.
    """

    keysets = np.asarray(waypoint_target_keys, dtype=np.float32).reshape(-1, NUM_KEYS)
    all_key_targets = np.asarray(key_targets, dtype=np.float32)
    previous_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
    neutral_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
    previous_fingertips = np.asarray(kin.fingertip_positions_for_qpos(previous_qpos), dtype=np.float32)
    previous_assignment = np.full((NUM_FINGERS,), -1, dtype=np.int32)

    fingertip_targets = np.full((keysets.shape[0], NUM_FINGERS, 3), np.nan, dtype=np.float32)
    active_masks = np.zeros((keysets.shape[0], NUM_FINGERS), dtype=np.float32)
    assignments = np.full((keysets.shape[0], NUM_FINGERS), -1, dtype=np.int32)
    previous_qpos_seeds = np.zeros((keysets.shape[0], HAND_STATE_DIM), dtype=np.float32)
    verified_qpos = np.zeros((keysets.shape[0], HAND_STATE_DIM), dtype=np.float32)
    dropped_rows: list[dict[str, Any]] = []
    repair_tier_counts: dict[str, int] = {"tier_1_cpu_exact_from_previous": 0}
    nfev_sum = 0
    max_residual = 0.0

    for index, keyset in enumerate(keysets):
        active_keys = np.flatnonzero(keyset[:NUM_KEYS] > float(config.threshold)).astype(np.int32)
        previous_qpos_seeds[index] = previous_qpos.astype(np.float32)
        contact_targets = kin.key_contact_targets(active_keys)
        assignment = assign_fingers_previous_pose(
            active_keys,
            previous_fingertips,
            contact_targets,
            config,
            previous_assignment=previous_assignment,
        )
        if assignment.count:
            assignment = replace(
                assignment,
                target_positions=all_key_targets[assignment.assigned_keys.astype(np.int64)].astype(np.float32),
            )
        dense_assignment = assignment.dense_key_by_finger()
        assignments[index] = dense_assignment.astype(np.int32)
        dense_targets = assignment.dense_targets_by_finger()
        fingertip_targets[index] = dense_targets.astype(np.float32)
        active_masks[index] = (dense_assignment >= 0).astype(np.float32)
        if assignment.unassigned_keys.size:
            dropped_rows.append(
                {
                    "waypoint_index": int(index),
                    "unassigned_keys": assignment.unassigned_keys.astype(int).tolist(),
                }
            )

        result = kin.solve_press_pose(
            assignment,
            previous_qpos,
            neutral_qpos=neutral_qpos,
            config=config,
        )
        repair_tier_counts["tier_1_cpu_exact_from_previous"] += 1
        nfev_sum += int(result.nfev)
        max_residual = max(max_residual, float(result.max_residual))
        previous_qpos = np.asarray(result.pose, dtype=np.float32)
        verified_qpos[index] = previous_qpos
        previous_fingertips = np.asarray(result.fingertip_positions, dtype=np.float32)
        previous_assignment = dense_assignment.astype(np.int32)

    dropped_key_count = int(sum(len(row["unassigned_keys"]) for row in dropped_rows))
    if dropped_key_count:
        raise RuntimeError(
            "Bagatelle waypoint assignment dropped target keys; "
            f"dropped_key_count={dropped_key_count}"
        )
    return {
        "fingertip_targets": fingertip_targets,
        "active_masks": active_masks,
        "assignments": assignments,
        "previous_qpos_seeds": previous_qpos_seeds,
        "verified_qpos": verified_qpos,
        "metadata": {
            "assignment_strategy": "bagatelle_legacy_previous_pose_sequential",
            "dropped_key_count": dropped_key_count,
            "dropped_rows": dropped_rows,
            "nfev_sum": int(nfev_sum),
            "max_residual_m": float(max_residual),
            "repair_tier_counts": repair_tier_counts,
        },
    }


def run_cmd(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    env: dict[str, str],
    timeout: int | None,
) -> float:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_path = log_dir / f"{name}.log"
    with log_path.open("ab") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} =====\n".encode())
        log.write((" ".join(command) + "\n").encode())
        log.flush()
        subprocess.run(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        seconds = time.time() - started
        log.write(f"===== finished {name} in {seconds:.1f}s =====\n".encode())
    return float(seconds)


def common_impromptu_args(
    *,
    song: dict[str, Any],
    run_parent: Path,
    run_name: str,
    max_duration_s: float | None,
    trajectory_mode: str,
    ik_max_nfev: int,
    anchor_stride: int,
) -> list[str]:
    args = [
        sys.executable,
        "Impromptu/scripts/plan_trajectory.py",
        "--midi-path",
        str(song["midi_path"]),
        "--output-root",
        str(run_parent),
        "--run-name",
        str(run_name),
        "--environment-name",
        ENV_NAME,
        "--trajectory-mode",
        str(trajectory_mode),
        "--disable-adaptive-complex-song-defaults",
        "--key-press-depth",
        "0.006",
        "--wrong-hand-penalty",
        "4.0",
        "--wrong-hand-split-key",
        "48",
        "--assignment-dynamic-hand-split",
        "--assignment-dynamic-hand-split-min-span",
        "12",
        "--assignment-dynamic-hand-split-min-keys",
        "3",
        "--same-key-same-finger-bonus",
        "0.25",
        "--ik-unassigned-fingertip-strategy",
        "avoid_mispresses",
        "--ik-unassigned-fingertip-avoidance-weight",
        "0.5",
        "--ik-unassigned-fingertip-avoidance-radius",
        "0.03",
        "--assignment-strategy",
        "legacy_previous_pose",
        "--assignment-fail-if-unassigned",
        "--anchor-stride",
        str(int(anchor_stride)),
        "--ik-max-nfev",
        str(int(ik_max_nfev)),
        "--residual-success-threshold",
        "0.02",
        "--ik-static-contact-validation",
        "--ik-static-contact-settle-steps",
        "1",
        "--ik-static-contact-missed-key-weight",
        "4.0",
        "--ik-static-contact-wrong-key-weight",
        "1.0",
    ]
    if max_duration_s is not None:
        args += ["--max-duration-s", str(float(max_duration_s))]
    return args


def variant_command(
    *,
    variant: str,
    song: dict[str, Any],
    run_parent: Path,
    max_duration_s: float | None,
    hard_window_max_duration_s: float,
    rhapsody_checkpoint: Path,
    rhapsody_device: str,
) -> tuple[list[str], dict[str, Any]]:
    run_name = str(song["run_id"])
    if variant == "impromptu_accuracy_baseline":
        args = common_impromptu_args(
            song=song,
            run_parent=run_parent,
            run_name=run_name,
            max_duration_s=max_duration_s,
            trajectory_mode=PRODUCTION_TRAJECTORY_MODE,
            ik_max_nfev=80,
            anchor_stride=2,
        )
        args += ["--disable-ik-multistart-on-failure"]
        return args, {"repair_tier": "tier_1_cpu_exact_from_previous", "uses_rhapsody": False}

    if variant == "maestroso_bagatelle_assign_cpu_verify":
        args = common_impromptu_args(
            song=song,
            run_parent=run_parent,
            run_name=run_name,
            max_duration_s=max_duration_s,
            trajectory_mode=PRODUCTION_TRAJECTORY_MODE,
            ik_max_nfev=80,
            anchor_stride=2,
        )
        return args, {"repair_tier": "tier_1_cpu_exact_from_previous", "uses_rhapsody": False}

    if variant == "maestroso_bagatelle_assign_gpu_seed_cpu_verify":
        args = common_impromptu_args(
            song=song,
            run_parent=run_parent,
            run_name=run_name,
            max_duration_s=max_duration_s,
            trajectory_mode=PRODUCTION_TRAJECTORY_MODE,
            ik_max_nfev=80,
            anchor_stride=2,
        )
        args += [
            "--disable-ik-multistart-on-failure",
            "--enable-rhapsody-ik",
            "--rhapsody-ik-checkpoint",
            str(rhapsody_checkpoint),
            "--rhapsody-ik-device",
            str(rhapsody_device),
            "--rhapsody-ik-refinement-steps",
            "0",
        ]
        return args, {"repair_tier": "tier_0_gpu_seed_tier_1_cpu_exact", "uses_rhapsody": True}

    if variant == "maestroso_bagatelle_assign_gpu_seed_multistart":
        args = common_impromptu_args(
            song=song,
            run_parent=run_parent,
            run_name=run_name,
            max_duration_s=max_duration_s,
            trajectory_mode=PRODUCTION_TRAJECTORY_MODE,
            ik_max_nfev=160,
            anchor_stride=2,
        )
        args += [
            "--ik-multistart-seed-count",
            "4",
            "--enable-rhapsody-ik",
            "--rhapsody-ik-checkpoint",
            str(rhapsody_checkpoint),
            "--rhapsody-ik-device",
            str(rhapsody_device),
            "--rhapsody-ik-refinement-steps",
            "0",
        ]
        return args, {"repair_tier": "tier_2_cpu_exact_multistart", "uses_rhapsody": True}

    if variant == "maestroso_bagatelle_assign_gpu_seed_topk":
        args = common_impromptu_args(
            song=song,
            run_parent=run_parent,
            run_name=run_name,
            max_duration_s=max_duration_s,
            trajectory_mode=PRODUCTION_TRAJECTORY_MODE,
            ik_max_nfev=160,
            anchor_stride=2,
        )
        args += [
            "--assignment-strategy",
            "ik_aware_topk",
            "--assignment-top-k",
            "4",
            "--assignment-ik-residual-weight",
            "1.0",
            "--assignment-ik-max-residual-weight",
            "10.0",
            "--assignment-motion-weight",
            "0.05",
            "--enable-rhapsody-ik",
            "--rhapsody-ik-checkpoint",
            str(rhapsody_checkpoint),
            "--rhapsody-ik-device",
            str(rhapsody_device),
            "--rhapsody-ik-refinement-steps",
            "0",
        ]
        return args, {"repair_tier": "tier_3_ik_aware_topk", "uses_rhapsody": True}

    if variant == "maestroso_hard_window_dense_repair":
        hard_duration = min(
            float(max_duration_s) if max_duration_s is not None else float(hard_window_max_duration_s),
            float(hard_window_max_duration_s),
        )
        args = common_impromptu_args(
            song=song,
            run_parent=run_parent,
            run_name=run_name,
            max_duration_s=hard_duration,
            trajectory_mode=HARD_WINDOW_TRAJECTORY_MODE,
            ik_max_nfev=160,
            anchor_stride=1,
        )
        args += ["--solve-contact-window-only", "--ik-multistart-seed-count", "4"]
        return args, {"repair_tier": "tier_4_dense_fingertip_ik_hard_window", "uses_rhapsody": False}

    raise ValueError(f"unknown recovery variant: {variant}")


def _metric_column(meta: dict[str, Any], values: np.ndarray, preferred: tuple[str, ...]) -> np.ndarray:
    columns = list(meta.get("sparse_press_ik_metric_columns") or [])
    if values.ndim != 2 or not columns:
        return np.zeros((0,), dtype=np.float32)
    for name in preferred:
        if name in columns:
            index = int(columns.index(name))
            if values.shape[1] > index:
                return values[:, index].astype(np.float32)
    return np.zeros((0,), dtype=np.float32)


def dropped_key_count_from_npz(data: Any) -> int:
    if "unassigned_keys" not in data:
        return 0
    unassigned = np.asarray(data["unassigned_keys"], dtype=np.int32)
    return int(np.count_nonzero(unassigned >= 0))


def all_waypoint_activation_validation(run_dir: Path, *, env: dict[str, str]) -> dict[str, Any]:
    started = time.time()
    trajectory_path = run_dir / "trajectory.npz"
    metadata = load_json(run_dir / "metadata.json")
    if not trajectory_path.exists():
        return {"enabled": True, "available": False, "error": "missing_trajectory"}
    old_mujoco_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    kin = None
    try:
        with np.load(trajectory_path, allow_pickle=False) as data:
            target_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :NUM_KEYS]
            hand = np.asarray(data["planned_hand_joints"], dtype=np.float32)
            waypoint_frames = np.asarray(data["waypoint_frames"], dtype=np.int64).reshape(-1)
            assignments = (
                np.asarray(data["assignments"], dtype=np.int32)
                if "assignments" in data
                else np.full((waypoint_frames.size, NUM_FINGERS), -1, dtype=np.int32)
            )
            sparse_metrics = (
                np.asarray(data["sparse_press_ik_metrics"], dtype=np.float32)
                if "sparse_press_ik_metrics" in data
                else np.zeros((0, 0), dtype=np.float32)
            )
            dropped = dropped_key_count_from_npz(data)
        frames = waypoint_frames[(waypoint_frames >= 0) & (waypoint_frames < min(target_keys.shape[0], hand.shape[0]))]
        if frames.size:
            active = np.any(target_keys[frames] > 0.5, axis=1)
            frames = frames[active]
        kin = BagatelleKinematics(
            BagatelleConfig(
                control_timestep=CONTROL_TIMESTEP,
                threshold=0.5,
                environment_name=ENV_NAME,
                key_press_depth=0.006,
            ),
            target_keys=target_keys,
            output_dir=run_dir / "all_waypoint_activation",
        )
        rows: list[dict[str, Any]] = []
        tp = fp = fn = 0
        assigned_missed_total = 0
        wrong_total = 0
        frame_to_waypoint = {int(frame): idx for idx, frame in enumerate(waypoint_frames.astype(np.int64).tolist())}
        for frame in frames.astype(np.int64):
            waypoint_index = int(frame_to_waypoint.get(int(frame), -1))
            activation = kin.activation_for_qpos(hand[int(frame)], settle_steps=1)[:NUM_KEYS]
            target_active = target_keys[int(frame)] > 0.5
            played_active = activation > 0.5
            row_tp = int(np.count_nonzero(target_active & played_active))
            row_fp = int(np.count_nonzero(~target_active & played_active))
            row_fn = int(np.count_nonzero(target_active & ~played_active))
            assigned = np.zeros((NUM_KEYS,), dtype=bool)
            if 0 <= waypoint_index < assignments.shape[0]:
                assigned_keys = assignments[waypoint_index]
                valid = assigned_keys[(assigned_keys >= 0) & (assigned_keys < NUM_KEYS)]
                assigned[valid.astype(np.int64)] = True
            else:
                assigned = target_active.copy()
            assigned_missed = int(np.count_nonzero(assigned & ~played_active))
            tp += row_tp
            fp += row_fp
            fn += row_fn
            assigned_missed_total += assigned_missed
            wrong_total += row_fp
            rows.append(
                {
                    "waypoint_index": int(waypoint_index),
                    "frame": int(frame),
                    "tp": row_tp,
                    "fp": row_fp,
                    "fn": row_fn,
                    "assigned_missed": assigned_missed,
                    "target_keys": np.flatnonzero(target_active).astype(int).tolist(),
                    "played_keys": np.flatnonzero(played_active).astype(int).tolist(),
                }
            )
        residuals = _metric_column(metadata, sparse_metrics, ("max_assigned_distance", "max_residual", "residual_norm"))
        max_residual = float(np.max(residuals)) if residuals.size else float(metadata.get("ik_max_residual_p95", 0.0) or 0.0)
        result = {
            "enabled": True,
            "available": True,
            "policy": "all_active_waypoint_frames_cpu_mujoco_activation",
            "sampled_frames": int(frames.size),
            "all_waypoint_activation_f1": f1_from_counts(tp, fp, fn),
            "all_waypoint_activation_tp": int(tp),
            "all_waypoint_activation_fp": int(fp),
            "all_waypoint_activation_fn": int(fn),
            "assigned_target_missed_count": int(assigned_missed_total),
            "wrong_key_count": int(wrong_total),
            "dropped_key_count": int(dropped),
            "max_residual_m": float(max_residual),
            "gate_passed": bool(
                assigned_missed_total == 0
                and wrong_total <= 0
                and dropped == 0
                and max_residual <= 0.02
            ),
            "gate": {
                "assigned_target_missed_count": 0,
                "wrong_key_tolerance": 0,
                "max_residual_m": 0.02,
                "dropped_key_count": 0,
            },
            "seconds": float(time.time() - started),
            "per_waypoint_json": str(run_dir / "all_waypoint_activation" / "per_waypoint_activation.json"),
        }
        save_json(
            run_dir / "all_waypoint_activation" / "per_waypoint_activation.json",
            {"rows": rows, "summary": result},
        )
        save_json(run_dir / "all_waypoint_activation_summary.json", result)
        return result
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seconds": float(time.time() - started),
        }
    finally:
        if kin is not None:
            kin.close()
        if old_mujoco_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = old_mujoco_gl


def run_retest(
    *,
    run_parent: Path,
    run_id: str,
    output_root: Path,
    log_dir: Path,
    env: dict[str, str],
    timeout: int | None,
) -> dict[str, Any] | None:
    command = [
        sys.executable,
        "retest_impromptu_rp1m_simulator.py",
        "--run-root",
        str(run_parent),
        "--output-root",
        str(output_root),
        "--only-run",
        str(run_id),
        "--environment-name",
        ENV_NAME,
        "--threshold",
        "0.5",
        "--set-hand-qvel",
    ]
    run_cmd("rp1m_retest", command, cwd=REPO, log_dir=log_dir, env=env, timeout=timeout)
    result_path = output_root / run_id / "impromptu_rp1m_retest_result.json"
    return load_json(result_path) if result_path.exists() else None


def extract_planning_metadata(run_dir: Path) -> dict[str, Any]:
    trajectory_path = run_dir / "trajectory.npz"
    metadata = load_json(run_dir / "metadata.json")
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "trajectory_npz": str(trajectory_path),
        "metadata": metadata,
        "dropped_key_count": 0,
        "accepted_raw_gpu_proposals": 0,
        "accepted_cpu_verified_waypoints": int(metadata.get("sparse_press_ik_success_count", metadata.get("ik_success_count", 0)) or 0),
        "failed_waypoints": 0,
    }
    if not trajectory_path.exists():
        return out
    with np.load(trajectory_path, allow_pickle=False) as data:
        if "target_keys" in data:
            target = np.asarray(data["target_keys"], dtype=np.float32)[:, :NUM_KEYS]
            out["target_steps"] = int(target.shape[0])
            active = target > 0.5
            starts = active & np.vstack([np.ones((1, NUM_KEYS), dtype=bool), ~active[:-1]])
            out["target_press_events"] = int(np.count_nonzero(starts)) if target.size else 0
        if "waypoint_frames" in data:
            out["waypoint_count"] = int(np.asarray(data["waypoint_frames"]).size)
        dropped = dropped_key_count_from_npz(data)
        out["dropped_key_count"] = int(dropped)
        out["failed_waypoints"] = int(dropped)
    return out


def recovery_metadata_fields(
    *,
    variant: str,
    variant_meta: dict[str, Any],
    planning: dict[str, Any],
    activation: dict[str, Any],
    chunk_overlap_seconds: float,
    stitch_blend_enabled: bool,
) -> dict[str, Any]:
    waypoint_count = int(planning.get("waypoint_count", 0) or 0)
    activation_failed = int(activation.get("assigned_target_missed_count", 0) or 0)
    residual_failed = 1 if float(activation.get("max_residual_m", 0.0) or 0.0) > 0.02 else 0
    dropped = int(planning.get("dropped_key_count", activation.get("dropped_key_count", 0)) or 0)
    uses_rhapsody = bool(variant_meta.get("uses_rhapsody"))
    tier = str(variant_meta.get("repair_tier", "tier_1_cpu_exact_from_previous"))
    repair_tier_counts = {
        "tier_0_gpu_rhapsody_seed": int(waypoint_count if uses_rhapsody else 0),
        "tier_1_cpu_exact_from_seed": int(waypoint_count if tier in {"tier_0_gpu_seed_tier_1_cpu_exact", "tier_1_cpu_exact_from_previous"} else 0),
        "tier_2_cpu_exact_multistart": int(waypoint_count if tier == "tier_2_cpu_exact_multistart" else 0),
        "tier_3_ik_aware_topk": int(waypoint_count if tier == "tier_3_ik_aware_topk" else 0),
        "tier_4_dense_fingertip_ik": int(waypoint_count if tier == "tier_4_dense_fingertip_ik_hard_window" else 0),
        "tier_5_offline_failure_rows": int(activation_failed + residual_failed + dropped > 0),
    }
    accepted_cpu = int(planning.get("accepted_cpu_verified_waypoints", 0) or 0)
    if accepted_cpu <= 0 and waypoint_count:
        accepted_cpu = max(0, waypoint_count - activation_failed - dropped)
    return {
        "dropped_key_count": int(dropped),
        "all_waypoint_activation_f1": float(activation.get("all_waypoint_activation_f1", 0.0) or 0.0),
        "all_waypoint_activation_tp": int(activation.get("all_waypoint_activation_tp", 0) or 0),
        "all_waypoint_activation_fp": int(activation.get("all_waypoint_activation_fp", 0) or 0),
        "all_waypoint_activation_fn": int(activation.get("all_waypoint_activation_fn", 0) or 0),
        "repair_tier_counts": repair_tier_counts,
        "accepted_raw_gpu_proposals": 0,
        "accepted_cpu_verified_waypoints": int(accepted_cpu),
        "failed_waypoints": int(activation_failed + dropped + residual_failed),
        "chunk_overlap_seconds": float(chunk_overlap_seconds),
        "stitch_blend_enabled": bool(stitch_blend_enabled),
        "rhapsody_role": "seed_only_cpu_verified" if uses_rhapsody else "not_used",
        "raw_gpu_proposal_acceptance_policy": "never_accept_without_exact_cpu_mujoco_validation",
        "online_rl_in_planner": False,
        "online_cem_in_planner": False,
        "failure_collection_file": "offline_training_failures.jsonl",
    }


def run_variant(
    *,
    variant: str,
    song: dict[str, Any],
    output_root: Path,
    max_duration_s: float,
    hard_window_max_duration_s: float,
    rhapsody_checkpoint: Path,
    rhapsody_device: str,
    env: dict[str, str],
    variant_timeout_s: int | None,
    rollout_timeout_s: int | None,
    chunk_overlap_seconds: float,
) -> dict[str, Any]:
    run_id = str(song["run_id"])
    variant_root = output_root / variant
    run_dir = variant_root / run_id
    log_dir = output_root / "logs" / variant / run_id
    started = time.time()
    result: dict[str, Any] = {
        "variant": variant,
        "run_id": run_id,
        "song": song,
        "ok": False,
    }
    try:
        command, variant_meta = variant_command(
            variant=variant,
            song=song,
            run_parent=variant_root,
            max_duration_s=max_duration_s,
            hard_window_max_duration_s=hard_window_max_duration_s,
            rhapsody_checkpoint=rhapsody_checkpoint,
            rhapsody_device=rhapsody_device,
        )
        plan_seconds = run_cmd("plan", command, cwd=REPO, log_dir=log_dir, env=env, timeout=variant_timeout_s)
        planning = extract_planning_metadata(run_dir)
        activation = all_waypoint_activation_validation(run_dir, env=env)
        retest = run_retest(
            run_parent=variant_root,
            run_id=run_id,
            output_root=variant_root / "rp1m_retest",
            log_dir=log_dir,
            env=env,
            timeout=rollout_timeout_s,
        )
        result.update(planning)
        result.update(
            recovery_metadata_fields(
                variant=variant,
                variant_meta=variant_meta,
                planning=planning,
                activation=activation,
                chunk_overlap_seconds=chunk_overlap_seconds,
                stitch_blend_enabled=False,
            )
        )
        result.update(
            {
                "plan_seconds": float(plan_seconds),
                "all_waypoint_activation": activation,
                "rp1m_retest": retest,
                "event_f1": None if retest is None else retest.get("event_f1"),
                "frame_f1": None if retest is None else retest.get("frame_f1"),
                "matched": None if retest is None else retest.get("matched"),
                "target": None if retest is None else retest.get("target"),
                "played": None if retest is None else retest.get("played"),
                "mispresses": None if retest is None else retest.get("mispresses"),
                "gate_passed": bool(activation.get("gate_passed", False)),
                "variant_command": command,
                "variant_policy": variant_meta,
                "ok": True,
            }
        )
    except Exception as exc:
        result.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    result["wall_seconds"] = float(time.time() - started)
    save_json(run_dir / "variant_result.json", result)
    return result


def collect_failure_manifest(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_root / "offline_training_failures.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            poor = (
                not bool(row.get("ok"))
                or not bool(row.get("gate_passed"))
                or float(row.get("event_f1") or 0.0) < 0.6
            )
            if not poor:
                continue
            handle.write(
                json.dumps(
                    {
                        "variant": row.get("variant"),
                        "run_id": row.get("run_id"),
                        "song": row.get("song"),
                        "trajectory_npz": row.get("trajectory_npz"),
                        "run_dir": row.get("run_dir"),
                        "event_f1": row.get("event_f1"),
                        "frame_f1": row.get("frame_f1"),
                        "dropped_key_count": row.get("dropped_key_count"),
                        "all_waypoint_activation_f1": row.get("all_waypoint_activation_f1"),
                        "failed_waypoints": row.get("failed_waypoints"),
                        "repair_tier_counts": row.get("repair_tier_counts"),
                        "offline_loop": "planner_run_collect_failures_train_policy_offline_freeze_checkpoint",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def summarize(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("event_f1") is not None]
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(row.get("variant")) for row in rows}):
        items = [row for row in rows if str(row.get("variant")) == variant]
        scored_items = [row for row in items if row.get("event_f1") is not None]
        by_variant[variant] = {
            "count": len(items),
            "ok_count": sum(1 for row in items if row.get("ok")),
            "gate_pass_count": sum(1 for row in items if row.get("gate_passed")),
            "mean_event_f1": float(np.mean([float(row["event_f1"]) for row in scored_items])) if scored_items else 0.0,
            "mean_frame_f1": float(np.mean([float(row["frame_f1"]) for row in scored_items])) if scored_items else 0.0,
            "mean_wall_seconds": float(np.mean([float(row.get("wall_seconds", 0.0)) for row in items])) if items else 0.0,
            "mean_plan_seconds": float(np.mean([float(row.get("plan_seconds", 0.0)) for row in items])) if items else 0.0,
        }
    failure_path = collect_failure_manifest(output_root, rows)
    summary = {
        "output_root": str(output_root),
        "created_at": time.time(),
        "recovery_mode": True,
        "successive_halving": False,
        "result_count": len(rows),
        "scored_count": len(scored),
        "mean_event_f1": float(np.mean([float(row["event_f1"]) for row in scored])) if scored else 0.0,
        "mean_frame_f1": float(np.mean([float(row["frame_f1"]) for row in scored])) if scored else 0.0,
        "by_variant": by_variant,
        "rows": rows,
        "offline_training_failures_jsonl": str(failure_path),
        "success_criterion": "match_or_exceed_impromptu_event_f1_on_short_windows_before_full_song_scaling",
    }
    save_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Maestroso accuracy recovery variants on short MAESTRO windows.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--maestro-root", default=str(MAESTRO_ROOT))
    parser.add_argument("--sample-manifest", default=None)
    parser.add_argument("--songs", type=int, default=2)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--hard-window-max-duration-s", type=float, default=20.0)
    parser.add_argument("--variants", default=",".join(RECOVERY_VARIANTS))
    parser.add_argument("--rhapsody-checkpoint", default=str(DEFAULT_RHAPSODY))
    parser.add_argument("--rhapsody-device", default="cuda")
    parser.add_argument("--variant-timeout-s", type=int, default=1800)
    parser.add_argument("--rollout-timeout-s", type=int, default=900)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=1.0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"

    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    unknown = sorted(set(variants) - set(RECOVERY_VARIANTS))
    if unknown:
        raise ValueError(f"unknown recovery variants: {unknown}")
    songs = read_maestro(Path(args.maestro_root), int(args.songs), Path(args.sample_manifest) if args.sample_manifest else None)
    manifest = {
        "songs": songs,
        "variants": variants,
        "max_duration_s": float(args.max_duration_s),
        "hard_window_max_duration_s": float(args.hard_window_max_duration_s),
        "successive_halving": False,
        "full_rp1m_retest_for_every_variant": True,
        "all_waypoint_sparse_activation_validation": True,
        "proposal_assignment_default": "Bagatelle sequential previous-pose",
        "debug_only_assignment": "build_proposal_fingertip_targets split-even",
        "production_trajectory_mode": PRODUCTION_TRAJECTORY_MODE,
        "hard_window_trajectory_mode": HARD_WINDOW_TRAJECTORY_MODE,
        "chunk_overlap_seconds": float(max(float(args.chunk_overlap_seconds), 1.0)),
        "stitch_blend_enabled": False,
        "rhapsody_role": "seed_only_then_cpu_exact_mujoco_validation",
        "online_rl_in_planner": False,
        "online_cem_in_planner": False,
        "rhapsody_checkpoint": str(Path(args.rhapsody_checkpoint)),
        "rhapsody_device": str(args.rhapsody_device),
    }
    save_json(output_root / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for song in songs:
        for variant in variants:
            print(f"=== recovery {song['run_id']} {variant} ===", flush=True)
            row = run_variant(
                variant=variant,
                song=song,
                output_root=output_root,
                max_duration_s=float(args.max_duration_s),
                hard_window_max_duration_s=float(args.hard_window_max_duration_s),
                rhapsody_checkpoint=Path(args.rhapsody_checkpoint),
                rhapsody_device=str(args.rhapsody_device),
                env=env,
                variant_timeout_s=int(args.variant_timeout_s),
                rollout_timeout_s=int(args.rollout_timeout_s),
                chunk_overlap_seconds=max(float(args.chunk_overlap_seconds), 1.0),
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            summarize(output_root, rows)
    summary = summarize(output_root, rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
