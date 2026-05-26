#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
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
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.keys import extract_waypoint_frames  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402


MAESTRO_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0")
DEFAULT_OUTPUT = Path("/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch")
RHAPSODY = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/"
    "random10_heldout2_prev_random_scale6_refine16_20260522/rhapsody_rpik.pt"
)
ENV_NAME = "RoboPianist-debug-NocturneRousseau-v0"
PRODUCTION_TRAJECTORY_MODE = "joint_space_straighten"
HARD_WINDOW_TRAJECTORY_MODE = "dense_fingertip_ik"
CONTROL_TIMESTEP = 0.05
HAND_STATE_DIM = 46
NUM_FINGERS = 10
NUM_KEYS = 88
RHAPSODY_Y_OFFSET = 0.08289646
PROXY_RESIDUAL_WARN_M = 0.05
PROXY_ACTIVATION_BAD_F1 = 0.45
DEFAULT_HALVING_ROUND1_S = 15.0


def save_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_save_json(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0.0))
    played = float(score.get("played_press_events", 0.0))
    target = float(score.get("target_press_events", 0.0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def target_event_count(target_keys: np.ndarray, threshold: float = 0.5) -> int:
    active = np.asarray(target_keys[:, :88], dtype=np.float32) > float(threshold)
    if active.shape[0] == 0:
        return 0
    starts = active & np.vstack([np.ones((1, 88), dtype=bool), ~active[:-1]])
    return int(np.count_nonzero(starts))


def waypoint_release_frames(target_keys: np.ndarray, waypoint_frames: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    keys = np.asarray(target_keys, dtype=np.float32)
    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    if frames.size == 0 or keys.size == 0:
        return np.zeros((0,), dtype=np.int64)
    active = keys[:, :NUM_KEYS] > float(threshold)
    releases: list[int] = []
    total = int(active.shape[0])
    for frame in frames:
        start = int(np.clip(int(frame), 0, max(total - 1, 0)))
        row = active[start]
        end = start
        while end + 1 < total and np.array_equal(active[end + 1], row):
            end += 1
        releases.append(int(end))
    return np.asarray(releases, dtype=np.int64)


def select_evenly(keys: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(keys, dtype=np.int32).reshape(-1)
    limit = max(int(count), 0)
    if values.size <= limit:
        return values, np.zeros((0,), dtype=np.int32)
    positions = np.rint(np.linspace(0, values.size - 1, num=limit)).astype(np.int64)
    selected = values[positions]
    keep = np.zeros((values.size,), dtype=bool)
    keep[positions] = True
    return selected.astype(np.int32), values[~keep].astype(np.int32)


def spread_fingers(fingers_low_to_high: tuple[int, ...], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(fingers_low_to_high):
        return list(fingers_low_to_high)
    slots = np.rint(np.linspace(0, len(fingers_low_to_high) - 1, num=count)).astype(np.int64)
    return [int(fingers_low_to_high[int(slot)]) for slot in slots]


def build_proposal_fingertip_targets(
    waypoint_target_keys: np.ndarray,
    key_targets: np.ndarray,
    *,
    threshold: float = 0.5,
    split_key: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    keysets = np.asarray(waypoint_target_keys, dtype=np.float32)
    targets = np.full((keysets.shape[0], NUM_FINGERS, 3), np.nan, dtype=np.float32)
    masks = np.zeros((keysets.shape[0], NUM_FINGERS), dtype=np.float32)
    assignments = np.full((keysets.shape[0], NUM_FINGERS), -1, dtype=np.int32)
    dropped_total = 0
    dropped_rows = 0
    max_dropped = 0
    left_fingers = (4, 3, 2, 1, 0)
    right_fingers = (5, 6, 7, 8, 9)
    for row, keyset in enumerate(keysets):
        active = np.flatnonzero(keyset[:NUM_KEYS] > float(threshold)).astype(np.int32)
        left_keys, left_dropped = select_evenly(np.sort(active[active < int(split_key)]), len(left_fingers))
        right_keys, right_dropped = select_evenly(np.sort(active[active >= int(split_key)]), len(right_fingers))
        dropped = int(left_dropped.size + right_dropped.size)
        dropped_total += dropped
        dropped_rows += int(dropped > 0)
        max_dropped = max(max_dropped, dropped)
        for finger, key in zip(spread_fingers(left_fingers, int(left_keys.size)), left_keys):
            targets[row, finger] = key_targets[int(key)]
            masks[row, finger] = 1.0
            assignments[row, finger] = int(key)
        for finger, key in zip(spread_fingers(right_fingers, int(right_keys.size)), right_keys):
            targets[row, finger] = key_targets[int(key)]
            masks[row, finger] = 1.0
            assignments[row, finger] = int(key)
    return targets, masks, assignments, {
        "assignment_strategy": "maestroso_split_even_finger_spread",
        "split_key": int(split_key),
        "dropped_key_count": int(dropped_total),
        "rows_with_dropped_keys": int(dropped_rows),
        "max_dropped_keys_in_row": int(max_dropped),
        "left_fingers_low_to_high": list(left_fingers),
        "right_fingers_low_to_high": list(right_fingers),
    }


def transform_bagatelle_targets_for_rhapsody(
    targets: np.ndarray,
    masks: np.ndarray,
    *,
    y_offset: float = RHAPSODY_Y_OFFSET,
) -> tuple[np.ndarray, np.ndarray]:
    target_values = np.asarray(targets, dtype=np.float32).reshape(-1, NUM_FINGERS, 3)
    mask_values = np.asarray(masks, dtype=np.float32).reshape(-1, NUM_FINGERS)
    transformed = np.empty_like(target_values, dtype=np.float32)
    transformed[:, :5] = target_values[:, 5:]
    transformed[:, 5:] = target_values[:, :5]
    transformed[:, :, 1] += np.float32(float(y_offset))
    transformed_mask = np.concatenate([mask_values[:, 5:], mask_values[:, :5]], axis=1).astype(np.float32)
    return transformed, transformed_mask


def solve_gpu_proposal_batch(
    targets: np.ndarray,
    masks: np.ndarray,
    *,
    neutral_qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    device: str,
    batch_size: int,
    passes: int,
    refinement_steps: int,
    refinement_lr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from rhapsody.solver import RhapsodyIKSolver

    count = int(np.asarray(targets).shape[0])
    if count == 0:
        return (
            np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    if str(device).startswith("cuda"):
        import torch

        if not bool(torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested for gpu_batch_proposal, but torch.cuda.is_available() is false")
    solver = RhapsodyIKSolver.from_checkpoint(RHAPSODY, device=device)
    proposal_targets, proposal_masks = transform_bagatelle_targets_for_rhapsody(targets, masks)
    batch = max(int(batch_size), 1)
    pass_count = max(int(passes), 1)
    previous = np.repeat(np.asarray(neutral_qpos, dtype=np.float32).reshape(1, -1), count, axis=0)
    qpos = previous.copy()
    mean_error = np.zeros((count,), dtype=np.float32)
    max_error = np.zeros((count,), dtype=np.float32)
    for _pass_index in range(pass_count):
        rows: list[np.ndarray] = []
        mean_rows: list[np.ndarray] = []
        max_rows: list[np.ndarray] = []
        for start in range(0, count, batch):
            end = min(start + batch, count)
            solution = solver.solve_batch(
                proposal_targets[start:end],
                active_mask=proposal_masks[start:end],
                previous_qpos=previous[start:end],
                refinement_steps=max(int(refinement_steps), 0),
                refinement_lr=float(refinement_lr),
            )
            rows.append(np.asarray(solution.qpos, dtype=np.float32))
            mean_rows.append(np.asarray(solution.mean_error_m, dtype=np.float32))
            max_rows.append(np.asarray(solution.max_error_m, dtype=np.float32))
        qpos = np.concatenate(rows, axis=0).astype(np.float32)
        qpos = np.clip(qpos, np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)).astype(np.float32)
        mean_error = np.concatenate(mean_rows, axis=0).astype(np.float32)
        max_error = np.concatenate(max_rows, axis=0).astype(np.float32)
        previous = np.repeat(np.asarray(neutral_qpos, dtype=np.float32).reshape(1, -1), count, axis=0)
        if count > 1:
            previous[1:] = qpos[:-1]
    return qpos, mean_error, max_error


def dense_assignment_result(
    keyset: np.ndarray,
    assignment_row: np.ndarray,
    key_targets: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Any:
    from bagatelle.assignment import FingerAssignmentResult

    active_keys = np.flatnonzero(np.asarray(keyset, dtype=np.float32)[:NUM_KEYS] > float(threshold)).astype(np.int32)
    dense = np.asarray(assignment_row, dtype=np.int32).reshape(NUM_FINGERS)
    assigned_fingers = np.flatnonzero(dense >= 0).astype(np.int32)
    assigned_keys = dense[assigned_fingers.astype(np.int64)].astype(np.int32)
    positions = []
    keep = []
    for index, key in enumerate(assigned_keys):
        matches = np.flatnonzero(active_keys == int(key))
        if matches.size:
            positions.append(int(matches[0]))
            keep.append(index)
    if len(keep) != int(assigned_keys.size):
        keep_array = np.asarray(keep, dtype=np.int64)
        assigned_fingers = assigned_fingers[keep_array]
        assigned_keys = assigned_keys[keep_array]
    assigned_key_positions = np.asarray(positions, dtype=np.int32)
    target_positions = (
        np.asarray(key_targets[assigned_keys.astype(np.int64)], dtype=np.float32)
        if assigned_keys.size
        else np.zeros((0, 3), dtype=np.float32)
    )
    unassigned = np.setdiff1d(active_keys, assigned_keys, assume_unique=False).astype(np.int32)
    cost_matrix = np.zeros((NUM_FINGERS, max(int(active_keys.size), 1)), dtype=np.float32)
    return FingerAssignmentResult(
        active_keys=active_keys,
        assigned_finger_indices=assigned_fingers.astype(np.int32),
        assigned_keys=assigned_keys.astype(np.int32),
        assigned_key_positions=assigned_key_positions.astype(np.int32),
        target_positions=target_positions.astype(np.float32),
        unassigned_keys=unassigned,
        cost_matrix=cost_matrix,
        total_cost=0.0,
        mean_cost=0.0,
        strategy="maestroso_gpu_batch_split_even_cpu_verify",
    )


def verify_sparse_waypoints_on_cpu(
    *,
    kin: Any,
    waypoint_target_keys: np.ndarray,
    proposal_assignments: np.ndarray,
    key_targets: np.ndarray,
    proposal_qpos: np.ndarray,
    use_bagatelle_assignment: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from dataclasses import replace

    from bagatelle.assignment import assign_fingers_previous_pose
    from bagatelle.config import BagatelleConfig

    verify_cfg = BagatelleConfig(
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
        ik_fingertip_weight=2.0,
        ik_key_front_weight=1.0,
        ik_key_width_weight=1.0,
        ik_key_height_weight=1.0,
        ik_smoothness_weight=0.05,
        ik_neutral_weight=0.005,
        ik_unassigned_fingertip_strategy="avoid_mispresses",
        ik_unassigned_fingertip_avoidance_weight=0.5,
        ik_unassigned_fingertip_avoidance_radius=0.03,
        ik_max_nfev=80,
        ik_multistart_on_failure=True,
        ik_multistart_seed_count=2,
        rhapsody_ik_enabled=False,
    )
    count = int(np.asarray(proposal_assignments).shape[0])
    if count == 0:
        return np.zeros((0, HAND_STATE_DIM), dtype=np.float32), np.zeros((0, NUM_FINGERS), dtype=np.int32), {
            "enabled": True,
            "assignment_mode": "bagatelle_previous_pose" if use_bagatelle_assignment else "fixed_proposal_assignment",
            "waypoints": 0,
            "success_count": 0,
            "optimizer_success_count": 0,
            "nfev_sum": 0,
            "mean_max_residual_m": 0.0,
            "max_residual_m": 0.0,
        }
    previous = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
    neutral = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
    rows: list[np.ndarray] = []
    assignment_rows: list[np.ndarray] = []
    residuals: list[float] = []
    nfev_sum = 0
    success_count = 0
    optimizer_success_count = 0
    previous_dense_assignment = np.full((NUM_FINGERS,), -1, dtype=np.int32)
    previous_fingertips = np.asarray(kin.fingertip_positions_for_qpos(previous), dtype=np.float32)
    contact_targets_all = kin.key_contact_targets(np.arange(NUM_KEYS, dtype=np.int32))
    started = time.time()
    for index in range(count):
        if bool(use_bagatelle_assignment):
            active_keys = np.flatnonzero(
                np.asarray(waypoint_target_keys[index], dtype=np.float32)[:NUM_KEYS] > 0.5
            ).astype(np.int32)
            contact_targets = contact_targets_all[active_keys.astype(np.int64)]
            assignment = assign_fingers_previous_pose(
                active_keys,
                previous_fingertips,
                contact_targets,
                verify_cfg,
                previous_assignment=previous_dense_assignment,
            )
            if assignment.count:
                assignment = replace(
                    assignment,
                    target_positions=key_targets[assignment.assigned_keys.astype(np.int64)].astype(np.float32),
                )
        else:
            assignment = dense_assignment_result(
                waypoint_target_keys[index],
                proposal_assignments[index],
                key_targets,
                threshold=0.5,
            )
        result = kin.solve_press_pose(
            assignment,
            previous,
            neutral_qpos=neutral,
            config=verify_cfg,
            rhapsody_seed_override=proposal_qpos[index],
        )
        pose = np.asarray(result.pose, dtype=np.float32)
        rows.append(pose)
        residuals.append(float(result.max_residual))
        nfev_sum += int(result.nfev)
        success_count += int(bool(result.success))
        optimizer_success_count += int(bool(result.optimizer_success))
        previous = pose
        previous_fingertips = np.asarray(result.fingertip_positions, dtype=np.float32)
        previous_dense_assignment = assignment.dense_key_by_finger()
        assignment_rows.append(previous_dense_assignment.astype(np.int32))
    residual_array = np.asarray(residuals, dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32), np.stack(assignment_rows, axis=0).astype(np.int32), {
        "enabled": True,
        "assignment_mode": "bagatelle_previous_pose" if use_bagatelle_assignment else "fixed_proposal_assignment",
        "waypoints": int(count),
        "seconds": float(time.time() - started),
        "success_count": int(success_count),
        "optimizer_success_count": int(optimizer_success_count),
        "nfev_sum": int(nfev_sum),
        "mean_max_residual_m": float(np.mean(residual_array)) if residual_array.size else 0.0,
        "max_residual_m": float(np.max(residual_array)) if residual_array.size else 0.0,
    }


def read_maestro(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (MAESTRO_ROOT / "maestro-v3.0.0.csv").open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            midi_path = MAESTRO_ROOT / row["midi_filename"]
            if not midi_path.is_file():
                continue
            rows.append(
                {
                    "run_id": f"maestro_{index:04d}",
                    "manifest_index": index,
                    "midi_path": str(midi_path),
                    **row,
                }
            )
            if len(rows) >= int(limit):
                break
    return rows


def run_cmd(name: str, command: list[str], *, cwd: Path, log_dir: Path, env: dict[str, str], timeout: int | None) -> float:
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log_path = log_dir / f"{name}.log"
    with log_path.open("ab") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} =====\n".encode())
        log.write((" ".join(command) + "\n").encode())
        log.flush()
        subprocess.run(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        seconds = time.time() - start
        log.write(f"===== finished {name} in {seconds:.1f}s =====\n".encode())
    return float(seconds)


def common_plan_args(
    song: dict[str, Any],
    run_parent: Path,
    run_name: str,
    max_duration_s: float | None,
    *,
    trajectory_mode: str = PRODUCTION_TRAJECTORY_MODE,
) -> list[str]:
    args = [
        sys.executable,
        "Impromptu/scripts/plan_trajectory.py",
        "--midi-path",
        str(song["midi_path"]),
        "--output-root",
        str(run_parent),
        "--run-name",
        run_name,
        "--environment-name",
        ENV_NAME,
        "--trajectory-mode",
        str(trajectory_mode),
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
        "--no-include-midpoint-anchors",
    ]
    if max_duration_s is not None:
        args += ["--max-duration-s", str(float(max_duration_s))]
    return args


def variant_command(
    variant: str,
    song: dict[str, Any],
    run_parent: Path,
    max_duration_s: float | None,
    *,
    hard_window_max_duration_s: float,
    round_profile: str = "standard",
) -> list[str]:
    run_name = str(song["run_id"])
    args = common_plan_args(song, run_parent, run_name, max_duration_s)
    cheap = str(round_profile) == "round1_proxy"
    anchor_stride = "12" if cheap else "6"
    topk_anchor_stride = "12" if cheap else "8"
    ik_nfev = "30" if cheap else "40"
    cpu_ik_nfev = "40" if cheap else "60"
    if variant == "cpu_fast":
        return args + [
            "--assignment-strategy",
            "legacy_previous_pose",
            "--anchor-stride",
            anchor_stride,
            "--ik-max-nfev",
            cpu_ik_nfev,
            "--disable-ik-multistart-on-failure",
        ]
    if variant == "gpu_rhapsody_seed":
        return args + [
            "--assignment-strategy",
            "legacy_previous_pose",
            "--anchor-stride",
            anchor_stride,
            "--ik-max-nfev",
            ik_nfev,
            "--disable-ik-multistart-on-failure",
            "--enable-rhapsody-ik",
            "--rhapsody-ik-checkpoint",
            str(RHAPSODY),
            "--rhapsody-ik-device",
            "cuda",
            "--rhapsody-ik-refinement-steps",
            "0",
        ]
    if variant == "gpu_rhapsody_refine":
        return args + [
            "--assignment-strategy",
            "legacy_previous_pose",
            "--anchor-stride",
            anchor_stride,
            "--ik-max-nfev",
            ik_nfev,
            "--disable-ik-multistart-on-failure",
            "--enable-rhapsody-ik",
            "--rhapsody-ik-checkpoint",
            str(RHAPSODY),
            "--rhapsody-ik-device",
            "cuda",
            "--rhapsody-ik-refinement-steps",
            "4",
        ]
    if variant == "gpu_rhapsody_topk":
        return args + [
            "--assignment-strategy",
            "ik_aware_topk",
            "--assignment-top-k",
            "4",
            "--anchor-stride",
            topk_anchor_stride,
            "--ik-max-nfev",
            ik_nfev,
            "--ik-static-contact-validation",
            "--ik-static-contact-missed-key-weight",
            "4.0",
            "--ik-static-contact-wrong-key-weight",
            "0.25",
            "--enable-rhapsody-ik",
            "--rhapsody-ik-checkpoint",
            str(RHAPSODY),
            "--rhapsody-ik-device",
            "cuda",
            "--rhapsody-ik-candidate-scoring",
            "--rhapsody-ik-refinement-steps",
            "0",
        ]
    if variant == "hard_window_dense_ik":
        hard_duration = min(
            float(max_duration_s) if max_duration_s is not None else float(hard_window_max_duration_s),
            float(hard_window_max_duration_s),
        )
        hard_args = common_plan_args(
            song,
            run_parent,
            run_name,
            hard_duration,
            trajectory_mode=HARD_WINDOW_TRAJECTORY_MODE,
        )
        return hard_args + [
            "--assignment-strategy",
            "legacy_previous_pose",
            "--anchor-stride",
            "1",
            "--solve-contact-window-only",
            "--ik-max-nfev",
            "80",
            "--disable-ik-multistart-on-failure",
            "--disable-attraction-forces",
            "--enable-rhapsody-ik",
            "--rhapsody-ik-checkpoint",
            str(RHAPSODY),
            "--rhapsody-ik-device",
            "cuda",
            "--rhapsody-ik-refinement-steps",
            "0",
        ]
    raise ValueError(f"unknown variant {variant}")


def extract_metrics(run_dir: Path) -> dict[str, Any]:
    traj = run_dir / "trajectory.npz"
    meta = load_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "trajectory_npz": str(traj),
        "metadata": meta,
    }
    if traj.exists():
        with np.load(traj, allow_pickle=False) as data:
            target = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
            result.update(
                {
                    "target_steps": int(target.shape[0]),
                    "target_press_events": target_event_count(target),
                    "unique_active_keys": int(np.count_nonzero(np.max(target, axis=0) > 0.5)) if target.size else 0,
                    "dense_frames": int(np.asarray(data["planned_hand_joints_dense"]).shape[0]),
                }
            )
    return result


def metric_column_values(data: Any, metadata: dict[str, Any], array_name: str, column_names: tuple[str, ...]) -> np.ndarray:
    if array_name not in data:
        return np.zeros((0,), dtype=np.float32)
    values = np.asarray(data[array_name], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if array_name == "sparse_press_ik_metrics":
        columns = metadata.get("sparse_press_ik_metric_columns") or []
    else:
        columns = metadata.get("ik_anchor_metric_columns") or []
    for name in column_names:
        if name in columns:
            return values[:, int(columns.index(name))].astype(np.float32)
    return np.zeros((0,), dtype=np.float32)


def frame_f1_counts(target: np.ndarray, played: np.ndarray, *, threshold: float = 0.5) -> tuple[int, int, int]:
    target_active = np.asarray(target, dtype=np.float32)[:NUM_KEYS] > float(threshold)
    played_active = np.asarray(played, dtype=np.float32)[:NUM_KEYS] > float(threshold)
    tp = int(np.count_nonzero(target_active & played_active))
    fp = int(np.count_nonzero(~target_active & played_active))
    fn = int(np.count_nonzero(target_active & ~played_active))
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = float(tp) / float(tp + fp) if tp + fp else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compute_planning_only_metrics(run_dir: Path) -> dict[str, Any]:
    traj = run_dir / "trajectory.npz"
    meta = load_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    metrics: dict[str, Any] = {
        "stage": "A_planning_only",
        "policy": "planning_metadata_only_no_mujoco_activation_no_rollout",
        "contact_cem_allowed": False,
        "online_policy_update_allowed": False,
    }
    if not traj.exists():
        metrics.update({"available": False, "proxy_badness": 1.0e9, "reason": "missing_trajectory"})
        return metrics

    residuals: list[np.ndarray] = []
    nfev_values: list[np.ndarray] = []
    anchor_count = int(meta.get("num_ik_anchor_frames", 0) or 0)
    waypoint_count = int(meta.get("num_waypoints", 0) or 0)
    unassigned_key_count = 0
    with np.load(traj, allow_pickle=False) as data:
        if not anchor_count and "ik_anchor_frames_dense" in data:
            anchor_count = int(np.asarray(data["ik_anchor_frames_dense"]).size)
        if not waypoint_count and "waypoint_frames" in data:
            waypoint_count = int(np.asarray(data["waypoint_frames"]).size)
        for array_name in ("ik_anchor_metrics", "sparse_press_ik_metrics"):
            residual = metric_column_values(
                data,
                meta,
                array_name,
                ("max_assigned_distance", "max_residual", "residual_norm"),
            )
            if residual.size:
                residuals.append(residual[np.isfinite(residual)])
            nfev = metric_column_values(data, meta, array_name, ("nfev",))
            if nfev.size:
                nfev_values.append(nfev[np.isfinite(nfev)])
        if "proposal_solver_max_error_m" in data:
            values = np.asarray(data["proposal_solver_max_error_m"], dtype=np.float32).reshape(-1)
            residuals.append(values[np.isfinite(values)])
        if "unassigned_keys" in data:
            unassigned = np.asarray(data["unassigned_keys"])
            unassigned_key_count = int(np.count_nonzero(unassigned >= 0))
        if "target_keys" in data:
            target_events = target_event_count(np.asarray(data["target_keys"], dtype=np.float32)[:, :NUM_KEYS])
        else:
            target_events = int(meta.get("target_press_events", 0) or 0)
    unassigned_key_count += int(meta.get("assignment", {}).get("dropped_key_count", 0) or 0)
    sparse_verification = meta.get("sparse_cpu_verification") if isinstance(meta.get("sparse_cpu_verification"), dict) else {}
    residual_array = np.concatenate(residuals, axis=0) if residuals else np.zeros((0,), dtype=np.float32)
    nfev_array = np.concatenate(nfev_values, axis=0) if nfev_values else np.zeros((0,), dtype=np.float32)
    ik_residual_p95 = (
        float(meta.get("ik_max_residual_p95"))
        if meta.get("ik_max_residual_p95") is not None
        else (
            float(np.percentile(residual_array, 95))
            if residual_array.size
            else float(sparse_verification.get("max_residual_m", 0.0) or 0.0)
        )
    )
    nfev_sum = (
        int(np.sum(nfev_array))
        if nfev_array.size
        else int(meta.get("sparse_cpu_verify_nfev_sum", sparse_verification.get("nfev_sum", 0)) or 0)
    )
    ik_success_count = int(
        meta.get("ik_success_count")
        if meta.get("ik_success_count") is not None
        else meta.get("sparse_press_ik_success_count", 0) or sparse_verification.get("success_count", 0) or 0
    )
    denominator = max(anchor_count or waypoint_count, 1)
    success_rate = float(ik_success_count) / float(denominator)
    planning_badness = (
        min(ik_residual_p95 / max(PROXY_RESIDUAL_WARN_M, 1e-8), 5.0)
        + (1.0 - min(max(success_rate, 0.0), 1.0))
        + min(float(unassigned_key_count) / float(max(target_events, 1)), 5.0)
        + 0.1 * min(float(nfev_sum) / float(max(denominator * 60, 1)), 5.0)
    )
    metrics.update(
        {
            "available": True,
            "anchor_count": int(anchor_count),
            "waypoint_count": int(waypoint_count),
            "ik_success_count": int(ik_success_count),
            "ik_success_rate": success_rate,
            "ik_max_residual_p95": float(ik_residual_p95),
            "ik_max_residual_p95_m": float(ik_residual_p95),
            "ik_nfev_mean": float(np.mean(nfev_array)) if nfev_array.size else 0.0,
            "ik_nfev_p95": float(np.percentile(nfev_array, 95)) if nfev_array.size else 0.0,
            "nfev_sum": int(nfev_sum),
            "unassigned_key_count": int(unassigned_key_count),
            "target_press_events": int(target_events),
            "planning_badness": float(planning_badness),
            "proxy_badness": float(planning_badness),
            "poor_proxy": bool(
                ik_residual_p95 > PROXY_RESIDUAL_WARN_M
                or unassigned_key_count > 0
                or success_rate < 0.5
            ),
        }
    )
    save_json(run_dir / "stage_a_planning_metrics.json", metrics)
    return metrics


def sparse_activation_proxy(run_dir: Path, *, max_frames: int, env: dict[str, str]) -> dict[str, Any]:
    if int(max_frames) <= 0:
        return {"enabled": False}
    traj = run_dir / "trajectory.npz"
    if not traj.exists():
        return {"enabled": False, "error": "missing_trajectory"}
    started = time.time()
    try:
        from bagatelle.config import BagatelleConfig
        from bagatelle.kinematics import BagatelleKinematics

        with np.load(traj, allow_pickle=False) as data:
            target_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :NUM_KEYS]
            hand = np.asarray(data["planned_hand_joints"], dtype=np.float32)
            waypoint_frames = (
                np.asarray(data["waypoint_frames"], dtype=np.int64).reshape(-1)
                if "waypoint_frames" in data
                else np.flatnonzero(np.any(target_keys > 0.5, axis=1)).astype(np.int64)
            )
        frames = waypoint_frames[(waypoint_frames >= 0) & (waypoint_frames < target_keys.shape[0])]
        frames = frames[np.any(target_keys[frames] > 0.5, axis=1)] if frames.size else frames
        if frames.size > int(max_frames):
            polyphony = np.count_nonzero(target_keys[frames] > 0.5, axis=1)
            order = np.argsort(-polyphony, kind="stable")
            frames = np.sort(frames[order[: int(max_frames)]])
        if frames.size == 0:
            return {"enabled": True, "sampled_frames": 0, "activation_f1": 0.0, "seconds": float(time.time() - started)}
        old_mujoco_gl = os.environ.get("MUJOCO_GL")
        os.environ["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
        kin = BagatelleKinematics(
            BagatelleConfig(
                control_timestep=CONTROL_TIMESTEP,
                threshold=0.5,
                environment_name=ENV_NAME,
                key_press_depth=0.006,
            ),
            target_keys=target_keys,
            output_dir=run_dir / "proxy_activation",
        )
        try:
            tp = fp = fn = 0
            for frame in frames.astype(np.int64):
                activation = kin.activation_for_qpos(hand[int(frame)], settle_steps=1)
                row_tp, row_fp, row_fn = frame_f1_counts(target_keys[int(frame)], activation, threshold=0.5)
                tp += row_tp
                fp += row_fp
                fn += row_fn
        finally:
            kin.close()
            if old_mujoco_gl is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = old_mujoco_gl
        return {
            "enabled": True,
            "stage": "B_sparse_mujoco_activation",
            "selection_policy": "waypoint_frames_high_polyphony_first",
            "sampled_frames": int(frames.size),
            "activation_f1": f1_from_counts(tp, fp, fn),
            "activation_tp": int(tp),
            "activation_fp": int(fp),
            "activation_fn": int(fn),
            "estimated_missed_notes": int(fn),
            "estimated_wrong_notes": int(fp),
            "seconds": float(time.time() - started),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seconds": float(time.time() - started),
        }


def compute_proxy_metrics(run_dir: Path, *, activation_frames: int, env: dict[str, str]) -> dict[str, Any]:
    planning = compute_planning_only_metrics(run_dir)
    if int(activation_frames) <= 0 or not bool(planning.get("available")):
        return planning
    activation = sparse_activation_proxy(run_dir, max_frames=int(activation_frames), env=env)
    activation_f1 = activation.get("activation_f1")
    target_events = int(planning.get("target_press_events", 0) or 0)
    estimated_missed = int(activation.get("estimated_missed_notes", planning.get("ik_residual_bad_count", 0) or 0))
    estimated_wrong = int(activation.get("estimated_wrong_notes", planning.get("unassigned_key_count", 0) or 0))
    if activation_f1 is None:
        activation_penalty = 1.0 if bool(planning.get("poor_proxy")) else 0.0
    else:
        activation_penalty = max(0.0, 1.0 - float(activation_f1))
    proxy_badness = (
        float(planning.get("planning_badness", 0.0))
        + activation_penalty
        + min(float(estimated_missed) / float(max(target_events, 1)), 5.0)
        + 0.5 * min(float(estimated_wrong) / float(max(target_events, 1)), 5.0)
    )
    poor = bool(
        (activation_f1 is not None and float(activation_f1) < PROXY_ACTIVATION_BAD_F1)
        or bool(planning.get("poor_proxy"))
        or estimated_missed > 0
        or estimated_wrong > 0
    )
    proxy = dict(planning)
    proxy.update(
        {
            "stage": "B_sparse_mujoco_activation",
            "available": True,
            "target_press_events": int(target_events),
            "estimated_missed_notes": estimated_missed,
            "estimated_wrong_notes": estimated_wrong,
            "activation_proxy": activation,
            "activation_f1": activation_f1,
            "proxy_badness": float(proxy_badness),
            "poor_proxy": poor,
            "expensive_repair_recommended": poor,
            "expensive_repair_policy": "offline_failure_collection_only; do not run CEM/RL inside planner",
        }
    )
    save_json(run_dir / "proxy_metrics.json", proxy)
    return proxy


def attach_proxy(
    result: dict[str, Any],
    *,
    run_dir: Path,
    activation_frames: int,
    env: dict[str, str],
    evaluation_stage: str,
) -> None:
    stage = str(evaluation_stage).upper()
    if stage == "B":
        proxy = compute_proxy_metrics(run_dir, activation_frames=int(activation_frames), env=env)
    else:
        proxy = compute_planning_only_metrics(run_dir)
        proxy["stage"] = f"{stage}_planning_context" if stage not in ("A", "") else "A_planning_only"
        proxy["policy"] = "planning_metadata_only_no_sparse_activation"
    proxy["evaluation_stage"] = stage
    result["evaluation_stage"] = stage
    result["proxy_metrics"] = proxy
    result["proxy_badness"] = proxy.get("proxy_badness")
    result["poor_proxy"] = proxy.get("poor_proxy")


def run_retest(
    run_parent: Path,
    run_id: str,
    retest_root: Path,
    log_dir: Path,
    timeout: int | None,
    env: dict[str, str],
    *,
    render_mp4: bool = False,
) -> dict[str, Any] | None:
    command = [
        sys.executable,
        "retest_impromptu_rp1m_simulator.py",
        "--run-root",
        str(run_parent),
        "--output-root",
        str(retest_root),
        "--only-run",
        run_id,
        "--environment-name",
        ENV_NAME,
        "--threshold",
        "0.5",
        "--set-hand-qvel",
    ]
    if bool(render_mp4):
        command.append("--render-mp4")
    run_cmd("rp1m_retest", command, cwd=REPO, log_dir=log_dir, env=env, timeout=timeout)
    result_path = retest_root / run_id / "impromptu_rp1m_retest_result.json"
    return load_json(result_path) if result_path.exists() else None


def run_plan_variant(
    *,
    variant: str,
    song: dict[str, Any],
    output_root: Path,
    max_duration_s: float | None,
    env: dict[str, str],
    timeout: int | None,
    rollout_timeout: int | None,
    skip_rollout: bool,
    hard_window_max_duration_s: float,
    proxy_activation_frames: int,
    evaluation_stage: str,
    round_profile: str = "standard",
    render_mp4: bool = False,
) -> dict[str, Any]:
    run_id = str(song["run_id"])
    variant_root = output_root / variant
    run_dir = variant_root / run_id
    log_dir = output_root / "logs" / variant / run_id
    started = time.time()
    command = variant_command(
        variant,
        song,
        variant_root,
        max_duration_s,
        hard_window_max_duration_s=hard_window_max_duration_s,
        round_profile=round_profile,
    )
    mode = HARD_WINDOW_TRAJECTORY_MODE if variant == "hard_window_dense_ik" else PRODUCTION_TRAJECTORY_MODE
    result: dict[str, Any] = {
        "variant": variant,
        "run_id": run_id,
        "song": song,
        "trajectory_mode": mode,
        "production_mode": mode == PRODUCTION_TRAJECTORY_MODE,
    }
    try:
        plan_seconds = run_cmd("plan", command, cwd=REPO, log_dir=log_dir, env=env, timeout=timeout)
        result.update(extract_metrics(run_dir))
        result["plan_seconds"] = plan_seconds
        attach_proxy(
            result,
            run_dir=run_dir,
            activation_frames=proxy_activation_frames,
            env=env,
            evaluation_stage=evaluation_stage,
        )
        if not skip_rollout:
            retest = run_retest(
                variant_root,
                run_id,
                variant_root / "rp1m_retest",
                log_dir,
                rollout_timeout,
                env,
                render_mp4=bool(render_mp4),
            )
            result["rp1m_retest"] = retest
            if retest:
                result.update(
                    {
                        "event_f1": retest.get("event_f1"),
                        "frame_f1": retest.get("frame_f1"),
                        "matched": retest.get("matched"),
                        "target": retest.get("target"),
                        "played": retest.get("played"),
                        "mispresses": retest.get("mispresses"),
                    }
                )
        result["ok"] = True
    except Exception as exc:
        result.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    result["wall_seconds"] = float(time.time() - started)
    save_json(run_dir / "variant_result.json", result)
    return result


def chunk_ranges(total: int, chunk_steps: int, overlap_steps: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    stride = max(chunk_steps - overlap_steps, 1)
    while start < total:
        end = min(start + chunk_steps, total)
        ranges.append((start, end))
        if end >= total:
            break
        start += stride
    return ranges


def plan_chunk(
    *,
    chunk_id: int,
    target: np.ndarray,
    variant_root: Path,
    run_id: str,
    env: dict[str, str],
    timeout: int | None,
) -> Path:
    chunk_input = variant_root / "chunk_inputs" / run_id / f"chunk_{chunk_id:04d}.npz"
    atomic_save_npz(chunk_input, target_keys=target.astype(np.float32))
    chunk_parent = variant_root / "chunk_plans" / run_id
    chunk_name = f"chunk_{chunk_id:04d}"
    log_dir = variant_root / "logs" / run_id / chunk_name
    command = [
        sys.executable,
        "Impromptu/scripts/plan_trajectory.py",
        "--target-keys-npz",
        str(chunk_input),
        "--output-root",
        str(chunk_parent),
        "--run-name",
        chunk_name,
        "--environment-name",
        ENV_NAME,
        "--trajectory-mode",
        PRODUCTION_TRAJECTORY_MODE,
        "--assignment-strategy",
        "legacy_previous_pose",
        "--anchor-stride",
        "4",
        "--ik-max-nfev",
        "40",
        "--disable-ik-multistart-on-failure",
        "--no-include-midpoint-anchors",
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
        "--enable-rhapsody-ik",
        "--rhapsody-ik-checkpoint",
        str(RHAPSODY),
        "--rhapsody-ik-device",
        "cuda",
        "--rhapsody-ik-refinement-steps",
        "0",
    ]
    run_cmd("plan_chunk", command, cwd=REPO, log_dir=log_dir, env=env, timeout=timeout)
    return chunk_parent / chunk_name


def stitch_chunks(chunk_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {}
    offset = 0
    waypoint_frames: list[np.ndarray] = []
    for chunk_dir in chunk_dirs:
        with np.load(chunk_dir / "trajectory.npz", allow_pickle=False) as data:
            for key in (
                "target_keys",
                "planned_hand_joints",
                "planned_hand_velocities",
                "planned_hand_joints_dense",
                "planned_hand_velocities_dense",
            ):
                arrays.setdefault(key, []).append(np.asarray(data[key]))
            if "waypoint_frames" in data:
                waypoint_frames.append(np.asarray(data["waypoint_frames"], dtype=np.int64) + offset)
            offset += int(np.asarray(data["target_keys"]).shape[0])
    payload = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    if waypoint_frames:
        payload["waypoint_frames"] = np.concatenate(waypoint_frames, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(output_dir / "trajectory.npz", **payload)
    metadata = {
        "planner": "maestroso_chunked_gpu_rhapsody",
        "trajectory_mode": PRODUCTION_TRAJECTORY_MODE,
        "chunk_count": len(chunk_dirs),
        "chunk_dirs": [str(path) for path in chunk_dirs],
        "target_keys_shape": list(payload["target_keys"].shape),
        "num_dense_frames": int(payload["planned_hand_joints_dense"].shape[0]),
        "environment_name": ENV_NAME,
        "trajectory_npz": str(output_dir / "trajectory.npz"),
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


def run_chunked_variant(
    *,
    song: dict[str, Any],
    output_root: Path,
    max_duration_s: float | None,
    chunk_seconds: float,
    chunk_overlap_seconds: float,
    chunk_workers: int,
    env: dict[str, str],
    timeout: int | None,
    rollout_timeout: int | None,
    skip_rollout: bool,
    proxy_activation_frames: int,
    evaluation_stage: str,
    render_mp4: bool = False,
) -> dict[str, Any]:
    variant = "chunked_gpu_rhapsody"
    run_id = str(song["run_id"])
    variant_root = output_root / variant
    run_dir = variant_root / run_id
    started = time.time()
    result: dict[str, Any] = {
        "variant": variant,
        "run_id": run_id,
        "song": song,
        "trajectory_mode": PRODUCTION_TRAJECTORY_MODE,
        "production_mode": True,
    }
    try:
        target_keys, midi_meta = load_target_keys_from_midi(
            song["midi_path"], control_timestep=0.05, max_duration_s=max_duration_s
        )
        ranges = chunk_ranges(
            int(target_keys.shape[0]),
            max(int(round(float(chunk_seconds) / 0.05)), 1),
            max(int(round(float(chunk_overlap_seconds) / 0.05)), 0),
        )
        chunk_dirs: list[Path] = []
        if int(chunk_workers) <= 1:
            for index, (start, end) in enumerate(ranges):
                chunk_dirs.append(
                    plan_chunk(
                        chunk_id=index,
                        target=target_keys[start:end],
                        variant_root=variant_root,
                        run_id=run_id,
                        env=env,
                        timeout=timeout,
                    )
                )
        else:
            # Chunk planning invokes Bagatelle/MuJoCo code. Use process workers
            # so each worker owns its own mutable physics state.
            with ProcessPoolExecutor(max_workers=int(chunk_workers)) as executor:
                futures = {
                    executor.submit(
                        plan_chunk,
                        chunk_id=index,
                        target=target_keys[start:end],
                        variant_root=variant_root,
                        run_id=run_id,
                        env=env,
                        timeout=timeout,
                    ): index
                    for index, (start, end) in enumerate(ranges)
                }
                done = [(futures[future], future.result()) for future in as_completed(futures)]
                chunk_dirs = [path for _, path in sorted(done)]
        metadata = stitch_chunks(chunk_dirs, run_dir)
        result.update(extract_metrics(run_dir))
        result.update({"chunk_count": len(chunk_dirs), "midi_meta": midi_meta, "chunk_metadata": metadata})
        result["plan_seconds"] = float(time.time() - started)
        attach_proxy(
            result,
            run_dir=run_dir,
            activation_frames=proxy_activation_frames,
            env=env,
            evaluation_stage=evaluation_stage,
        )
        if not skip_rollout:
            retest = run_retest(
                variant_root,
                run_id,
                variant_root / "rp1m_retest",
                output_root / "logs" / variant / run_id,
                rollout_timeout,
                env,
                render_mp4=bool(render_mp4),
            )
            result["rp1m_retest"] = retest
            if retest:
                result.update(
                    {
                        "event_f1": retest.get("event_f1"),
                        "frame_f1": retest.get("frame_f1"),
                        "matched": retest.get("matched"),
                        "target": retest.get("target"),
                        "played": retest.get("played"),
                        "mispresses": retest.get("mispresses"),
                    }
                )
        result["ok"] = True
    except Exception as exc:
        result.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    result["wall_seconds"] = float(time.time() - started)
    save_json(run_dir / "variant_result.json", result)
    return result


def run_gpu_batch_proposal_variant(
    *,
    variant: str,
    song: dict[str, Any],
    output_root: Path,
    max_duration_s: float | None,
    env: dict[str, str],
    rollout_timeout: int | None,
    skip_rollout: bool,
    proposal_batch_size: int,
    proposal_passes: int,
    proposal_refinement_steps: int,
    proposal_refinement_lr: float,
    proposal_device: str,
    proposal_cpu_verify_sparse_ik: bool,
    proxy_activation_frames: int,
    evaluation_stage: str,
    render_mp4: bool = False,
) -> dict[str, Any]:
    from bagatelle.config import BagatelleConfig
    from bagatelle.kinematics import BagatelleKinematics
    from impromptu.config import ImpromptuConfig
    from impromptu.joint_space_trajectory import build_joint_space_straightened_trajectory
    from intermezzo.planner import compute_hand_velocities

    use_bagatelle_assignment = variant == "gpu_batch_proposal_bagatelle_assign"
    run_id = str(song["run_id"])
    variant_root = output_root / variant
    run_dir = variant_root / run_id
    log_dir = output_root / "logs" / variant / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result: dict[str, Any] = {
        "variant": variant,
        "run_id": run_id,
        "song": song,
        "trajectory_mode": PRODUCTION_TRAJECTORY_MODE,
        "production_mode": True,
        "proposal_device": str(proposal_device),
        "cpu_exact_verification": not bool(skip_rollout),
    }
    kin = None
    try:
        plan_started = time.time()
        target_keys, midi_meta = load_target_keys_from_midi(
            song["midi_path"], control_timestep=CONTROL_TIMESTEP, max_duration_s=max_duration_s
        )
        active = np.asarray(target_keys[:, :NUM_KEYS], dtype=np.float32) > 0.5
        waypoint_frames = extract_waypoint_frames(target_keys[:, :NUM_KEYS], threshold=0.5)
        if waypoint_frames.size:
            keep = np.any(active[np.asarray(waypoint_frames, dtype=np.int64)], axis=1)
            waypoint_frames = np.asarray(waypoint_frames, dtype=np.int64)[keep]
        waypoint_target_keys = (
            np.asarray(target_keys[waypoint_frames, :NUM_KEYS], dtype=np.float32)
            if waypoint_frames.size
            else np.zeros((0, NUM_KEYS), dtype=np.float32)
        )
        bag_cfg = BagatelleConfig(
            control_timestep=CONTROL_TIMESTEP,
            threshold=0.5,
            environment_name=ENV_NAME,
            key_press_depth=0.006,
            key_target_front_offset=0.35,
            key_target_top_offset=0.5,
        )
        kin = BagatelleKinematics(
            bag_cfg,
            target_keys=target_keys[:, :NUM_KEYS],
            output_dir=run_dir / "bagatelle_targets",
        )
        key_targets = kin.key_press_targets(np.arange(NUM_KEYS, dtype=np.int32), press_depth=0.006)
        fingertip_targets, active_masks, assignments, assignment_meta = build_proposal_fingertip_targets(
            waypoint_target_keys,
            key_targets,
            threshold=0.5,
            split_key=48,
        )
        proposal_started = time.time()
        waypoint_qpos, proposal_mean_error, proposal_max_error = solve_gpu_proposal_batch(
            fingertip_targets,
            active_masks,
            neutral_qpos=np.asarray(kin.neutral_qpos, dtype=np.float32),
            lower=np.asarray(kin.joint_lower, dtype=np.float32),
            upper=np.asarray(kin.joint_upper, dtype=np.float32),
            device=str(proposal_device),
            batch_size=int(proposal_batch_size),
            passes=int(proposal_passes),
            refinement_steps=int(proposal_refinement_steps),
            refinement_lr=float(proposal_refinement_lr),
        )
        proposal_seconds = float(time.time() - proposal_started)
        proposal_qpos = waypoint_qpos.astype(np.float32, copy=True)
        proposal_assignments = assignments.astype(np.int32, copy=True)
        sparse_verify_metadata = {"enabled": False}
        if bool(proposal_cpu_verify_sparse_ik):
            waypoint_qpos, assignments, sparse_verify_metadata = verify_sparse_waypoints_on_cpu(
                kin=kin,
                waypoint_target_keys=waypoint_target_keys,
                proposal_assignments=proposal_assignments,
                key_targets=key_targets,
                proposal_qpos=proposal_qpos,
                use_bagatelle_assignment=bool(use_bagatelle_assignment),
            )
        cfg = ImpromptuConfig(
            control_timestep=CONTROL_TIMESTEP,
            threshold=0.5,
            environment_name=ENV_NAME,
            trajectory_mode=PRODUCTION_TRAJECTORY_MODE,
            interpolation_substeps=10,
            approach_s=0.055,
            hold_s=0.008,
            release_s=0.025,
            key_press_depth=0.006,
            clearance_height=0.04,
            joint_space_straight_value=0.0,
            joint_space_release_fraction=0.25,
            joint_space_approach_fraction=0.35,
            joint_space_straighten_idle_fingers_at_waypoints=True,
        )
        joint_plan = build_joint_space_straightened_trajectory(
            total_steps=int(target_keys.shape[0]),
            waypoint_frames=waypoint_frames,
            waypoint_release_frames=waypoint_release_frames(target_keys, waypoint_frames, threshold=0.5),
            waypoint_qpos=waypoint_qpos,
            assignments=assignments,
            neutral_qpos=np.asarray(kin.neutral_qpos, dtype=np.float32),
            config=cfg,
            kinematics=kin,
        )
        substeps = max(int(cfg.interpolation_substeps), 1)
        dense = np.asarray(joint_plan.dense_qpos, dtype=np.float32)
        dense_velocities = compute_hand_velocities(dense, control_timestep=CONTROL_TIMESTEP / float(substeps))
        planned = dense[::substeps][: int(target_keys.shape[0])].astype(np.float32, copy=True)
        if planned.shape[0] < target_keys.shape[0]:
            pad = planned[-1:] if planned.size else np.asarray(kin.neutral_qpos, dtype=np.float32).reshape(1, -1)
            planned = np.concatenate([planned, np.repeat(pad, target_keys.shape[0] - planned.shape[0], axis=0)], axis=0)
        planned_velocities = compute_hand_velocities(planned, control_timestep=CONTROL_TIMESTEP)
        segment_ids = np.asarray(joint_plan.segment_ids[::substeps][: int(target_keys.shape[0])], dtype=np.int32)
        if segment_ids.shape[0] < target_keys.shape[0]:
            pad_value = int(segment_ids[-1]) if segment_ids.size else -1
            segment_ids = np.concatenate(
                [segment_ids, np.full((target_keys.shape[0] - segment_ids.shape[0],), pad_value, dtype=np.int32)],
                axis=0,
            )
        payload = {
            "target_keys": np.asarray(target_keys, dtype=np.float32),
            "waypoint_frames": np.asarray(waypoint_frames, dtype=np.int64),
            "waypoint_target_keys": np.asarray(waypoint_target_keys, dtype=np.float32),
            "waypoint_hand_joints": np.asarray(joint_plan.waypoint_qpos, dtype=np.float32),
            "planned_hand_joints": planned.astype(np.float32),
            "planned_hand_velocities": planned_velocities.astype(np.float32),
            "planned_hand_joints_dense": dense.astype(np.float32),
            "planned_hand_velocities_dense": dense_velocities.astype(np.float32),
            "segment_ids": segment_ids.astype(np.int32),
            "segment_ids_dense": np.asarray(joint_plan.segment_ids, dtype=np.int32),
            "ik_anchor_frames_dense": np.asarray(joint_plan.anchor_frames, dtype=np.int64),
            "ik_anchor_frames_control": (np.asarray(joint_plan.anchor_frames, dtype=np.int64) // substeps).astype(np.int64),
            "ik_anchor_qpos": np.asarray(joint_plan.anchor_qpos, dtype=np.float32),
            "proposal_hand_joints": np.asarray(proposal_qpos, dtype=np.float32),
            "proposal_assignments": np.asarray(proposal_assignments, dtype=np.int32),
            "proposal_active_mask": np.asarray(active_masks, dtype=np.float32),
            "proposal_solver_mean_error_m": np.asarray(proposal_mean_error, dtype=np.float32),
            "proposal_solver_max_error_m": np.asarray(proposal_max_error, dtype=np.float32),
            "assignments": np.asarray(assignments, dtype=np.int32),
        }
        atomic_save_npz(run_dir / "trajectory.npz", **payload)
        metadata = {
            "planner": "maestroso_gpu_batch_proposal",
            "trajectory_mode": PRODUCTION_TRAJECTORY_MODE,
            "proposal_engine": "RhapsodyIKSolver.solve_batch",
            "proposal_device": str(proposal_device),
            "cpu_exact_verifier": "retest_impromptu_rp1m_simulator.py",
            "rhapsody_checkpoint": str(RHAPSODY),
            "proposal_batch_size": int(proposal_batch_size),
            "proposal_passes": int(proposal_passes),
            "proposal_refinement_steps": int(proposal_refinement_steps),
            "proposal_refinement_lr": float(proposal_refinement_lr),
            "proposal_cpu_verify_sparse_ik": bool(proposal_cpu_verify_sparse_ik),
            "proposal_cpu_assignment_mode": (
                "bagatelle_previous_pose" if bool(use_bagatelle_assignment) else "fixed_proposal_assignment"
            ),
            "proposal_frames": int(waypoint_frames.size),
            "proposal_active_finger_targets": int(np.count_nonzero(active_masks > 0.5)),
            "target_keys_shape": list(np.asarray(target_keys).shape),
            "num_dense_frames": int(dense.shape[0]),
            "dense_control_timestep": CONTROL_TIMESTEP / float(substeps),
            "environment_name": ENV_NAME,
            "midi_meta": midi_meta,
            "assignment": assignment_meta,
            "sparse_cpu_verification": sparse_verify_metadata,
            "joint_space_trajectory": dict(joint_plan.metadata),
            "target_source": "midi target key activations only",
            "simulation_state_source": "CPU MuJoCo retest restores generated hand states; it does not define piano targets",
            "trajectory_npz": str(run_dir / "trajectory.npz"),
        }
        save_json(run_dir / "metadata.json", metadata)
        result.update(extract_metrics(run_dir))
        result.update(
            {
                "midi_meta": midi_meta,
                "proposal_frames": int(waypoint_frames.size),
                "proposal_active_finger_targets": int(np.count_nonzero(active_masks > 0.5)),
                "dropped_key_count": int(assignment_meta["dropped_key_count"]),
                "proposal_mean_error_m": float(np.mean(proposal_mean_error)) if proposal_mean_error.size else 0.0,
                "proposal_max_error_m": float(np.max(proposal_max_error)) if proposal_max_error.size else 0.0,
                "proposal_seconds": proposal_seconds,
                "sparse_cpu_verify_seconds": sparse_verify_metadata.get("seconds"),
                "sparse_cpu_verify_success_count": sparse_verify_metadata.get("success_count"),
                "sparse_cpu_verify_nfev_sum": sparse_verify_metadata.get("nfev_sum"),
                "sparse_cpu_verify_max_residual_m": sparse_verify_metadata.get("max_residual_m"),
                "plan_seconds": float(time.time() - plan_started),
            }
        )
        attach_proxy(
            result,
            run_dir=run_dir,
            activation_frames=proxy_activation_frames,
            env=env,
            evaluation_stage=evaluation_stage,
        )
        if not skip_rollout:
            retest = run_retest(
                variant_root,
                run_id,
                variant_root / "rp1m_retest",
                log_dir,
                rollout_timeout,
                env,
                render_mp4=bool(render_mp4),
            )
            result["rp1m_retest"] = retest
            if retest:
                result.update(
                    {
                        "event_f1": retest.get("event_f1"),
                        "frame_f1": retest.get("frame_f1"),
                        "matched": retest.get("matched"),
                        "target": retest.get("target"),
                        "played": retest.get("played"),
                        "mispresses": retest.get("mispresses"),
                    }
                )
        result["ok"] = True
    except Exception as exc:
        result.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if kin is not None:
            kin.close()
    result["wall_seconds"] = float(time.time() - started)
    save_json(run_dir / "variant_result.json", result)
    return result


def rank_rows_for_halving(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float]:
        if row.get("event_f1") is not None:
            return (-float(row.get("event_f1") or 0.0), float(row.get("wall_seconds") or 0.0))
        badness = row.get("proxy_badness")
        return (float(badness) if badness is not None else 1.0e9, float(row.get("wall_seconds") or 0.0))

    return sorted(rows, key=key)


def keep_top_variants(rows: list[dict[str, Any]], *, keep: int) -> list[str]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    scored: list[tuple[float, str]] = []
    for variant, items in by_variant.items():
        ranked = rank_rows_for_halving(items)
        values = []
        for item in ranked:
            if item.get("event_f1") is not None:
                values.append(-float(item.get("event_f1") or 0.0))
            else:
                values.append(float(item.get("proxy_badness") if item.get("proxy_badness") is not None else 1.0e9))
        scored.append((float(np.mean(values)) if values else 1.0e9, variant))
    scored.sort(key=lambda item: item[0])
    return [variant for _score, variant in scored[: max(int(keep), 1)]]


def run_variant_once(
    *,
    variant: str,
    song: dict[str, Any],
    output_root: Path,
    max_duration_s: float | None,
    env: dict[str, str],
    args: argparse.Namespace,
    skip_rollout: bool,
    round_profile: str,
    proxy_activation_frames: int,
    evaluation_stage: str,
    render_mp4: bool,
) -> dict[str, Any]:
    if "cem" in variant.lower() or "rl" in variant.lower():
        run_id = str(song["run_id"])
        run_dir = output_root / variant / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "variant": variant,
            "run_id": run_id,
            "song": song,
            "ok": False,
            "error_type": "DisallowedOnlineRepair",
            "error": "Online RL/CEM variants are not allowed inside production planning sweeps.",
            "online_repair_policy": "collect failures for offline training, freeze checkpoint, then rerun planner",
        }
        save_json(run_dir / "variant_result.json", row)
        return row
    if variant == "chunked_gpu_rhapsody":
        return run_chunked_variant(
            song=song,
            output_root=output_root,
            max_duration_s=max_duration_s,
            chunk_seconds=args.chunk_seconds,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
            chunk_workers=args.chunk_workers,
            env=env,
            timeout=args.variant_timeout_s,
            rollout_timeout=args.rollout_timeout_s,
            skip_rollout=bool(skip_rollout),
            proxy_activation_frames=int(proxy_activation_frames),
            evaluation_stage=evaluation_stage,
            render_mp4=bool(render_mp4),
        )
    if variant in ("gpu_batch_proposal", "gpu_batch_proposal_bagatelle_assign"):
        return run_gpu_batch_proposal_variant(
            variant=variant,
            song=song,
            output_root=output_root,
            max_duration_s=max_duration_s,
            env=env,
            rollout_timeout=args.rollout_timeout_s,
            skip_rollout=bool(skip_rollout),
            proposal_batch_size=int(args.proposal_batch_size),
            proposal_passes=int(args.proposal_passes),
            proposal_refinement_steps=int(args.proposal_refinement_steps),
            proposal_refinement_lr=float(args.proposal_refinement_lr),
            proposal_device=str(args.proposal_device),
            proposal_cpu_verify_sparse_ik=not bool(args.no_proposal_cpu_verify_sparse_ik),
            proxy_activation_frames=int(proxy_activation_frames),
            evaluation_stage=evaluation_stage,
            render_mp4=bool(render_mp4),
        )
    return run_plan_variant(
        variant=variant,
        song=song,
        output_root=output_root,
        max_duration_s=max_duration_s,
        env=env,
        timeout=args.variant_timeout_s,
        rollout_timeout=args.rollout_timeout_s,
        skip_rollout=bool(skip_rollout),
        hard_window_max_duration_s=float(args.hard_window_max_duration_s),
        proxy_activation_frames=int(proxy_activation_frames),
        evaluation_stage=evaluation_stage,
        round_profile=str(round_profile),
        render_mp4=bool(render_mp4),
    )


def run_round(
    *,
    label: str,
    variants: list[str],
    songs: list[dict[str, Any]],
    output_root: Path,
    max_duration_s: float | None,
    env: dict[str, str],
    args: argparse.Namespace,
    skip_rollout: bool,
    round_profile: str,
    proxy_activation_frames: int,
    evaluation_stage: str,
    render_mp4: bool,
) -> list[dict[str, Any]]:
    round_root = output_root / label
    round_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for song in songs:
        for variant in variants:
            print(f"=== {label} {song['run_id']} {variant} ===", flush=True)
            row = run_variant_once(
                variant=variant,
                song=song,
                output_root=round_root,
                max_duration_s=max_duration_s,
                env=env,
                args=args,
                skip_rollout=bool(skip_rollout),
                round_profile=round_profile,
                proxy_activation_frames=int(proxy_activation_frames),
                evaluation_stage=evaluation_stage,
                render_mp4=bool(render_mp4),
            )
            row["halving_round"] = label
            row["stage"] = str(evaluation_stage).upper()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            summarize(round_root, rows)
    summarize(round_root, rows)
    return rows


def collect_failure_manifest(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    failure_path = output_root / "offline_training_failures.jsonl"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            proxy = row.get("proxy_metrics") or {}
            if not bool(proxy.get("poor_proxy")) and row.get("event_f1") is not None and float(row.get("event_f1") or 0.0) >= 0.6:
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
                        "proxy_metrics": proxy,
                        "offline_loop": "planner_run_collect_failures_train_policy_offline_freeze_checkpoint",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return failure_path


def run_successive_halving(
    *,
    output_root: Path,
    songs: list[dict[str, Any]],
    variants: list[str],
    env: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage_a_duration = (
        min(float(args.max_duration_s), float(args.round1_max_duration_s))
        if args.max_duration_s is not None
        else float(args.round1_max_duration_s)
    )
    stage_a_rows = run_round(
        label="stage_a_planning",
        variants=variants,
        songs=songs,
        output_root=output_root,
        max_duration_s=stage_a_duration,
        env=env,
        args=args,
        skip_rollout=True,
        round_profile="round1_proxy",
        proxy_activation_frames=0,
        evaluation_stage="A",
        render_mp4=False,
    )
    stage_b_variants = keep_top_variants(stage_a_rows, keep=int(args.round2_keep))
    stage_b_rows = run_round(
        label="stage_b_sparse_activation",
        variants=stage_b_variants,
        songs=songs,
        output_root=output_root,
        max_duration_s=args.max_duration_s,
        env=env,
        args=args,
        skip_rollout=True,
        round_profile="standard",
        proxy_activation_frames=int(args.proxy_activation_frames),
        evaluation_stage="B",
        render_mp4=False,
    )
    stage_c_variants = keep_top_variants(stage_b_rows, keep=int(args.round3_keep))
    stage_c_rows = run_round(
        label="stage_c_dense_rollout",
        variants=stage_c_variants,
        songs=songs,
        output_root=output_root,
        max_duration_s=args.max_duration_s,
        env=env,
        args=args,
        skip_rollout=False,
        round_profile="standard",
        proxy_activation_frames=0,
        evaluation_stage="C",
        render_mp4=False,
    )
    stage_d_variants = keep_top_variants(stage_c_rows, keep=int(args.round3_keep))
    stage_d_rows: list[dict[str, Any]] = []
    if bool(args.round3_render):
        stage_d_rows = run_round(
            label="stage_d_render",
            variants=stage_d_variants,
            songs=songs,
            output_root=output_root,
            max_duration_s=args.max_duration_s,
            env=env,
            args=args,
            skip_rollout=False,
            round_profile="standard",
            proxy_activation_frames=0,
            evaluation_stage="D",
            render_mp4=True,
        )
    all_rows = [*stage_a_rows, *stage_b_rows, *stage_c_rows, *stage_d_rows]
    failure_path = collect_failure_manifest(output_root, all_rows)
    summary = {
        "output_root": str(output_root),
        "successive_halving": True,
        "evaluation_order": [
            "Stage A: planning-only metrics",
            "Stage B: sparse MuJoCo activation checks",
            "Stage C: full dense rollout without video",
            "Stage D: render video for final configs",
        ],
        "stage_a_variants": variants,
        "stage_b_variants": stage_b_variants,
        "stage_c_variants": stage_c_variants,
        "stage_d_variants": stage_d_variants,
        "stage_a_count": len(stage_a_rows),
        "stage_b_count": len(stage_b_rows),
        "stage_c_count": len(stage_c_rows),
        "stage_d_count": len(stage_d_rows),
        "round1_variants": variants,
        "round2_variants": stage_b_variants,
        "round3_variants": stage_c_variants,
        "round1_count": len(stage_a_rows),
        "round2_count": len(stage_b_rows),
        "round3_count": len(stage_c_rows),
        "rows": all_rows,
        "offline_training_failures_jsonl": str(failure_path),
        "online_rl_in_planner": False,
        "online_cem_in_planner": False,
        "repair_policy": "only flagged windows should enter offline failure collection or an explicit repair job",
    }
    save_json(output_root / "successive_halving_summary.json", summary)
    save_json(output_root / "summary.json", summary)
    return summary


def summarize(output_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    ok = [row for row in rows if row.get("ok")]
    scored = [row for row in ok if row.get("event_f1") is not None]
    summary = {
        "output_root": str(output_root),
        "created_at": time.time(),
        "result_count": len(rows),
        "ok_count": len(ok),
        "scored_count": len(scored),
        "rows": rows,
    }
    if scored:
        summary.update(
            {
                "mean_event_f1": float(np.mean([float(row["event_f1"]) for row in scored])),
                "mean_wall_seconds": float(np.mean([float(row["wall_seconds"]) for row in scored])),
                "by_variant": {
                    variant: {
                        "count": len(items),
                        "mean_event_f1": float(np.mean([float(row["event_f1"]) for row in items])),
                        "mean_wall_seconds": float(np.mean([float(row["wall_seconds"]) for row in items])),
                        "mean_plan_seconds": float(np.mean([float(row.get("plan_seconds", 0.0)) for row in items])),
                    }
                    for variant, items in sorted(
                        {
                            row["variant"]: [item for item in scored if item["variant"] == row["variant"]]
                            for row in scored
                        }.items()
                    )
                },
            }
        )
    save_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run accelerated Maestroso full-song planning variants.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--songs", type=int, default=2)
    parser.add_argument("--max-duration-s", type=float, default=45.0)
    parser.add_argument(
        "--variants",
        default=(
            "cpu_fast,gpu_batch_proposal,gpu_batch_proposal_bagatelle_assign,"
            "gpu_rhapsody_seed,gpu_rhapsody_refine,chunked_gpu_rhapsody"
        ),
    )
    parser.add_argument("--variant-timeout-s", type=int, default=900)
    parser.add_argument("--rollout-timeout-s", type=int, default=600)
    parser.add_argument("--chunk-seconds", type=float, default=15.0)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=0.0)
    parser.add_argument("--chunk-workers", type=int, default=1)
    parser.add_argument("--hard-window-max-duration-s", type=float, default=15.0)
    parser.add_argument("--proposal-device", default="cuda")
    parser.add_argument("--proposal-batch-size", type=int, default=512)
    parser.add_argument("--proposal-passes", type=int, default=2)
    parser.add_argument("--proposal-refinement-steps", type=int, default=0)
    parser.add_argument("--proposal-refinement-lr", type=float, default=0.05)
    parser.add_argument("--no-proposal-cpu-verify-sparse-ik", action="store_true")
    parser.add_argument("--proxy-activation-frames", type=int, default=32)
    parser.add_argument("--successive-halving", action="store_true")
    parser.add_argument("--round1-max-duration-s", type=float, default=DEFAULT_HALVING_ROUND1_S)
    parser.add_argument("--round2-keep", "--stage-b-keep", dest="round2_keep", type=int, default=3)
    parser.add_argument("--round3-keep", "--stage-c-keep", dest="round3_keep", type=int, default=1)
    parser.add_argument("--round3-render", "--stage-d-render", dest="round3_render", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    songs = read_maestro(args.songs)
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    save_json(
        output_root / "manifest.json",
        {
            "songs": songs,
            "variants": variants,
            "production_trajectory_mode": PRODUCTION_TRAJECTORY_MODE,
            "hard_window_trajectory_mode": HARD_WINDOW_TRAJECTORY_MODE,
            "hard_window_max_duration_s": args.hard_window_max_duration_s,
            "max_duration_s": args.max_duration_s,
            "chunk_seconds": args.chunk_seconds,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
            "chunk_workers": args.chunk_workers,
            "chunk_parallelism": "process",
            "proposal_device": args.proposal_device,
            "proposal_batch_size": args.proposal_batch_size,
            "proposal_passes": args.proposal_passes,
            "proposal_refinement_steps": args.proposal_refinement_steps,
            "proposal_refinement_lr": args.proposal_refinement_lr,
            "proposal_cpu_verify_sparse_ik": not bool(args.no_proposal_cpu_verify_sparse_ik),
            "proposal_parallelism": "batched_gpu_tensors",
            "exact_verification": "cpu_mujoco_rp1m_retest",
            "proxy_activation_frames": args.proxy_activation_frames,
            "successive_halving": bool(args.successive_halving),
            "stage_a_planning_only": True,
            "stage_a_max_duration_s": args.round1_max_duration_s,
            "stage_b_sparse_activation_frames": args.proxy_activation_frames,
            "stage_b_keep": args.round2_keep,
            "stage_c_dense_rollout_keep": args.round3_keep,
            "stage_d_render": bool(args.round3_render),
            "round1_max_duration_s": args.round1_max_duration_s,
            "round2_keep": args.round2_keep,
            "round3_keep": args.round3_keep,
            "round3_render": bool(args.round3_render),
            "online_rl_in_planner": False,
            "online_cem_in_planner": False,
            "repair_policy": "collect failures, train/update policy offline, freeze checkpoint, rerun planner",
            "rhapsody_checkpoint": str(RHAPSODY),
            "environment_name": ENV_NAME,
        },
    )

    if bool(args.successive_halving):
        summary = run_successive_halving(
            output_root=output_root,
            songs=songs,
            variants=variants,
            env=env,
            args=args,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    results: list[dict[str, Any]] = []
    for song in songs:
        for variant in variants:
            print(f"=== {song['run_id']} {variant} ===", flush=True)
            row = run_variant_once(
                variant=variant,
                song=song,
                output_root=output_root,
                max_duration_s=args.max_duration_s,
                env=env,
                args=args,
                skip_rollout=bool(args.skip_rollout),
                round_profile="standard",
                proxy_activation_frames=int(args.proxy_activation_frames),
                evaluation_stage="B",
                render_mp4=False,
            )
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            summarize(output_root, results)
    collect_failure_manifest(output_root, results)
    summary = summarize(output_root, results)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
