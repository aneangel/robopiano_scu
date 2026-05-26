#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
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

from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.joint_space_trajectory import build_joint_space_straightened_trajectory  # noqa: E402
from intermezzo.keys import extract_waypoint_frames  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rhapsody.solver import RhapsodyIKSolver  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "ROBOPIANO_ARTIFACT_ROOT",
        r"D:\WAVE\datasets\ccoelho_lab-jlanders\robopiano_local_artifacts",
    )
)
DEFAULT_MAESTRO_ROOT = DEFAULT_ARTIFACT_ROOT / "maestro-v3.0.0" / "maestro-v3.0.0"
DEFAULT_RHAPSODY = (
    DEFAULT_ARTIFACT_ROOT
    / "Rhapsody"
    / "random10_heldout2_prev_random_scale6_refine16_20260522"
    / "rhapsody_rpik.pt"
)
DEFAULT_KEY_TARGETS = DEFAULT_ARTIFACT_ROOT / "key_press_targets_006.npy"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "local_gpu_plans"
CONTROL_TIMESTEP = 0.05
HAND_STATE_DIM = 46
NUM_FINGERS = 10
NUM_KEYS = 88
RHAPSODY_Y_OFFSET = 0.08289646


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def target_event_count(target_keys: np.ndarray, threshold: float = 0.5) -> int:
    active = np.asarray(target_keys[:, :NUM_KEYS], dtype=np.float32) > float(threshold)
    if active.shape[0] == 0:
        return 0
    starts = active & np.vstack([np.ones((1, NUM_KEYS), dtype=bool), ~active[:-1]])
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


def build_fingertip_targets(
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
        "assignment_strategy": "local_split_even_finger_spread",
        "split_key": int(split_key),
        "dropped_key_count": int(dropped_total),
        "rows_with_dropped_keys": int(dropped_rows),
        "max_dropped_keys_in_row": int(max_dropped),
    }


def transform_for_rhapsody(targets: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(targets, dtype=np.float32).reshape(-1, NUM_FINGERS, 3)
    mask_values = np.asarray(masks, dtype=np.float32).reshape(-1, NUM_FINGERS)
    transformed = np.empty_like(values, dtype=np.float32)
    transformed[:, :5] = values[:, 5:]
    transformed[:, 5:] = values[:, :5]
    transformed[:, :, 1] += np.float32(RHAPSODY_Y_OFFSET)
    transformed_mask = np.concatenate([mask_values[:, 5:], mask_values[:, :5]], axis=1).astype(np.float32)
    return transformed, transformed_mask


def solve_proposals(
    solver: RhapsodyIKSolver,
    targets: np.ndarray,
    masks: np.ndarray,
    *,
    previous_qpos: np.ndarray,
    batch_size: int,
    passes: int,
    refinement_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(targets.shape[0])
    if count == 0:
        return (
            np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    proposal_targets, proposal_masks = transform_for_rhapsody(targets, masks)
    previous = np.repeat(np.asarray(previous_qpos, dtype=np.float32).reshape(1, -1), count, axis=0)
    qpos = previous.copy()
    mean_error = np.zeros((count,), dtype=np.float32)
    max_error = np.zeros((count,), dtype=np.float32)
    for _ in range(max(int(passes), 1)):
        rows: list[np.ndarray] = []
        mean_rows: list[np.ndarray] = []
        max_rows: list[np.ndarray] = []
        for start in range(0, count, max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), count)
            solution = solver.solve_batch(
                proposal_targets[start:end],
                active_mask=proposal_masks[start:end],
                previous_qpos=previous[start:end],
                refinement_steps=int(refinement_steps),
            )
            rows.append(solution.qpos)
            mean_rows.append(solution.mean_error_m)
            max_rows.append(solution.max_error_m)
        qpos = np.concatenate(rows, axis=0).astype(np.float32)
        mean_error = np.concatenate(mean_rows, axis=0).astype(np.float32)
        max_error = np.concatenate(max_rows, axis=0).astype(np.float32)
        previous = np.repeat(np.asarray(previous_qpos, dtype=np.float32).reshape(1, -1), count, axis=0)
        if count > 1:
            previous[1:] = qpos[:-1]
    return qpos, mean_error, max_error


def read_sample(maestro_root: Path, limit: int, sample_manifest: Path | None = None) -> list[dict[str, Any]]:
    manifest = sample_manifest or (maestro_root / "local_sample10_manifest.json")
    if manifest.exists() and manifest.suffix.lower() == ".json":
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        csv_path = maestro_root / "maestro-v3.0.0.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows[: max(int(limit), 0)], start=1):
        midi_path = maestro_root / str(row["midi_filename"])
        out.append({**row, "run_id": f"local_maestro_{index:04d}", "midi_path": str(midi_path)})
    return out


def plan_one(
    *,
    solver: RhapsodyIKSolver,
    song: dict[str, Any],
    output_root: Path,
    key_targets: np.ndarray,
    max_duration_s: float | None,
    batch_size: int,
    passes: int,
    refinement_steps: int,
    checkpoint: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = output_root / str(song["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    midi_started = time.perf_counter()
    target_keys, midi_meta = load_target_keys_from_midi(
        song["midi_path"],
        control_timestep=CONTROL_TIMESTEP,
        max_duration_s=max_duration_s,
    )
    midi_seconds = time.perf_counter() - midi_started
    waypoint_frames = extract_waypoint_frames(target_keys[:, :NUM_KEYS], threshold=0.5)
    if waypoint_frames.size:
        active = np.asarray(target_keys[:, :NUM_KEYS], dtype=np.float32) > 0.5
        waypoint_frames = np.asarray(waypoint_frames, dtype=np.int64)[np.any(active[waypoint_frames], axis=1)]
    waypoint_target_keys = (
        np.asarray(target_keys[waypoint_frames, :NUM_KEYS], dtype=np.float32)
        if waypoint_frames.size
        else np.zeros((0, NUM_KEYS), dtype=np.float32)
    )
    targets, masks, assignments, assignment_meta = build_fingertip_targets(waypoint_target_keys, key_targets)
    neutral_qpos = solver.normalizer.qpos_mean.detach().cpu().numpy().astype(np.float32)
    proposal_started = time.perf_counter()
    waypoint_qpos, mean_error, max_error = solve_proposals(
        solver,
        targets,
        masks,
        previous_qpos=neutral_qpos,
        batch_size=batch_size,
        passes=passes,
        refinement_steps=refinement_steps,
    )
    proposal_seconds = time.perf_counter() - proposal_started
    cfg = ImpromptuConfig(
        control_timestep=CONTROL_TIMESTEP,
        threshold=0.5,
        environment_name="local_gpu_proposal_no_mujoco",
        trajectory_mode="joint_space_straighten",
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
    joint_started = time.perf_counter()
    joint_plan = build_joint_space_straightened_trajectory(
        total_steps=int(target_keys.shape[0]),
        waypoint_frames=waypoint_frames,
        waypoint_release_frames=waypoint_release_frames(target_keys, waypoint_frames, threshold=0.5),
        waypoint_qpos=waypoint_qpos,
        assignments=assignments,
        neutral_qpos=neutral_qpos,
        config=cfg,
        kinematics=None,
    )
    substeps = max(int(cfg.interpolation_substeps), 1)
    dense = np.asarray(joint_plan.dense_qpos, dtype=np.float32)
    planned = dense[::substeps][: int(target_keys.shape[0])].astype(np.float32, copy=True)
    if planned.shape[0] < target_keys.shape[0]:
        pad = planned[-1:] if planned.size else neutral_qpos.reshape(1, -1)
        planned = np.concatenate([planned, np.repeat(pad, target_keys.shape[0] - planned.shape[0], axis=0)], axis=0)
    dense_vel = compute_hand_velocities(dense, control_timestep=CONTROL_TIMESTEP / float(substeps))
    planned_vel = compute_hand_velocities(planned, control_timestep=CONTROL_TIMESTEP)
    joint_seconds = time.perf_counter() - joint_started
    write_started = time.perf_counter()
    np.savez(
        run_dir / "trajectory.npz",
        target_keys=np.asarray(target_keys, dtype=np.float32),
        waypoint_frames=np.asarray(waypoint_frames, dtype=np.int64),
        waypoint_target_keys=np.asarray(waypoint_target_keys, dtype=np.float32),
        waypoint_hand_joints=np.asarray(joint_plan.waypoint_qpos, dtype=np.float32),
        planned_hand_joints=planned,
        planned_hand_velocities=planned_vel.astype(np.float32),
        planned_hand_joints_dense=dense,
        planned_hand_velocities_dense=dense_vel.astype(np.float32),
        segment_ids_dense=np.asarray(joint_plan.segment_ids, dtype=np.int32),
        ik_anchor_frames_dense=np.asarray(joint_plan.anchor_frames, dtype=np.int64),
        ik_anchor_qpos=np.asarray(joint_plan.anchor_qpos, dtype=np.float32),
        assignments=np.asarray(assignments, dtype=np.int32),
        proposal_active_mask=np.asarray(masks, dtype=np.float32),
        proposal_solver_mean_error_m=np.asarray(mean_error, dtype=np.float32),
        proposal_solver_max_error_m=np.asarray(max_error, dtype=np.float32),
    )
    write_seconds = time.perf_counter() - write_started
    total_seconds = time.perf_counter() - started
    metadata = {
        "planner": "local_gpu_rhapsody_proposal",
        "target_source": "local MAESTRO MIDI target key activations",
        "simulation_used_for_planning": False,
        "online_rollout_included": False,
        "trajectory_mode": "joint_space_straighten",
        "midi_meta": midi_meta,
        "assignment": assignment_meta,
        "rhapsody_checkpoint": str(checkpoint),
        "target_steps": int(target_keys.shape[0]),
        "duration_s": float(target_keys.shape[0]) * CONTROL_TIMESTEP,
        "target_press_events": target_event_count(target_keys),
        "unique_active_keys": int(np.count_nonzero(np.max(target_keys[:, :NUM_KEYS], axis=0) > 0.5)) if target_keys.size else 0,
        "num_waypoints": int(waypoint_frames.size),
        "num_dense_frames": int(dense.shape[0]),
        "proposal_active_finger_targets": int(np.count_nonzero(masks > 0.5)),
        "proposal_mean_error_m": float(np.mean(mean_error)) if mean_error.size else 0.0,
        "proposal_max_error_m": float(np.max(max_error)) if max_error.size else 0.0,
        "timing": {
            "midi_seconds": float(midi_seconds),
            "proposal_seconds": float(proposal_seconds),
            "joint_trajectory_seconds": float(joint_seconds),
            "write_seconds": float(write_seconds),
            "plan_seconds_excluding_rollout": float(total_seconds),
        },
        "joint_space_trajectory": dict(joint_plan.metadata),
        "trajectory_npz": str(run_dir / "trajectory.npz"),
    }
    save_json(run_dir / "metadata.json", metadata)
    return {
        "run_id": song["run_id"],
        "midi_filename": song["midi_filename"],
        "duration_s": metadata["duration_s"],
        "target_press_events": metadata["target_press_events"],
        "num_waypoints": metadata["num_waypoints"],
        "num_dense_frames": metadata["num_dense_frames"],
        "proposal_mean_error_m": metadata["proposal_mean_error_m"],
        "proposal_max_error_m": metadata["proposal_max_error_m"],
        "plan_seconds_excluding_rollout": float(total_seconds),
        "trajectory_npz": str(run_dir / "trajectory.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GTX 1660 Super full-song Rhapsody planning benchmark.")
    parser.add_argument("--songs", type=int, default=1)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--maestro-root", default=str(DEFAULT_MAESTRO_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_RHAPSODY))
    parser.add_argument("--key-targets", default=str(DEFAULT_KEY_TARGETS))
    parser.add_argument("--sample-manifest", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--refinement-steps", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    key_targets = np.load(Path(args.key_targets)).astype(np.float32)
    solver = RhapsodyIKSolver.from_checkpoint(checkpoint, device=str(args.device))
    rows = []
    sample_manifest = Path(args.sample_manifest) if args.sample_manifest else None
    for song in read_sample(Path(args.maestro_root), args.songs, sample_manifest=sample_manifest):
        print(f"=== {song['run_id']} {song['midi_filename']} ===", flush=True)
        row = plan_one(
            solver=solver,
            song=song,
            output_root=output_root,
            key_targets=key_targets,
            max_duration_s=args.max_duration_s,
            batch_size=args.batch_size,
            passes=args.passes,
            refinement_steps=args.refinement_steps,
            checkpoint=checkpoint,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        save_json(output_root / "summary.json", {"rows": rows})
    seconds = [float(row["plan_seconds_excluding_rollout"]) for row in rows]
    summary = {
        "rows": rows,
        "song_count": len(rows),
        "mean_plan_seconds_excluding_rollout": float(np.mean(seconds)) if seconds else 0.0,
        "max_plan_seconds_excluding_rollout": float(np.max(seconds)) if seconds else 0.0,
        "under_60s_count": int(sum(value < 60.0 for value in seconds)),
        "output_root": str(output_root),
    }
    save_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
