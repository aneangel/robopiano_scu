#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import ctypes
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from impromptu.active_window import crop_active_window  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from partita.evaluation.rollout import (  # noqa: E402
    _capture_piano_activation,
    _load_env,
    _locate_task_physics_piano,
    _set_reduced_hand_qpos,
    candidate_environment_names,
    piano_roll_to_midi_events,
    render_frame,
    write_goals_proto,
    write_video,
)


def _preload_conda_libstdcxx() -> None:
    lib = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not lib.is_file():
        return
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render dense Impromptu hand-state playback in RoboPianist.")
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--interpolation-substeps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--active-window-last-s", type=float, default=None)
    parser.add_argument("--active-window-preroll-s", type=float, default=0.5)
    parser.add_argument("--active-window-postroll-s", type=float, default=0.25)
    parser.add_argument(
        "--disable-gravity",
        action="store_true",
        help="Set MuJoCo gravity to zero before dense direct-pose playback.",
    )
    return parser


def _load_dense_payload(path: Path, control_timestep: float, interpolation_substeps: int | None) -> tuple[np.ndarray, np.ndarray, float, int]:
    data = np.load(path, allow_pickle=False)
    if "planned_hand_joints_dense" not in data:
        raise ValueError(f"{path} does not contain planned_hand_joints_dense")
    dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
    target_keys = np.asarray(data["target_keys"], dtype=np.float32) if "target_keys" in data else np.zeros((dense.shape[0], 88), dtype=np.float32)
    if interpolation_substeps is None:
        if target_keys.shape[0] > 0 and dense.shape[0] % target_keys.shape[0] == 0:
            substeps = max(int(dense.shape[0] // target_keys.shape[0]), 1)
        else:
            substeps = 1
    else:
        substeps = max(int(interpolation_substeps), 1)
    if target_keys.shape[0] * substeps == dense.shape[0]:
        target_keys_dense = np.repeat(target_keys[:, :88], substeps, axis=0).astype(np.float32)
    elif target_keys.shape[0] == dense.shape[0]:
        target_keys_dense = target_keys[:, :88].astype(np.float32)
    else:
        raise ValueError(f"Cannot align target_keys shape {target_keys.shape} with dense hand states {dense.shape}")
    dense_dt = float(control_timestep) / float(substeps)
    return dense, target_keys_dense, dense_dt, substeps


def _write_key_count_csv(path: Path, counts: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "frames"])
        for key, value in enumerate(np.asarray(counts, dtype=np.int64).reshape(-1)[:88]):
            writer.writerow([int(key), int(value)])


def _lagged_score(target: np.ndarray, played: np.ndarray, *, lag: int, dt: float, threshold: float) -> dict[str, Any]:
    if lag > 0:
        aligned_target = target[:-lag]
        aligned_played = played[lag:]
    elif lag < 0:
        aligned_target = target[-lag:]
        aligned_played = played[:lag]
    else:
        aligned_target = target
        aligned_played = played
    steps = min(int(aligned_target.shape[0]), int(aligned_played.shape[0]))
    if steps <= 0:
        aligned_target = target[:0]
        aligned_played = played[:0]
    else:
        aligned_target = aligned_target[:steps]
        aligned_played = aligned_played[:steps]
    score = score_rollout(
        target_keys=aligned_target,
        played_keys=aligned_played,
        dt=dt,
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    return {
        "lag_frames": int(lag),
        "lag_s": float(lag) * float(dt),
        "frame_f1": float(score["frame_f1"]),
        "frame_precision": float(score["frame_precision"]),
        "frame_recall": float(score["frame_recall"]),
        "scored_steps": int(score["scored_steps"]),
    }


def render_dense_playback(
    *,
    trajectory_npz: str | Path,
    output_dir: str | Path | None,
    environment_name: str,
    control_timestep: float,
    interpolation_substeps: int | None,
    fps: int,
    width: int,
    height: int,
    seed: int,
    threshold: float,
    active_window_last_s: float | None = None,
    active_window_preroll_s: float = 0.5,
    active_window_postroll_s: float = 0.25,
    disable_gravity: bool = False,
) -> dict[str, Any]:
    npz_path = Path(trajectory_npz).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else npz_path.parent / "render_200fps"
    out_dir.mkdir(parents=True, exist_ok=True)
    hand_states, target_keys_dense, dense_dt, substeps = _load_dense_payload(npz_path, control_timestep, interpolation_substeps)
    active_window = None
    if active_window_last_s is not None:
        crop = crop_active_window(
            target_keys_dense,
            dt=dense_dt,
            threshold=float(threshold),
            active_window_last_s=active_window_last_s,
            active_window_preroll_s=float(active_window_preroll_s),
            active_window_postroll_s=float(active_window_postroll_s),
        )
        hand_states = hand_states[crop.start_frame : crop.end_frame]
        target_keys_dense = crop.target_keys
        active_window = crop.metadata
    else:
        with np.load(npz_path, allow_pickle=False) as data:
            if "active_window_crop_start_frame" in data:
                original_steps = int(np.asarray(data["active_window_original_steps"]).reshape(()))
                cropped_steps = int(np.asarray(data["active_window_cropped_steps"]).reshape(()))
                start_frame = int(np.asarray(data["active_window_crop_start_frame"]).reshape(()))
                end_frame = int(np.asarray(data["active_window_crop_end_frame"]).reshape(()))
                active_window = {
                    "original_steps": original_steps,
                    "cropped_steps": cropped_steps,
                    "crop_start_frame": start_frame,
                    "crop_end_frame": end_frame,
                    "crop_start_s": float(start_frame) * float(control_timestep),
                    "crop_end_s": float(end_frame) * float(control_timestep),
                    "active_window_last_s": None,
                }
    midi_proto = write_goals_proto(target_keys_dense[:, :88], out_dir / "impromptu_dense_target_goals.proto", dt=dense_dt, title="Impromptu dense playback render")
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(environment_name),
        midi_proto_path=midi_proto,
        control_timestep=dense_dt,
        seed=int(seed),
        reduced_action_space=True,
        extra_task_kwargs={
            "disable_forearm_reward": True,
            "disable_fingering_reward": True,
            "disable_colorization": True,
            "disable_hand_collisions": False,
            "wrong_press_termination": False,
        },
        suite_load_kwargs=None,
        prefer_canonical_midi=False,
    )
    played_roll: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    render_error = None
    gravity_error = None
    terminated = False
    restored_count = 0
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        if bool(disable_gravity):
            try:
                physics.model.opt.gravity[:] = 0.0
                if hasattr(physics, "forward"):
                    physics.forward()
            except Exception as exc:
                gravity_error = str(exc)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        for hand_state in hand_states:
            restored_count = _set_reduced_hand_qpos(task, physics, hand_state)
            if hasattr(physics, "forward"):
                physics.forward()
            timestep = env.step(zero_action)
            update_key_state = getattr(piano, "_update_key_state", None)
            if callable(update_key_state):
                update_key_state(physics)
            update_key_color = getattr(piano, "_update_key_color", None)
            if callable(update_key_color):
                update_key_color(physics)
            activation = _capture_piano_activation(env)
            if activation is None:
                activation = np.asarray(getattr(piano, "activation"), dtype=np.float32).reshape(-1)[:88]
            played_roll.append(np.asarray(activation[:88], dtype=np.float32))
            if render_error is None:
                try:
                    frames.append(render_frame(env, height=int(height), width=int(width)))
                except Exception as exc:
                    render_error = str(exc)
            if timestep.last():
                terminated = True
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    played = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
    steps = min(int(played.shape[0]), int(target_keys_dense.shape[0]))
    rollout_path = out_dir / "dense_playback_rollout.npz"
    atomic_save_npz(rollout_path, target_keys=target_keys_dense[:steps], played_keys=played[:steps])
    score = score_rollout(
        target_keys=target_keys_dense[:steps],
        played_keys=played[:steps],
        dt=dense_dt,
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    atomic_save_json(out_dir / "score.json", score)
    target_binary = target_keys_dense[:steps, :88] > float(threshold)
    played_binary = played[:steps, :88] > float(threshold)
    fp_by_key = np.logical_and(~target_binary, played_binary).sum(axis=0).astype(np.int64)
    fn_by_key = np.logical_and(target_binary, ~played_binary).sum(axis=0).astype(np.int64)
    _write_key_count_csv(out_dir / "fp_by_key.csv", fp_by_key)
    _write_key_count_csv(out_dir / "fn_by_key.csv", fn_by_key)
    threshold_sweep = {
        "thresholds": [
            {
                "threshold": float(value),
                **score_rollout(
                    target_keys=target_keys_dense[:steps],
                    played_keys=played[:steps],
                    dt=dense_dt,
                    threshold=float(value),
                    timing_tolerance_s=0.15,
                ),
            }
            for value in (0.3, 0.4, 0.5, 0.6, 0.7)
        ]
    }
    atomic_save_json(out_dir / "threshold_sweep.json", threshold_sweep)
    lag_rows = [
        _lagged_score(target_keys_dense[:steps], played[:steps], lag=lag, dt=dense_dt, threshold=float(threshold))
        for lag in range(-20, 21)
    ]
    best_lag = max(lag_rows, key=lambda row: (row["frame_f1"], row["frame_precision"], row["frame_recall"])) if lag_rows else None
    lag_sweep = {
        "lags": lag_rows,
        "best": best_lag,
        "best_frame_f1": None if best_lag is None else float(best_lag["frame_f1"]),
        "best_precision": None if best_lag is None else float(best_lag["frame_precision"]),
        "best_recall": None if best_lag is None else float(best_lag["frame_recall"]),
        "best_lag_s": None if best_lag is None else float(best_lag["lag_s"]),
    }
    atomic_save_json(out_dir / "lag_sweep.json", lag_sweep)
    active_window_summary = active_window or {
        "original_steps": int(target_keys_dense.shape[0]),
        "cropped_steps": int(target_keys_dense.shape[0]),
        "crop_start_frame": 0,
        "crop_end_frame": int(target_keys_dense.shape[0]),
        "crop_start_s": 0.0,
        "crop_end_s": float(target_keys_dense.shape[0]) * float(dense_dt),
        "active_window_last_s": None,
    }
    atomic_save_json(out_dir / "active_window_summary.json", active_window_summary)
    events = piano_roll_to_midi_events(played[:steps], dt=dense_dt, threshold=float(threshold))
    video_path = None
    video_format = None
    video_audio_warning = None
    if render_error is None and frames:
        video_path, video_format, video_audio_warning = write_video(frames, out_dir / "rollout_video.mp4", fps=int(fps), audio_events=events)
    summary: dict[str, Any] = {
        "trajectory_npz": str(npz_path),
        "run_dir": str(out_dir),
        "rollout_npz": str(rollout_path),
        "midi_proto_path": str(midi_proto),
        "environment_name": env_name,
        "load_info": load_info,
        "dense_control_timestep": dense_dt,
        "interpolation_substeps": int(substeps),
        "fps": int(fps),
        "video_path": str(video_path) if video_path is not None else None,
        "video_format": video_format,
        "video_audio_warning": video_audio_warning,
        "render_error": render_error,
        "disable_gravity": bool(disable_gravity),
        "gravity_error": gravity_error,
        "attraction_forces": "none_in_direct_hand_state_playback",
        "rendered_frames": int(len(frames)),
        "terminated": bool(terminated),
        "restored_hand_joint_count": int(restored_count),
        "score": score,
        "active_window": active_window_summary,
        "score_json": str(out_dir / "score.json"),
        "fp_by_key_csv": str(out_dir / "fp_by_key.csv"),
        "fn_by_key_csv": str(out_dir / "fn_by_key.csv"),
        "threshold_sweep_json": str(out_dir / "threshold_sweep.json"),
        "lag_sweep_json": str(out_dir / "lag_sweep.json"),
    }
    atomic_save_json(out_dir / "render_summary.json", summary)
    return summary


def main() -> None:
    _preload_conda_libstdcxx()
    args = build_parser().parse_args()
    summary = render_dense_playback(
        trajectory_npz=args.trajectory_npz,
        output_dir=args.output_dir,
        environment_name=str(args.environment_name),
        control_timestep=float(args.control_timestep),
        interpolation_substeps=args.interpolation_substeps,
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        seed=int(args.seed),
        threshold=float(args.threshold),
        active_window_last_s=args.active_window_last_s,
        active_window_preroll_s=float(args.active_window_preroll_s),
        active_window_postroll_s=float(args.active_window_postroll_s),
        disable_gravity=bool(args.disable_gravity),
    )
    print(f"Wrote Impromptu dense playback render: {summary.get('video_path')}")
    print(
        "matched="
        f"{summary['score']['matched_press_events']}/{summary['score']['target_press_events']} "
        f"missed={summary['score']['missed_key_presses']} "
        f"mispresses={summary['score']['mispresses']}"
    )


if __name__ == "__main__":
    main()
