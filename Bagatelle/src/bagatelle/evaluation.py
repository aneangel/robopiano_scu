from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bagatelle.config import BagatelleConfig
from bagatelle.paths import ensure_repo_paths

ensure_repo_paths()
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.keys import validate_target_keys  # noqa: E402
from intermezzo.online_eval import RolloutConfig, score_rollout  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/evaluation")


def load_trajectory_npz(path: str | Path) -> dict[str, np.ndarray]:
    trajectory_path = Path(path).expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Bagatelle trajectory NPZ not found: {trajectory_path}")
    data = np.load(trajectory_path, allow_pickle=False)
    required = ["target_keys", "planned_hand_joints"]
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Trajectory NPZ is missing required arrays: {missing}")
    return {name: np.asarray(data[name]) for name in data.files}


def fingertip_summary_from_trajectory(payload: dict[str, np.ndarray], *, success_threshold: float) -> dict[str, Any]:
    targets = np.asarray(payload.get("fingertip_targets", np.zeros((0, 10, 3))), dtype=np.float32)
    measured = np.asarray(payload.get("waypoint_fingertips", np.zeros((0, 10, 3))), dtype=np.float32)
    if targets.shape != measured.shape or targets.ndim != 3:
        return {"fingertip_note": f"Cannot compare fingertip targets {targets.shape} and measured {measured.shape}."}
    mask = np.isfinite(targets).all(axis=2)
    if not np.any(mask):
        return {
            "fingertip_assignments": 0,
            "fingertip_distance_mean": None,
            "fingertip_distance_median": None,
            "fingertip_distance_p95": None,
            "fingertip_distance_max": None,
            "fingertip_width_distance_mean": None,
            "fingertip_width_distance_median": None,
            "fingertip_width_distance_p95": None,
            "fingertip_width_distance_max": None,
            "fingertip_success_rate": None,
            "fingertip_width_success_rate": None,
            "fingertip_success_threshold_m": float(success_threshold),
        }
    distances = np.linalg.norm(measured[mask] - targets[mask], axis=1)
    # In RoboPianist/Bagatelle key pitch runs along world y; x is front/back.
    width_distances = np.abs(measured[mask, 1] - targets[mask, 1])
    return {
        "fingertip_assignments": int(distances.size),
        "fingertip_distance_mean": float(np.mean(distances)),
        "fingertip_distance_median": float(np.median(distances)),
        "fingertip_distance_p95": float(np.percentile(distances, 95)),
        "fingertip_distance_max": float(np.max(distances)),
        "fingertip_width_distance_mean": float(np.mean(width_distances)),
        "fingertip_width_distance_median": float(np.median(width_distances)),
        "fingertip_width_distance_p95": float(np.percentile(width_distances, 95)),
        "fingertip_width_distance_max": float(np.max(width_distances)),
        "fingertip_success_rate": float(np.mean(distances <= float(success_threshold))),
        "fingertip_width_success_rate": float(np.mean(width_distances <= float(success_threshold))),
        "fingertip_success_threshold_m": float(success_threshold),
    }


def _refresh_piano_activation(piano: Any, physics: Any) -> None:
    update_key_state = getattr(piano, "_update_key_state", None)
    if callable(update_key_state):
        update_key_state(physics)
    update_key_color = getattr(piano, "_update_key_color", None)
    if callable(update_key_color):
        update_key_color(physics)


def rollout_bagatelle_hand_targets_headless(
    *,
    hand_targets: np.ndarray,
    target_keys: np.ndarray,
    output_dir: str | Path,
    label: str,
    config: RolloutConfig,
    settle_steps: int = 0,
) -> dict[str, Any]:
    from partita.evaluation.rollout import (
        _capture_piano_activation,
        _load_env,
        _locate_task_physics_piano,
        _set_reduced_hand_qpos,
        candidate_environment_names,
        write_goals_proto,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(hand_targets, dtype=np.float32)
    keys = validate_target_keys(target_keys)
    if targets.ndim != 2:
        raise ValueError(f"hand_targets must be [T, joints], got {targets.shape}")
    steps = min(int(targets.shape[0]), int(keys.shape[0]))

    midi_proto = write_goals_proto(
        keys[:steps],
        output / f"{label}_target_goals.proto",
        dt=float(config.control_timestep),
        title=f"Bagatelle online eval {label}",
    )
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(config.environment_name),
        midi_proto_path=midi_proto,
        control_timestep=float(config.control_timestep),
        seed=int(config.seed),
        reduced_action_space=bool(config.reduced_action_space),
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

    piano_roll: list[np.ndarray] = []
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        restored_hand_joint_count = 0
        pose_frames_applied = 0
        terminated = False
        for step in range(steps):
            restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, targets[step])
            if hasattr(physics, "forward"):
                physics.forward()
            for _ in range(max(int(settle_steps), 0)):
                restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, targets[step])
                if hasattr(physics, "step"):
                    physics.step()
                elif hasattr(physics, "forward"):
                    physics.forward()
            if hasattr(physics, "forward"):
                physics.forward()
            _refresh_piano_activation(piano, physics)
            pose_frames_applied += 1
            activation = _capture_piano_activation(env)
            if activation is not None:
                piano_roll.append(np.asarray(activation[:88], dtype=np.float32))
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    played = np.stack(piano_roll, axis=0) if piano_roll else np.zeros((0, 88), dtype=np.float32)
    score = score_rollout(
        target_keys=keys[: played.shape[0]],
        played_keys=played,
        dt=float(config.control_timestep),
        threshold=float(config.threshold),
        timing_tolerance_s=float(config.timing_tolerance_s),
    )
    atomic_save_npz(output / f"{label}_online_rollout.npz", target_keys=keys[: played.shape[0]], played_keys=played)
    result = {
        "label": label,
        "environment_name": env_name,
        "midi_proto_path": str(midi_proto),
        "load_info": load_info,
        "hand_targets_shape": list(targets.shape),
        "target_keys_shape": list(keys.shape),
        "played_keys_shape": list(played.shape),
        "control_mode": "direct_hand_qpos_pose_injection_with_settle",
        "settle_steps": int(settle_steps),
        "action_dim": None,
        "mapped_actuators": 0,
        "unmapped_actuators": 0,
        "actions_executed": 0,
        "pose_frames_applied": int(pose_frames_applied),
        "restored_hand_joint_count": int(restored_hand_joint_count),
        "terminated": bool(terminated),
        "rollout_config": {
            "control_timestep": float(config.control_timestep),
            "threshold": float(config.threshold),
            "timing_tolerance_s": float(config.timing_tolerance_s),
            "seed": int(config.seed),
            "environment_name": str(config.environment_name),
            "reduced_action_space": bool(config.reduced_action_space),
        },
        "score": score,
    }
    atomic_save_json(output / f"{label}_online_rollout.json", result)
    return result


def evaluate_bagatelle_trajectory(
    trajectory_npz: str | Path,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_name: str | None = None,
    label: str = "bagatelle",
    config: BagatelleConfig | None = None,
    timing_tolerance_s: float = 0.15,
) -> dict[str, Any]:
    cfg = config or BagatelleConfig()
    payload = load_trajectory_npz(trajectory_npz)
    run_dir = create_unique_run_dir(output_root, run_name=run_name, prefix="bagatelle_eval")
    rollout = rollout_bagatelle_hand_targets_headless(
        hand_targets=np.asarray(payload["planned_hand_joints"], dtype=np.float32),
        target_keys=np.asarray(payload["target_keys"], dtype=np.float32),
        output_dir=run_dir,
        label=label,
        config=RolloutConfig(
            control_timestep=float(cfg.control_timestep),
            threshold=float(cfg.threshold),
            timing_tolerance_s=float(timing_tolerance_s),
            seed=int(cfg.seed),
            environment_name=str(cfg.environment_name),
            reduced_action_space=bool(cfg.reduced_action_space),
        ),
        settle_steps=int(cfg.settle_steps),
    )
    summary = {
        "run_dir": str(run_dir),
        "trajectory_npz": str(Path(trajectory_npz).expanduser().resolve()),
        "label": str(label),
        "config": cfg.to_dict(),
        "timing_tolerance_s": float(timing_tolerance_s),
        "target_keys_shape": list(payload["target_keys"].shape),
        "planned_hand_joints_shape": list(payload["planned_hand_joints"].shape),
        "fingertips": fingertip_summary_from_trajectory(
            payload,
            success_threshold=float(cfg.residual_success_threshold),
        ),
        "rollout": rollout,
    }
    atomic_save_json(run_dir / f"{label}_evaluation.json", summary)
    return summary
