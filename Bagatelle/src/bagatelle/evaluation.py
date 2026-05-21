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


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] <= 1 or dt <= 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return np.gradient(array, float(dt), axis=0).astype(np.float32)


def _resolve_env_dt(env: Any, *, fallback_dt: float) -> float:
    for attr in ("control_timestep", "dt", "timestep", "physics_timestep"):
        value = getattr(env, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is None:
            continue
        try:
            dt = float(value)
        except (TypeError, ValueError):
            continue
        if dt > 0.0:
            return dt
    return float(fallback_dt)


def _build_reference_schedule(
    *,
    horizon: int,
    trajectory_dt: float,
    env_dt: float,
    max_steps: int | None,
) -> np.ndarray:
    if horizon <= 0:
        return np.zeros(0, dtype=np.int32)
    if trajectory_dt <= 0.0 or env_dt <= 0.0:
        steps = horizon if max_steps is None else min(horizon, int(max_steps))
        return np.arange(max(steps, 0), dtype=np.int32)
    total_steps = int(np.ceil(horizon * trajectory_dt / env_dt))
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    sim_times = np.arange(total_steps, dtype=np.float32) * float(env_dt)
    indices = np.floor(sim_times / float(trajectory_dt)).astype(np.int32)
    return np.clip(indices, 0, horizon - 1)


def _align_reference(reference: np.ndarray, reference_indices: np.ndarray | None) -> np.ndarray:
    array = np.asarray(reference, dtype=np.float32)
    if reference_indices is None:
        return array
    if array.shape[0] == 0:
        return array
    indices = np.asarray(reference_indices, dtype=np.int64).reshape(-1)
    indices = np.clip(indices, 0, array.shape[0] - 1)
    return array[indices]


def _array_error_metrics(observed: np.ndarray, reference: np.ndarray, *, prefix: str) -> dict[str, float]:
    current = np.asarray(observed, dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    steps = min(int(current.shape[0]), int(target.shape[0]))
    if steps == 0:
        return {}
    current = current[:steps]
    target = target[:steps]
    error = current - target
    l2 = np.linalg.norm(error.reshape(steps, -1), axis=1)
    abs_error = np.abs(error)
    return {
        f"{prefix}_steps": int(steps),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(error)))),
        f"{prefix}_mean_abs": float(np.mean(abs_error)),
        f"{prefix}_max_abs": float(np.max(abs_error)),
        f"{prefix}_mean_l2": float(np.mean(l2)),
        f"{prefix}_final_l2": float(l2[-1]),
    }


def _action_summary(actions: np.ndarray, action_spec: Any) -> dict[str, Any]:
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[0] == 0:
        return {
            "actions_executed": 0,
            "action_abs_mean": None,
            "action_abs_p95": None,
            "action_abs_max": None,
            "action_bound_hit_rate": None,
        }
    low = np.asarray(getattr(action_spec, "minimum", np.full(action_array.shape[1], -np.inf)), dtype=np.float32).reshape(-1)
    high = np.asarray(getattr(action_spec, "maximum", np.full(action_array.shape[1], np.inf)), dtype=np.float32).reshape(-1)
    low = low[: action_array.shape[1]]
    high = high[: action_array.shape[1]]
    finite_low = np.isfinite(low)
    finite_high = np.isfinite(high)
    hits = np.zeros(action_array.shape, dtype=bool)
    if finite_low.any():
        hits[:, finite_low] |= np.isclose(action_array[:, finite_low], low[finite_low], atol=1e-6)
    if finite_high.any():
        hits[:, finite_high] |= np.isclose(action_array[:, finite_high], high[finite_high], atol=1e-6)
    abs_actions = np.abs(action_array)
    return {
        "actions_executed": int(action_array.shape[0]),
        "action_abs_mean": float(np.mean(abs_actions)),
        "action_abs_p95": float(np.percentile(abs_actions, 95)),
        "action_abs_max": float(np.max(abs_actions)),
        "action_bound_hit_rate": float(np.mean(np.any(hits, axis=1))),
    }


def metric_validity_for_rollout(
    rollout: dict[str, Any],
    *,
    target_keys: np.ndarray | None = None,
    played_keys: np.ndarray | None = None,
) -> dict[str, bool]:
    control_mode = str(rollout.get("control_mode", ""))
    action_dim = rollout.get("action_dim")
    mapped_actuators = int(rollout.get("mapped_actuators") or 0)
    actions_executed = int(rollout.get("actions_executed") or 0)
    pose_frames = int(rollout.get("pose_frames_applied") or 0)
    is_actuated = bool(rollout.get("is_actuated_playback", False))
    target_copied = False
    if target_keys is not None and played_keys is not None:
        target = np.asarray(target_keys, dtype=np.float32)
        played = np.asarray(played_keys, dtype=np.float32)
        steps = min(int(target.shape[0]), int(played.shape[0]))
        if steps > 0 and target.shape[1:] == played.shape[1:]:
            target_slice = target[:steps]
            played_slice = played[:steps]
            has_piano_roll_content = bool(np.any(target_slice) or np.any(played_slice))
            target_copied = has_piano_roll_content and bool(np.array_equal(target_slice, played_slice))
    action_space_present = action_dim is not None and int(action_dim) > 0
    return {
        "is_actuated_playback": is_actuated,
        "actions_executed_positive": actions_executed > 0,
        "action_space_present": action_space_present,
        "mapped_actuators_or_action_dim_positive": mapped_actuators > 0 or action_space_present,
        "no_direct_pose_injection_in_scoring_loop": "direct_hand_qpos" not in control_mode and pose_frames == 0,
        "target_not_copied_to_played_roll": not target_copied,
        "controller_input_uses_colorization": bool(rollout.get("controller_input_uses_colorization", False)),
        "no_colorization_signal_used_as_control_input": not bool(rollout.get("controller_input_uses_colorization", False)),
    }


def _make_pd_controller(controller_family: str, mapping: Any, *, kp: float, kd: float, lookahead_steps: int) -> Any:
    family = str(controller_family).lower()
    if family == "pd":
        from etude.controllers.pd import PDController

        return PDController(mapping, kp=float(kp), kd=float(kd), lookahead=int(lookahead_steps), clip=True)
    if family in {"scheduled_pd", "pd_scheduled"}:
        from etude.controllers.pd_scheduled import ScheduledPDController

        return ScheduledPDController(
            mapping,
            kp=float(kp),
            kd=float(kd),
            mode="scalar",
            lookahead_steps=int(lookahead_steps),
            action_clip=True,
        )
    raise ValueError(f"Unsupported Bagatelle actuated controller family: {controller_family}")


def rollout_bagatelle_actuated_headless(
    *,
    hand_targets: np.ndarray,
    target_keys: np.ndarray,
    output_dir: str | Path,
    label: str,
    config: RolloutConfig,
    controller_family: str = "pd",
    kp: float = 12.0,
    kd: float = 0.6,
    lookahead_steps: int = 1,
    max_steps: int | None = None,
) -> dict[str, Any]:
    from etude.robopianist.observation import extract_tracking_observation
    from etude.robopianist.state_mapping import resolve_mapping_from_env
    from partita.evaluation.rollout import (
        _capture_piano_activation,
        _load_env,
        _locate_task_physics_piano,
        candidate_environment_names,
        write_goals_proto,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(hand_targets, dtype=np.float32)
    keys = validate_target_keys(target_keys)
    if targets.ndim != 2:
        raise ValueError(f"hand_targets must be [T, joints], got {targets.shape}")
    if targets.shape[1] != 46:
        raise ValueError(f"Actuated PD playback requires 46D hand targets, got {targets.shape}")
    steps = min(int(targets.shape[0]), int(keys.shape[0]))

    midi_proto = write_goals_proto(
        keys[:steps],
        output / f"{label}_target_goals.proto",
        dt=float(config.control_timestep),
        title=f"Bagatelle actuated eval {label}",
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
    actions: list[np.ndarray] = []
    observed_q: list[np.ndarray] = []
    observed_qdot: list[np.ndarray] = []
    observed_fingertips: list[np.ndarray] = []
    reference_indices: list[int] = []
    terminated = False
    action_spec = None
    controller = None
    env_dt = float(config.control_timestep)
    try:
        time_step = env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        del task
        action_spec = env.action_spec()
        mapping = resolve_mapping_from_env(env)
        action_dim = int(mapping.action_dim)
        controller = _make_pd_controller(
            controller_family,
            mapping,
            kp=float(kp),
            kd=float(kd),
            lookahead_steps=int(lookahead_steps),
        )
        q_ref = targets[:steps]
        qdot_ref = _finite_difference(q_ref, float(config.control_timestep))
        env_dt = _resolve_env_dt(env, fallback_dt=float(config.control_timestep))
        schedule = _build_reference_schedule(
            horizon=steps,
            trajectory_dt=float(config.control_timestep),
            env_dt=float(env_dt),
            max_steps=max_steps,
        )
        controller.reset(q_ref, qdot_ref, metadata={"dt": float(config.control_timestep), "target_keys": keys[:steps]})
        for ref_t in schedule:
            obs = extract_tracking_observation(time_step.observation, mapping, include_key_state=False)
            action = np.asarray(controller.act(obs, int(ref_t)), dtype=np.float32).reshape(-1)
            time_step = env.step(action)
            actions.append(action)
            reference_indices.append(int(ref_t))
            observed_q.append(np.asarray(obs["q"], dtype=np.float32).reshape(-1))
            observed_qdot.append(np.asarray(obs["qdot"], dtype=np.float32).reshape(-1))
            if "fingertips" in obs:
                observed_fingertips.append(np.asarray(obs["fingertips"], dtype=np.float32).reshape(-1))
            _refresh_piano_activation(piano, physics)
            activation = _capture_piano_activation(env)
            if activation is not None:
                piano_roll.append(np.asarray(activation[:88], dtype=np.float32))
            if getattr(time_step, "last", lambda: False)():
                terminated = True
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    played = np.stack(piano_roll, axis=0) if piano_roll else np.zeros((0, 88), dtype=np.float32)
    actions_array = np.stack(actions, axis=0) if actions else np.zeros((0, 0), dtype=np.float32)
    ref_indices_array = np.asarray(reference_indices[: played.shape[0]], dtype=np.int32)
    target_scoring = _align_reference(keys[:steps], ref_indices_array)[: played.shape[0]]
    score = score_rollout(
        target_keys=target_scoring,
        played_keys=played,
        dt=float(env_dt),
        threshold=float(config.threshold),
        timing_tolerance_s=float(config.timing_tolerance_s),
    )
    q_observed_array = np.stack(observed_q, axis=0) if observed_q else np.zeros((0, 46), dtype=np.float32)
    qdot_observed_array = np.stack(observed_qdot, axis=0) if observed_qdot else np.zeros((0, 46), dtype=np.float32)
    q_ref_aligned = _align_reference(targets[:steps], np.asarray(reference_indices, dtype=np.int32))[: q_observed_array.shape[0]]
    qdot_ref_aligned = _finite_difference(q_ref_aligned, float(env_dt)) if q_ref_aligned.size else np.zeros_like(qdot_observed_array)
    tracking = _array_error_metrics(q_observed_array, q_ref_aligned, prefix="tracking_qpos")
    tracking.update(_array_error_metrics(qdot_observed_array, qdot_ref_aligned, prefix="tracking_qvel"))
    if observed_fingertips:
        tracking["fingertip_tracking_frames_observed"] = int(len(observed_fingertips))

    if action_spec is None:
        action_dim = 0
        mapped_actuators = 0
        action_summary = _action_summary(actions_array, type("Spec", (), {"minimum": [], "maximum": []})())
    else:
        action_dim = int(np.asarray(getattr(action_spec, "minimum", np.zeros(actions_array.shape[1] if actions_array.ndim == 2 else 0))).size)
        mapped_actuators = int(action_dim)
        action_summary = _action_summary(actions_array, action_spec)

    controller_diagnostics: dict[str, Any] = {}
    diagnostics = getattr(controller, "diagnostics", None)
    if callable(diagnostics):
        controller_diagnostics = dict(diagnostics())

    npz_payload = {
        "target_keys": target_scoring,
        "played_keys": played,
        "actions": actions_array,
        "reference_indices": np.asarray(reference_indices, dtype=np.int32),
        "q_observed": q_observed_array,
        "q_reference": q_ref_aligned,
    }
    atomic_save_npz(output / f"{label}_actuated_rollout.npz", **npz_payload)
    result = {
        "label": label,
        "environment_name": env_name,
        "midi_proto_path": str(midi_proto),
        "load_info": load_info,
        "hand_targets_shape": list(targets.shape),
        "target_keys_shape": list(keys.shape),
        "played_keys_shape": list(played.shape),
        "control_mode": f"actuated_env_step_{controller_family}",
        "controller_family": str(controller_family),
        "controller_input_uses_colorization": False,
        "is_actuated_playback": True,
        "action_dim": int(action_dim),
        "mapped_actuators": int(mapped_actuators),
        "unmapped_actuators": 0,
        "pose_frames_applied": 0,
        "terminated": bool(terminated),
        "trajectory_dt_s": float(config.control_timestep),
        "env_dt_s": float(env_dt),
        "rollout_config": {
            "control_timestep": float(config.control_timestep),
            "threshold": float(config.threshold),
            "timing_tolerance_s": float(config.timing_tolerance_s),
            "seed": int(config.seed),
            "environment_name": str(config.environment_name),
            "reduced_action_space": bool(config.reduced_action_space),
            "max_steps": None if max_steps is None else int(max_steps),
            "kp": float(kp),
            "kd": float(kd),
            "lookahead_steps": int(lookahead_steps),
        },
        **action_summary,
        "tracking": tracking,
        "controller_diagnostics": controller_diagnostics,
        "score": score,
    }
    result["metric_validity"] = metric_validity_for_rollout(result, target_keys=target_scoring, played_keys=played)
    atomic_save_json(output / f"{label}_actuated_rollout.json", result)
    return result


def evaluate_bagatelle_actuated_trajectory(
    trajectory_npz: str | Path,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_name: str | None = None,
    label: str = "bagatelle",
    config: BagatelleConfig | None = None,
    controller_family: str = "pd",
    kp: float = 12.0,
    kd: float = 0.6,
    lookahead_steps: int = 1,
    timing_tolerance_s: float = 0.15,
    max_steps: int | None = None,
    include_pose_replay_diagnostic: bool = True,
) -> dict[str, Any]:
    cfg = config or BagatelleConfig()
    payload = load_trajectory_npz(trajectory_npz)
    run_dir = create_unique_run_dir(output_root, run_name=run_name, prefix="bagatelle_actuated_eval")
    rollout_config = RolloutConfig(
        control_timestep=float(cfg.control_timestep),
        threshold=float(cfg.threshold),
        timing_tolerance_s=float(timing_tolerance_s),
        seed=int(cfg.seed),
        environment_name=str(cfg.environment_name),
        reduced_action_space=bool(cfg.reduced_action_space),
    )
    pose_replay = None
    if include_pose_replay_diagnostic:
        pose_replay = rollout_bagatelle_hand_targets_headless(
            hand_targets=np.asarray(payload["planned_hand_joints"], dtype=np.float32),
            target_keys=np.asarray(payload["target_keys"], dtype=np.float32),
            output_dir=run_dir,
            label=f"{label}_diagnostic",
            config=rollout_config,
            settle_steps=int(cfg.settle_steps),
        )
    actuated = rollout_bagatelle_actuated_headless(
        hand_targets=np.asarray(payload["planned_hand_joints"], dtype=np.float32),
        target_keys=np.asarray(payload["target_keys"], dtype=np.float32),
        output_dir=run_dir,
        label=label,
        config=rollout_config,
        controller_family=str(controller_family),
        kp=float(kp),
        kd=float(kd),
        lookahead_steps=int(lookahead_steps),
        max_steps=max_steps,
    )
    summary = {
        "run_dir": str(run_dir),
        "trajectory_npz": str(Path(trajectory_npz).expanduser().resolve()),
        "label": str(label),
        "config": cfg.to_dict(),
        "controller_family": str(controller_family),
        "timing_tolerance_s": float(timing_tolerance_s),
        "target_keys_shape": list(payload["target_keys"].shape),
        "planned_hand_joints_shape": list(payload["planned_hand_joints"].shape),
        "fingertips": fingertip_summary_from_trajectory(
            payload,
            success_threshold=float(cfg.residual_success_threshold),
        ),
        "pose_replay_diagnostics": pose_replay,
        "actuated_rollout_metrics": actuated,
        "metric_validity": {
            "pose_replay_is_diagnostic_only": pose_replay is None or not bool(pose_replay.get("is_actuated_playback", False)),
            "actuated_passes_leakage_checks": all(
                value
                for key, value in actuated.get("metric_validity", {}).items()
                if key != "controller_input_uses_colorization"
            ),
        },
    }
    atomic_save_json(run_dir / f"{label}_actuated_evaluation.json", summary)
    return summary


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
        title=f"Bagatelle diagnostic pose replay {label}",
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
    atomic_save_npz(output / f"{label}_diagnostic_pose_replay.npz", target_keys=keys[: played.shape[0]], played_keys=played)
    result = {
        "label": label,
        "environment_name": env_name,
        "midi_proto_path": str(midi_proto),
        "load_info": load_info,
        "hand_targets_shape": list(targets.shape),
        "target_keys_shape": list(keys.shape),
        "played_keys_shape": list(played.shape),
        "control_mode": "diagnostic_pose_replay_direct_hand_qpos_pose_injection_with_settle",
        "is_actuated_playback": False,
        "diagnostic_warning": "Diagnostic pose replay restores hand qpos directly. These metrics are not actuator-driven playback scores.",
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
    result["metric_validity"] = metric_validity_for_rollout(result, target_keys=keys[: played.shape[0]], played_keys=played)
    atomic_save_json(output / f"{label}_diagnostic_pose_replay.json", result)
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
        "pose_replay_diagnostics": rollout,
        "actuated_rollout_metrics": None,
        "metric_validity": {
            "pose_replay_is_diagnostic_only": not bool(rollout.get("is_actuated_playback", False)),
            "actuated_passes_leakage_checks": False,
        },
        # Backward-compatible alias for older callers. This remains diagnostic-only.
        "rollout": rollout,
    }
    atomic_save_json(run_dir / f"{label}_evaluation.json", summary)
    return summary
