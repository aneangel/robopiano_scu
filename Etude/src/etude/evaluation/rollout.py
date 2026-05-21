from __future__ import annotations

from typing import Any

import numpy as np

from etude.controllers.base import TrajectoryFollower
from etude.robopianist.observation import extract_raw_key_state, extract_tracking_observation
from etude.robopianist.state_mapping import StateMapping


def _metadata_timestep(value: Any, t: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim <= 1:
        return array.astype(np.float32)
    if array.ndim == 2 and array.shape[-1] != 3:
        index = int(np.clip(t, 0, array.shape[0] - 1))
        return array[index].astype(np.float32)
    if array.ndim == 2 and array.shape == (10, 3):
        return array.astype(np.float32)
    index = int(np.clip(t, 0, array.shape[0] - 1))
    return array[index].astype(np.float32)


def rollout_controller(
    env: Any,
    controller: TrajectoryFollower,
    mapping: StateMapping,
    q_ref: np.ndarray,
    qdot_ref: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    max_steps: int | None = None,
    render_video: bool = False,
    render_height: int = 480,
    render_width: int = 640,
    render_every: int = 1,
    camera_id: int | str | None = None,
) -> dict[str, np.ndarray]:
    """Run a controller in a dm_control-style environment."""
    time_step = env.reset()
    rollout_metadata = dict(metadata or {})
    controller.reset(q_ref, qdot_ref, metadata=rollout_metadata)
    trajectory_dt = float(rollout_metadata.get("dt", 0.005))
    env_dt = _resolve_env_dt(env, fallback_dt=trajectory_dt)
    reference_indices = _build_reference_schedule(
        horizon=q_ref.shape[0],
        trajectory_dt=trajectory_dt,
        env_dt=env_dt,
        max_steps=max_steps,
    )
    actions = []
    q = []
    qdot = []
    applied_reference_indices = []
    key_state = []
    fingertips = []
    desired_fingertips = []
    fingertip_error = []
    active_finger_mask = []
    frames = []
    for step_idx, ref_t in enumerate(reference_indices):
        obs = extract_tracking_observation(time_step.observation, mapping, include_key_state=False)
        for key in ("target_keys", "desired_fingertips", "fingertip_ref", "time_to_next_active_key"):
            if key in rollout_metadata:
                obs[key] = rollout_metadata[key]
        synthesized_fingertips = False
        if "fingertips" not in obs and "desired_fingertips" in rollout_metadata:
            desired_t = _metadata_timestep(rollout_metadata.get("desired_fingertips"), ref_t)
            if desired_t is not None:
                obs["fingertips"] = np.asarray(desired_t, dtype=np.float32).reshape(-1)
                synthesized_fingertips = True
        action = controller.act(obs, ref_t)
        if render_video and step_idx % max(int(render_every), 1) == 0:
            frame = _render_frame(env, height=render_height, width=render_width, camera_id=camera_id)
            if frame is not None:
                frames.append(frame)
        actions.append(action)
        q.append(obs["q"])
        qdot.append(obs["qdot"])
        applied_reference_indices.append(ref_t)
        observed_key_state = extract_raw_key_state(time_step.observation)
        if observed_key_state is not None:
            key_state.append(observed_key_state)
        current_fingertips = obs.get("fingertips")
        desired = _metadata_timestep(rollout_metadata.get("desired_fingertips"), ref_t)
        active_mask_t = _metadata_timestep(rollout_metadata.get("active_finger_mask"), ref_t)
        if current_fingertips is not None and not synthesized_fingertips:
            fingertips.append(np.asarray(current_fingertips, dtype=np.float32))
        if desired is not None:
            desired_fingertips.append(np.asarray(desired, dtype=np.float32))
        if current_fingertips is not None and desired is not None:
            current_array = np.asarray(current_fingertips, dtype=np.float32)
            desired_array = np.asarray(desired, dtype=np.float32)
            if current_array.shape == desired_array.shape:
                fingertip_error.append(desired_array - current_array)
        if active_mask_t is not None:
            active_finger_mask.append(np.asarray(active_mask_t, dtype=np.float32))
        time_step = env.step(action)
        if getattr(time_step, "last", lambda: False)():
            break
    rollout = {
        "actions": np.asarray(actions, dtype=np.float32),
        "q": np.asarray(q, dtype=np.float32),
        "qdot": np.asarray(qdot, dtype=np.float32),
        "reference_indices": np.asarray(applied_reference_indices, dtype=np.int32),
        "trajectory_dt": np.asarray(trajectory_dt, dtype=np.float32),
        "env_dt": np.asarray(env_dt, dtype=np.float32),
    }
    if key_state:
        rollout["key_state"] = np.asarray(key_state, dtype=np.float32)
    if fingertips:
        rollout["fingertips"] = np.asarray(fingertips, dtype=np.float32)
    if desired_fingertips:
        rollout["desired_fingertips"] = np.asarray(desired_fingertips, dtype=np.float32)
    if fingertip_error:
        rollout["fingertip_error"] = np.asarray(fingertip_error, dtype=np.float32)
    if active_finger_mask:
        rollout["active_finger_mask"] = np.asarray(active_finger_mask, dtype=np.float32)
    if frames:
        rollout["frames"] = np.asarray(frames, dtype=np.uint8)
    return rollout


def _render_frame(
    env: Any,
    *,
    height: int,
    width: int,
    camera_id: int | str | None,
) -> np.ndarray | None:
    physics = getattr(env, "physics", None)
    if physics is None and hasattr(env, "_env"):
        physics = getattr(env._env, "physics", None)
    if physics is None or not hasattr(physics, "render"):
        return None
    kwargs: dict[str, Any] = {"height": int(height), "width": int(width)}
    if camera_id is not None:
        kwargs["camera_id"] = camera_id
    return np.asarray(physics.render(**kwargs), dtype=np.uint8)


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
    physics = getattr(env, "physics", None)
    if physics is None and hasattr(env, "_env"):
        physics = getattr(env._env, "physics", None)
    timestep_fn = getattr(physics, "timestep", None)
    if callable(timestep_fn):
        try:
            dt = float(timestep_fn())
        except (TypeError, ValueError):
            dt = 0.0
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
    sim_times = np.arange(total_steps, dtype=np.float32) * env_dt
    ref_indices = np.floor(sim_times / trajectory_dt).astype(np.int32)
    return np.clip(ref_indices, 0, horizon - 1)
