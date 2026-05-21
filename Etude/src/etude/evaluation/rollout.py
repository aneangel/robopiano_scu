from __future__ import annotations

from typing import Any

import numpy as np

from etude.controllers.base import TrajectoryFollower
from etude.robopianist.observation import extract_tracking_observation
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
) -> dict[str, np.ndarray]:
    """Run a controller in a dm_control-style environment."""
    time_step = env.reset()
    rollout_metadata = dict(metadata or {})
    controller.reset(q_ref, qdot_ref, metadata=rollout_metadata)
    steps = min(q_ref.shape[0], max_steps or q_ref.shape[0])
    actions = []
    q = []
    qdot = []
    key_state = []
    fingertips = []
    desired_fingertips = []
    fingertip_error = []
    active_finger_mask = []
    for t in range(steps):
        obs = extract_tracking_observation(time_step.observation, mapping)
        for key in ("target_keys", "desired_fingertips", "fingertip_ref", "time_to_next_active_key"):
            if key in rollout_metadata:
                obs[key] = rollout_metadata[key]
        action = controller.act(obs, t)
        actions.append(action)
        q.append(obs["q"])
        qdot.append(obs["qdot"])
        if "key_state" in obs:
            key_state.append(np.asarray(obs["key_state"], dtype=np.float32))
        current_fingertips = obs.get("fingertips")
        desired = _metadata_timestep(rollout_metadata.get("desired_fingertips"), t)
        active_mask_t = _metadata_timestep(rollout_metadata.get("active_finger_mask"), t)
        if current_fingertips is not None:
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
    return rollout
