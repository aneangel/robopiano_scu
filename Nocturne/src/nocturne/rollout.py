from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from nocturne.controller_dataset import build_online_feature, finite_difference
from nocturne.controller_model import load_policy
from nocturne.io import ensure_dir, save_json, save_npz
from nocturne.offline_eval import evaluate_rollout


def evaluate_policy_online(
    *,
    checkpoint: str | Path,
    trajectory: str | Path,
    output_root: str | Path,
    song_name: str | None = None,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run a learned policy through RoboPianist env.step(action).

    This function deliberately does not set piano key states or teleport hands
    during rollout. Simulator handles are imported lazily so unit tests do not
    require MuJoCo/RoboPianist.
    """
    _preload_conda_libstdcxx()
    from etude.robopianist.observation import extract_tracking_observation
    from etude.robopianist.state_mapping import resolve_mapping_from_env
    from partita.evaluation.rollout import (
        _capture_piano_activation,
        _load_env,
        candidate_environment_names,
        write_goals_proto,
    )

    output = ensure_dir(output_root)
    with np.load(Path(trajectory), allow_pickle=False) as data:
        goals = np.asarray(data["goals"], dtype=np.float32)
        q_ref = np.asarray(data["hand_joints"], dtype=np.float32)
        actions_ref = np.asarray(data["actions"], dtype=np.float32)
        dt = float(np.asarray(data["dt"]).reshape(())) if "dt" in data else 0.05
    qdot_ref = finite_difference(q_ref, dt)
    env_song = song_name or _metadata_song_name(trajectory) or "RoboPianist-repertoire-150-v0"
    midi_proto = write_goals_proto(goals[:, :88], output / "target_goals.proto", dt=dt, title="Nocturne online rollout")
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(env_song),
        midi_proto_path=midi_proto,
        control_timestep=dt,
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
    model = load_policy(checkpoint, map_location="cpu")
    previous_action = np.zeros((actions_ref.shape[1],), dtype=np.float32)
    played = []
    executed = []
    try:
        timestep = env.reset()
        mapping = resolve_mapping_from_env(env)
        for t in range(q_ref.shape[0]):
            obs = extract_tracking_observation(timestep.observation, mapping, include_key_state=False)
            feature = build_online_feature(
                q=np.asarray(obs["q"], dtype=np.float32),
                qdot=np.asarray(obs["qdot"], dtype=np.float32),
                previous_action=previous_action,
                q_ref=q_ref,
                goals=goals,
                t=t,
            )
            with torch.no_grad():
                action = model(torch.from_numpy(feature[None]).float()).cpu().numpy()[0].astype(np.float32)
            previous_action = action
            timestep = env.step(action)
            executed.append(action)
            activation = _capture_piano_activation(env)
            if activation is not None:
                played.append(np.asarray(activation[:88], dtype=np.float32))
            if timestep.last():
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    played_keys = np.stack(played, axis=0) if played else np.zeros((0, 88), dtype=np.float32)
    actions = np.stack(executed, axis=0) if executed else np.zeros((0, actions_ref.shape[1]), dtype=np.float32)
    metrics = evaluate_rollout(goals[: played_keys.shape[0], :88], played_keys, dt=dt, threshold=threshold)
    result = {
        "checkpoint": str(checkpoint),
        "trajectory": str(trajectory),
        "environment_name": env_name,
        "load_info": load_info,
        "control_mode": "learned_action_policy_env_step",
        "actions_executed": int(actions.shape[0]),
        "played_shape": list(played_keys.shape),
        "metrics": metrics,
    }
    save_npz(output / "rollout.npz", played_keys=played_keys, target_keys=goals[: played_keys.shape[0], :88], actions=actions)
    save_json(output / "online_rollout_eval.json", result)
    return result


def _metadata_song_name(_trajectory: str | Path) -> str | None:
    return None


def _preload_conda_libstdcxx() -> None:
    import ctypes
    import sys

    lib = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not lib.is_file():
        return
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return
