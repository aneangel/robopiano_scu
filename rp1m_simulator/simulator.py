from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

RolloutMode = Literal["hand_state", "action"]
ActionSourceScale = Literal["normalized_minus_one_to_one", "actuator_units"]
ActionMapping = Literal["as_is", "swap_hands", "zero_sustain", "invert_sustain", "swap_hands_zero_sustain"]
ActionSubstepPolicy = Literal["zero_pad_hold", "zero_control", "zero_source", "repeat"]
HandStateActionSource = Literal["recorded", "zero"]
HandAnchorApplication = Literal["compile_time", "post_reset"]
WristActionPolicy = Literal["recorded", "hold_initial"]

ACTION_SOURCE_SCALES: tuple[ActionSourceScale, ...] = ("normalized_minus_one_to_one", "actuator_units")
ACTION_MAPPINGS: tuple[ActionMapping, ...] = (
    "as_is",
    "swap_hands",
    "zero_sustain",
    "invert_sustain",
    "swap_hands_zero_sustain",
)
ACTION_SUBSTEP_POLICIES: tuple[ActionSubstepPolicy, ...] = (
    "zero_pad_hold",
    "zero_control",
    "zero_source",
    "repeat",
)

DEFAULT_RP1M_ROOT = "/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr"
DEFAULT_OUTPUT_ROOT = "/WAVE/datasets/ccoelho_lab-jlanders/Fugue/runs/rp1m_simulator"
DEFAULT_HAND_ANCHOR_Y_OFFSET = -0.0490857549

_HAND_ANCHOR_DEFAULT_POSITIONS: dict[str, tuple[float, float, float]] | None = None


@dataclass(slots=True)
class RP1MTrajectory:
    song_key: str
    environment_name: str
    demo_id: int
    actions: np.ndarray
    goals: np.ndarray
    hand_joints: np.ndarray
    hand_fingertips: np.ndarray | None = None
    reference_piano_states: np.ndarray | None = None


@dataclass(slots=True)
class RolloutConfig:
    mode: RolloutMode = "hand_state"
    dataset_timestep: float = 0.05
    simulation_timestep: float = 0.005
    hand_anchor_y_offset: float | None = DEFAULT_HAND_ANCHOR_Y_OFFSET
    auto_hand_anchor_y_offset: bool = True
    hand_anchor_application: HandAnchorApplication = "post_reset"
    reduced_action_space: bool = True
    action_source_scale: ActionSourceScale = "normalized_minus_one_to_one"
    action_mapping: ActionMapping = "as_is"
    action_substep_policy: ActionSubstepPolicy = "zero_pad_hold"
    wrist_action_policy: WristActionPolicy = "hold_initial"
    hand_state_action_source: HandStateActionSource = "recorded"
    restore_initial_hand: bool = True
    set_hand_qvel: bool = True
    initial_hand_qvel_scale: float = 0.5
    hand_resync_interval: int | None = None
    gravity_compensation: bool = False
    primitive_fingertip_collisions: bool = False
    disable_hand_collisions: bool = False
    mujoco_integrator: int | None = None
    mujoco_solver: int | None = None
    mujoco_cone: int | None = None
    mujoco_jacobian: int | None = None
    mujoco_iterations: int | None = None
    mujoco_ls_iterations: int | None = None
    seed: int = 0
    threshold: float = 0.5
    max_source_steps: int | None = None
    render_mp4: bool = False
    render_audio: bool = True
    render_every_source_step: int = 1
    width: int = 640
    height: int = 480
    camera_id: str | int | None = None
    fps: int = 20


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonicalize_environment_name(song_key: str) -> str:
    return re.sub(r"(-v\d+)_\d+$", r"\1", str(song_key))


def candidate_environment_names(song_key: str) -> list[str]:
    canonical = canonicalize_environment_name(song_key)
    candidates = [canonical]
    if "repertoire-150" in canonical:
        candidates.append(canonical.replace("repertoire-150", "etude-12"))
    seen: set[str] = set()
    out: list[str] = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def open_rp1m_root(path: str | Path):
    import zarr

    root_path = Path(path)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    return zarr.open(str(root_path), mode="r")


def load_rp1m_trajectory(
    rp1m_root: str | Path,
    song_key: str,
    demo_id: int,
    *,
    include_reference_piano_states: bool = True,
) -> RP1MTrajectory:
    root = open_rp1m_root(rp1m_root)
    if song_key not in root:
        raise KeyError(f"RP1M song key not found: {song_key}")
    group = root[song_key]
    required = ["actions", "goals", "hand_joints"]
    missing = [name for name in required if name not in group]
    if missing:
        raise KeyError(f"RP1M song {song_key} is missing arrays: {missing}")
    actions = np.asarray(group["actions"][demo_id], dtype=np.float32)
    goals = np.asarray(group["goals"][demo_id], dtype=np.float32)
    hand_joints = np.asarray(group["hand_joints"][demo_id], dtype=np.float32)
    hand_fingertips = None
    if "hand_fingertips" in group:
        hand_fingertips = np.asarray(group["hand_fingertips"][demo_id], dtype=np.float32)
    reference = None
    if include_reference_piano_states and "piano_states" in group:
        reference = np.asarray(group["piano_states"][demo_id], dtype=np.float32)
    return RP1MTrajectory(
        song_key=song_key,
        environment_name=canonicalize_environment_name(song_key),
        demo_id=int(demo_id),
        actions=actions,
        goals=goals,
        hand_joints=hand_joints,
        hand_fingertips=hand_fingertips,
        reference_piano_states=reference,
    )


def make_rp1m_trajectory_from_arrays(
    *,
    song_key: str,
    demo_id: int,
    actions: np.ndarray,
    goals: np.ndarray,
    hand_joints: np.ndarray,
    hand_fingertips: np.ndarray | None = None,
    reference_piano_states: np.ndarray | None = None,
    environment_name: str | None = None,
) -> RP1MTrajectory:
    """Build an RP1M trajectory object without treating piano states as simulator input."""
    return RP1MTrajectory(
        song_key=str(song_key),
        environment_name=canonicalize_environment_name(environment_name or song_key),
        demo_id=int(demo_id),
        actions=np.asarray(actions, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        hand_joints=np.asarray(hand_joints, dtype=np.float32),
        hand_fingertips=None if hand_fingertips is None else np.asarray(hand_fingertips, dtype=np.float32),
        reference_piano_states=(
            None if reference_piano_states is None else np.asarray(reference_piano_states, dtype=np.float32)
        ),
    )


def find_high_f1_examples(
    rp1m_root: str | Path,
    *,
    max_songs: int = 8,
    examples: int = 4,
    min_recorded_f1: float = 0.85,
    threshold: float = 0.5,
) -> list[tuple[str, int, float]]:
    root = open_rp1m_root(rp1m_root)
    candidates: list[tuple[str, int, float]] = []
    for song_key in sorted(root.keys())[: max(int(max_songs), 1)]:
        group = root[song_key]
        if not hasattr(group, "keys") or "goals" not in group or "piano_states" not in group:
            continue
        goals = np.asarray(group["goals"][:, :, :88] > threshold, dtype=bool)
        states = np.asarray(group["piano_states"][:, :, :88] > threshold, dtype=bool)
        tp = np.logical_and(goals, states).sum(axis=(1, 2))
        fp = np.logical_and(~goals, states).sum(axis=(1, 2))
        fn = np.logical_and(goals, ~states).sum(axis=(1, 2))
        denom = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)
        for demo_id in np.argsort(f1)[::-1][: max(int(examples), 1)]:
            score = float(f1[int(demo_id)])
            if score >= float(min_recorded_f1):
                candidates.append((str(song_key), int(demo_id), score))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[: max(int(examples), 1)]


def key_metrics(target: np.ndarray, played: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    target_active = np.asarray(target, dtype=np.float32) > float(threshold)
    played_active = np.asarray(played, dtype=np.float32) > float(threshold)
    steps = min(target_active.shape[0], played_active.shape[0])
    keys = min(target_active.shape[1], played_active.shape[1], 88)
    if steps <= 0 or keys <= 0:
        return {"key_precision": 0.0, "key_recall": 0.0, "key_f1": 0.0, "mispress_rate": 0.0}
    target_active = target_active[:steps, :keys]
    played_active = played_active[:steps, :keys]
    tp = int(np.logical_and(target_active, played_active).sum())
    fp = int(np.logical_and(~target_active, played_active).sum())
    fn = int(np.logical_and(target_active, ~played_active).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    false_frames = np.logical_and(~target_active, played_active).any(axis=1)
    return {
        "key_precision": precision,
        "key_recall": recall,
        "key_f1": f1,
        "mispress_rate": float(false_frames.mean()) if false_frames.size else 0.0,
    }


def _load_music_pb2():
    import site

    candidates = []
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        candidates.append(Path(root) / "note_seq" / "protobuf" / "music_pb2.py")
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("rp1m_simulator_note_seq_music_pb2", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate note_seq/protobuf/music_pb2.py.")


def write_goals_proto(goals: np.ndarray, path: str | Path, *, dt: float, title: str) -> Path:
    music_pb2 = _load_music_pb2()
    active = np.asarray(goals, dtype=np.float32)[:, :88] > 0.5
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = title
    seq.sequence_metadata.artist = "rp1m_simulator"
    for key in range(active.shape[1]):
        start = None
        for t, is_active in enumerate(active[:, key]):
            if is_active and start is None:
                start = t
            if start is not None and (not is_active or t == active.shape[0] - 1):
                end_t = t + 1 if is_active and t == active.shape[0] - 1 else t
                if end_t > start:
                    note = seq.notes.add()
                    note.pitch = int(21 + key)
                    note.velocity = 80
                    note.start_time = float(start * dt)
                    note.end_time = float(max(end_t * dt, start * dt + dt))
                    note.part = 0
                start = None
    seq.total_time = float(active.shape[0] * dt)
    seq.tempos.add(qpm=60)
    out = Path(path)
    ensure_dir(out.parent)
    out.write_bytes(seq.SerializeToString())
    return out


def _apply_hand_anchor_y_offset(offset: float | None) -> dict[str, Any]:
    from robopianist.suite.tasks import base as task_base

    global _HAND_ANCHOR_DEFAULT_POSITIONS
    if _HAND_ANCHOR_DEFAULT_POSITIONS is None:
        _HAND_ANCHOR_DEFAULT_POSITIONS = {
            "left": tuple(getattr(task_base, "_LEFT_HAND_POSITION")),
            "right": tuple(getattr(task_base, "_RIGHT_HAND_POSITION")),
        }
    left_default = _HAND_ANCHOR_DEFAULT_POSITIONS["left"]
    right_default = _HAND_ANCHOR_DEFAULT_POSITIONS["right"]
    offset_value = 0.0 if offset is None else float(offset)
    left_position = (left_default[0], left_default[1] + offset_value, left_default[2])
    right_position = (right_default[0], right_default[1] + offset_value, right_default[2])
    task_base._LEFT_HAND_POSITION = left_position
    task_base._RIGHT_HAND_POSITION = right_position
    return {
        "hand_anchor_y_offset": None if offset is None else offset_value,
        "effective_hand_anchor_y_offset": offset_value,
        "left_hand_position": list(left_position),
        "right_hand_position": list(right_position),
    }


def _load_env(config: RolloutConfig, midi_proto_path: Path, environment_name: str, control_timestep: float):
    from robopianist import suite

    if config.hand_anchor_application == "compile_time":
        hand_patch = _apply_hand_anchor_y_offset(config.hand_anchor_y_offset)
        post_reset_hand_anchor_y_offset = None
    elif config.hand_anchor_application == "post_reset":
        hand_patch = _apply_hand_anchor_y_offset(None)
        post_reset_hand_anchor_y_offset = config.hand_anchor_y_offset
    else:
        raise ValueError(f"Unknown hand_anchor_application: {config.hand_anchor_application}")
    task_kwargs = {
        "control_timestep": float(control_timestep),
        "n_steps_lookahead": 1,
        "disable_colorization": True,
        "disable_fingering_reward": True,
        "disable_forearm_reward": True,
        "disable_hand_collisions": bool(config.disable_hand_collisions),
        "wrong_press_termination": False,
        "reduced_action_space": bool(config.reduced_action_space),
        "gravity_compensation": bool(config.gravity_compensation),
        "primitive_fingertip_collisions": bool(config.primitive_fingertip_collisions),
    }
    last_error: Exception | None = None
    for env_name in candidate_environment_names(environment_name):
        try:
            env = suite.load(
                environment_name=env_name,
                midi_file=midi_proto_path,
                seed=int(config.seed),
                task_kwargs=task_kwargs,
            )
            return env, {
                "task_kwargs": task_kwargs,
                "hand_anchor_patch": hand_patch,
                "post_reset_hand_anchor_y_offset": post_reset_hand_anchor_y_offset,
                "midi_proto_path": str(midi_proto_path),
                "resolved_environment_name": env_name,
            }
        except Exception as exc:  # pragma: no cover - depends on local RoboPianist assets.
            last_error = exc
    raise RuntimeError(f"Could not load RoboPianist environment from {environment_name!r}: {last_error}")


def _iter_wrapped_envs(env: Any) -> list[Any]:
    current = env
    wrapped = []
    seen = set()
    while current is not None and id(current) not in seen:
        wrapped.append(current)
        seen.add(id(current))
        current = getattr(current, "_environment", None)
    return wrapped


def _locate_task_physics_piano(env: Any) -> tuple[Any, Any, Any]:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        physics = getattr(current, "physics", None)
        piano = getattr(task, "piano", None)
        if task is not None and physics is not None and piano is not None:
            return task, physics, piano
    raise RuntimeError("Could not locate RoboPianist task, physics, and piano handles.")


def _hand_joint_handles(task: Any) -> list[Any]:
    joints: list[Any] = []
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name, None)
        hand_joints = getattr(hand, "joints", None)
        if hand_joints is not None:
            joints.extend(list(hand_joints))
    return joints


def _set_hand_qpos(task: Any, physics: Any, hand_qpos: np.ndarray) -> int:
    values = np.asarray(hand_qpos, dtype=np.float32).reshape(-1)
    joints = _hand_joint_handles(task)
    if values.size < len(joints):
        raise ValueError(f"hand_joints has {values.size} values but env expects {len(joints)} hand joints")
    for joint, value in zip(joints, values[: len(joints)]):
        physics.bind(joint).qpos = float(value)
    return len(joints)


def _set_hand_qvel(task: Any, physics: Any, hand_qvel: np.ndarray) -> int:
    values = np.asarray(hand_qvel, dtype=np.float32).reshape(-1)
    joints = _hand_joint_handles(task)
    width = min(values.size, len(joints))
    failed = 0
    for joint, value in zip(joints[:width], values[:width]):
        try:
            physics.bind(joint).qvel = float(value)
        except Exception:
            failed += 1
    return int(width - failed)


def _apply_post_reset_hand_anchor_y_offset(task: Any, physics: Any, offset: float | None) -> dict[str, Any]:
    if offset is None:
        return {"applied": False, "hand_anchor_y_offset": None}
    offset_value = float(offset)
    shifted = 0
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name, None)
        shift_pose = getattr(hand, "shift_pose", None)
        if callable(shift_pose):
            shift_pose(physics, (0.0, offset_value, 0.0))
            shifted += 1
    if hasattr(physics, "forward"):
        physics.forward()
    return {
        "applied": bool(shifted),
        "hand_anchor_y_offset": offset_value,
        "hands_shifted": int(shifted),
    }


def _capture_hand_qpos(task: Any, physics: Any) -> np.ndarray | None:
    joints = _hand_joint_handles(task)
    if not joints:
        return None
    values: list[float] = []
    for joint in joints:
        q = np.asarray(physics.bind(joint).qpos, dtype=np.float64).reshape(-1)
        values.append(float(q[0]) if q.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _capture_hand_qvel(task: Any, physics: Any) -> np.ndarray | None:
    joints = _hand_joint_handles(task)
    if not joints:
        return None
    values: list[float] = []
    for joint in joints:
        q = np.asarray(physics.bind(joint).qvel, dtype=np.float64).reshape(-1)
        values.append(float(q[0]) if q.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _element_name(element: Any) -> str:
    for attr in ("full_identifier", "identifier", "name"):
        value = getattr(element, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        if value:
            return str(value)
    return str(element)


def _hand_joint_names(task: Any) -> list[str]:
    return [_element_name(joint) for joint in _hand_joint_handles(task)]


def _hand_actuator_names(task: Any) -> list[str]:
    names: list[str] = []
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name, None)
        actuators = getattr(hand, "actuators", None)
        if actuators is not None:
            names.extend(_element_name(actuator) for actuator in actuators)
    names.append("sustain")
    return names


def _capture_fingertips(task: Any, physics: Any) -> np.ndarray | None:
    sites = []
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name, None)
        fingertip_sites = getattr(hand, "fingertip_sites", None)
        if fingertip_sites is not None:
            sites.extend(list(fingertip_sites))
    if not sites:
        return None
    return np.asarray([physics.bind(site).xpos.copy() for site in sites], dtype=np.float32).reshape(-1)


def _calibrate_hand_anchor_y_offset(
    trajectory: RP1MTrajectory,
    config: RolloutConfig,
    midi_proto_path: Path,
    control_timestep: float,
) -> dict[str, Any]:
    if not config.auto_hand_anchor_y_offset:
        return {"enabled": False, "effective_hand_anchor_y_offset": config.hand_anchor_y_offset}
    if trajectory.hand_fingertips is None:
        return {
            "enabled": True,
            "status": "missing_hand_fingertips",
            "effective_hand_anchor_y_offset": config.hand_anchor_y_offset,
        }
    calibration_config = replace(config, hand_anchor_y_offset=0.0, auto_hand_anchor_y_offset=False, render_mp4=False)
    env, _info = _load_env(calibration_config, midi_proto_path, trajectory.environment_name, control_timestep)
    try:
        env.reset()
        task, physics, _piano = _locate_task_physics_piano(env)
        _set_hand_qpos(task, physics, trajectory.hand_joints[0])
        if hasattr(physics, "forward"):
            physics.forward()
        simulated = _capture_fingertips(task, physics)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    if simulated is None:
        return {
            "enabled": True,
            "status": "simulated_fingertips_unavailable",
            "effective_hand_anchor_y_offset": config.hand_anchor_y_offset,
        }
    reference = np.asarray(trajectory.hand_fingertips[0], dtype=np.float32).reshape(-1)
    width = min(simulated.size, reference.size)
    diff = simulated[:width].reshape(-1, 3) - reference[:width].reshape(-1, 3)
    offset = -float(np.mean(diff[:, 1]))
    return {
        "enabled": True,
        "status": "ok",
        "effective_hand_anchor_y_offset": offset,
        "mean_sim_minus_reference_xyz": diff.mean(axis=0).tolist(),
        "mean_tip_l2_before_offset": float(np.linalg.norm(diff, axis=1).mean()),
    }


def _capture_piano_activation(env: Any) -> np.ndarray | None:
    for current in _iter_wrapped_envs(env):
        piano = getattr(getattr(current, "task", None), "piano", None)
        activation = getattr(piano, "activation", None)
        if activation is not None:
            return np.asarray(activation, dtype=np.float32).reshape(-1)[:88]
    return None


def _action_runtime_config(config: RolloutConfig, action_dim: int) -> tuple[RolloutConfig, dict[str, Any]]:
    """Resolve action-only defaults that should not alter hand-state playback."""
    if config.mode != "action":
        return config, {"mode": config.mode, "adjustments": {}}

    updates: dict[str, Any] = {}
    adjustments: dict[str, str] = {}
    if int(action_dim) == 39 and not config.reduced_action_space:
        updates["reduced_action_space"] = True
        adjustments["reduced_action_space"] = "forced_true_for_39d_compressed_rp1m_actions"
    if config.restore_initial_hand:
        updates["restore_initial_hand"] = False
        adjustments["restore_initial_hand"] = "disabled_for_action_mode_default_reset_hand_state"
    if config.set_hand_qvel:
        updates["set_hand_qvel"] = False
        adjustments["set_hand_qvel"] = "disabled_because_action_mode_uses_default_reset_hand_state"
    if config.hand_resync_interval is not None:
        updates["hand_resync_interval"] = None
        adjustments["hand_resync_interval"] = "disabled_because_action_mode_must_not_resync_from_rp1m_hand_state"

    runtime_config = replace(config, **updates) if updates else config
    return runtime_config, {
        "mode": config.mode,
        "action_input_dim": int(action_dim),
        "action_input_format": "compressed_39d" if int(action_dim) == 39 else "environment_action_spec",
        "initial_hand_state": "robopianist_reset_default_qpos_qvel",
        "hand_anchor_policy": "preserve_configured_anchor_or_auto_calibration",
        "adjustments": adjustments,
    }


def _scale_action_to_spec(action: np.ndarray, action_spec: Any, *, source: ActionSourceScale) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    control = np.zeros_like(minimum, dtype=np.float32)
    width = min(control.size, values.size)
    if width <= 0:
        return control
    if source == "normalized_minus_one_to_one":
        clipped = np.clip(values[:width], -1.0, 1.0)
        control[:width] = minimum[:width] + 0.5 * (clipped + 1.0) * (maximum[:width] - minimum[:width])
    elif source == "actuator_units":
        control[:width] = values[:width]
    else:
        raise ValueError(f"Unknown action source scale: {source}")
    return np.clip(control, minimum, maximum)


def map_action_channels(action: np.ndarray, *, mapping: ActionMapping) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if mapping == "as_is" or values.size != 39:
        return values
    right = values[:19]
    left = values[19:38]
    sustain = values[38:39]
    if mapping == "swap_hands":
        return np.concatenate([left, right, sustain]).astype(np.float32)
    if mapping == "zero_sustain":
        return np.concatenate([right, left, np.zeros_like(sustain)]).astype(np.float32)
    if mapping == "invert_sustain":
        return np.concatenate([right, left, -sustain]).astype(np.float32)
    if mapping == "swap_hands_zero_sustain":
        return np.concatenate([left, right, np.zeros_like(sustain)]).astype(np.float32)
    raise ValueError(f"Unknown action mapping: {mapping}")


def _prepare_control(action: np.ndarray, action_spec: Any, config: RolloutConfig) -> np.ndarray:
    mapped = map_action_channels(action, mapping=config.action_mapping)
    return _scale_action_to_spec(mapped, action_spec, source=config.action_source_scale)


def _initial_wrist_control_overrides(
    initial_hand_qpos: np.ndarray,
    action_spec: Any,
    config: RolloutConfig,
) -> dict[int, float]:
    if config.wrist_action_policy == "recorded":
        return {}
    if config.wrist_action_policy != "hold_initial":
        raise ValueError(f"Unknown wrist_action_policy: {config.wrist_action_policy}")
    values = np.asarray(initial_hand_qpos, dtype=np.float32).reshape(-1)
    if values.size < 25 or int(action_spec.shape[0]) < 21:
        return {}
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    overrides = {
        0: float(values[0]),
        1: float(values[1]),
        19: float(values[23]),
        20: float(values[24]),
    }
    return {idx: float(np.clip(value, minimum[idx], maximum[idx])) for idx, value in overrides.items()}


def _apply_control_overrides(control: np.ndarray, overrides: dict[int, float]) -> np.ndarray:
    if not overrides:
        return control
    for idx, value in overrides.items():
        if 0 <= idx < control.size:
            control[idx] = value
    return control


def _controller_metadata(action_controller: Any | None) -> dict[str, Any] | None:
    if action_controller is None:
        return None
    metadata = getattr(action_controller, "metadata", None)
    if callable(metadata):
        try:
            return _jsonable(metadata())
        except Exception as exc:
            return {"metadata_error": f"{type(exc).__name__}: {exc}"}
    return {"controller_type": type(action_controller).__name__}


def _reset_action_controller(
    action_controller: Any,
    *,
    action_spec: Any,
    initial_hand_qpos: np.ndarray | None,
    initial_hand_qvel: np.ndarray | None,
    trajectory_hand_qpos: np.ndarray | None,
    hand_joint_names: list[str],
    actuator_names: list[str],
    config: RolloutConfig,
) -> None:
    reset = getattr(action_controller, "reset", None)
    if not callable(reset):
        return
    reset(
        action_spec=action_spec,
        initial_hand_qpos=initial_hand_qpos,
        initial_hand_qvel=initial_hand_qvel,
        trajectory_hand_qpos=trajectory_hand_qpos,
        hand_joint_names=hand_joint_names,
        actuator_names=actuator_names,
        config=config,
    )


def _compute_controller_control(
    action_controller: Any,
    *,
    action_spec: Any,
    source_t: int,
    substep: int,
    substeps: int,
    control_timestep: float,
    dataset_timestep: float,
    current_hand_qpos: np.ndarray,
    current_hand_qvel: np.ndarray | None,
    source_hand_qpos: np.ndarray,
    target_hand_qpos: np.ndarray,
    previous_control: np.ndarray | None,
) -> np.ndarray:
    compute = getattr(action_controller, "compute_control", None)
    if not callable(compute):
        raise TypeError("action_controller must provide compute_control(...)")
    control = compute(
        action_spec=action_spec,
        source_t=int(source_t),
        substep=int(substep),
        substeps=int(substeps),
        time_s=float((int(source_t) * int(substeps) + int(substep)) * float(control_timestep)),
        simulation_timestep=float(control_timestep),
        dataset_timestep=float(dataset_timestep),
        current_hand_qpos=current_hand_qpos,
        current_hand_qvel=current_hand_qvel,
        source_hand_qpos=source_hand_qpos,
        target_hand_qpos=target_hand_qpos,
        previous_control=previous_control,
    )
    values = np.asarray(control, dtype=np.float32).reshape(-1)
    if values.shape != tuple(action_spec.shape):
        raise ValueError(f"Controller returned control shape {values.shape}, expected {action_spec.shape}")
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    return np.clip(values, minimum, maximum).astype(np.float32)


def _apply_mujoco_options(physics: Any, config: RolloutConfig) -> dict[str, int]:
    option_updates = {
        "integrator": config.mujoco_integrator,
        "solver": config.mujoco_solver,
        "cone": config.mujoco_cone,
        "jacobian": config.mujoco_jacobian,
        "iterations": config.mujoco_iterations,
        "ls_iterations": config.mujoco_ls_iterations,
    }
    applied: dict[str, int] = {}
    opt = getattr(getattr(physics, "model", None), "opt", None)
    if opt is None:
        return applied
    for name, value in option_updates.items():
        if value is None:
            continue
        if not hasattr(opt, name):
            continue
        setattr(opt, name, int(value))
        applied[name] = int(getattr(opt, name))
    return applied


def render_frame(env: Any, *, height: int, width: int, camera_id: str | int | None) -> np.ndarray:
    errors = []
    for current in _iter_wrapped_envs(env):
        physics = getattr(current, "physics", None)
        if physics is None or not hasattr(physics, "render"):
            continue
        attempts: list[dict[str, Any]] = []
        if camera_id is not None:
            attempts.append({"height": height, "width": width, "camera_id": camera_id})
        attempts.extend([{"height": height, "width": width}, {"height": height, "width": width, "camera_id": 0}])
        for kwargs in attempts:
            try:
                frame = physics.render(**kwargs)
                arr = np.asarray(frame, dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    return arr[..., :3]
            except Exception as exc:
                errors.append(f"render({kwargs}): {exc}")
    raise RuntimeError("Unable to render RoboPianist frame. " + " | ".join(errors[:4]))


def piano_roll_to_midi_events(piano_roll: np.ndarray, *, dt: float, threshold: float) -> list[Any]:
    from robopianist.music import midi_message

    active = np.asarray(piano_roll, dtype=np.float32)[:, :88] > float(threshold)
    events: list[Any] = []
    for key in range(active.shape[1]):
        was_active = False
        for step, is_active in enumerate(active[:, key]):
            time_value = float(step * dt)
            if is_active and not was_active:
                events.append(midi_message.NoteOn(note=int(21 + key), velocity=127, time=time_value))
            elif was_active and not is_active:
                events.append(midi_message.NoteOff(note=int(21 + key), time=time_value))
            was_active = bool(is_active)
        if was_active:
            events.append(midi_message.NoteOff(note=int(21 + key), time=float(active.shape[0] * dt)))
    events.sort(key=lambda event: (float(getattr(event, "time", 0.0)), 0 if type(event).__name__ == "NoteOff" else 1))
    return events


def _default_soundfont_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "robopianist" / "soundfonts" / "SalamanderGrandPiano.sf2",
        repo_root / "robopianist" / "soundfonts" / "TimGM6mb.sf2",
        repo_root / "robopianist" / "third_party" / "soundfonts" / "TimGM6mb.sf2",
        repo_root / "third_party" / "soundfonts" / "TimGM6mb.sf2",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                if candidate.read_bytes()[:4] == b"RIFF":
                    return candidate
            except OSError:
                pass
    return None


def _write_waveform(path: Path, waveform: np.ndarray, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(waveform, dtype=np.int16).tobytes())


def _clone_note_events(events: Sequence[Any]) -> list[Any]:
    cloned = []
    for event in events:
        time_value = float(getattr(event, "time"))
        note = getattr(event, "note", None)
        velocity = getattr(event, "velocity", None)
        if note is not None and velocity is not None:
            cloned.append(type(event)(note=int(note), velocity=int(velocity), time=time_value))
        elif note is not None:
            cloned.append(type(event)(note=int(note), time=time_value))
    return cloned


def write_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    *,
    fps: int,
    audio_events: list[Any] | None,
) -> tuple[Path | None, str | None]:
    if not frames:
        return None, "No frames captured."
    import imageio.v2 as imageio

    output = Path(output_path)
    ensure_dir(output.parent)
    imageio.mimwrite(output, frames, fps=max(int(fps), 1), codec="libx264", quality=7, macro_block_size=None)
    if not audio_events:
        return output, "Audio skipped because rollout produced no note events."
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return output, "Audio skipped because ffmpeg was not found."
    soundfont = _default_soundfont_path()
    if soundfont is None:
        return output, "Audio skipped because no valid soundfont was found."
    try:
        from robopianist.music import synthesizer
    except Exception as exc:
        return output, f"Audio skipped because RoboPianist synthesizer import failed: {exc}"
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output.stem}_audio_", dir=str(output.parent)))
    wav_path = temp_dir / f"{output.stem}.wav"
    temp_video = temp_dir / output.name
    try:
        synth = synthesizer.Synthesizer(soundfont_path=soundfont)
        try:
            waveform = synth.get_samples(_clone_note_events(audio_events))
        finally:
            synth.stop()
        _write_waveform(wav_path, waveform)
        shutil.copyfile(output, temp_video)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_video),
                "-i",
                str(wav_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output, None
    except Exception as exc:
        return output, f"Audio mux failed: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _score_rollout(
    *,
    played: np.ndarray,
    goals: np.ndarray,
    reference_piano_states: np.ndarray | None,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"source_scored_steps": int(played.shape[0])}
    if played.size:
        steps = min(int(played.shape[0]), int(goals.shape[0]))
        keys = min(int(played.shape[1]), int(goals.shape[1]), 88)
        result["against_goals"] = {
            **key_metrics(goals[:steps, :keys], played[:steps, :keys], threshold=threshold),
            "scored_steps": int(steps),
            "scored_keys": int(keys),
        }
    if played.size and reference_piano_states is not None:
        ref = np.asarray(reference_piano_states, dtype=np.float32)
        steps = min(int(played.shape[0]), int(ref.shape[0]))
        keys = min(int(played.shape[1]), int(ref.shape[1]), 88)
        result["against_reference_piano_states"] = {
            **key_metrics(ref[:steps, :keys], played[:steps, :keys], threshold=threshold),
            "scored_steps": int(steps),
            "scored_keys": int(keys),
            "reference_use": "scoring_only_not_simulator_input",
        }
    return result


def _recorded_reference_score(traj: RP1MTrajectory, threshold: float) -> dict[str, Any] | None:
    if traj.reference_piano_states is None:
        return None
    steps = min(int(traj.reference_piano_states.shape[0]), int(traj.goals.shape[0]))
    keys = min(int(traj.reference_piano_states.shape[1]), int(traj.goals.shape[1]), 88)
    return {
        **key_metrics(traj.goals[:steps, :keys], traj.reference_piano_states[:steps, :keys], threshold=threshold),
        "scored_steps": int(steps),
        "scored_keys": int(keys),
    }


def _qvel_between(hand_joints: np.ndarray, source_t: int, dt: float) -> np.ndarray:
    next_t = min(source_t + 1, hand_joints.shape[0] - 1)
    return ((hand_joints[next_t] - hand_joints[source_t]) / float(dt)).astype(np.float32)


def simulate_rp1m_rollout(
    trajectory: RP1MTrajectory,
    config: RolloutConfig,
    output_dir: str | Path,
    *,
    action_controller: Any | None = None,
) -> dict[str, Any]:
    """Roll out RP1M through RoboPianist without ever restoring piano key states."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    output = ensure_dir(output_dir)
    actions = np.asarray(trajectory.actions, dtype=np.float32)
    goals = np.asarray(trajectory.goals, dtype=np.float32)
    hand_joints = np.asarray(trajectory.hand_joints, dtype=np.float32)
    source_steps = min(int(actions.shape[0]), int(goals.shape[0]), int(hand_joints.shape[0]))
    if trajectory.reference_piano_states is not None:
        source_steps = min(source_steps, int(trajectory.reference_piano_states.shape[0]))
    if config.max_source_steps is not None:
        source_steps = min(source_steps, int(config.max_source_steps))
    if source_steps < 2:
        raise ValueError(f"Need at least two RP1M source steps, got {source_steps}")

    if config.mode not in {"hand_state", "action"}:
        raise ValueError(f"Unknown rollout mode: {config.mode}")
    if action_controller is not None and config.mode != "action":
        raise ValueError("action_controller is only valid with RolloutConfig(mode='action')")
    runtime_config, runtime_policy = _action_runtime_config(config, int(actions.shape[-1]))
    if runtime_config.max_source_steps is not None:
        source_steps = min(source_steps, int(runtime_config.max_source_steps))

    control_timestep = float(runtime_config.simulation_timestep)
    substeps = int(round(float(runtime_config.dataset_timestep) / float(runtime_config.simulation_timestep)))
    if substeps < 1 or not np.isclose(
        substeps * control_timestep,
        float(runtime_config.dataset_timestep),
        rtol=1e-4,
        atol=1e-7,
    ):
        raise ValueError("dataset_timestep must be an integer multiple of simulation_timestep")
    if runtime_config.mode == "action":
        if runtime_config.action_substep_policy not in ACTION_SUBSTEP_POLICIES:
            raise ValueError(f"Unknown action_substep_policy: {runtime_config.action_substep_policy}")
        runtime_policy["action_substep_policy"] = runtime_config.action_substep_policy
        runtime_policy["action_controller"] = _controller_metadata(action_controller)
        runtime_policy["action_source_substeps_per_source_step"] = (
            int(substeps) if runtime_config.action_substep_policy == "repeat" else 1
        )
        runtime_policy["action_zero_padded_substeps_per_source_step"] = (
            0 if runtime_config.action_substep_policy == "repeat" else max(int(substeps) - 1, 0)
        )
        runtime_policy["action_substep_policy_note"] = {
            "repeat": "repeat each RP1M source action on every simulator substep",
            "zero_control": "send actuator-unit zero controls after the first source substep",
            "zero_source": "send normalized-source zero actions after the first source substep",
            "zero_pad_hold": "zero-pad sparse RP1M source commands but hold the previous actuator target between commands",
        }[runtime_config.action_substep_policy]
    dense_goals = np.repeat(goals[:source_steps, :88], substeps, axis=0)

    midi_proto = write_goals_proto(
        dense_goals,
        output / "target_goals.proto",
        dt=control_timestep,
        title=f"RP1M simulator {trajectory.song_key} demo {trajectory.demo_id}",
    )
    hand_anchor_calibration = _calibrate_hand_anchor_y_offset(trajectory, runtime_config, midi_proto, control_timestep)
    effective_config = replace(
        runtime_config,
        hand_anchor_y_offset=hand_anchor_calibration.get("effective_hand_anchor_y_offset"),
        auto_hand_anchor_y_offset=False,
    )
    env, load_info = _load_env(effective_config, midi_proto, trajectory.environment_name, control_timestep)
    frames: list[np.ndarray] = []
    source_played: list[np.ndarray] = []
    dense_played: list[np.ndarray] = []
    source_hand: list[np.ndarray] = []
    dense_hand: list[np.ndarray] = []
    dense_action_controls: list[np.ndarray] = []
    total_reward = 0.0
    actions_executed = 0
    render_error = None
    terminated = False
    qpos_restored = 0
    qvel_restored = 0
    hand_resync_count = 0
    post_reset_hand_anchor = {"applied": False, "hand_anchor_y_offset": None}
    mujoco_options_applied: dict[str, int] = {}
    control_overrides: dict[int, float] = {}
    initial_hand_policy: dict[str, Any] = {
        "state_source": "rp1m_hand_joints_0" if effective_config.restore_initial_hand else "robopianist_reset_default",
        "qpos_restored": False,
        "qvel_restored": False,
        "wrist_override_source": None,
    }
    try:
        env.reset()
        task, physics, _piano = _locate_task_physics_piano(env)
        mujoco_options_applied = _apply_mujoco_options(physics, effective_config)
        if effective_config.hand_anchor_application == "post_reset":
            post_reset_hand_anchor = _apply_post_reset_hand_anchor_y_offset(
                task,
                physics,
                effective_config.hand_anchor_y_offset,
            )
        if effective_config.restore_initial_hand:
            qpos_restored = _set_hand_qpos(task, physics, hand_joints[0])
            initial_hand_policy["qpos_restored"] = bool(qpos_restored)
            if effective_config.set_hand_qvel:
                initial_qvel = float(effective_config.initial_hand_qvel_scale) * _qvel_between(
                    hand_joints,
                    0,
                    effective_config.dataset_timestep,
                )
                qvel_restored = _set_hand_qvel(task, physics, initial_qvel)
                initial_hand_policy["qvel_restored"] = bool(qvel_restored)
            if hasattr(physics, "forward"):
                physics.forward()
        action_spec = env.action_spec()
        action_dim = int(actions.shape[-1])
        action_spec_dim = int(action_spec.shape[0])
        load_info["action_spec_shape"] = [action_spec_dim]
        load_info["action_input_dim"] = action_dim
        load_info["action_input_format"] = runtime_policy.get("action_input_format")
        if action_dim != action_spec_dim:
            hint = " Use reduced_action_space=True for compressed 39D RP1M actions." if action_dim == 39 else ""
            raise ValueError(f"Action dimension mismatch: RP1M has {action_dim}, environment expects {action_spec_dim}.{hint}")
        if effective_config.mode == "action":
            wrist_qpos = _capture_hand_qpos(task, physics)
            if effective_config.restore_initial_hand:
                wrist_qpos = hand_joints[0]
                initial_hand_policy["wrist_override_source"] = "rp1m_hand_joints_0"
            elif wrist_qpos is not None:
                initial_hand_policy["wrist_override_source"] = "robopianist_reset_default"
            else:
                wrist_qpos = np.zeros((0,), dtype=np.float32)
                initial_hand_policy["wrist_override_source"] = "unavailable"
            control_overrides = _initial_wrist_control_overrides(wrist_qpos, action_spec, effective_config)
            if action_controller is not None:
                _reset_action_controller(
                    action_controller,
                    action_spec=action_spec,
                    initial_hand_qpos=_capture_hand_qpos(task, physics),
                    initial_hand_qvel=_capture_hand_qvel(task, physics),
                    trajectory_hand_qpos=hand_joints,
                    hand_joint_names=_hand_joint_names(task),
                    actuator_names=_hand_actuator_names(task),
                    config=effective_config,
                )
                load_info["action_controller_reset"] = True
                load_info["action_controller_hand_joint_names"] = _hand_joint_names(task)
                load_info["action_controller_actuator_names"] = _hand_actuator_names(task)
        render_stride = max(int(effective_config.render_every_source_step), 1)
        zero_action = np.zeros_like(actions[0])
        held_action_control: np.ndarray | None = None
        for source_t in range(source_steps):
            interval: list[np.ndarray] = []
            next_t = min(source_t + 1, source_steps - 1)
            if (
                effective_config.mode == "action"
                and effective_config.hand_resync_interval is not None
                and int(effective_config.hand_resync_interval) > 0
                and source_t > 0
                and source_t % int(effective_config.hand_resync_interval) == 0
            ):
                task, physics, _piano = _locate_task_physics_piano(env)
                qpos_restored = _set_hand_qpos(task, physics, hand_joints[source_t])
                hand_resync_count += 1
                if effective_config.set_hand_qvel:
                    qvel = float(effective_config.initial_hand_qvel_scale) * _qvel_between(
                        hand_joints,
                        source_t,
                        effective_config.dataset_timestep,
                    )
                    qvel_restored = _set_hand_qvel(task, physics, qvel)
                if hasattr(physics, "forward"):
                    physics.forward()
            for substep in range(substeps):
                if effective_config.mode == "hand_state":
                    alpha = float(substep) / float(substeps)
                    qpos = ((1.0 - alpha) * hand_joints[source_t] + alpha * hand_joints[next_t]).astype(np.float32)
                    qvel = (
                        (hand_joints[next_t] - hand_joints[source_t]) / float(effective_config.dataset_timestep)
                    ).astype(np.float32)
                    task, physics, _piano = _locate_task_physics_piano(env)
                    qpos_restored = _set_hand_qpos(task, physics, qpos)
                    if effective_config.set_hand_qvel:
                        qvel_restored = _set_hand_qvel(task, physics, qvel)
                    if hasattr(physics, "forward"):
                        physics.forward()
                    action = actions[source_t] if effective_config.hand_state_action_source == "recorded" else zero_action
                else:
                    action = actions[source_t]
                if effective_config.mode == "action" and action_controller is not None:
                    task, physics, _piano = _locate_task_physics_piano(env)
                    current_hand = _capture_hand_qpos(task, physics)
                    if current_hand is None:
                        raise RuntimeError("Controller rollout could not capture current hand qpos")
                    control = _compute_controller_control(
                        action_controller,
                        action_spec=action_spec,
                        source_t=source_t,
                        substep=substep,
                        substeps=substeps,
                        control_timestep=control_timestep,
                        dataset_timestep=effective_config.dataset_timestep,
                        current_hand_qpos=current_hand,
                        current_hand_qvel=_capture_hand_qvel(task, physics),
                        source_hand_qpos=hand_joints[source_t],
                        target_hand_qpos=hand_joints[next_t],
                        previous_control=held_action_control,
                    )
                    held_action_control = control.copy()
                elif effective_config.mode == "action" and substep > 0:
                    if effective_config.action_substep_policy == "repeat":
                        control = _prepare_control(action, action_spec, effective_config)
                    elif effective_config.action_substep_policy == "zero_control":
                        control = np.zeros(action_spec.shape, dtype=np.float32)
                    elif effective_config.action_substep_policy == "zero_source":
                        control = _prepare_control(zero_action, action_spec, effective_config)
                    elif effective_config.action_substep_policy == "zero_pad_hold":
                        if held_action_control is None:
                            held_action_control = _prepare_control(action, action_spec, effective_config)
                        control = held_action_control.copy()
                    else:
                        raise ValueError(f"Unknown action_substep_policy: {effective_config.action_substep_policy}")
                else:
                    control = _prepare_control(action, action_spec, effective_config)
                    if effective_config.mode == "action":
                        held_action_control = control.copy()
                if effective_config.mode == "action":
                    control = _apply_control_overrides(control, control_overrides)
                dense_action_controls.append(np.asarray(control, dtype=np.float32).reshape(-1))
                timestep = env.step(control)
                total_reward += float(timestep.reward or 0.0)
                actions_executed += 1
                activation = _capture_piano_activation(env)
                if activation is not None:
                    activation = np.asarray(activation, dtype=np.float32)[:88]
                    interval.append(activation)
                    dense_played.append(activation)
                if action_controller is not None:
                    task, physics, _piano = _locate_task_physics_piano(env)
                    dense_hand_qpos = _capture_hand_qpos(task, physics)
                    if dense_hand_qpos is not None:
                        dense_hand.append(dense_hand_qpos.astype(np.float32))
                if timestep.last():
                    terminated = True
                    break
            task, physics, _piano = _locate_task_physics_piano(env)
            hand = _capture_hand_qpos(task, physics)
            if hand is not None:
                source_hand.append(hand.astype(np.float32))
            if interval:
                source_played.append(np.max(np.stack(interval, axis=0), axis=0).astype(np.float32))
            if effective_config.render_mp4 and source_t % render_stride == 0 and render_error is None:
                try:
                    frames.append(
                        render_frame(
                            env,
                            height=effective_config.height,
                            width=effective_config.width,
                            camera_id=effective_config.camera_id,
                        )
                    )
                except Exception as exc:
                    render_error = str(exc)
            if terminated:
                break
        if effective_config.render_mp4 and render_error is None:
            try:
                frames.append(
                    render_frame(
                        env,
                        height=effective_config.height,
                        width=effective_config.width,
                        camera_id=effective_config.camera_id,
                    )
                )
            except Exception as exc:
                render_error = str(exc)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    source_played_arr = np.stack(source_played, axis=0) if source_played else np.zeros((0, 88), dtype=np.float32)
    dense_played_arr = np.stack(dense_played, axis=0) if dense_played else np.zeros((0, 88), dtype=np.float32)
    source_hand_arr = np.stack(source_hand, axis=0) if source_hand else np.zeros((0, hand_joints.shape[-1]), dtype=np.float32)
    dense_hand_arr = np.stack(dense_hand, axis=0) if dense_hand else np.zeros((0, hand_joints.shape[-1]), dtype=np.float32)
    dense_action_control_arr = (
        np.stack(dense_action_controls, axis=0)
        if dense_action_controls
        else np.zeros((0, int(actions.shape[-1])), dtype=np.float32)
    )
    npz_path = output / "rollout.npz"
    np.savez_compressed(
        npz_path,
        source_played_piano=source_played_arr,
        dense_played_piano=dense_played_arr,
        source_hand_after_step=source_hand_arr,
        dense_hand_after_step=dense_hand_arr,
        dense_action_controls=dense_action_control_arr,
        goals=goals[: source_played_arr.shape[0], :88],
        actions=actions[:source_steps],
        reference_piano_states_for_scoring=(
            np.asarray(trajectory.reference_piano_states[: source_played_arr.shape[0], :88], dtype=np.float32)
            if trajectory.reference_piano_states is not None
            else np.zeros((0, 88), dtype=np.float32)
        ),
    )

    video_path = None
    audio_warning = None
    if effective_config.render_mp4 and render_error is None:
        audio_roll = dense_played_arr
        audio_dt = control_timestep
        events = (
            piano_roll_to_midi_events(audio_roll, dt=audio_dt, threshold=effective_config.threshold)
            if effective_config.render_audio
            else None
        )
        video_path, audio_warning = write_video(frames, output / "rollout.mp4", fps=effective_config.fps, audio_events=events)

    hand_l2 = None
    if source_hand_arr.size:
        if effective_config.mode == "action" and hand_joints.shape[0] > 1:
            reference_hand = hand_joints[1:, : source_hand_arr.shape[1]]
            alignment = "after_action_t_vs_rp1m_hand_joints_t_plus_1"
        else:
            reference_hand = hand_joints[:, : source_hand_arr.shape[1]]
            alignment = "source_t_vs_rp1m_hand_joints_t"
        n = min(source_hand_arr.shape[0], reference_hand.shape[0])
        hand_l2_values = np.linalg.norm(source_hand_arr[:n] - reference_hand[:n], axis=1)
        hand_l2 = {
            "mean": float(hand_l2_values.mean()),
            "median": float(np.median(hand_l2_values)),
            "max": float(hand_l2_values.max()),
            "scored_steps": int(n),
            "alignment": alignment,
        }

    summary: dict[str, Any] = {
        "song_key": trajectory.song_key,
        "environment_name": trajectory.environment_name,
        "demo_id": int(trajectory.demo_id),
        "mode": config.mode,
        "piano_state_policy": "not_restored_or_used_by_simulator",
        "reference_piano_state_policy": "scoring_only" if trajectory.reference_piano_states is not None else "not_loaded",
        "config": asdict(config),
        "effective_config": asdict(effective_config),
        "runtime_policy": runtime_policy,
        "action_controller": _controller_metadata(action_controller),
        "initial_hand_policy": initial_hand_policy,
        "hand_anchor_calibration": hand_anchor_calibration,
        "post_reset_hand_anchor": post_reset_hand_anchor,
        "mujoco_options_applied": mujoco_options_applied,
        "action_control_overrides": {
            "wrist_action_policy": effective_config.wrist_action_policy,
            "indices": sorted(int(idx) for idx in control_overrides),
            "values": {str(idx): float(value) for idx, value in sorted(control_overrides.items())},
        },
        "load_info": load_info,
        "source_steps_requested": int(source_steps),
        "source_steps_played": int(source_played_arr.shape[0]),
        "dense_steps_played": int(dense_played_arr.shape[0]),
        "substeps_per_source_step": int(substeps),
        "actions_executed": int(actions_executed),
        "terminated": bool(terminated),
        "total_reward": float(total_reward),
        "qpos_restored_count": int(qpos_restored),
        "qvel_restored_count": int(qvel_restored),
        "hand_resync_policy": {
            "interval": (
                None
                if effective_config.hand_resync_interval is None
                else int(effective_config.hand_resync_interval)
            ),
            "uses_rp1m_hand_joints": bool(effective_config.hand_resync_interval is not None),
            "resync_count": int(hand_resync_count),
        },
        "hand_qpos_l2_vs_reference": hand_l2,
        "rollout_npz": str(npz_path),
        "video_path": None if video_path is None else str(video_path),
        "rendered_frames": int(len(frames)),
        "render_error": render_error,
        "audio_warning": audio_warning,
        "recorded_reference_against_goals": _recorded_reference_score(trajectory, effective_config.threshold),
        **_score_rollout(
            played=source_played_arr,
            goals=goals,
            reference_piano_states=trajectory.reference_piano_states,
            threshold=effective_config.threshold,
        ),
    }
    save_json(output / "summary.json", summary)
    return summary


def parse_example(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Examples must be SONG_KEY:DEMO_ID")
    song_key, demo = value.rsplit(":", 1)
    try:
        return song_key, int(demo)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid demo id in {value!r}") from exc


def write_validation_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    fields = [
        "song_key",
        "demo_id",
        "mode",
        "recorded_goal_f1",
        "rollout_goal_f1",
        "rollout_reference_f1",
        "precision",
        "recall",
        "source_steps_played",
        "video_path",
        "summary_path",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return out


def row_from_summary(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    goals = summary.get("against_goals") or {}
    ref = summary.get("against_reference_piano_states") or {}
    recorded = summary.get("recorded_reference_against_goals") or {}
    return {
        "song_key": summary.get("song_key"),
        "demo_id": summary.get("demo_id"),
        "mode": summary.get("mode"),
        "recorded_goal_f1": recorded.get("key_f1"),
        "rollout_goal_f1": goals.get("key_f1"),
        "rollout_reference_f1": ref.get("key_f1"),
        "precision": goals.get("key_precision"),
        "recall": goals.get("key_recall"),
        "source_steps_played": summary.get("source_steps_played"),
        "video_path": summary.get("video_path"),
        "summary_path": str(summary_path),
    }
