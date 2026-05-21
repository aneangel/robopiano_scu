from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from partita.evaluation.metrics import key_metrics
from partita.utils.io import ensure_dir, save_json


ACTION_MAPPINGS: tuple[str, ...] = (
    "as_is",
    "swap_hands",
    "zero_sustain",
    "invert_sustain",
    "swap_hands_zero_sustain",
)

ACTION_SOURCE_SCALES: tuple[str, ...] = (
    "normalized_minus_one_to_one",
    "actuator_units",
)


def canonicalize_rp1m_environment_name(song_name: str) -> str:
    return re.sub(r"(-v\d+)_\d+$", r"\1", str(song_name))


def candidate_environment_names(song_name: str) -> list[str]:
    canonical = canonicalize_rp1m_environment_name(song_name)
    candidates = [canonical]
    if "repertoire-150" in canonical:
        candidates.append(canonical.replace("repertoire-150", "etude-12"))
    seen = set()
    out = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _load_music_pb2():
    # Avoid importing top-level note_seq here. On some WAVE nodes that path imports
    # fluidsynth and can hit a system libstdc++ mismatch before rollout starts.
    import importlib.util
    import site

    candidates = []
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        candidates.append(Path(root) / "note_seq" / "protobuf" / "music_pb2.py")
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("partita_note_seq_music_pb2", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not find note_seq/protobuf/music_pb2.py for generated MIDI proto writing.")


def goals_to_note_sequence(goals: np.ndarray, *, dt: float = 0.05, title: str = "Partita target goals"):
    music_pb2 = _load_music_pb2()

    goal = np.asarray(goals) > 0.5
    if goal.ndim != 2:
        raise ValueError(f"Expected goals with shape [T, keys], got {goal.shape}")
    key_dim = min(goal.shape[1], 88)
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = title
    seq.sequence_metadata.artist = "partita"
    for key in range(key_dim):
        active = goal[:, key]
        start = None
        for t, is_active in enumerate(active):
            if is_active and start is None:
                start = t
            if start is not None and (not is_active or t == len(active) - 1):
                end_t = t + 1 if is_active and t == len(active) - 1 else t
                if end_t > start:
                    note = seq.notes.add()
                    note.pitch = int(21 + key)  # A0 is MIDI 21.
                    note.velocity = 80
                    note.start_time = float(start * dt)
                    note.end_time = float(max(end_t * dt, start * dt + dt))
                    note.part = 0  # Leave fingering unspecified; RoboPianist will use OT fingering reward.
                start = None
    seq.total_time = float(goal.shape[0] * dt)
    seq.tempos.add(qpm=60)
    return seq


def write_goals_proto(goals: np.ndarray, path: str | Path, *, dt: float, title: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    seq = goals_to_note_sequence(goals, dt=dt, title=title)
    with path.open("wb") as f:
        f.write(seq.SerializeToString())
    return path


def _iter_wrapped_envs(env: Any) -> list[Any]:
    current = env
    wrapped = []
    seen = set()
    while current is not None and id(current) not in seen:
        wrapped.append(current)
        seen.add(id(current))
        current = getattr(current, "_environment", None)
    return wrapped


def render_frame(env: Any, *, height: int, width: int) -> np.ndarray:
    errors = []
    for current in _iter_wrapped_envs(env):
        physics = getattr(current, "physics", None)
        if physics is not None and hasattr(physics, "render"):
            for kwargs in ({"height": height, "width": width}, {"height": height, "width": width, "camera_id": 0}):
                try:
                    frame = physics.render(**kwargs)
                    arr = np.asarray(frame, dtype=np.uint8)
                    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                        return arr[..., :3]
                except Exception as exc:
                    errors.append(f"physics.render({kwargs}): {exc}")
    raise RuntimeError("Unable to render RoboPianist frame. Try MUJOCO_GL=egl or MUJOCO_GL=osmesa. " + " | ".join(errors[:4]))


def write_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    *,
    fps: int,
    audio_events: list[Any] | None = None,
) -> tuple[Path, str, str | None]:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if not frames:
        raise ValueError("No frames were captured.")
    audio_warning = None
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(output_path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=None)
        if audio_events:
            audio_warning = _attach_keypress_audio(output_path, audio_events)
        return output_path, "mp4", audio_warning
    except Exception as mp4_exc:
        gif_path = output_path.with_suffix(".gif")
        import imageio.v2 as imageio

        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path, f"gif fallback after MP4 failure: {mp4_exc}", None



def _scale_action_to_spec(action: np.ndarray, action_spec: Any, *, source: str) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    control = np.zeros_like(minimum, dtype=np.float32)
    width = min(control.size, values.size)
    if width <= 0:
        return control
    clipped = np.clip(values[:width], -1.0, 1.0)
    if source == "normalized_minus_one_to_one":
        control[:width] = minimum[:width] + 0.5 * (clipped + 1.0) * (maximum[:width] - minimum[:width])
    elif source == "actuator_units":
        control[:width] = values[:width]
    else:
        raise ValueError(f"Unknown action source scale: {source}")
    return np.clip(control, minimum, maximum)


def _action_spec_statistics(actions: np.ndarray, action_spec: Any) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float32)
    flat = values.reshape(-1)
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    input_dim = int(values.shape[-1]) if values.ndim >= 2 else int(flat.size)
    env_dim = int(minimum.size)
    if flat.size:
        input_stats: dict[str, Any] = {
            "input_actions_min": float(np.min(flat)),
            "input_actions_max": float(np.max(flat)),
            "input_actions_mean": float(np.mean(flat)),
            "input_actions_std": float(np.std(flat)),
            "input_actions_abs_p95": float(np.percentile(np.abs(flat), 95)),
        }
    else:
        input_stats = {
            "input_actions_min": None,
            "input_actions_max": None,
            "input_actions_mean": None,
            "input_actions_std": None,
            "input_actions_abs_p95": None,
        }
    return {
        **input_stats,
        "env_action_spec_min": float(np.min(minimum)) if minimum.size else None,
        "env_action_spec_max": float(np.max(maximum)) if maximum.size else None,
        "env_action_dim": env_dim,
        "input_action_dim": input_dim,
        "action_dim_width_used": int(min(env_dim, input_dim)),
    }


def _canonical_midi_available(env_name: str) -> bool:
    try:
        from robopianist.suite import ALL as _ROBO_ALL
    except Exception:
        return False
    return env_name in _ROBO_ALL


def _load_env(
    *,
    environment_names: list[str],
    midi_proto_path: Path,
    control_timestep: float,
    seed: int,
    reduced_action_space: bool = True,
    extra_task_kwargs: dict[str, Any] | None = None,
    suite_load_kwargs: dict[str, Any] | None = None,
    prefer_canonical_midi: bool = False,
) -> tuple[str, Any, dict[str, Any]]:
    import sys

    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from robopianist import suite

    base_task_kwargs: dict[str, Any] = {
        "control_timestep": float(control_timestep),
        "n_steps_lookahead": 1,
        "disable_colorization": False,
        "disable_hand_collisions": False,
        "reduced_action_space": bool(reduced_action_space),
    }
    if extra_task_kwargs:
        for key, value in extra_task_kwargs.items():
            if value is None:
                continue
            base_task_kwargs[key] = value

    suite_kwargs: dict[str, Any] = {}
    if suite_load_kwargs:
        for key, value in suite_load_kwargs.items():
            if value is None:
                continue
            suite_kwargs[key] = value

    last_error = None
    for env_name in environment_names:
        used_canonical = bool(prefer_canonical_midi) and _canonical_midi_available(env_name)
        load_kwargs = dict(suite_kwargs)
        if not used_canonical:
            load_kwargs["midi_file"] = midi_proto_path
        try:
            env = suite.load(
                environment_name=env_name,
                seed=seed,
                task_kwargs=base_task_kwargs,
                **load_kwargs,
            )
            suite_kwargs_record = {}
            for key, value in load_kwargs.items():
                if isinstance(value, Path):
                    suite_kwargs_record[key] = str(value)
                else:
                    suite_kwargs_record[key] = value
            load_info = {
                "task_kwargs": dict(base_task_kwargs),
                "suite_load_kwargs": suite_kwargs_record,
                "used_canonical_midi": bool(used_canonical),
            }
            return env_name, env, load_info
        except Exception as exc:  # pragma: no cover - depends on local RoboPianist assets.
            last_error = exc
    raise RuntimeError(f"Could not load RoboPianist environment from candidates {environment_names}: {last_error}")


def _collect_metrics(env: Any, *, terminated: bool) -> dict[str, float | str]:
    if terminated:
        for current in _iter_wrapped_envs(env):
            get_metrics = getattr(current, "get_musical_metrics", None)
            if callable(get_metrics):
                try:
                    return {k: float(v) for k, v in dict(get_metrics()).items()}
                except Exception as exc:
                    return {"metrics_error": str(exc)}
    return {"metrics_note": "Episode did not terminate before action sequence ended; final musical metrics unavailable."}


def _capture_piano_activation(env: Any) -> np.ndarray | None:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        piano = getattr(task, "piano", None)
        activation = getattr(piano, "activation", None)
        if activation is not None:
            return np.asarray(activation, dtype=np.float32).reshape(-1)
    return None


def _find_piano_midi_events(env: Any) -> list[Any]:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        piano = getattr(task, "piano", None)
        midi_module = getattr(piano, "midi_module", None)
        get_all = getattr(midi_module, "get_all_midi_messages", None)
        if callable(get_all):
            return list(get_all())
    return []


def _note_midi_events(events: list[Any]) -> list[Any]:
    return [event for event in events if type(event).__name__ in {"NoteOn", "NoteOff"}]


def _rollout_key_metrics(goals: np.ndarray, piano_roll: list[np.ndarray]) -> dict[str, float | str | int]:
    if not piano_roll:
        return {"rollout_key_metrics_note": "No piano activation frames were captured from RoboPianist rollout."}
    played = np.stack(piano_roll, axis=0)
    target = np.asarray(goals, dtype=np.float32)
    steps = min(int(target.shape[0]), int(played.shape[0]))
    keys = min(int(target.shape[-1]), int(played.shape[-1]), 88)
    metrics = key_metrics(target[:steps, :keys], played[:steps, :keys], threshold=0.5)
    return {
        "rollout_key_precision": metrics["key_precision"],
        "rollout_key_recall": metrics["key_recall"],
        "rollout_key_f1": metrics["key_f1"],
        "rollout_mispress_rate": metrics["mispress_rate"],
        "rollout_scored_steps": int(steps),
        "rollout_scored_keys": int(keys),
        "rollout_scoring_source": "robopianist_piano_activation_from_replayed_actions",
    }


def _locate_task_physics_piano(env: Any) -> tuple[Any, Any, Any]:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        physics = getattr(current, "physics", None)
        piano = getattr(task, "piano", None)
        if task is not None and physics is not None and piano is not None:
            return task, physics, piano
    raise RuntimeError("Could not locate RoboPianist task, physics, and piano handles.")


def _set_reduced_hand_qpos(task: Any, physics: Any, hand_qpos: np.ndarray) -> int:
    values = np.asarray(hand_qpos, dtype=np.float32).reshape(-1)
    joints = []
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name, None)
        hand_joints = getattr(hand, "joints", None)
        if hand_joints is not None:
            joints.extend(list(hand_joints))
    if values.size < len(joints):
        raise ValueError(f"RP1M hand_joints has {values.size} values but environment expects {len(joints)} joints.")
    for joint, value in zip(joints, values[: len(joints)]):
        physics.bind(joint).qpos = float(value)
    return len(joints)


def _set_piano_qpos_from_state(piano: Any, physics: Any, piano_state: np.ndarray, threshold: float) -> np.ndarray:
    state = np.asarray(piano_state, dtype=np.float32).reshape(-1)
    key_active = state[:88] > float(threshold)
    qpos_range = np.asarray(getattr(piano, "_qpos_range"), dtype=np.float64)
    inactive = np.maximum(qpos_range[:, 0], 0.0)
    active = qpos_range[:, 1]
    physics.bind(piano.joints).qpos = np.where(key_active, active, inactive)
    if state.size > 88:
        getattr(piano, "_sustain_state")[0] = float(state[88])
    else:
        getattr(piano, "_sustain_state")[0] = 0.0
    if hasattr(physics, "forward"):
        physics.forward()
    piano._update_key_state(physics)
    piano._update_key_color(physics)
    return np.asarray(piano.activation, dtype=np.float32).reshape(-1)[:88]


def _hand_joint_handles(task: Any) -> list[Any]:
    joints: list[Any] = []
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name, None)
        hand_joints = getattr(hand, "joints", None)
        if hand_joints is not None:
            joints.extend(list(hand_joints))
    return joints


def _capture_hand_qpos(task: Any, physics: Any) -> np.ndarray | None:
    joints = _hand_joint_handles(task)
    if not joints:
        return None
    values: list[float] = []
    for joint in joints:
        try:
            q = np.asarray(physics.bind(joint).qpos, dtype=np.float64).reshape(-1)
            values.append(float(q[0]) if q.size else 0.0)
        except Exception:
            return None
    return np.asarray(values, dtype=np.float32)


def _zero_joint_velocities(physics: Any, joints: list[Any]) -> str | None:
    failed = 0
    for joint in joints:
        try:
            physics.bind(joint).qvel = 0.0
        except Exception:
            failed += 1
    if hasattr(physics, "forward"):
        try:
            physics.forward()
        except Exception as exc:
            return f"physics.forward after qvel zeroing failed: {exc}"
    if failed:
        return f"failed to zero qvel on {failed} joints"
    return None


def _restore_initial_rp1m_state(
    env: Any,
    *,
    hand_joints_t0: np.ndarray | None,
    piano_state_t0: np.ndarray | None = None,
    sustain_t0: float | None = None,
    key_threshold: float = 0.5,
    zero_velocities: bool = True,
) -> dict[str, Any]:
    """Copy the RP1M dataset's t=0 physics state into the simulator after env.reset().

    Restores hand joint qpos, optionally piano key qpos / sustain pedal, and zeros velocities so
    open-loop replay starts from the same configuration RP1M did at collection time. Required
    before stepping recorded RP1M actions if you want them to track recorded states.
    """
    info: dict[str, Any] = {
        "hand_joints_restored": 0,
        "piano_state_restored": False,
        "sustain_restored": False,
        "qvel_zeroed": False,
    }
    task, physics, piano = _locate_task_physics_piano(env)
    if hand_joints_t0 is not None:
        info["hand_joints_restored"] = _set_reduced_hand_qpos(task, physics, hand_joints_t0)
    if piano_state_t0 is not None:
        _set_piano_qpos_from_state(piano, physics, piano_state_t0, key_threshold)
        info["piano_state_restored"] = True
        if np.asarray(piano_state_t0).reshape(-1).size > 88:
            info["sustain_restored"] = True
    elif sustain_t0 is not None:
        try:
            getattr(piano, "_sustain_state")[0] = float(sustain_t0)
            info["sustain_restored"] = True
        except Exception as exc:
            info["sustain_warning"] = str(exc)
    if zero_velocities:
        joints = _hand_joint_handles(task)
        try:
            joints.extend(list(piano.joints))
        except Exception:
            pass
        warning = _zero_joint_velocities(physics, joints)
        if warning is None:
            info["qvel_zeroed"] = True
        else:
            info["qvel_warning"] = warning
    elif hasattr(physics, "forward"):
        try:
            physics.forward()
        except Exception:
            pass
    return info


def map_action_channels(action: np.ndarray, *, mapping: str) -> np.ndarray:
    """Reinterpret a 39D RP1M action vector under different channel-layout assumptions."""
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if mapping == "as_is":
        return values
    if values.size != 39:
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


def _prepare_control(
    action: np.ndarray,
    action_spec: Any,
    *,
    mapping: str,
    source_scale: str,
) -> np.ndarray:
    mapped = map_action_channels(action, mapping=mapping)
    return _scale_action_to_spec(mapped, action_spec, source=source_scale)


def piano_roll_to_midi_events(piano_roll: np.ndarray, *, dt: float, threshold: float) -> list[Any]:
    from robopianist.music import midi_file, midi_message

    active = np.asarray(piano_roll, dtype=np.float32)[:, :88] > float(threshold)
    events: list[Any] = []
    for key in range(active.shape[1]):
        was_active = False
        for step, is_active in enumerate(active[:, key]):
            time_value = float(step * dt)
            if is_active and not was_active:
                events.append(
                    midi_message.NoteOn(
                        note=midi_file.key_number_to_midi_number(key),
                        velocity=127,
                        time=time_value,
                    )
                )
            elif was_active and not is_active:
                events.append(
                    midi_message.NoteOff(
                        note=midi_file.key_number_to_midi_number(key),
                        time=time_value,
                    )
                )
            was_active = bool(is_active)
        if was_active:
            events.append(
                midi_message.NoteOff(
                    note=midi_file.key_number_to_midi_number(key),
                    time=float(active.shape[0] * dt),
                )
            )
    events.sort(key=lambda event: (float(getattr(event, "time", 0.0)), 0 if type(event).__name__ == "NoteOff" else 1))
    return events


def rollout_recorded_rp1m_episode_with_robopianist(
    *,
    hand_joints: np.ndarray,
    piano_states: np.ndarray,
    goals: np.ndarray,
    song_name: str,
    output_dir: str | Path,
    label: str = "rp1m_recorded_state",
    control_timestep: float = 0.05,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    max_steps: int | None = None,
    render_every: int = 1,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Render RP1M recorded hand/key states directly in RoboPianist.

    This validates dataset playback independently from open-loop action replay. Audio is
    generated only from transitions in recorded RP1M piano key states.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = ensure_dir(output_dir)
    hand_joints = np.asarray(hand_joints, dtype=np.float32)
    piano_states = np.asarray(piano_states, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    if hand_joints.ndim != 2:
        raise ValueError(f"Expected hand_joints [T, joints], got {hand_joints.shape}")
    if piano_states.ndim != 2:
        raise ValueError(f"Expected piano_states [T, keys], got {piano_states.shape}")
    steps = min(int(hand_joints.shape[0]), int(piano_states.shape[0]), int(goals.shape[0]))
    if max_steps is not None:
        steps = min(steps, int(max_steps))

    midi_proto_path = write_goals_proto(
        goals,
        output_dir / f"{label}_target_goals.proto",
        dt=control_timestep,
        title=f"Partita {label} {song_name}",
    )
    env_name, env, _load_info = _load_env(
        environment_names=candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=control_timestep,
        seed=seed,
        reduced_action_space=True,
    )
    frames: list[np.ndarray] = []
    played_roll: list[np.ndarray] = []
    render_error = None
    restored_hand_joint_count = 0
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        for step_index in range(steps):
            restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, hand_joints[step_index])
            activation = _set_piano_qpos_from_state(piano, physics, piano_states[step_index], threshold)
            played_roll.append(activation)
            if step_index % max(int(render_every), 1) == 0 and render_error is None:
                try:
                    frames.append(render_frame(env, height=height, width=width))
                except Exception as exc:
                    render_error = str(exc)
        video_path = None
        video_format = None
        audio_warning = None
        audio_events = piano_roll_to_midi_events(piano_states[:steps], dt=control_timestep, threshold=threshold)
        if render_error is None:
            video_path, video_format, audio_warning = write_video(
                frames,
                output_dir / f"{label}_playback.mp4",
                fps=max(int(fps / max(render_every, 1)), 1),
                audio_events=audio_events,
            )
        played = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
        goal_steps = min(int(goals.shape[0]), int(played.shape[0]))
        goal_keys = min(int(goals.shape[-1]), int(played.shape[-1]), 88)
        state_steps = min(int(piano_states.shape[0]), int(played.shape[0]))
        state_keys = min(int(piano_states.shape[-1]), int(played.shape[-1]), 88)
        against_goals = key_metrics(goals[:goal_steps, :goal_keys], played[:goal_steps, :goal_keys], threshold=threshold)
        against_states = key_metrics(piano_states[:state_steps, :state_keys], played[:state_steps, :state_keys], threshold=threshold)
        result = {
            "label": label,
            "song_name": song_name,
            "environment_name": env_name,
            "playback_mode": "recorded_rp1m_hand_joints_and_piano_states",
            "audio_source": "recorded_rp1m_piano_state_key_transitions",
            "midi_proto_path": str(midi_proto_path),
            "hand_joints_shape": list(hand_joints.shape),
            "piano_states_shape": list(piano_states.shape),
            "steps_rendered": int(steps),
            "restored_hand_joint_count": int(restored_hand_joint_count),
            "rendered_frames": int(len(frames)),
            "render_every": int(render_every),
            "render_error": render_error,
            "video_path": str(video_path) if video_path is not None else None,
            "video_format": video_format,
            "audio_warning": audio_warning,
            "audio_midi_note_event_count": int(len(_note_midi_events(audio_events))),
            "against_goals": {
                **against_goals,
                "scored_steps": int(goal_steps),
                "scored_keys": int(goal_keys),
            },
            "against_rp1m_piano_states": {
                **against_states,
                "scored_steps": int(state_steps),
                "scored_keys": int(state_keys),
            },
        }
        save_json(output_dir / f"{label}_playback.json", result)
        return result
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _default_soundfont_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "robopianist" / "soundfonts" / "SalamanderGrandPiano.sf2",
        repo_root / "robopianist" / "soundfonts" / "TimGM6mb.sf2",
        repo_root / "robopianist" / "third_party" / "soundfonts" / "TimGM6mb.sf2",
        repo_root / "third_party" / "soundfonts" / "TimGM6mb.sf2",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                with candidate.open("rb") as handle:
                    if handle.read(4) == b"RIFF":
                        return candidate
            except OSError:
                pass
    return None


def _clone_midi_events(events: list[Any]) -> list[Any]:
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


def _write_waveform(path: Path, waveform: np.ndarray, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(waveform, dtype=np.int16).tobytes())


def _attach_keypress_audio(video_path: Path, audio_events: list[Any]) -> str | None:
    note_events = _note_midi_events(audio_events)
    if not note_events:
        return "Audio mux skipped because rollout produced no piano key press MIDI events."
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return "Audio mux skipped because ffmpeg was not found on PATH."
    soundfont_path = _default_soundfont_path()
    if soundfont_path is None:
        return "Audio mux skipped because no valid soundfont file was found."
    try:
        from robopianist.music import synthesizer
    except Exception as exc:
        return f"Audio mux skipped because RoboPianist synthesizer import failed: {exc}"

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_audio_", dir=str(video_path.parent)))
    wav_path = temp_dir / f"{video_path.stem}.wav"
    temp_video = temp_dir / video_path.name
    try:
        synth = synthesizer.Synthesizer(soundfont_path=soundfont_path)
        try:
            waveform = synth.get_samples(_clone_midi_events(note_events))
        finally:
            synth.stop()
        _write_waveform(wav_path, waveform)
        shutil.copyfile(video_path, temp_video)
        subprocess.run(
            [
                ffmpeg_path,
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
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return None
    except Exception as exc:
        return f"Audio mux failed: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _hand_qpos_l2(sim: np.ndarray | None, ref: np.ndarray | None) -> float | None:
    if sim is None or ref is None:
        return None
    sim_arr = np.asarray(sim, dtype=np.float32).reshape(-1)
    ref_arr = np.asarray(ref, dtype=np.float32).reshape(-1)
    width = min(sim_arr.size, ref_arr.size)
    if width == 0:
        return None
    diff = sim_arr[:width] - ref_arr[:width]
    return float(np.linalg.norm(diff))


def _active_indices(frame: np.ndarray | None, threshold: float) -> list[int]:
    if frame is None:
        return []
    arr = np.asarray(frame).reshape(-1)[:88]
    return [int(i) for i in np.flatnonzero(arr > threshold)]


def _write_fidelity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fields = [
        "step",
        "hand_qpos_l2",
        "piano_state_iou",
        "sim_keys",
        "ref_keys",
        "goal_keys",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _summarize_fidelity(
    *,
    sim_hand: list[np.ndarray],
    sim_piano: list[np.ndarray],
    ref_hand: np.ndarray | None,
    ref_piano: np.ndarray | None,
    goals: np.ndarray | None,
    threshold: float,
    early_window: int = 50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hand_l2_series: list[float] = []
    piano_iou_series: list[float] = []
    steps = len(sim_piano)
    for step_index in range(steps):
        sim_pose = sim_hand[step_index] if step_index < len(sim_hand) else None
        ref_pose = (
            ref_hand[step_index]
            if ref_hand is not None and step_index < ref_hand.shape[0]
            else None
        )
        l2 = _hand_qpos_l2(sim_pose, ref_pose)
        sim_frame = sim_piano[step_index]
        ref_frame = ref_piano[step_index] if ref_piano is not None and step_index < ref_piano.shape[0] else None
        goal_frame = goals[step_index] if goals is not None and step_index < goals.shape[0] else None
        sim_keys = _active_indices(sim_frame, threshold)
        ref_keys = _active_indices(ref_frame, threshold) if ref_frame is not None else []
        goal_keys = _active_indices(goal_frame, threshold) if goal_frame is not None else []
        iou: float | None = None
        if ref_frame is not None:
            sim_set = set(sim_keys)
            ref_set = set(ref_keys)
            union = sim_set | ref_set
            iou = float(len(sim_set & ref_set) / max(len(union), 1))
            piano_iou_series.append(iou)
        if l2 is not None:
            hand_l2_series.append(l2)
        rows.append(
            {
                "step": int(step_index),
                "hand_qpos_l2": "" if l2 is None else float(l2),
                "piano_state_iou": "" if iou is None else float(iou),
                "sim_keys": json.dumps(sim_keys),
                "ref_keys": json.dumps(ref_keys),
                "goal_keys": json.dumps(goal_keys),
            }
        )
    summary: dict[str, Any] = {
        "scored_steps": int(steps),
        "hand_qpos_available": bool(hand_l2_series),
        "piano_states_available": bool(piano_iou_series),
    }
    if hand_l2_series:
        arr = np.asarray(hand_l2_series, dtype=np.float64)
        early = arr[: max(int(early_window), 1)]
        summary.update(
            {
                "hand_qpos_l2_mean": float(arr.mean()),
                "hand_qpos_l2_max": float(arr.max()),
                "hand_qpos_l2_mean_first_n": float(early.mean()) if early.size else None,
                "hand_qpos_l2_first_n_window": int(min(int(early_window), arr.size)),
            }
        )
    if piano_iou_series:
        ious = np.asarray(piano_iou_series, dtype=np.float64)
        summary["piano_state_iou_mean"] = float(ious.mean())
    if sim_piano and ref_piano is not None:
        played = np.stack(sim_piano, axis=0)
        state_steps = min(int(ref_piano.shape[0]), int(played.shape[0]))
        state_keys = min(int(ref_piano.shape[-1]), int(played.shape[-1]), 88)
        if state_steps > 0 and state_keys > 0:
            against_states = key_metrics(
                ref_piano[:state_steps, :state_keys],
                played[:state_steps, :state_keys],
                threshold=threshold,
            )
            summary["against_rp1m_piano_states"] = {
                **against_states,
                "scored_steps": int(state_steps),
                "scored_keys": int(state_keys),
            }
    if sim_piano and goals is not None:
        played = np.stack(sim_piano, axis=0)
        goal_steps = min(int(goals.shape[0]), int(played.shape[0]))
        goal_keys = min(int(goals.shape[-1]), int(played.shape[-1]), 88)
        if goal_steps > 0 and goal_keys > 0:
            against_goals = key_metrics(
                goals[:goal_steps, :goal_keys],
                played[:goal_steps, :goal_keys],
                threshold=threshold,
            )
            summary["against_goals"] = {
                **against_goals,
                "scored_steps": int(goal_steps),
                "scored_keys": int(goal_keys),
            }
    return summary, rows


def diagnose_rp1m_one_step_consistency(
    *,
    actions: np.ndarray,
    hand_joints: np.ndarray,
    piano_states: np.ndarray,
    goals: np.ndarray,
    song_name: str,
    output_dir: str | Path,
    control_timestep: float = 0.05,
    seed: int = 0,
    reduced_action_space: bool = True,
    action_source_scale: str = "normalized_minus_one_to_one",
    action_mapping: str = "as_is",
    extra_task_kwargs: dict[str, Any] | None = None,
    suite_load_kwargs: dict[str, Any] | None = None,
    prefer_canonical_midi: bool = False,
    key_threshold: float = 0.5,
    max_pairs: int | None = None,
    start_t: int = 0,
    require_exact_action_dim: bool = True,
) -> dict[str, Any]:
    """Isolate open-loop dynamics mismatch without trajectory accumulation.

    For each timestep ``t`` in ``[start_t, min(T - 2, start_t + max_pairs - 1)]``:

    1. ``env.reset()``.
    2. Restore RP1M ``hand_joints[t]`` and ``piano_states[t]`` (plus zero velocities).
    3. Apply ``actions[t]`` once via ``_prepare_control`` / ``env.step``.
    4. Compare simulator ``hand_qpos`` to RP1M ``hand_joints[t + 1]`` and piano activation to
       ``piano_states[t + 1]`` (first 88 keys, thresholded F1).

    Convention matches standard MDP rollout indexing:
    ``(state_t, action_t) -> state_{t+1}``. If your RP1M export aligns differently,
    compare aggregates across offsets externally.

    Writes ``rp1m_one_step_consistency.csv`` and ``rp1m_one_step_consistency_summary.json``
    under ``output_dir``.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = ensure_dir(Path(output_dir))
    actions = np.asarray(actions, dtype=np.float32)
    hand_joints = np.asarray(hand_joints, dtype=np.float32)
    piano_states = np.asarray(piano_states, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    if action_mapping not in ACTION_MAPPINGS:
        raise ValueError(f"Unknown action_mapping '{action_mapping}'")
    if action_source_scale not in ACTION_SOURCE_SCALES:
        raise ValueError(f"Unknown action_source_scale '{action_source_scale}'")

    length = min(
        int(actions.shape[0]),
        int(hand_joints.shape[0]),
        int(piano_states.shape[0]),
        int(goals.shape[0]),
    )
    if length < 2:
        raise ValueError(f"Need at least two timesteps for one-step diagnosis; got length={length}")
    end_t = length - 1
    if max_pairs is not None:
        end_t = min(end_t, start_t + max(int(max_pairs), 1))

    midi_proto_path = write_goals_proto(
        goals,
        output_dir / "rp1m_one_step_target_goals.proto",
        dt=control_timestep,
        title=f"Partita one-step diagnostic {song_name}",
    )
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=control_timestep,
        seed=seed,
        reduced_action_space=reduced_action_space,
        extra_task_kwargs=extra_task_kwargs,
        suite_load_kwargs=suite_load_kwargs,
        prefer_canonical_midi=prefer_canonical_midi,
    )

    rows: list[dict[str, Any]] = []
    try:
        for t in range(start_t, end_t):
            env.reset()
            _restore_initial_rp1m_state(
                env,
                hand_joints_t0=hand_joints[t],
                piano_state_t0=piano_states[t],
                key_threshold=key_threshold,
                zero_velocities=True,
            )
            action_spec = env.action_spec()
            if require_exact_action_dim and int(actions.shape[-1]) != int(action_spec.shape[0]):
                raise ValueError(
                    f"Action dim {actions.shape[-1]} != env {action_spec.shape[0]} at step {t}; "
                    "align rollout.reduced_action_space."
                )
            control = _prepare_control(
                actions[t],
                action_spec,
                mapping=action_mapping,
                source_scale=action_source_scale,
            )
            timestep = env.step(control)
            task, physics, _piano = _locate_task_physics_piano(env)
            sim_hand = _capture_hand_qpos(task, physics)
            ref_hand_next = hand_joints[t + 1]
            hand_l2 = _hand_qpos_l2(sim_hand, ref_hand_next)
            sim_activation = _capture_piano_activation(env)
            ref_next = piano_states[t + 1]
            piano_km = {"key_f1": float("nan"), "key_precision": float("nan"), "key_recall": float("nan")}
            piano_iou: float | None = None
            sim_keys: list[int] = []
            ref_keys: list[int] = []
            if sim_activation is not None and ref_next.size >= 88:
                ref88 = np.asarray(ref_next[:88], dtype=np.float32).reshape(1, -1)
                sim88 = np.asarray(sim_activation[:88], dtype=np.float32).reshape(1, -1)
                piano_km = key_metrics(ref88, sim88, threshold=key_threshold)
                sim_keys = _active_indices(sim_activation, key_threshold)
                ref_keys = _active_indices(ref88.reshape(-1), key_threshold)
                union = set(sim_keys) | set(ref_keys)
                piano_iou = float(len(set(sim_keys) & set(ref_keys)) / max(len(union), 1))

            rows.append(
                {
                    "step_t": int(t),
                    "hand_qpos_l2": "" if hand_l2 is None else float(hand_l2),
                    "piano_key_f1": piano_km.get("key_f1"),
                    "piano_precision": piano_km.get("key_precision"),
                    "piano_recall": piano_km.get("key_recall"),
                    "piano_iou": "" if piano_iou is None else float(piano_iou),
                    "sim_keys": json.dumps(sim_keys),
                    "ref_keys": json.dumps(ref_keys),
                    "episode_done": bool(timestep.last()),
                }
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    csv_path = output_dir / "rp1m_one_step_consistency.csv"
    fields = [
        "step_t",
        "hand_qpos_l2",
        "piano_key_f1",
        "piano_precision",
        "piano_recall",
        "piano_iou",
        "sim_keys",
        "ref_keys",
        "episode_done",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    hand_vals = np.asarray(
        [float(row["hand_qpos_l2"]) for row in rows if row.get("hand_qpos_l2") != ""],
        dtype=np.float64,
    )
    f1_vals = np.asarray(
        [
            float(row["piano_key_f1"])
            for row in rows
            if row.get("piano_key_f1") is not None and np.isfinite(float(row["piano_key_f1"]))
        ],
        dtype=np.float64,
    )
    iou_vals = np.asarray(
        [float(row["piano_iou"]) for row in rows if row.get("piano_iou") != ""],
        dtype=np.float64,
    )

    def _pct(arr: np.ndarray, q: float) -> float | None:
        if arr.size == 0:
            return None
        return float(np.percentile(arr, q))

    summary: dict[str, Any] = {
        "song_name": song_name,
        "environment_name": env_name,
        "steps_diagnosed": int(len(rows)),
        "start_t": int(start_t),
        "max_pairs": max_pairs,
        "convention": (
            "restore_RP1M_hand_joints[t]_and_piano_states[t], env.step(actions[t]), "
            "compare_simulator_to_RP1M_hand_joints[t+1]_and_piano_states[t+1]"
        ),
        "action_source_scale": action_source_scale,
        "action_mapping": action_mapping,
        "prefer_canonical_midi": bool(prefer_canonical_midi),
        "used_canonical_midi": bool(load_info.get("used_canonical_midi", False)),
        "task_kwargs": load_info.get("task_kwargs"),
        "suite_load_kwargs": load_info.get("suite_load_kwargs"),
        "hand_qpos_l2_vs_rp1m_next": {
            "mean": float(hand_vals.mean()) if hand_vals.size else None,
            "median": float(np.median(hand_vals)) if hand_vals.size else None,
            "p95": _pct(hand_vals, 95),
            "max": float(hand_vals.max()) if hand_vals.size else None,
        },
        "piano_next_step_key_f1_vs_rp1m": {
            "mean": float(f1_vals.mean()) if f1_vals.size else None,
            "median": float(np.median(f1_vals)) if f1_vals.size else None,
            "p05": _pct(f1_vals, 5),
            "min": float(f1_vals.min()) if f1_vals.size else None,
        },
        "piano_next_step_iou": {
            "mean": float(iou_vals.mean()) if iou_vals.size else None,
            "median": float(np.median(iou_vals)) if iou_vals.size else None,
        },
        "csv_path": str(csv_path),
        "midi_proto_path": str(midi_proto_path),
    }
    summary_path = output_dir / "rp1m_one_step_consistency_summary.json"
    save_json(summary_path, summary)
    summary["summary_json_path"] = str(summary_path)
    return summary


def calibrate_action_scale(
    *,
    actions: np.ndarray,
    reference_hand_joints: np.ndarray,
    reference_piano_states: np.ndarray | None,
    song_name: str,
    control_timestep: float = 0.05,
    seed: int = 0,
    reduced_action_space: bool = True,
    extra_task_kwargs: dict[str, Any] | None = None,
    suite_load_kwargs: dict[str, Any] | None = None,
    prefer_canonical_midi: bool = False,
    candidates_scale: tuple[str, ...] = ACTION_SOURCE_SCALES,
    candidates_mapping: tuple[str, ...] = ACTION_MAPPINGS,
    probe_steps: int = 5,
    output_dir: str | Path | None = None,
    label: str = "calibration",
    key_threshold: float = 0.5,
) -> dict[str, Any]:
    """Sweep action_source_scale x action_mapping and pick the one with lowest hand-qpos drift.

    Runs `probe_steps` of replay per candidate after restoring RP1M's t=0 state and returns the
    candidate with the smallest mean hand_qpos L2 to the reference. Result is JSON-serializable
    and may be written to disk for later reuse.
    """
    actions = np.asarray(actions, dtype=np.float32)
    reference_hand_joints = np.asarray(reference_hand_joints, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"Expected non-empty actions [T, action_dim], got {actions.shape}")
    if reference_hand_joints.ndim != 2 or reference_hand_joints.shape[0] == 0:
        raise ValueError(
            f"Expected non-empty reference_hand_joints [T, joints], got {reference_hand_joints.shape}"
        )
    probe_steps = max(int(probe_steps), 1)
    probe_steps = min(probe_steps, int(actions.shape[0]), int(reference_hand_joints.shape[0]) - 1)
    if probe_steps < 1:
        raise ValueError("Not enough reference steps to calibrate; need at least 2 reference frames.")

    midi_dir = ensure_dir(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="partita_calib_"))
    midi_proto_path = write_goals_proto(
        np.zeros((max(int(probe_steps + 2), 4), 88), dtype=np.float32),
        midi_dir / f"{label}_target_goals.proto",
        dt=control_timestep,
        title=f"Partita calibration {song_name}",
    )

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scale in candidates_scale:
        for mapping in candidates_mapping:
            env_name, env, _info = _load_env(
                environment_names=candidate_environment_names(song_name),
                midi_proto_path=midi_proto_path,
                control_timestep=control_timestep,
                seed=seed,
                reduced_action_space=reduced_action_space,
                extra_task_kwargs=extra_task_kwargs,
                suite_load_kwargs=suite_load_kwargs,
                prefer_canonical_midi=prefer_canonical_midi,
            )
            try:
                env.reset()
                _restore_initial_rp1m_state(
                    env,
                    hand_joints_t0=reference_hand_joints[0],
                    piano_state_t0=reference_piano_states[0]
                    if reference_piano_states is not None and reference_piano_states.shape[0] > 0
                    else None,
                    key_threshold=key_threshold,
                    zero_velocities=True,
                )
                task, physics, _piano = _locate_task_physics_piano(env)
                action_spec = env.action_spec()
                step_l2: list[float] = []
                for step_index in range(probe_steps):
                    control = _prepare_control(
                        actions[step_index],
                        action_spec,
                        mapping=mapping,
                        source_scale=scale,
                    )
                    env.step(control)
                    sim_pose = _capture_hand_qpos(task, physics)
                    ref_pose = reference_hand_joints[step_index + 1]
                    l2 = _hand_qpos_l2(sim_pose, ref_pose)
                    if l2 is not None:
                        step_l2.append(l2)
                if step_l2:
                    mean_l2 = float(np.mean(step_l2))
                    max_l2 = float(np.max(step_l2))
                else:
                    mean_l2 = float("inf")
                    max_l2 = float("inf")
                entry = {
                    "action_source_scale": scale,
                    "action_mapping": mapping,
                    "probe_steps": int(probe_steps),
                    "hand_qpos_l2_mean": mean_l2,
                    "hand_qpos_l2_max": max_l2,
                    "environment_name": env_name,
                }
                results.append(entry)
                if best is None or mean_l2 < best["hand_qpos_l2_mean"]:
                    best = entry
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    summary = {
        "song_name": song_name,
        "candidates": results,
        "best": best,
    }
    if output_dir is not None and best is not None:
        save_json(Path(output_dir) / f"{label}.json", summary)
    return summary


def rollout_reconstructed_actions_with_robopianist(
    *,
    actions: np.ndarray,
    goals: np.ndarray,
    song_name: str,
    output_dir: str | Path,
    label: str,
    control_timestep: float = 0.05,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    max_steps: int | None = None,
    render_every: int = 1,
    seed: int = 0,
    reduced_action_space: bool = True,
    action_source_scale: str = "normalized_minus_one_to_one",
    require_exact_action_dim: bool = True,
    action_mapping: str = "as_is",
    reference_hand_joints: np.ndarray | None = None,
    reference_piano_states: np.ndarray | None = None,
    restore_initial_state: bool = False,
    initial_hand_joints: np.ndarray | None = None,
    key_threshold: float = 0.5,
    extra_task_kwargs: dict[str, Any] | None = None,
    suite_load_kwargs: dict[str, Any] | None = None,
    prefer_canonical_midi: bool = False,
    fidelity_window: int = 50,
) -> dict[str, Any]:
    """
    Replay a Partita action sequence in RoboPianist / DM Control and render a video.

    When `reference_hand_joints` / `reference_piano_states` are supplied (e.g. RP1M target
    pose stream), per-step pose error and piano-state IoU are recorded. With
    `restore_initial_state=True` and a reference, the simulator is initialized to the RP1M
    t=0 state (hand qpos, piano qpos, sustain, zero velocities) before stepping; this is
    required for action replay to track the recorded states.

    The MIDI target is synthesized from the RP1M goal pianoroll because this checkout may
    not include the full PIG MIDI/proto asset library. Set `prefer_canonical_midi=True`
    when the song is in `robopianist.suite.ALL` to load the canonical PIG MIDI instead.
    robopianist_root is expected at /WAVE/projects/ECEN-524-Wi26/robopiano/robopianist.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = ensure_dir(output_dir)
    actions = np.asarray(actions, dtype=np.float32)
    goals = np.asarray(goals)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions [T, action_dim], got {actions.shape}")
    if action_mapping not in ACTION_MAPPINGS:
        raise ValueError(
            f"Unknown action_mapping '{action_mapping}'. Allowed: {list(ACTION_MAPPINGS)}"
        )
    if action_source_scale not in ACTION_SOURCE_SCALES:
        raise ValueError(
            f"Unknown action_source_scale '{action_source_scale}'. Allowed: {list(ACTION_SOURCE_SCALES)}"
        )
    ref_hand = (
        np.asarray(reference_hand_joints, dtype=np.float32)
        if reference_hand_joints is not None
        else None
    )
    ref_piano = (
        np.asarray(reference_piano_states, dtype=np.float32)
        if reference_piano_states is not None
        else None
    )
    midi_proto_path = write_goals_proto(
        goals,
        output_dir / f"{label}_target_goals.proto",
        dt=control_timestep,
        title=f"Partita {label} {song_name}",
    )
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=control_timestep,
        seed=seed,
        reduced_action_space=reduced_action_space,
        extra_task_kwargs=extra_task_kwargs,
        suite_load_kwargs=suite_load_kwargs,
        prefer_canonical_midi=prefer_canonical_midi,
    )
    frames: list[np.ndarray] = []
    piano_roll: list[np.ndarray] = []
    sim_hand: list[np.ndarray] = []
    total_reward = 0.0
    actions_executed = 0
    render_error = None
    terminated = False
    restore_info: dict[str, Any] | None = None
    restored_initial_hand_joint_count = 0
    try:
        timestep = env.reset()
        if restore_initial_state and ref_hand is not None and ref_hand.shape[0] > 0:
            restore_info = _restore_initial_rp1m_state(
                env,
                hand_joints_t0=ref_hand[0],
                piano_state_t0=ref_piano[0] if ref_piano is not None and ref_piano.shape[0] > 0 else None,
                key_threshold=key_threshold,
                zero_velocities=True,
            )
            restored_initial_hand_joint_count = int(restore_info.get("hand_joints_restored", 0))
        elif initial_hand_joints is not None:
            task, physics, _piano = _locate_task_physics_piano(env)
            restored_initial_hand_joint_count = _set_reduced_hand_qpos(task, physics, initial_hand_joints)
            if hasattr(physics, "forward"):
                physics.forward()
        try:
            frames.append(render_frame(env, height=height, width=width))
        except Exception as exc:
            render_error = str(exc)
        action_spec = env.action_spec()
        action_dim = int(action_spec.shape[0])
        action_stats = _action_spec_statistics(actions, action_spec)
        if require_exact_action_dim and int(actions.shape[-1]) != action_dim:
            raise ValueError(
                "Action dimension mismatch for rollout: "
                f"actions have dim {actions.shape[-1]}, environment expects {action_dim}. "
                "Use rollout.reduced_action_space/action_source_scale settings matching the action source."
            )
        task, physics, _piano = _locate_task_physics_piano(env)
        steps = actions if max_steps is None else actions[: int(max_steps)]
        for step_index, action in enumerate(steps):
            control = _prepare_control(
                action,
                action_spec,
                mapping=action_mapping,
                source_scale=action_source_scale,
            )
            timestep = env.step(control)
            total_reward += float(timestep.reward or 0.0)
            actions_executed += 1
            piano_activation = _capture_piano_activation(env)
            if piano_activation is not None:
                piano_roll.append(piano_activation)
            sim_pose = _capture_hand_qpos(task, physics)
            if sim_pose is not None:
                sim_hand.append(sim_pose)
            if render_error is None and (step_index + 1) % max(int(render_every), 1) == 0:
                try:
                    frames.append(render_frame(env, height=height, width=width))
                except Exception as exc:
                    render_error = str(exc)
            if timestep.last():
                terminated = True
                break
        video_path = None
        video_format = None
        audio_warning = None
        audio_events = _find_piano_midi_events(env)
        if render_error is None:
            video_path, video_format, audio_warning = write_video(
                frames,
                output_dir / f"{label}_rollout.mp4",
                fps=max(int(fps / max(render_every, 1)), 1),
                audio_events=audio_events,
            )
        fidelity_summary, fidelity_rows = _summarize_fidelity(
            sim_hand=sim_hand,
            sim_piano=piano_roll,
            ref_hand=ref_hand,
            ref_piano=ref_piano,
            goals=np.asarray(goals, dtype=np.float32),
            threshold=key_threshold,
            early_window=fidelity_window,
        )
        if fidelity_rows:
            _write_fidelity_csv(output_dir / f"{label}_fidelity_frames.csv", fidelity_rows)
        result = {
            "label": label,
            "song_name": song_name,
            "environment_name": env_name,
            "midi_proto_path": str(midi_proto_path),
            "actions_shape": list(actions.shape),
            "action_dim_environment": action_dim,
            "reduced_action_space": bool(reduced_action_space),
            "action_source_scale": action_source_scale,
            "action_mapping": action_mapping,
            "require_exact_action_dim": bool(require_exact_action_dim),
            "restore_initial_state": bool(restore_initial_state),
            "restore_info": restore_info,
            "restored_initial_hand_joint_count": int(restored_initial_hand_joint_count),
            "prefer_canonical_midi": bool(prefer_canonical_midi),
            "used_canonical_midi": bool(load_info.get("used_canonical_midi", False)),
            "task_kwargs": load_info.get("task_kwargs"),
            "suite_load_kwargs": load_info.get("suite_load_kwargs"),
            "actions_executed": int(actions_executed),
            "terminated": bool(terminated),
            "total_reward": float(total_reward),
            "rendered_frames": int(len(frames)),
            "render_error": render_error,
            "video_path": str(video_path) if video_path is not None else None,
            "video_format": video_format,
            "audio_warning": audio_warning,
            "audio_source": "robopianist_piano_midi_keypress_events",
            "audio_midi_note_event_count": int(len(_note_midi_events(audio_events))),
            "fidelity": fidelity_summary,
            **action_stats,
            **_rollout_key_metrics(goals, piano_roll),
            **_collect_metrics(env, terminated=terminated),
        }
        save_json(output_dir / f"{label}_rollout.json", result)
        return result
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
