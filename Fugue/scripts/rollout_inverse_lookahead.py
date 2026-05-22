#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT / "partita" / "src", REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from fugue.constants import DEFAULT_ENVIRONMENT_NAME, DEFAULT_RP1M_ROOT  # noqa: E402
from fugue.data import NormalizationStats, SampleConfig, load_demo_arrays, open_zarr_root, standardize, unstandardize  # noqa: E402
from fugue.evaluation import load_checkpoint_model, save_npz_prediction  # noqa: E402
from fugue.training import resolve_device, save_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop rollout for Model C inverse lookahead checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--dataset-artifact-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--demo-id", type=int, default=None)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--environment-name", default=DEFAULT_ENVIRONMENT_NAME)
    parser.add_argument("--target-mode", choices=["repeat-next", "future-window"], default="repeat-next")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(Path(args.dataset_artifact_root) / "manifest.csv")
    split_rows = manifest[manifest["split"].astype(str) == str(args.split)].sort_values("demo_id")
    if split_rows.empty:
        raise ValueError(f"No demos found for split={args.split!r}")
    demo_id = int(args.demo_id) if args.demo_id is not None else int(split_rows.iloc[int(args.demo_index)]["demo_id"])
    model, checkpoint = load_checkpoint_model(args.checkpoint, device=args.device)
    sample_config = SampleConfig.from_dict(checkpoint["sample_config"])
    if sample_config.feature_mode != "inverse":
        raise ValueError(f"Expected feature_mode='inverse', got {sample_config.feature_mode!r}")
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    root = open_zarr_root(args.rp1m_root)
    group = root[str(checkpoint["song_key"])]
    raw = load_demo_arrays(group, demo_id=demo_id, dt=float(checkpoint.get("dt", stats.dt)))
    rollout = rollout_inverse_lookahead_with_robopianist(
        model=model,
        checkpoint=checkpoint,
        sample_config=sample_config,
        stats=stats,
        raw_episode=raw,
        song_name=str(args.environment_name),
        output_dir=output_dir / "rollout",
        device=args.device,
        target_mode=str(args.target_mode),
        max_steps=args.max_steps,
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        render_every=int(args.render_every),
        seed=int(args.seed),
    )
    prediction_path = save_npz_prediction(
        output_dir / f"inverse_lookahead_{args.target_mode}.npz",
        {
            "predicted_actions": rollout.pop("predicted_actions"),
            "predicted_actions_normalized": rollout.pop("predicted_actions_normalized"),
            "sim_hand_joints": rollout.pop("sim_hand_joints"),
            "target_hand_joints": raw["q"],
            "reference_actions": raw["actions"],
            "goals": raw["goals"],
            "piano_states": raw.get("piano_states"),
            "checkpoint": str(args.checkpoint),
            "split": str(args.split),
            "demo_id": int(demo_id),
            "target_mode": str(args.target_mode),
        },
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "demo_id": int(demo_id),
        "split": str(args.split),
        "prediction_npz": str(prediction_path),
        "rollout": rollout,
    }
    save_json(output_dir / f"inverse_lookahead_{args.target_mode}_summary.json", summary)
    print(f"wrote_prediction_npz={prediction_path}")
    print(f"rollout_json={rollout.get('rollout_json_path')}")
    print(f"video_path={rollout.get('video_path')}")
    print(f"audio_source={rollout.get('audio_source')}")


def rollout_inverse_lookahead_with_robopianist(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sample_config: SampleConfig,
    stats: NormalizationStats,
    raw_episode: dict[str, np.ndarray | None],
    song_name: str,
    output_dir: str | Path,
    device: str,
    target_mode: str,
    max_steps: int | None,
    fps: int,
    width: int,
    height: int,
    render_every: int,
    seed: int,
) -> dict[str, Any]:
    import os

    from partita.evaluation import rollout as pr

    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    q_ref = np.asarray(raw_episode["q"], dtype=np.float32)
    qvel_ref = np.asarray(raw_episode["qvel"], dtype=np.float32)
    goals = np.asarray(raw_episode["goals"], dtype=np.float32)
    piano_states = raw_episode.get("piano_states")
    piano_ref = None if piano_states is None else np.asarray(piano_states, dtype=np.float32)
    dt = float(checkpoint.get("dt", stats.dt))
    horizon = int(sample_config.goal_horizon)
    steps = max(min(int(q_ref.shape[0]) - 1, int(goals.shape[0]) - 1), 0)
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    if steps <= 0:
        raise ValueError(f"No valid inverse-lookahead rollout steps for q shape {q_ref.shape}")

    midi_proto_path = pr.write_goals_proto(
        goals,
        output_dir / f"inverse_lookahead_{target_mode}_target_goals.proto",
        dt=dt,
        title=f"Fugue inverse lookahead {target_mode} {song_name}",
    )
    env_name, env, load_info = pr._load_env(
        environment_names=pr.candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=dt,
        seed=seed,
        reduced_action_space=True,
    )

    frames: list[np.ndarray] = []
    piano_roll: list[np.ndarray] = []
    sim_hand: list[np.ndarray] = []
    predicted_actions: list[np.ndarray] = []
    predicted_norm_actions: list[np.ndarray] = []
    total_reward = 0.0
    actions_executed = 0
    render_error = None
    terminated = False
    restore_info: dict[str, Any] | None = None
    model_device = resolve_device(device)
    model.to(model_device)
    model.eval()
    try:
        env.reset()
        restore_info = pr._restore_initial_rp1m_state(
            env,
            hand_joints_t0=q_ref[0],
            piano_state_t0=piano_ref[0] if piano_ref is not None and piano_ref.shape[0] > 0 else None,
            key_threshold=0.5,
            zero_velocities=True,
        )
        task, physics, _piano = pr._locate_task_physics_piano(env)
        current_q = pr._capture_hand_qpos(task, physics)
        if current_q is None:
            current_q = q_ref[0].copy()
        previous_q = current_q.copy()
        try:
            frames.append(pr.render_frame(env, height=height, width=width))
        except Exception as exc:
            render_error = str(exc)
        action_spec = env.action_spec()
        if int(checkpoint["action_dim"]) != int(action_spec.shape[0]):
            raise ValueError(
                f"Checkpoint action_dim={checkpoint['action_dim']} does not match env action_dim={action_spec.shape[0]}"
            )
        action_stats = pr._action_spec_statistics(np.zeros((steps, int(action_spec.shape[0])), dtype=np.float32), action_spec)
        with torch.no_grad():
            for step_index in range(steps):
                current_qvel = np.zeros_like(current_q, dtype=np.float32) if step_index == 0 else (current_q - previous_q) / dt
                feature = _build_inverse_online_feature(
                    current_q=current_q,
                    current_qvel=current_qvel,
                    q_ref=q_ref,
                    qvel_ref=qvel_ref,
                    goals=goals,
                    step_index=step_index,
                    horizon=horizon,
                    target_mode=target_mode,
                    sample_config=sample_config,
                    stats=stats,
                )
                feature_tensor = torch.from_numpy(feature[None]).to(model_device).float()
                pred_norm = model(feature_tensor).detach().cpu().numpy()[0, 0]
                pred_action = unstandardize(pred_norm.reshape(1, -1), stats.action_mean, stats.action_std)[0]
                pred_action = np.clip(pred_action, -1.0, 1.0).astype(np.float32)
                control = pr._prepare_control(
                    pred_action,
                    action_spec,
                    mapping="as_is",
                    source_scale="normalized_minus_one_to_one",
                )
                start_q = current_q.copy()
                timestep = env.step(control)
                total_reward += float(timestep.reward or 0.0)
                actions_executed += 1
                predicted_norm_actions.append(pred_norm.astype(np.float32))
                predicted_actions.append(pred_action)
                piano_activation = pr._capture_piano_activation(env)
                if piano_activation is not None:
                    piano_roll.append(piano_activation)
                sim_pose = pr._capture_hand_qpos(task, physics)
                if sim_pose is not None:
                    current_q = sim_pose.astype(np.float32)
                    sim_hand.append(current_q)
                previous_q = start_q
                if render_error is None and (step_index + 1) % max(int(render_every), 1) == 0:
                    try:
                        frames.append(pr.render_frame(env, height=height, width=width))
                    except Exception as exc:
                        render_error = str(exc)
                if timestep.last():
                    terminated = True
                    break
        actions_array = np.asarray(predicted_actions, dtype=np.float32)
        norm_actions_array = np.asarray(predicted_norm_actions, dtype=np.float32)
        sim_hand_array = np.asarray(sim_hand, dtype=np.float32)
        if actions_array.size:
            action_stats = pr._action_spec_statistics(actions_array, action_spec)
        video_path = None
        video_format = None
        audio_warning = None
        audio_events = pr._find_piano_midi_events(env)
        if render_error is None:
            video_path, video_format, audio_warning = pr.write_video(
                frames,
                output_dir / f"inverse_lookahead_{target_mode}_rollout.mp4",
                fps=max(int(fps / max(render_every, 1)), 1),
                audio_events=audio_events,
            )
        fidelity_summary, fidelity_rows = pr._summarize_fidelity(
            sim_hand=sim_hand,
            sim_piano=piano_roll,
            ref_hand=q_ref,
            ref_piano=piano_ref,
            goals=goals,
            threshold=0.5,
        )
        if fidelity_rows:
            pr._write_fidelity_csv(output_dir / f"inverse_lookahead_{target_mode}_fidelity_frames.csv", fidelity_rows)
        result = {
            "label": f"inverse_lookahead_{target_mode}",
            "song_name": song_name,
            "environment_name": env_name,
            "feature_mode": sample_config.feature_mode,
            "goal_horizon": int(sample_config.goal_horizon),
            "target_mode": target_mode,
            "midi_proto_path": str(midi_proto_path),
            "actions_shape": list(actions_array.shape),
            "action_dim_environment": int(action_spec.shape[0]),
            "closed_loop_conditioning": (
                "current hand qpos/qvel is captured from RoboPianist after the previous action; "
                "desired hand-state lookahead is read from a held-out demo trajectory"
            ),
            "planner_source": "held_out_demo_hand_joints",
            "restore_initial_state": True,
            "restore_info": restore_info,
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
            "audio_midi_note_event_count": int(len(pr._note_midi_events(audio_events))),
            "fidelity": fidelity_summary,
            **action_stats,
            **pr._rollout_key_metrics(goals, piano_roll),
            **pr._collect_metrics(env, terminated=terminated),
            "predicted_actions": actions_array,
            "predicted_actions_normalized": norm_actions_array,
            "sim_hand_joints": sim_hand_array,
        }
        summary = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        summary["rollout_json_path"] = str(output_dir / f"inverse_lookahead_{target_mode}_rollout.json")
        save_json(output_dir / f"inverse_lookahead_{target_mode}_rollout.json", summary)
        result["rollout_json_path"] = summary["rollout_json_path"]
        return result
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _build_inverse_online_feature(
    *,
    current_q: np.ndarray,
    current_qvel: np.ndarray,
    q_ref: np.ndarray,
    qvel_ref: np.ndarray,
    goals: np.ndarray,
    step_index: int,
    horizon: int,
    target_mode: str,
    sample_config: SampleConfig,
    stats: NormalizationStats,
) -> np.ndarray:
    current_q_norm = standardize(np.asarray(current_q, dtype=np.float32).reshape(1, -1), stats.q_mean, stats.q_std)[0]
    current_qvel_norm = standardize(np.asarray(current_qvel, dtype=np.float32).reshape(1, -1), stats.qvel_mean, stats.qvel_std)[0]
    if target_mode == "repeat-next":
        next_index = min(int(step_index) + 1, int(q_ref.shape[0]) - 1)
        target_q = np.repeat(q_ref[next_index : next_index + 1], int(horizon), axis=0)
        target_qvel = np.repeat(qvel_ref[next_index : next_index + 1], int(horizon), axis=0)
    elif target_mode == "future-window":
        indices = np.clip(
            np.arange(int(step_index) + 1, int(step_index) + 1 + int(horizon)),
            0,
            max(int(q_ref.shape[0]) - 1, 0),
        )
        target_q = q_ref[indices]
        target_qvel = qvel_ref[indices]
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    parts = []
    if sample_config.include_qpos:
        parts.append(current_q_norm)
    if sample_config.include_qvel:
        parts.append(current_qvel_norm)
    target_q_norm = standardize(target_q, stats.q_mean, stats.q_std).reshape(-1)
    parts.append(target_q_norm)
    if sample_config.include_future_qvel:
        parts.append(standardize(target_qvel, stats.qvel_mean, stats.qvel_std).reshape(-1))
    if sample_config.include_goals:
        indices = np.clip(
            np.arange(int(step_index), int(step_index) + int(sample_config.goal_horizon)),
            0,
            max(int(goals.shape[0]) - 1, 0),
        )
        parts.append(standardize(goals[indices], stats.goal_mean, stats.goal_std).reshape(-1))
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts], axis=0).astype(np.float32)


if __name__ == "__main__":
    main()
