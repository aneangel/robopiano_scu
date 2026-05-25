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
from fugue.data import (  # noqa: E402
    NormalizationStats,
    SampleConfig,
    build_planner_next_feature,
    build_planner_sequence_feature,
    load_demo_arrays,
    open_zarr_root,
    standardize,
    unstandardize,
)
from fugue.evaluation import load_checkpoint_model, save_npz_prediction  # noqa: E402
from fugue.training import resolve_device, save_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop Fugue planner-next RoboPianist rollout.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--dataset-artifact-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--demo-id", type=int, default=None)
    parser.add_argument("--demo-song-key", default=None)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--environment-name", default=DEFAULT_ENVIRONMENT_NAME)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-execution", default="first", choices=("first", "temporal_aggregate"))
    parser.add_argument("--temporal-agg-decay", type=float, default=0.7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(Path(args.dataset_artifact_root) / "manifest.csv")
    split_rows = manifest[manifest["split"].astype(str) == str(args.split)].copy()
    if args.demo_song_key is not None:
        if "song_key" not in split_rows.columns:
            raise ValueError("--demo-song-key requires a manifest with a song_key column")
        split_rows = split_rows[split_rows["song_key"].astype(str) == str(args.demo_song_key)]
    sort_cols = ["song_key", "demo_id"] if "song_key" in split_rows.columns else ["demo_id"]
    split_rows = split_rows.sort_values(sort_cols)
    if split_rows.empty:
        raise ValueError(f"No demos found for split={args.split!r} demo_song_key={args.demo_song_key!r}")
    if args.demo_id is not None:
        candidates = split_rows[split_rows["demo_id"].astype(int) == int(args.demo_id)]
        if candidates.empty:
            raise ValueError(
                f"No demo_id={args.demo_id} found for split={args.split!r} "
                f"demo_song_key={args.demo_song_key!r}"
            )
        demo_row = candidates.iloc[0]
    else:
        demo_row = split_rows.iloc[int(args.demo_index)]
    demo_id = int(demo_row["demo_id"])
    demo_song_key = str(demo_row["song_key"]) if "song_key" in demo_row.index else None
    model, checkpoint = load_checkpoint_model(args.checkpoint, device=args.device)
    sample_config = SampleConfig.from_dict(checkpoint["sample_config"])
    if sample_config.feature_mode not in {"planner_next", "planner_sequence"}:
        raise ValueError(
            "rollout_planner_next.py requires a checkpoint trained with "
            f"feature_mode='planner_next' or 'planner_sequence', got {sample_config.feature_mode!r}"
        )
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    root = open_zarr_root(args.rp1m_root)
    planner_song_key = demo_song_key or str(checkpoint["song_key"])
    group = root[planner_song_key]
    raw = load_demo_arrays(group, demo_id=demo_id, dt=float(checkpoint.get("dt", stats.dt)))
    song_name = str(args.environment_name)
    if song_name == DEFAULT_ENVIRONMENT_NAME and demo_song_key is not None:
        song_name = _environment_name_from_song_key(demo_song_key)
    rollout = rollout_planner_next_with_robopianist(
        model=model,
        checkpoint=checkpoint,
        sample_config=sample_config,
        stats=stats,
        raw_episode=raw,
        song_name=song_name,
        output_dir=output_dir / "rollout",
        device=args.device,
        max_steps=args.max_steps,
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        render_every=int(args.render_every),
        seed=int(args.seed),
        chunk_execution=args.chunk_execution,
        temporal_agg_decay=float(args.temporal_agg_decay),
    )
    prediction_path = save_npz_prediction(
        output_dir / "planner_next_closed_loop.npz",
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
            "planner_song_key": planner_song_key,
            "environment_name": song_name,
        },
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "demo_id": int(demo_id),
        "planner_song_key": planner_song_key,
        "split": str(args.split),
        "prediction_npz": str(prediction_path),
        "rollout": rollout,
    }
    save_json(output_dir / "planner_next_rollout_summary.json", summary)
    print(f"wrote_prediction_npz={prediction_path}")
    print(f"rollout_json={rollout.get('rollout_json_path')}")
    print(f"video_path={rollout.get('video_path')}")
    print(f"audio_source={rollout.get('audio_source')}")


def _environment_name_from_song_key(song_key: str) -> str:
    text = str(song_key)
    return text[:-2] if text.endswith("_0") else text


def rollout_planner_next_with_robopianist(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sample_config: SampleConfig,
    stats: NormalizationStats,
    raw_episode: dict[str, np.ndarray | None],
    song_name: str,
    output_dir: str | Path,
    device: str,
    max_steps: int | None,
    fps: int,
    width: int,
    height: int,
    render_every: int,
    seed: int,
    chunk_execution: str = "first",
    temporal_agg_decay: float = 0.7,
) -> dict[str, Any]:
    import os

    from partita.evaluation import rollout as pr

    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    q_ref = np.asarray(raw_episode["q"], dtype=np.float32)
    goals = np.asarray(raw_episode["goals"], dtype=np.float32)
    piano_states = raw_episode.get("piano_states")
    piano_ref = None if piano_states is None else np.asarray(piano_states, dtype=np.float32)
    dt = float(checkpoint.get("dt", stats.dt))
    lookahead = int(sample_config.lookahead)
    target_horizon = int(sample_config.goal_horizon)
    steps = max(min(int(q_ref.shape[0]) - lookahead, int(goals.shape[0]) - lookahead), 0)
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    if steps <= 0:
        raise ValueError(f"No valid planner-next rollout steps for q shape {q_ref.shape} and lookahead={lookahead}")
    chunk_execution = str(chunk_execution)
    if chunk_execution not in {"first", "temporal_aggregate"}:
        raise ValueError(f"Unsupported chunk_execution={chunk_execution!r}")
    temporal_agg_decay = float(temporal_agg_decay)
    if chunk_execution == "temporal_aggregate" and not (0.0 < temporal_agg_decay <= 1.0):
        raise ValueError(f"temporal_agg_decay must be in (0, 1], got {temporal_agg_decay}")

    midi_proto_path = pr.write_goals_proto(
        goals,
        output_dir / "planner_next_target_goals.proto",
        dt=dt,
        title=f"Fugue planner-next {song_name}",
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
    chunk_votes_norm: list[tuple[int, int, np.ndarray]] = []
    chunk_vote_counts: list[int] = []
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
            piano_state_t0=None,
            key_threshold=0.5,
            zero_velocities=True,
        )
        restore_info["reference_piano_state_policy"] = "scoring_only_not_simulator_input"
        task, physics, _piano = pr._locate_task_physics_piano(env)
        current_q = pr._capture_hand_qpos(task, physics)
        if current_q is None:
            current_q = q_ref[0].copy()
        previous_q = current_q.copy()
        q_history: list[np.ndarray] = []
        qvel_history: list[np.ndarray] = []
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
                target_indices = np.clip(
                    np.arange(step_index + lookahead, step_index + lookahead + target_horizon),
                    0,
                    q_ref.shape[0] - 1,
                )
                current_qvel = np.zeros_like(current_q, dtype=np.float32) if step_index == 0 else (current_q - previous_q) / dt
                feature = _build_online_feature(
                    current_q=current_q,
                    current_qvel=current_qvel,
                    q_history=q_history,
                    qvel_history=qvel_history,
                    target_q=q_ref[target_indices],
                    target_qvel=raw_episode["qvel"][target_indices] if sample_config.include_future_qvel else None,
                    goals=goals,
                    step_index=step_index,
                    sample_config=sample_config,
                    stats=stats,
                    predicted_norm_actions=predicted_norm_actions,
                )
                feature_tensor = torch.from_numpy(feature[None]).to(model_device).float()
                pred_norm_chunk = model(feature_tensor).detach().cpu().numpy()[0].astype(np.float32)
                for offset, pred_norm_row in enumerate(pred_norm_chunk):
                    target_step = int(step_index + int(sample_config.delta) + int(offset))
                    if target_step >= step_index:
                        chunk_votes_norm.append((target_step, step_index, pred_norm_row.copy()))
                pred_norm, vote_count = _select_online_chunk_action(
                    pred_norm_chunk=pred_norm_chunk,
                    chunk_votes_norm=chunk_votes_norm,
                    step_index=step_index,
                    chunk_execution=chunk_execution,
                    temporal_agg_decay=temporal_agg_decay,
                )
                chunk_votes_norm = [vote for vote in chunk_votes_norm if int(vote[0]) > step_index]
                chunk_vote_counts.append(int(vote_count))
                pred_action = unstandardize(pred_norm.reshape(1, -1), stats.action_mean, stats.action_std)[0]
                pred_action = np.clip(pred_action, -1.0, 1.0).astype(np.float32)
                control = pr._prepare_control(
                    pred_action,
                    action_spec,
                    mapping="as_is",
                    source_scale="normalized_minus_one_to_one",
                )
                start_q = current_q.copy()
                q_history.append(start_q.astype(np.float32))
                qvel_history.append(current_qvel.astype(np.float32))
                q_history = q_history[-max(int(sample_config.history) - 1, 0) :]
                qvel_history = qvel_history[-max(int(sample_config.history) - 1, 0) :]
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
        vote_counts_array = np.asarray(chunk_vote_counts, dtype=np.float32)
        if actions_array.size:
            action_stats = pr._action_spec_statistics(actions_array, action_spec)
        video_path = None
        video_format = None
        audio_warning = None
        audio_events = pr._find_piano_midi_events(env)
        if render_error is None:
            video_path, video_format, audio_warning = pr.write_video(
                frames,
                output_dir / "planner_next_closed_loop_rollout.mp4",
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
            pr._write_fidelity_csv(output_dir / "planner_next_closed_loop_fidelity_frames.csv", fidelity_rows)
        result = {
            "label": "planner_next_closed_loop",
            "song_name": song_name,
            "environment_name": env_name,
            "feature_mode": sample_config.feature_mode,
            "lookahead": int(sample_config.lookahead),
            "target_horizon": int(sample_config.goal_horizon),
            "chunk_horizon": int(sample_config.chunk_horizon),
            "chunk_execution": chunk_execution,
            "temporal_agg_decay": float(temporal_agg_decay),
            "midi_proto_path": str(midi_proto_path),
            "actions_shape": list(actions_array.shape),
            "action_dim_environment": int(action_spec.shape[0]),
            "reduced_action_space": True,
            "action_source_scale": "normalized_minus_one_to_one",
            "action_mapping": "as_is",
            "require_exact_action_dim": True,
            "closed_loop_conditioning": (
                "current hand qpos is captured from RoboPianist after the previous action; "
                "target hand qpos trajectory is read from the held-out demo trajectory as planner output"
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
            "chunk_vote_count_mean": None if vote_counts_array.size == 0 else float(vote_counts_array.mean()),
            "chunk_vote_count_min": None if vote_counts_array.size == 0 else int(vote_counts_array.min()),
            "chunk_vote_count_max": None if vote_counts_array.size == 0 else int(vote_counts_array.max()),
            "fidelity": fidelity_summary,
            **action_stats,
            **pr._rollout_key_metrics(goals, piano_roll),
            **pr._collect_metrics(env, terminated=terminated),
            "predicted_actions": actions_array,
            "predicted_actions_normalized": norm_actions_array,
            "sim_hand_joints": sim_hand_array,
        }
        summary = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        summary["rollout_json_path"] = str(output_dir / "planner_next_closed_loop_rollout.json")
        save_json(output_dir / "planner_next_closed_loop_rollout.json", summary)
        result["rollout_json_path"] = summary["rollout_json_path"]
        return result
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _select_online_chunk_action(
    *,
    pred_norm_chunk: np.ndarray,
    chunk_votes_norm: list[tuple[int, int, np.ndarray]],
    step_index: int,
    chunk_execution: str,
    temporal_agg_decay: float,
) -> tuple[np.ndarray, int]:
    if pred_norm_chunk.ndim != 2:
        raise ValueError(f"Expected prediction chunk [C, action_dim], got {pred_norm_chunk.shape}")
    if pred_norm_chunk.shape[0] < 1:
        raise ValueError("Prediction chunk must contain at least one action")
    if chunk_execution == "first" or pred_norm_chunk.shape[0] == 1:
        return pred_norm_chunk[0].astype(np.float32), 1
    if chunk_execution != "temporal_aggregate":
        raise ValueError(f"Unsupported chunk_execution={chunk_execution!r}")
    votes = [vote for vote in chunk_votes_norm if int(vote[0]) == int(step_index)]
    if not votes:
        return pred_norm_chunk[0].astype(np.float32), 0
    weights = np.asarray(
        [float(temporal_agg_decay) ** max(int(step_index) - int(source_step), 0) for _, source_step, _ in votes],
        dtype=np.float32,
    )
    actions = np.stack([np.asarray(action, dtype=np.float32).reshape(-1) for _, _, action in votes], axis=0)
    total = float(weights.sum())
    if total <= 0.0:
        return pred_norm_chunk[0].astype(np.float32), 0
    aggregated = (actions * weights[:, None]).sum(axis=0) / total
    return aggregated.astype(np.float32), int(len(votes))


def _build_online_feature(
    *,
    current_q: np.ndarray,
    current_qvel: np.ndarray,
    q_history: list[np.ndarray],
    qvel_history: list[np.ndarray],
    target_q: np.ndarray,
    target_qvel: np.ndarray | None,
    goals: np.ndarray,
    step_index: int,
    sample_config: SampleConfig,
    stats: NormalizationStats,
    predicted_norm_actions: list[np.ndarray],
) -> np.ndarray:
    current_q_norm = standardize(np.asarray(current_q, dtype=np.float32).reshape(1, -1), stats.q_mean, stats.q_std)[0]
    current_qvel_norm = standardize(np.asarray(current_qvel, dtype=np.float32).reshape(1, -1), stats.qvel_mean, stats.qvel_std)[0]
    target_q_norm = standardize(np.asarray(target_q, dtype=np.float32), stats.q_mean, stats.q_std)
    target_qvel_norm = None
    if sample_config.include_future_qvel and target_qvel is not None:
        target_qvel_norm = standardize(np.asarray(target_qvel, dtype=np.float32), stats.qvel_mean, stats.qvel_std)
    action_history_norm = None
    if sample_config.include_action_history:
        action_dim = len(stats.action_mean)
        rows = []
        missing = max(int(sample_config.history) - len(predicted_norm_actions), 0)
        rows.extend(np.zeros((action_dim,), dtype=np.float32) for _ in range(missing))
        rows.extend(predicted_norm_actions[-int(sample_config.history) :])
        action_history_norm = np.concatenate(rows, axis=0).astype(np.float32)
    goals_norm = None
    if sample_config.include_goals:
        indices = np.clip(
            np.arange(int(step_index), int(step_index) + int(sample_config.goal_horizon)),
            0,
            max(int(goals.shape[0]) - 1, 0),
        )
        goal_window = np.asarray(goals[indices], dtype=np.float32)
        goals_norm = standardize(goal_window, stats.goal_mean, stats.goal_std).reshape(-1)
    if sample_config.feature_mode == "planner_sequence":
        q_hist = _online_history_window(q_history, current_q, history=int(sample_config.history), pad="current")
        qvel_hist = _online_history_window(qvel_history, current_qvel, history=int(sample_config.history), pad="zero")
        q_hist_norm = standardize(q_hist, stats.q_mean, stats.q_std)
        qvel_hist_norm = standardize(qvel_hist, stats.qvel_mean, stats.qvel_std)
        action_history_matrix = None
        if sample_config.include_action_history:
            action_dim = len(stats.action_mean)
            rows = []
            missing = max(int(sample_config.history) - len(predicted_norm_actions), 0)
            rows.extend(np.zeros((action_dim,), dtype=np.float32) for _ in range(missing))
            rows.extend(predicted_norm_actions[-int(sample_config.history) :])
            action_history_matrix = np.stack(rows, axis=0).astype(np.float32)
        goal_matrix = None
        if goals_norm is not None:
            goal_matrix = goals_norm.reshape(int(sample_config.goal_horizon), -1)
        return build_planner_sequence_feature(
            current_q_history_norm=q_hist_norm,
            current_qvel_history_norm=qvel_hist_norm,
            target_q_norm=target_q_norm,
            config=sample_config,
            target_qvel_norm=target_qvel_norm,
            action_history_norm=action_history_matrix,
            goals_norm=goal_matrix,
        )
    if sample_config.feature_mode != "planner_next":
        raise ValueError(f"Unsupported planner rollout feature_mode={sample_config.feature_mode!r}")
    return build_planner_next_feature(
        current_q_norm=current_q_norm,
        current_qvel_norm=current_qvel_norm,
        target_q_norm=target_q_norm,
        config=sample_config,
        target_qvel_norm=target_qvel_norm,
        action_history_norm=action_history_norm,
        goals_norm=goals_norm,
    )


def _online_history_window(
    prior_rows: list[np.ndarray],
    current: np.ndarray,
    *,
    history: int,
    pad: str,
) -> np.ndarray:
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in prior_rows]
    rows.append(np.asarray(current, dtype=np.float32).reshape(-1))
    rows = rows[-int(history) :]
    if len(rows) < int(history):
        if pad == "zero":
            pad_value = np.zeros_like(rows[-1], dtype=np.float32)
        elif pad == "current":
            pad_value = rows[0].copy()
        else:
            raise ValueError(f"Unsupported pad mode: {pad}")
        rows = [pad_value.copy() for _ in range(int(history) - len(rows))] + rows
    return np.stack(rows, axis=0).astype(np.float32)


if __name__ == "__main__":
    main()
