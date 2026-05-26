from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fugue.data import (
    NormalizationStats,
    SampleConfig,
    build_planner_next_feature,
    build_planner_sequence_feature,
    load_demo_arrays,
    open_zarr_root,
    standardize,
    unstandardize,
)
from fugue.evaluation import load_checkpoint_model
from fugue.training import resolve_device
from rp1m_simulator import simulator as rp1m
from rp1m_simulator.simulator import RolloutConfig


@dataclass(slots=True)
class LoadedPolicy:
    model: torch.nn.Module
    checkpoint: dict[str, Any]
    sample_config: SampleConfig
    stats: NormalizationStats
    device: torch.device
    checkpoint_path: str


@dataclass(slots=True)
class ClosedLoopParams:
    chunk_execution: str = "first"
    temporal_agg_decay: float = 0.7
    action_gain: float = 1.0
    right_gain: float = 1.0
    left_gain: float = 1.0
    sustain_gain: float = 1.0
    action_smoothing_alpha: float = 0.0
    max_action_delta: float | None = None
    target_lead: int = 0


@dataclass(slots=True)
class OnlineAdaptationConfig:
    enabled: bool = False
    window: int = 20
    interval: int = 10
    target_recall: float = 0.35
    max_mispress_rate: float = 0.25
    min_precision_for_gain: float = 0.45
    gain_step: float = 0.05
    smoothing_step: float = 0.05
    decay_step: float = 0.05
    min_action_gain: float = 0.4
    max_action_gain: float = 1.6
    min_smoothing_alpha: float = 0.0
    max_smoothing_alpha: float = 0.85
    min_temporal_agg_decay: float = 0.2
    max_temporal_agg_decay: float = 0.98


def load_policy(checkpoint_path: str | Path, *, device: str = "cuda") -> LoadedPolicy:
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
    sample_config = SampleConfig.from_dict(checkpoint["sample_config"])
    if sample_config.feature_mode not in {"planner_next", "planner_sequence"}:
        raise ValueError(
            "RP1M simulator closed-loop rollout requires feature_mode "
            f"'planner_next' or 'planner_sequence', got {sample_config.feature_mode!r}"
        )
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    model_device = resolve_device(device)
    model.to(model_device)
    model.eval()
    return LoadedPolicy(
        model=model,
        checkpoint=checkpoint,
        sample_config=sample_config,
        stats=stats,
        device=model_device,
        checkpoint_path=str(checkpoint_path),
    )


def resolve_manifest_examples(
    *,
    dataset_artifact_root: str | Path,
    split: str,
    demo_id: int | None = None,
    demo_song_key: str | None = None,
    demo_index: int = 0,
    limit: int = 1,
) -> list[dict[str, Any]]:
    manifest = pd.read_csv(Path(dataset_artifact_root) / "manifest.csv")
    rows = manifest[manifest["split"].astype(str) == str(split)].copy()
    if demo_song_key is not None:
        if "song_key" not in rows.columns:
            raise ValueError("--demo-song-key requires a manifest with a song_key column")
        rows = rows[rows["song_key"].astype(str) == str(demo_song_key)]
    sort_cols = ["song_key", "demo_id"] if "song_key" in rows.columns else ["demo_id"]
    rows = rows.sort_values(sort_cols)
    if demo_id is not None:
        rows = rows[rows["demo_id"].astype(int) == int(demo_id)]
    else:
        start = max(int(demo_index), 0)
        rows = rows.iloc[start : start + max(int(limit), 1)]
    if rows.empty:
        raise ValueError(
            f"No manifest rows matched split={split!r} demo_id={demo_id!r} "
            f"demo_song_key={demo_song_key!r} demo_index={demo_index!r}"
        )
    examples: list[dict[str, Any]] = []
    for _, row in rows.head(max(int(limit), 1)).iterrows():
        song_key = str(row["song_key"]) if "song_key" in row.index else None
        if song_key is None:
            raise ValueError("Manifest row does not include song_key; multi-song rollout requires song_key")
        examples.append({"song_key": song_key, "demo_id": int(row["demo_id"]), "split": str(split)})
    return examples


def load_raw_episode(
    *,
    rp1m_root: str | Path,
    song_key: str,
    demo_id: int,
    dt: float,
) -> dict[str, np.ndarray | None]:
    root = open_zarr_root(rp1m_root)
    if str(song_key) not in root:
        raise KeyError(f"RP1M song key not found: {song_key}")
    return load_demo_arrays(root[str(song_key)], demo_id=int(demo_id), dt=float(dt))


def make_rollout_config(
    *,
    dataset_timestep: float,
    simulation_timestep: float | None = 0.005,
    seed: int = 0,
    threshold: float = 0.5,
    render_mp4: bool = False,
    render_audio: bool = False,
    render_every_source_step: int = 1,
    width: int = 640,
    height: int = 480,
    fps: int = 20,
    action_source_scale: str = "normalized_minus_one_to_one",
    action_mapping: str = "as_is",
    action_substep_policy: str = "zero_pad_hold",
    wrist_action_policy: str = "hold_initial",
    reduced_action_space: bool = True,
    hand_anchor_y_offset: float | None = rp1m.DEFAULT_HAND_ANCHOR_Y_OFFSET,
    auto_hand_anchor_y_offset: bool = False,
    restore_initial_hand: bool = False,
    set_hand_qvel: bool = False,
    gravity_compensation: bool = False,
    primitive_fingertip_collisions: bool = False,
    disable_hand_collisions: bool = False,
) -> RolloutConfig:
    return RolloutConfig(
        mode="action",
        dataset_timestep=float(dataset_timestep),
        simulation_timestep=float(dataset_timestep if simulation_timestep is None else simulation_timestep),
        hand_anchor_y_offset=hand_anchor_y_offset,
        auto_hand_anchor_y_offset=bool(auto_hand_anchor_y_offset),
        reduced_action_space=bool(reduced_action_space),
        action_source_scale=str(action_source_scale),  # type: ignore[arg-type]
        action_mapping=str(action_mapping),  # type: ignore[arg-type]
        action_substep_policy=str(action_substep_policy),  # type: ignore[arg-type]
        wrist_action_policy=str(wrist_action_policy),  # type: ignore[arg-type]
        hand_state_action_source="recorded",
        restore_initial_hand=bool(restore_initial_hand),
        set_hand_qvel=bool(set_hand_qvel),
        gravity_compensation=bool(gravity_compensation),
        primitive_fingertip_collisions=bool(primitive_fingertip_collisions),
        disable_hand_collisions=bool(disable_hand_collisions),
        seed=int(seed),
        threshold=float(threshold),
        max_source_steps=None,
        render_mp4=bool(render_mp4),
        render_audio=bool(render_audio),
        render_every_source_step=max(int(render_every_source_step), 1),
        width=int(width),
        height=int(height),
        camera_id=None,
        fps=int(fps),
    )


def rollout_loaded_policy_with_rp1m_simulator(
    *,
    policy: LoadedPolicy,
    raw_episode: dict[str, np.ndarray | None],
    song_key: str,
    demo_id: int,
    output_dir: str | Path,
    params: ClosedLoopParams | None = None,
    adaptation: OnlineAdaptationConfig | None = None,
    rollout_config: RolloutConfig | None = None,
    environment_name: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    params = replace(params) if params is not None else ClosedLoopParams()
    adaptation = adaptation or OnlineAdaptationConfig()
    _validate_params(params)
    _validate_adaptation(adaptation)
    q_ref = np.asarray(raw_episode["q"], dtype=np.float32)
    qvel_ref = np.asarray(raw_episode["qvel"], dtype=np.float32)
    reference_actions = np.asarray(raw_episode["actions"], dtype=np.float32)
    goals = np.asarray(raw_episode["goals"], dtype=np.float32)[..., :88]
    piano_ref = raw_episode.get("piano_states")
    if piano_ref is not None:
        piano_ref = np.asarray(piano_ref, dtype=np.float32)[..., :88]
    fingertips = raw_episode.get("fingertips")
    if fingertips is not None:
        fingertips = np.asarray(fingertips, dtype=np.float32)
    dt = float(policy.checkpoint.get("dt", policy.stats.dt))
    source_steps = min(int(q_ref.shape[0]), int(qvel_ref.shape[0]), int(reference_actions.shape[0]), int(goals.shape[0]))
    if piano_ref is not None:
        source_steps = min(source_steps, int(piano_ref.shape[0]))
    if max_steps is not None:
        source_steps = min(source_steps, int(max_steps))
    lookahead = int(policy.sample_config.lookahead)
    if source_steps <= lookahead + 1:
        raise ValueError(f"Not enough steps for closed-loop rollout: {source_steps}")

    output = rp1m.ensure_dir(output_dir)
    sim_config = rollout_config or make_rollout_config(dataset_timestep=dt)
    sim_config = replace(sim_config, mode="action", dataset_timestep=dt)
    control_dt = float(sim_config.simulation_timestep)
    substeps = int(round(dt / control_dt))
    if substeps < 1 or not np.isclose(substeps * control_dt, dt, rtol=1e-4, atol=1e-7):
        raise ValueError("Fugue rollout dataset timestep must be an integer multiple of simulation_timestep")
    if sim_config.action_substep_policy not in rp1m.ACTION_SUBSTEP_POLICIES:
        raise ValueError(f"Unknown action_substep_policy: {sim_config.action_substep_policy}")
    environment = environment_name or rp1m.canonicalize_environment_name(song_key)
    dense_goals = np.repeat(goals[:source_steps], substeps, axis=0)
    midi_proto = rp1m.write_goals_proto(
        dense_goals,
        output / "target_goals.proto",
        dt=control_dt,
        title=f"Fugue closed-loop {song_key} demo {demo_id}",
    )
    trajectory = rp1m.make_rp1m_trajectory_from_arrays(
        song_key=song_key,
        demo_id=int(demo_id),
        actions=reference_actions[:source_steps],
        goals=goals[:source_steps],
        hand_joints=q_ref[:source_steps],
        hand_fingertips=None if fingertips is None else fingertips[:source_steps],
        reference_piano_states=None if piano_ref is None else piano_ref[:source_steps],
        environment_name=environment,
    )
    hand_anchor_calibration = rp1m._calibrate_hand_anchor_y_offset(trajectory, sim_config, midi_proto, control_dt)
    effective_config = replace(
        sim_config,
        hand_anchor_y_offset=hand_anchor_calibration.get("effective_hand_anchor_y_offset"),
        auto_hand_anchor_y_offset=False,
    )
    env, load_info = rp1m._load_env(effective_config, midi_proto, trajectory.environment_name, control_dt)

    source_played: list[np.ndarray] = []
    dense_played: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    sim_hand: list[np.ndarray] = []
    predicted_actions: list[np.ndarray] = []
    raw_predicted_actions: list[np.ndarray] = []
    executed_norm_actions: list[np.ndarray] = []
    chunk_votes_norm: list[tuple[int, int, np.ndarray]] = []
    chunk_vote_counts: list[int] = []
    adaptation_events: list[dict[str, Any]] = []
    q_history: list[np.ndarray] = []
    qvel_history: list[np.ndarray] = []
    total_reward = 0.0
    terminated = False
    render_error = None
    previous_action: np.ndarray | None = None
    actions_executed = 0
    qpos_restored = 0
    qvel_restored = 0
    post_reset_hand_anchor = {"applied": False, "hand_anchor_y_offset": None}
    control_overrides: dict[int, float] = {}

    try:
        env.reset()
        task, physics, _piano = rp1m._locate_task_physics_piano(env)
        if effective_config.hand_anchor_application == "post_reset":
            post_reset_hand_anchor = rp1m._apply_post_reset_hand_anchor_y_offset(
                task,
                physics,
                effective_config.hand_anchor_y_offset,
            )
        if effective_config.restore_initial_hand:
            qpos_restored = rp1m._set_hand_qpos(task, physics, q_ref[0])
            if effective_config.set_hand_qvel:
                qvel_restored = rp1m._set_hand_qvel(task, physics, _qvel_between(q_ref, 0, dt))
            if hasattr(physics, "forward"):
                physics.forward()
        current_q = rp1m._capture_hand_qpos(task, physics)
        if current_q is None:
            current_q = q_ref[0].copy()
        current_q = np.asarray(current_q, dtype=np.float32)
        previous_q = current_q.copy()
        action_spec = env.action_spec()
        wrist_qpos = q_ref[0] if effective_config.restore_initial_hand else current_q
        control_overrides = rp1m._initial_wrist_control_overrides(wrist_qpos, action_spec, effective_config)
        if int(policy.checkpoint["action_dim"]) != int(action_spec.shape[0]):
            raise ValueError(
                f"Checkpoint action_dim={policy.checkpoint['action_dim']} does not match "
                f"environment action_dim={action_spec.shape[0]}"
            )
        if effective_config.render_mp4:
            try:
                frames.append(
                    rp1m.render_frame(
                        env,
                        height=int(effective_config.height),
                        width=int(effective_config.width),
                        camera_id=effective_config.camera_id,
                    )
                )
            except Exception as exc:  # pragma: no cover - depends on EGL/rendering.
                render_error = str(exc)
        with torch.no_grad():
            for step_index in range(source_steps - 1):
                current_qvel = np.zeros_like(current_q, dtype=np.float32) if step_index == 0 else (current_q - previous_q) / dt
                feature = _build_online_feature(
                    current_q=current_q,
                    current_qvel=current_qvel,
                    q_history=q_history,
                    qvel_history=qvel_history,
                    target_q=q_ref[_target_indices(step_index, q_ref.shape[0], policy.sample_config, params.target_lead)],
                    target_qvel=(
                        qvel_ref[_target_indices(step_index, qvel_ref.shape[0], policy.sample_config, params.target_lead)]
                        if policy.sample_config.include_future_qvel
                        else None
                    ),
                    goals=goals,
                    step_index=step_index,
                    goal_step_index=step_index + int(params.target_lead),
                    sample_config=policy.sample_config,
                    stats=policy.stats,
                    executed_norm_actions=executed_norm_actions,
                )
                feature_tensor = torch.from_numpy(feature[None]).to(policy.device).float()
                pred_norm_chunk = policy.model(feature_tensor).detach().cpu().numpy()[0].astype(np.float32)
                for offset, pred_norm_row in enumerate(pred_norm_chunk):
                    target_step = int(step_index + int(policy.sample_config.delta) + int(offset))
                    if target_step >= step_index:
                        chunk_votes_norm.append((target_step, step_index, pred_norm_row.copy()))
                pred_norm, vote_count = _select_online_chunk_action(
                    pred_norm_chunk=pred_norm_chunk,
                    chunk_votes_norm=chunk_votes_norm,
                    step_index=step_index,
                    chunk_execution=params.chunk_execution,
                    temporal_agg_decay=float(params.temporal_agg_decay),
                )
                chunk_votes_norm = [vote for vote in chunk_votes_norm if int(vote[0]) > step_index]
                chunk_vote_counts.append(int(vote_count))
                raw_action = unstandardize(pred_norm.reshape(1, -1), policy.stats.action_mean, policy.stats.action_std)[0]
                action = _apply_action_parameters(raw_action, previous_action=previous_action, params=params)
                executed_norm = standardize(action.reshape(1, -1), policy.stats.action_mean, policy.stats.action_std)[0]
                start_q = current_q.copy()
                q_history.append(start_q.astype(np.float32))
                qvel_history.append(current_qvel.astype(np.float32))
                q_history = q_history[-max(int(policy.sample_config.history) - 1, 0) :]
                qvel_history = qvel_history[-max(int(policy.sample_config.history) - 1, 0) :]
                prepared_control = rp1m._prepare_control(action, action_spec, effective_config)
                held_control = prepared_control.copy()
                interval_played: list[np.ndarray] = []
                timestep = None
                for substep in range(substeps):
                    if substep == 0 or effective_config.action_substep_policy == "repeat":
                        control = prepared_control.copy()
                    elif effective_config.action_substep_policy == "zero_control":
                        control = np.zeros(action_spec.shape, dtype=np.float32)
                    elif effective_config.action_substep_policy == "zero_source":
                        control = rp1m._prepare_control(np.zeros_like(action), action_spec, effective_config)
                    elif effective_config.action_substep_policy == "zero_pad_hold":
                        control = held_control.copy()
                    else:
                        raise ValueError(f"Unknown action_substep_policy: {effective_config.action_substep_policy}")
                    control = rp1m._apply_control_overrides(control, control_overrides)
                    timestep = env.step(control)
                    total_reward += float(timestep.reward or 0.0)
                    actions_executed += 1
                    activation = rp1m._capture_piano_activation(env)
                    if activation is not None:
                        activation = np.asarray(activation, dtype=np.float32)[:88]
                        interval_played.append(activation)
                        dense_played.append(activation)
                    if timestep.last():
                        terminated = True
                        break
                raw_predicted_actions.append(raw_action.astype(np.float32))
                predicted_actions.append(action.astype(np.float32))
                executed_norm_actions.append(executed_norm.astype(np.float32))
                previous_action = action.astype(np.float32)
                if interval_played:
                    source_played.append(np.max(np.stack(interval_played, axis=0), axis=0).astype(np.float32))
                task, physics, _piano = rp1m._locate_task_physics_piano(env)
                pose = rp1m._capture_hand_qpos(task, physics)
                if pose is not None:
                    current_q = np.asarray(pose, dtype=np.float32)
                    sim_hand.append(current_q.copy())
                previous_q = start_q
                if effective_config.render_mp4 and render_error is None:
                    if (step_index + 1) % max(int(effective_config.render_every_source_step), 1) == 0:
                        try:
                            frames.append(
                                rp1m.render_frame(
                                    env,
                                    height=int(effective_config.height),
                                    width=int(effective_config.width),
                                    camera_id=effective_config.camera_id,
                                )
                            )
                        except Exception as exc:  # pragma: no cover - depends on EGL/rendering.
                            render_error = str(exc)
                _maybe_adapt_online_parameters(
                    params=params,
                    config=adaptation,
                    goals=goals,
                    played=source_played,
                    step_index=step_index,
                    events=adaptation_events,
                    threshold=float(effective_config.threshold),
                )
                if timestep is not None and timestep.last():
                    break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    played_arr = np.stack(source_played, axis=0) if source_played else np.zeros((0, 88), dtype=np.float32)
    dense_played_arr = np.stack(dense_played, axis=0) if dense_played else np.zeros((0, 88), dtype=np.float32)
    pred_arr = np.stack(predicted_actions, axis=0) if predicted_actions else np.zeros((0, reference_actions.shape[-1]), dtype=np.float32)
    raw_pred_arr = np.stack(raw_predicted_actions, axis=0) if raw_predicted_actions else np.zeros_like(pred_arr)
    norm_arr = np.stack(executed_norm_actions, axis=0) if executed_norm_actions else np.zeros_like(pred_arr)
    sim_hand_arr = np.stack(sim_hand, axis=0) if sim_hand else np.zeros((0, q_ref.shape[-1]), dtype=np.float32)
    chunk_vote_arr = np.asarray(chunk_vote_counts, dtype=np.float32)
    npz_path = output / "closed_loop_rollout.npz"
    np.savez_compressed(
        npz_path,
        played_piano=played_arr,
        dense_played_piano=dense_played_arr,
        predicted_actions=pred_arr,
        raw_predicted_actions=raw_pred_arr,
        executed_actions_normalized=norm_arr,
        sim_hand_joints=sim_hand_arr,
        target_hand_joints=q_ref[: max(source_steps, 1)],
        reference_actions=reference_actions[: max(pred_arr.shape[0], 1)],
        goals=goals[: max(played_arr.shape[0], 1)],
        reference_piano_states_for_scoring=(
            np.zeros((0, 88), dtype=np.float32) if piano_ref is None else piano_ref[: max(played_arr.shape[0], 1)]
        ),
    )

    video_path = None
    audio_warning = None
    if effective_config.render_mp4 and render_error is None:
        events = (
            rp1m.piano_roll_to_midi_events(dense_played_arr, dt=control_dt, threshold=float(effective_config.threshold))
            if effective_config.render_audio
            else None
        )
        video_path, audio_warning = rp1m.write_video(
            frames,
            output / "closed_loop_rollout.mp4",
            fps=int(effective_config.fps),
            audio_events=events,
        )

    action_summary = _action_summary(pred_arr)
    hand_summary = _hand_summary(sim_hand_arr, q_ref)
    summary: dict[str, Any] = {
        "label": "fugue_rp1m_simulator_closed_loop",
        "checkpoint": str(policy.checkpoint_path),
        "song_key": str(song_key),
        "environment_name": str(environment),
        "demo_id": int(demo_id),
        "feature_mode": policy.sample_config.feature_mode,
        "lookahead": int(policy.sample_config.lookahead),
        "goal_horizon": int(policy.sample_config.goal_horizon),
        "chunk_horizon": int(policy.sample_config.chunk_horizon),
        "closed_loop_conditioning": (
            "current_q/current_qvel are captured from RoboPianist through rp1m_simulator; "
            "future target hand states are read from the held-out RP1M demo as planner output"
        ),
        "planner_source": "held_out_rp1m_hand_joints",
        "rollout_backend": "rp1m_simulator",
        "piano_state_policy": "not_restored_or_used_by_simulator",
        "reference_piano_state_policy": "scoring_only" if piano_ref is not None else "not_loaded",
        "initial_hand_restored": bool(effective_config.restore_initial_hand),
        "qpos_restored_count": int(qpos_restored),
        "qvel_restored_count": int(qvel_restored),
        "post_reset_hand_anchor": post_reset_hand_anchor,
        "action_control_overrides": {
            "wrist_action_policy": effective_config.wrist_action_policy,
            "indices": sorted(int(idx) for idx in control_overrides),
            "values": {str(idx): float(value) for idx, value in sorted(control_overrides.items())},
        },
        "source_steps_requested": int(source_steps),
        "source_steps_played": int(played_arr.shape[0]),
        "dense_steps_played": int(dense_played_arr.shape[0]),
        "control_timestep": float(control_dt),
        "substeps_per_source_step": int(substeps),
        "action_substep_policy": str(effective_config.action_substep_policy),
        "actions_executed": int(actions_executed),
        "terminated": bool(terminated),
        "total_reward": float(total_reward),
        "params_initial": asdict(params if not adaptation_events else _params_from_event_initial(adaptation_events, params)),
        "params_final": asdict(params),
        "online_adaptation": asdict(adaptation),
        "online_adaptation_events": adaptation_events,
        "rollout_config": asdict(sim_config),
        "effective_rollout_config": asdict(effective_config),
        "hand_anchor_calibration": hand_anchor_calibration,
        "load_info": load_info,
        "chunk_vote_count_mean": None if chunk_vote_arr.size == 0 else float(chunk_vote_arr.mean()),
        "chunk_vote_count_min": None if chunk_vote_arr.size == 0 else int(chunk_vote_arr.min()),
        "chunk_vote_count_max": None if chunk_vote_arr.size == 0 else int(chunk_vote_arr.max()),
        "action_summary": action_summary,
        "hand_qpos_l2_vs_next_reference": hand_summary,
        "against_goals": _score_against(goals, played_arr, threshold=float(effective_config.threshold)),
        "against_reference_piano_states": None
        if piano_ref is None
        else _score_against(piano_ref, played_arr, threshold=float(effective_config.threshold)),
        "recorded_reference_against_goals": None
        if piano_ref is None
        else _score_against(goals, piano_ref[:source_steps], threshold=float(effective_config.threshold)),
        "rollout_npz": str(npz_path),
        "video_path": None if video_path is None else str(video_path),
        "rendered_frames": int(len(frames)),
        "render_error": render_error,
        "audio_warning": audio_warning,
    }
    summary["closed_loop_score"] = closed_loop_score(summary)
    summary_path = rp1m.save_json(output / "closed_loop_summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def rollout_checkpoint_with_rp1m_simulator(
    *,
    checkpoint_path: str | Path,
    raw_episode: dict[str, np.ndarray | None],
    song_key: str,
    demo_id: int,
    output_dir: str | Path,
    device: str = "cuda",
    params: ClosedLoopParams | None = None,
    adaptation: OnlineAdaptationConfig | None = None,
    rollout_config: RolloutConfig | None = None,
    environment_name: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    policy = load_policy(checkpoint_path, device=device)
    return rollout_loaded_policy_with_rp1m_simulator(
        policy=policy,
        raw_episode=raw_episode,
        song_key=song_key,
        demo_id=int(demo_id),
        output_dir=output_dir,
        params=params,
        adaptation=adaptation,
        rollout_config=rollout_config,
        environment_name=environment_name,
        max_steps=max_steps,
    )


def closed_loop_score(summary: dict[str, Any]) -> float:
    goals = summary.get("against_goals") or {}
    action = summary.get("action_summary") or {}
    hand = summary.get("hand_qpos_l2_vs_next_reference") or {}
    key_f1 = _finite(goals.get("key_f1"), default=0.0)
    mispress = _finite(goals.get("mispress_rate"), default=1.0)
    saturation = _finite(action.get("action_saturation_fraction"), default=0.0)
    hand_mean = _finite(hand.get("mean"), default=0.0)
    termination_penalty = 0.5 if summary.get("terminated") else 0.0
    return float(key_f1 - 0.5 * mispress - 0.25 * saturation - 0.02 * hand_mean - termination_penalty)


def row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    goals = summary.get("against_goals") or {}
    ref = summary.get("against_reference_piano_states") or {}
    hand = summary.get("hand_qpos_l2_vs_next_reference") or {}
    action = summary.get("action_summary") or {}
    params = summary.get("params_final") or {}
    return {
        "checkpoint": summary.get("checkpoint"),
        "song_key": summary.get("song_key"),
        "demo_id": summary.get("demo_id"),
        "score": summary.get("closed_loop_score"),
        "goal_f1": goals.get("key_f1"),
        "goal_precision": goals.get("key_precision"),
        "goal_recall": goals.get("key_recall"),
        "mispress_rate": goals.get("mispress_rate"),
        "reference_f1": ref.get("key_f1"),
        "hand_l2_mean": hand.get("mean"),
        "action_saturation_fraction": action.get("action_saturation_fraction"),
        "actions_executed": summary.get("actions_executed"),
        "terminated": summary.get("terminated"),
        "action_gain": params.get("action_gain"),
        "right_gain": params.get("right_gain"),
        "left_gain": params.get("left_gain"),
        "sustain_gain": params.get("sustain_gain"),
        "action_smoothing_alpha": params.get("action_smoothing_alpha"),
        "max_action_delta": params.get("max_action_delta"),
        "chunk_execution": params.get("chunk_execution"),
        "temporal_agg_decay": params.get("temporal_agg_decay"),
        "target_lead": params.get("target_lead"),
        "summary_path": summary.get("summary_path"),
        "video_path": summary.get("video_path"),
    }


def write_summary_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    out = Path(path)
    rp1m.ensure_dir(out.parent)
    fields = sorted({key for row in rows for key in row.keys()})
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _target_indices(step_index: int, total_steps: int, sample_config: SampleConfig, target_lead: int) -> np.ndarray:
    start = int(step_index) + int(sample_config.lookahead) + int(target_lead)
    return np.clip(
        np.arange(start, start + int(sample_config.goal_horizon)),
        0,
        max(int(total_steps) - 1, 0),
    )


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
    goal_step_index: int,
    sample_config: SampleConfig,
    stats: NormalizationStats,
    executed_norm_actions: list[np.ndarray],
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
        rows: list[np.ndarray] = []
        missing = max(int(sample_config.history) - len(executed_norm_actions), 0)
        rows.extend(np.zeros((action_dim,), dtype=np.float32) for _ in range(missing))
        rows.extend(executed_norm_actions[-int(sample_config.history) :])
        action_history_norm = np.stack(rows, axis=0).astype(np.float32)
    goals_norm = None
    if sample_config.include_goals:
        indices = np.clip(
            np.arange(int(goal_step_index), int(goal_step_index) + int(sample_config.goal_horizon)),
            0,
            max(int(goals.shape[0]) - 1, 0),
        )
        goal_window = np.asarray(goals[indices], dtype=np.float32)
        goals_norm = standardize(goal_window, stats.goal_mean, stats.goal_std)
    if sample_config.feature_mode == "planner_sequence":
        q_hist = _online_history_window(q_history, current_q, history=int(sample_config.history), pad="current")
        qvel_hist = _online_history_window(qvel_history, current_qvel, history=int(sample_config.history), pad="zero")
        return build_planner_sequence_feature(
            current_q_history_norm=standardize(q_hist, stats.q_mean, stats.q_std),
            current_qvel_history_norm=standardize(qvel_hist, stats.qvel_mean, stats.qvel_std),
            target_q_norm=target_q_norm,
            config=sample_config,
            target_qvel_norm=target_qvel_norm,
            action_history_norm=action_history_norm,
            goals_norm=goals_norm,
        )
    if sample_config.feature_mode != "planner_next":
        raise ValueError(f"Unsupported feature_mode={sample_config.feature_mode!r}")
    return build_planner_next_feature(
        current_q_norm=current_q_norm,
        current_qvel_norm=current_qvel_norm,
        target_q_norm=target_q_norm,
        config=sample_config,
        target_qvel_norm=target_qvel_norm,
        action_history_norm=None if action_history_norm is None else action_history_norm.reshape(-1),
        goals_norm=None if goals_norm is None else goals_norm.reshape(-1),
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


def _select_online_chunk_action(
    *,
    pred_norm_chunk: np.ndarray,
    chunk_votes_norm: list[tuple[int, int, np.ndarray]],
    step_index: int,
    chunk_execution: str,
    temporal_agg_decay: float,
) -> tuple[np.ndarray, int]:
    if pred_norm_chunk.ndim != 2 or pred_norm_chunk.shape[0] < 1:
        raise ValueError(f"Expected non-empty prediction chunk [C, action_dim], got {pred_norm_chunk.shape}")
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
    return ((actions * weights[:, None]).sum(axis=0) / total).astype(np.float32), int(len(votes))


def _apply_action_parameters(
    action: np.ndarray,
    *,
    previous_action: np.ndarray | None,
    params: ClosedLoopParams,
) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    values *= float(params.action_gain)
    if values.size == 39:
        values[:19] *= float(params.right_gain)
        values[19:38] *= float(params.left_gain)
        values[38:39] *= float(params.sustain_gain)
    if previous_action is not None:
        prev = np.asarray(previous_action, dtype=np.float32).reshape(-1)
        alpha = float(np.clip(params.action_smoothing_alpha, 0.0, 0.99))
        if alpha > 0.0:
            values = alpha * prev + (1.0 - alpha) * values
        if params.max_action_delta is not None:
            delta = abs(float(params.max_action_delta))
            values = prev + np.clip(values - prev, -delta, delta)
    return np.clip(values, -1.0, 1.0).astype(np.float32)


def _maybe_adapt_online_parameters(
    *,
    params: ClosedLoopParams,
    config: OnlineAdaptationConfig,
    goals: np.ndarray,
    played: list[np.ndarray],
    step_index: int,
    events: list[dict[str, Any]],
    threshold: float,
) -> None:
    if not config.enabled:
        return
    if (int(step_index) + 1) % max(int(config.interval), 1) != 0:
        return
    if len(played) < max(int(config.window), 1):
        return
    end = len(played)
    start = max(end - int(config.window), 0)
    played_window = np.stack(played[start:end], axis=0)
    goal_window = goals[start:end, : played_window.shape[-1]]
    metrics = rp1m.key_metrics(goal_window, played_window, threshold=threshold)
    before = asdict(params)
    reason = "hold"
    if metrics["mispress_rate"] > float(config.max_mispress_rate):
        params.action_gain = max(float(config.min_action_gain), float(params.action_gain) * (1.0 - float(config.gain_step)))
        params.action_smoothing_alpha = min(
            float(config.max_smoothing_alpha),
            float(params.action_smoothing_alpha) + float(config.smoothing_step),
        )
        params.temporal_agg_decay = min(
            float(config.max_temporal_agg_decay),
            float(params.temporal_agg_decay) + float(config.decay_step),
        )
        reason = "reduce_gain_for_mispress"
    elif metrics["key_recall"] < float(config.target_recall) and metrics["key_precision"] >= float(config.min_precision_for_gain):
        params.action_gain = min(float(config.max_action_gain), float(params.action_gain) * (1.0 + float(config.gain_step)))
        params.action_smoothing_alpha = max(
            float(config.min_smoothing_alpha),
            float(params.action_smoothing_alpha) - float(config.smoothing_step),
        )
        reason = "increase_gain_for_low_recall"
    if reason != "hold":
        events.append(
            {
                "step": int(step_index),
                "window_start": int(start),
                "window_end": int(end),
                "reason": reason,
                "metrics": metrics,
                "before": before,
                "after": asdict(params),
            }
        )


def _score_against(target: np.ndarray, played: np.ndarray, *, threshold: float) -> dict[str, Any]:
    steps = min(int(target.shape[0]), int(played.shape[0]))
    keys = min(int(target.shape[-1]), int(played.shape[-1]), 88)
    if steps <= 0 or keys <= 0:
        metrics = {"key_precision": 0.0, "key_recall": 0.0, "key_f1": 0.0, "mispress_rate": 0.0}
    else:
        metrics = rp1m.key_metrics(target[:steps, :keys], played[:steps, :keys], threshold=threshold)
    return {**metrics, "scored_steps": int(steps), "scored_keys": int(keys)}


def _action_summary(actions: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return {
            "input_actions_min": None,
            "input_actions_max": None,
            "input_actions_mean": None,
            "input_actions_std": None,
            "input_actions_abs_p95": None,
            "action_delta_abs_p95": None,
            "action_saturation_fraction": None,
        }
    flat = arr.reshape(-1)
    deltas = np.diff(arr, axis=0).reshape(-1) if arr.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    return {
        "input_actions_min": float(flat.min()),
        "input_actions_max": float(flat.max()),
        "input_actions_mean": float(flat.mean()),
        "input_actions_std": float(flat.std()),
        "input_actions_abs_p95": float(np.percentile(np.abs(flat), 95)),
        "action_delta_abs_p95": None if deltas.size == 0 else float(np.percentile(np.abs(deltas), 95)),
        "action_saturation_fraction": float(np.mean(np.abs(flat) >= 0.999)),
    }


def _hand_summary(sim_hand: np.ndarray, q_ref: np.ndarray) -> dict[str, Any] | None:
    if sim_hand.size == 0 or q_ref.shape[0] < 2:
        return None
    n = min(int(sim_hand.shape[0]), int(q_ref.shape[0]) - 1)
    if n <= 0:
        return None
    width = min(int(sim_hand.shape[-1]), int(q_ref.shape[-1]))
    values = np.linalg.norm(sim_hand[:n, :width] - q_ref[1 : n + 1, :width], axis=1)
    return {
        "alignment": "sim_hand_after_action_t_vs_reference_hand_t_plus_1",
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _qvel_between(hand_joints: np.ndarray, source_t: int, dt: float) -> np.ndarray:
    next_t = min(int(source_t) + 1, int(hand_joints.shape[0]) - 1)
    return ((hand_joints[next_t] - hand_joints[int(source_t)]) / float(dt)).astype(np.float32)


def _validate_params(params: ClosedLoopParams) -> None:
    if params.chunk_execution not in {"first", "temporal_aggregate"}:
        raise ValueError(f"Unsupported chunk_execution={params.chunk_execution!r}")
    if not (0.0 < float(params.temporal_agg_decay) <= 1.0):
        raise ValueError("temporal_agg_decay must be in (0, 1]")
    if not (0.0 <= float(params.action_smoothing_alpha) < 1.0):
        raise ValueError("action_smoothing_alpha must be in [0, 1)")
    if params.max_action_delta is not None and float(params.max_action_delta) <= 0.0:
        raise ValueError("max_action_delta must be positive when set")


def _validate_adaptation(config: OnlineAdaptationConfig) -> None:
    if int(config.window) <= 0 or int(config.interval) <= 0:
        raise ValueError("Online adaptation window and interval must be positive")
    if not (0.0 <= float(config.target_recall) <= 1.0):
        raise ValueError("target_recall must be in [0, 1]")
    if not (0.0 <= float(config.max_mispress_rate) <= 1.0):
        raise ValueError("max_mispress_rate must be in [0, 1]")


def _finite(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _params_from_event_initial(events: list[dict[str, Any]], fallback: ClosedLoopParams) -> ClosedLoopParams:
    if not events:
        return fallback
    return ClosedLoopParams(**events[0].get("before", asdict(fallback)))
