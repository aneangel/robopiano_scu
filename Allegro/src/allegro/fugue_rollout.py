from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from allegro.alignment import (
    AllegroAligner,
    HighFrequencyAlignmentConfig,
    finite_difference_target,
    interpolate_source_target,
)
from fugue.data import standardize, unstandardize
from fugue.rp1m_closed_loop import (
    ClosedLoopParams,
    LoadedPolicy,
    _action_summary,
    _apply_action_parameters,
    _build_online_feature,
    _hand_summary,
    _qvel_between,
    _score_against,
    _select_online_chunk_action,
    _target_indices,
    closed_loop_score,
    make_rollout_config,
)
from rp1m_simulator import simulator as rp1m
from rp1m_simulator.simulator import RolloutConfig


def make_allegro_rollout_config(
    *,
    dataset_timestep: float,
    alignment: HighFrequencyAlignmentConfig,
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
    reduced_action_space: bool = True,
    hand_anchor_y_offset: float | None = rp1m.DEFAULT_HAND_ANCHOR_Y_OFFSET,
    auto_hand_anchor_y_offset: bool = True,
    wrist_action_policy: str = "hold_initial",
    initial_hand_qvel_scale: float = 0.5,
    hand_resync_interval: int | None = None,
    restore_initial_hand: bool = False,
    set_hand_qvel: bool = False,
    gravity_compensation: bool = False,
    primitive_fingertip_collisions: bool = False,
    disable_hand_collisions: bool = False,
) -> RolloutConfig:
    base = make_rollout_config(
        dataset_timestep=float(dataset_timestep),
        seed=int(seed),
        threshold=float(threshold),
        render_mp4=bool(render_mp4),
        render_audio=bool(render_audio),
        render_every_source_step=max(int(render_every_source_step), 1),
        width=int(width),
        height=int(height),
        fps=int(fps),
        action_source_scale=str(action_source_scale),
        action_mapping=str(action_mapping),
        action_substep_policy=str(action_substep_policy),
        reduced_action_space=bool(reduced_action_space),
        hand_anchor_y_offset=hand_anchor_y_offset,
        auto_hand_anchor_y_offset=bool(auto_hand_anchor_y_offset),
        restore_initial_hand=bool(restore_initial_hand),
        set_hand_qvel=bool(set_hand_qvel),
        gravity_compensation=bool(gravity_compensation),
        primitive_fingertip_collisions=bool(primitive_fingertip_collisions),
        disable_hand_collisions=bool(disable_hand_collisions),
    )
    return replace(
        base,
        mode="action",
        simulation_timestep=float(alignment.control_dt),
        wrist_action_policy=str(wrist_action_policy),
        initial_hand_qvel_scale=float(initial_hand_qvel_scale),
        hand_resync_interval=hand_resync_interval,
    )


def rollout_loaded_policy_with_allegro(
    *,
    policy: LoadedPolicy,
    raw_episode: dict[str, np.ndarray | None],
    song_key: str,
    demo_id: int,
    output_dir: str | Path,
    alignment: HighFrequencyAlignmentConfig | None = None,
    params: ClosedLoopParams | None = None,
    rollout_config: RolloutConfig | None = None,
    environment_name: str | None = None,
    max_steps: int | None = None,
    alignment_target_mode: str = "linear",
    base_action_source: str = "fugue",
    alignment_target_power: float = 1.0,
    dense_target_hand_joints: np.ndarray | None = None,
    dense_target_hand_velocities: np.ndarray | None = None,
    dense_goals_override: np.ndarray | None = None,
    dense_target_source: str | None = None,
    state_correction_gain: float = 0.0,
    state_correction_qvel_scale: float = 1.0,
    post_step_state_correction: bool = False,
) -> dict[str, Any]:
    alignment = alignment or HighFrequencyAlignmentConfig()
    alignment.validate()
    params = replace(params) if params is not None else ClosedLoopParams()
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
    if not np.isclose(float(alignment.source_dt), dt, rtol=1e-4, atol=1e-7):
        raise ValueError(
            f"alignment source_hz={alignment.source_hz} implies dt={alignment.source_dt}, "
            f"but Fugue checkpoint dt={dt}"
        )
    substeps = int(alignment.substeps_per_source_step)
    control_dt = float(alignment.control_dt)
    source_steps = min(int(q_ref.shape[0]), int(qvel_ref.shape[0]), int(reference_actions.shape[0]), int(goals.shape[0]))
    if piano_ref is not None:
        source_steps = min(source_steps, int(piano_ref.shape[0]))
    if max_steps is not None:
        source_steps = min(source_steps, int(max_steps))
    dense_target_q_ref = None
    dense_target_qvel_ref = None
    dense_goal_ref = None
    if dense_target_hand_joints is not None:
        dense_target_q_ref = np.asarray(dense_target_hand_joints, dtype=np.float32)
        if dense_target_q_ref.ndim != 2:
            raise ValueError("dense_target_hand_joints must be a 2D array")
        if int(dense_target_q_ref.shape[-1]) != int(q_ref.shape[-1]):
            raise ValueError(
                "dense_target_hand_joints width does not match source hand joints: "
                f"{dense_target_q_ref.shape[-1]} != {q_ref.shape[-1]}"
            )
        max_source_steps_from_dense = int(dense_target_q_ref.shape[0] // substeps) + 1
        source_steps = min(source_steps, max_source_steps_from_dense)
    if dense_target_hand_velocities is not None:
        dense_target_qvel_ref = np.asarray(dense_target_hand_velocities, dtype=np.float32)
        if dense_target_qvel_ref.ndim != 2:
            raise ValueError("dense_target_hand_velocities must be a 2D array")
        if int(dense_target_qvel_ref.shape[-1]) != int(q_ref.shape[-1]):
            raise ValueError(
                "dense_target_hand_velocities width does not match source hand joints: "
                f"{dense_target_qvel_ref.shape[-1]} != {q_ref.shape[-1]}"
            )
    if dense_goals_override is not None:
        dense_goal_ref = np.asarray(dense_goals_override, dtype=np.float32)[..., :88]
        if dense_goal_ref.ndim != 2 or int(dense_goal_ref.shape[-1]) != 88:
            raise ValueError("dense_goals_override must be a 2D piano-roll array with 88 keys")
    if source_steps < 2:
        raise ValueError(f"Need at least two source steps, got {source_steps}")
    lookahead = int(policy.sample_config.lookahead)
    if source_steps <= lookahead + 1:
        raise ValueError(f"Not enough steps for Fugue lookahead={lookahead}: {source_steps}")
    if alignment_target_mode not in {"linear", "endpoint", "dense"}:
        raise ValueError(f"Unsupported alignment_target_mode={alignment_target_mode!r}")
    if alignment_target_mode == "dense" and dense_target_q_ref is None:
        raise ValueError("alignment_target_mode='dense' requires dense_target_hand_joints")
    if base_action_source not in {"fugue", "reference", "zero", "target_qpos_sum", "target_qpos_mean"}:
        raise ValueError(f"Unsupported base_action_source={base_action_source!r}")
    if float(alignment_target_power) <= 0.0:
        raise ValueError("alignment_target_power must be positive")
    state_correction_gain = float(state_correction_gain)
    state_correction_qvel_scale = float(state_correction_qvel_scale)
    if state_correction_gain < 0.0 or state_correction_gain > 1.0:
        raise ValueError("state_correction_gain must be in [0, 1]")
    if state_correction_qvel_scale < 0.0:
        raise ValueError("state_correction_qvel_scale must be nonnegative")

    output = rp1m.ensure_dir(output_dir)
    if dense_goal_ref is None:
        dense_goals = np.repeat(goals[:source_steps, :88], substeps, axis=0)
        dense_goal_source = "repeat_source_goals"
    else:
        dense_goal_steps = min(int(dense_goal_ref.shape[0]), int(source_steps * substeps))
        dense_goals = dense_goal_ref[:dense_goal_steps]
        dense_goal_source = "external_dense_goals"
    effective_alignment_target_mode = "dense" if dense_target_q_ref is not None else str(alignment_target_mode)
    sim_config = rollout_config or make_allegro_rollout_config(dataset_timestep=dt, alignment=alignment)
    sim_config = replace(sim_config, mode="action", dataset_timestep=dt, simulation_timestep=control_dt)
    environment = environment_name or rp1m.canonicalize_environment_name(song_key)
    midi_proto = rp1m.write_goals_proto(
        dense_goals,
        output / "allegro_target_goals.proto",
        dt=control_dt,
        title=f"Allegro 200Hz alignment {song_key} demo {demo_id}",
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
    dense_hand: list[np.ndarray] = []
    dense_target_hand: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    sim_hand: list[np.ndarray] = []
    base_actions_source: list[np.ndarray] = []
    adjusted_actions_source: list[np.ndarray] = []
    base_actions_dense: list[np.ndarray] = []
    adjusted_actions_dense: list[np.ndarray] = []
    residuals_dense: list[np.ndarray] = []
    feedback_residuals_dense: list[np.ndarray] = []
    learned_residuals_dense: list[np.ndarray] = []
    raw_predicted_actions: list[np.ndarray] = []
    executed_norm_actions: list[np.ndarray] = []
    chunk_votes_norm: list[tuple[int, int, np.ndarray]] = []
    chunk_vote_counts: list[int] = []
    q_history: list[np.ndarray] = []
    qvel_history: list[np.ndarray] = []
    alignment_diagnostics: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = False
    render_error = None
    previous_source_q: np.ndarray | None = None
    previous_base_action: np.ndarray | None = None
    actions_executed = 0
    qpos_restored = 0
    qvel_restored = 0
    hand_resync_count = 0
    state_correction_count = 0
    state_correction_pre_count = 0
    state_correction_post_count = 0
    state_correction_qpos_restored = 0
    state_correction_qvel_restored = 0
    post_reset_hand_anchor = {"applied": False, "hand_anchor_y_offset": None}
    mujoco_options_applied: dict[str, int] = {}
    control_overrides: dict[int, float] = {}
    aligner: AllegroAligner | None = None

    try:
        env.reset()
        task, physics, _piano = rp1m._locate_task_physics_piano(env)
        mujoco_options_applied = rp1m._apply_mujoco_options(physics, effective_config)
        if effective_config.hand_anchor_application == "post_reset":
            post_reset_hand_anchor = rp1m._apply_post_reset_hand_anchor_y_offset(
                task,
                physics,
                effective_config.hand_anchor_y_offset,
            )
        initial_target_q = dense_target_q_ref[0] if dense_target_q_ref is not None else q_ref[0]
        if effective_config.restore_initial_hand:
            qpos_restored = rp1m._set_hand_qpos(task, physics, initial_target_q)
            if effective_config.set_hand_qvel:
                if dense_target_qvel_ref is not None and dense_target_qvel_ref.shape[0] > 0:
                    initial_qvel = float(effective_config.initial_hand_qvel_scale) * dense_target_qvel_ref[0]
                else:
                    initial_qvel = float(effective_config.initial_hand_qvel_scale) * _qvel_between(q_ref, 0, dt)
                qvel_restored = rp1m._set_hand_qvel(task, physics, initial_qvel)
            if hasattr(physics, "forward"):
                physics.forward()
        current_q = rp1m._capture_hand_qpos(task, physics)
        if current_q is None:
            current_q = q_ref[0].copy()
        current_q = np.asarray(current_q, dtype=np.float32)
        previous_source_q = current_q.copy()
        action_spec = env.action_spec()
        if int(policy.checkpoint["action_dim"]) != int(action_spec.shape[0]):
            raise ValueError(
                f"Checkpoint action_dim={policy.checkpoint['action_dim']} does not match "
                f"environment action_dim={action_spec.shape[0]}"
            )
        wrist_qpos = initial_target_q if effective_config.restore_initial_hand else current_q
        control_overrides = rp1m._initial_wrist_control_overrides(wrist_qpos, action_spec, effective_config)
        aligner = AllegroAligner(
            config=alignment,
            q_dim=int(current_q.size),
            action_dim=int(action_spec.shape[0]),
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

        def apply_state_correction(
            current: np.ndarray,
            target: np.ndarray,
            target_velocity: np.ndarray,
        ) -> np.ndarray:
            nonlocal state_correction_count
            nonlocal state_correction_qpos_restored, state_correction_qvel_restored
            if state_correction_gain <= 0.0:
                return np.asarray(current, dtype=np.float32)
            corrected = np.asarray(current, dtype=np.float32) + state_correction_gain * (
                np.asarray(target, dtype=np.float32) - np.asarray(current, dtype=np.float32)
            )
            task, physics, _piano = rp1m._locate_task_physics_piano(env)
            state_correction_qpos_restored += int(rp1m._set_hand_qpos(task, physics, corrected))
            target_velocity = np.asarray(target_velocity, dtype=np.float32)
            state_correction_qvel_restored += int(
                rp1m._set_hand_qvel(task, physics, state_correction_qvel_scale * target_velocity)
            )
            if hasattr(physics, "forward"):
                physics.forward()
            pose = rp1m._capture_hand_qpos(task, physics)
            state_correction_count += 1
            return corrected.astype(np.float32) if pose is None else np.asarray(pose, dtype=np.float32)

        for source_t in range(source_steps - 1):
            resynced_this_step = False
            if (
                effective_config.hand_resync_interval is not None
                and int(effective_config.hand_resync_interval) > 0
                and source_t > 0
                and source_t % int(effective_config.hand_resync_interval) == 0
            ):
                task, physics, _piano = rp1m._locate_task_physics_piano(env)
                resync_target_q = q_ref[source_t]
                if dense_target_q_ref is not None:
                    resync_target_q = dense_target_q_ref[min(source_t * substeps, dense_target_q_ref.shape[0] - 1)]
                qpos_restored = rp1m._set_hand_qpos(task, physics, resync_target_q)
                hand_resync_count += 1
                if dense_target_qvel_ref is not None and dense_target_qvel_ref.shape[0] > 0:
                    qvel = float(effective_config.initial_hand_qvel_scale) * dense_target_qvel_ref[
                        min(source_t * substeps, dense_target_qvel_ref.shape[0] - 1)
                    ]
                else:
                    qvel = float(effective_config.initial_hand_qvel_scale) * _qvel_between(q_ref, source_t, dt)
                qvel_restored = rp1m._set_hand_qvel(task, physics, qvel)
                if hasattr(physics, "forward"):
                    physics.forward()
                pose = rp1m._capture_hand_qpos(task, physics)
                current_q = q_ref[source_t].copy() if pose is None else np.asarray(pose, dtype=np.float32)
                previous_source_q = q_ref[source_t - 1].astype(np.float32)
                resynced_this_step = True
            if resynced_this_step and qvel_ref.size:
                current_qvel_source = qvel_ref[source_t].astype(np.float32)
            else:
                current_qvel_source = (
                    np.zeros_like(current_q, dtype=np.float32)
                    if source_t == 0 or previous_source_q is None
                    else (current_q - previous_source_q) / dt
                )
            feature = _build_online_feature(
                current_q=current_q,
                current_qvel=current_qvel_source,
                q_history=q_history,
                qvel_history=qvel_history,
                target_q=q_ref[_target_indices(source_t, q_ref.shape[0], policy.sample_config, params.target_lead)],
                target_qvel=(
                    qvel_ref[_target_indices(source_t, qvel_ref.shape[0], policy.sample_config, params.target_lead)]
                    if policy.sample_config.include_future_qvel
                    else None
                ),
                goals=goals,
                step_index=source_t,
                goal_step_index=source_t + int(params.target_lead),
                sample_config=policy.sample_config,
                stats=policy.stats,
                executed_norm_actions=executed_norm_actions,
            )
            with torch.no_grad():
                feature_tensor = torch.from_numpy(feature[None]).to(policy.device).float()
                pred_norm_chunk = policy.model(feature_tensor).detach().cpu().numpy()[0].astype(np.float32)
            for offset, pred_norm_row in enumerate(pred_norm_chunk):
                target_step = int(source_t + int(policy.sample_config.delta) + int(offset))
                if target_step >= source_t:
                    chunk_votes_norm.append((target_step, source_t, pred_norm_row.copy()))
            pred_norm, vote_count = _select_online_chunk_action(
                pred_norm_chunk=pred_norm_chunk,
                chunk_votes_norm=chunk_votes_norm,
                step_index=source_t,
                chunk_execution=params.chunk_execution,
                temporal_agg_decay=float(params.temporal_agg_decay),
            )
            chunk_votes_norm = [vote for vote in chunk_votes_norm if int(vote[0]) > source_t]
            chunk_vote_counts.append(int(vote_count))
            fugue_raw_action = unstandardize(pred_norm.reshape(1, -1), policy.stats.action_mean, policy.stats.action_std)[0]
            if base_action_source == "reference":
                raw_action = reference_actions[source_t]
            elif base_action_source == "zero":
                raw_action = np.zeros_like(reference_actions[source_t], dtype=np.float32)
            elif base_action_source in {"target_qpos_sum", "target_qpos_mean"}:
                raw_action = np.zeros_like(reference_actions[source_t], dtype=np.float32)
            else:
                raw_action = fugue_raw_action
            base_action = _apply_action_parameters(raw_action, previous_action=previous_base_action, params=params)
            base_actions_source.append(base_action.astype(np.float32))
            raw_predicted_actions.append(fugue_raw_action.astype(np.float32))

            source_start_q = current_q.copy()
            q_history.append(source_start_q.astype(np.float32))
            qvel_history.append(current_qvel_source.astype(np.float32))
            q_history = q_history[-max(int(policy.sample_config.history) - 1, 0) :]
            qvel_history = qvel_history[-max(int(policy.sample_config.history) - 1, 0) :]

            interval_played: list[np.ndarray] = []
            interval_actions: list[np.ndarray] = []
            dense_previous_q = current_q.copy()
            for substep in range(substeps):
                dense_qvel = (
                    current_qvel_source
                    if substep == 0
                    else (current_q - dense_previous_q) / control_dt
                )
                if dense_target_q_ref is not None:
                    dense_target_index = min(source_t * substeps + substep, dense_target_q_ref.shape[0] - 1)
                    target_q = dense_target_q_ref[dense_target_index]
                    if dense_target_qvel_ref is not None and dense_target_qvel_ref.shape[0] > 0:
                        target_qvel = dense_target_qvel_ref[min(dense_target_index, dense_target_qvel_ref.shape[0] - 1)]
                    else:
                        next_index = min(dense_target_index + 1, dense_target_q_ref.shape[0] - 1)
                        target_qvel = (dense_target_q_ref[next_index] - dense_target_q_ref[dense_target_index]) / control_dt
                elif alignment_target_mode == "endpoint":
                    target_index = min(source_t + 1, q_ref.shape[0] - 1)
                    target_q = q_ref[target_index]
                    target_qvel = qvel_ref[target_index] if qvel_ref.size else finite_difference_target(
                        q_ref,
                        source_step=source_t,
                        source_dt=dt,
                    )
                else:
                    target_q = interpolate_source_target(
                        q_ref,
                        source_step=source_t,
                        substep=substep,
                        substeps_per_source_step=substeps,
                        phase_power=float(alignment_target_power),
                    )
                    if qvel_ref.size:
                        target_qvel = interpolate_source_target(
                            qvel_ref,
                            source_step=source_t,
                            substep=substep,
                            substeps_per_source_step=substeps,
                            phase_power=float(alignment_target_power),
                        )
                    else:
                        target_qvel = finite_difference_target(q_ref, source_step=source_t, source_dt=dt)
                if state_correction_gain > 0.0:
                    current_q = apply_state_correction(current_q, target_q, target_qvel)
                    dense_qvel = (state_correction_qvel_scale * target_qvel).astype(np.float32)
                    state_correction_pre_count += 1
                if base_action_source in {"target_qpos_sum", "target_qpos_mean"}:
                    target_mode = "mean" if base_action_source.endswith("mean") else "sum"
                    substep_base_action = _hand_qpos_to_normalized_action(
                        target_q,
                        action_spec=action_spec,
                        tendon_mode=target_mode,
                    )
                    substep_base_action = np.clip(substep_base_action * float(params.action_gain), -1.0, 1.0)
                else:
                    substep_base_action = base_action
                alignment_result = aligner.act(
                    base_action=substep_base_action,
                    current_q=current_q,
                    current_qvel=dense_qvel,
                    target_q=target_q,
                    target_qvel=target_qvel,
                    source_step=source_t,
                    substep=substep,
                )
                control = rp1m._prepare_control(alignment_result.action, action_spec, effective_config)
                control = rp1m._apply_control_overrides(control, control_overrides)
                dense_start_q = current_q.copy()
                timestep = env.step(control)
                total_reward += float(timestep.reward or 0.0)
                actions_executed += 1
                base_actions_dense.append(substep_base_action.astype(np.float32))
                adjusted_actions_dense.append(alignment_result.action.astype(np.float32))
                residuals_dense.append(alignment_result.residual.astype(np.float32))
                feedback_residuals_dense.append(alignment_result.feedback_residual.astype(np.float32))
                learned_residuals_dense.append(alignment_result.learned_residual.astype(np.float32))
                alignment_diagnostics.append(dict(alignment_result.diagnostics))
                interval_actions.append(alignment_result.action.astype(np.float32))
                activation = rp1m._capture_piano_activation(env)
                if activation is not None:
                    activation = np.asarray(activation, dtype=np.float32)[:88]
                    interval_played.append(activation)
                    dense_played.append(activation)
                task, physics, _piano = rp1m._locate_task_physics_piano(env)
                pose = rp1m._capture_hand_qpos(task, physics)
                if pose is not None:
                    current_q = np.asarray(pose, dtype=np.float32)
                    aligner.observe_transition(
                        action=alignment_result.action,
                        q_before=dense_start_q,
                        q_after=current_q,
                    )
                if post_step_state_correction and state_correction_gain > 0.0:
                    current_q = apply_state_correction(current_q, target_q, target_qvel)
                    state_correction_post_count += 1
                dense_hand.append(current_q.astype(np.float32))
                dense_target_hand.append(target_q.astype(np.float32))
                dense_previous_q = dense_start_q
                if timestep.last():
                    terminated = True
                    break
            if interval_played:
                source_played.append(np.max(np.stack(interval_played, axis=0), axis=0).astype(np.float32))
            if interval_actions:
                adjusted_mean = np.mean(np.stack(interval_actions, axis=0), axis=0).astype(np.float32)
            else:
                adjusted_mean = base_action.astype(np.float32)
            adjusted_actions_source.append(adjusted_mean)
            executed_norm = standardize(
                adjusted_mean.reshape(1, -1),
                policy.stats.action_mean,
                policy.stats.action_std,
            )[0]
            executed_norm_actions.append(executed_norm.astype(np.float32))
            previous_base_action = base_action.astype(np.float32)
            sim_hand.append(current_q.copy())
            previous_source_q = source_start_q
            if effective_config.render_mp4 and render_error is None:
                if (source_t + 1) % max(int(effective_config.render_every_source_step), 1) == 0:
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
            if terminated:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    played_arr = np.stack(source_played, axis=0) if source_played else np.zeros((0, 88), dtype=np.float32)
    dense_played_arr = np.stack(dense_played, axis=0) if dense_played else np.zeros((0, 88), dtype=np.float32)
    dense_hand_arr = np.stack(dense_hand, axis=0) if dense_hand else np.zeros((0, q_ref.shape[-1]), dtype=np.float32)
    dense_target_arr = (
        np.stack(dense_target_hand, axis=0) if dense_target_hand else np.zeros((0, q_ref.shape[-1]), dtype=np.float32)
    )
    base_source_arr = _stack_or_empty(base_actions_source, reference_actions.shape[-1])
    adjusted_source_arr = _stack_or_empty(adjusted_actions_source, reference_actions.shape[-1])
    base_dense_arr = _stack_or_empty(base_actions_dense, reference_actions.shape[-1])
    adjusted_dense_arr = _stack_or_empty(adjusted_actions_dense, reference_actions.shape[-1])
    residual_dense_arr = _stack_or_empty(residuals_dense, reference_actions.shape[-1])
    feedback_dense_arr = _stack_or_empty(feedback_residuals_dense, reference_actions.shape[-1])
    learned_dense_arr = _stack_or_empty(learned_residuals_dense, reference_actions.shape[-1])
    raw_pred_arr = _stack_or_empty(raw_predicted_actions, reference_actions.shape[-1])
    norm_arr = _stack_or_empty(executed_norm_actions, reference_actions.shape[-1])
    sim_hand_arr = np.stack(sim_hand, axis=0) if sim_hand else np.zeros((0, q_ref.shape[-1]), dtype=np.float32)
    chunk_vote_arr = np.asarray(chunk_vote_counts, dtype=np.float32)
    npz_path = output / "allegro_200hz_rollout.npz"
    np.savez_compressed(
        npz_path,
        source_played_piano=played_arr,
        dense_played_piano=dense_played_arr,
        base_actions_source=base_source_arr,
        adjusted_actions_source=adjusted_source_arr,
        base_actions_dense=base_dense_arr,
        adjusted_actions_dense=adjusted_dense_arr,
        residuals_dense=residual_dense_arr,
        feedback_residuals_dense=feedback_dense_arr,
        learned_residuals_dense=learned_dense_arr,
        raw_predicted_actions=raw_pred_arr,
        executed_actions_normalized=norm_arr,
        sim_hand_joints=sim_hand_arr,
        target_hand_joints=q_ref[:source_steps],
        dense_sim_hand_joints=dense_hand_arr,
        dense_target_hand_joints=dense_target_arr,
        dense_goals=dense_goals[: max(dense_played_arr.shape[0], 1)],
        reference_actions=reference_actions[: max(adjusted_source_arr.shape[0], 1)],
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
            output / "allegro_200hz_rollout.mp4",
            fps=int(effective_config.fps),
            audio_events=events,
        )

    action_summary = _action_summary(adjusted_source_arr)
    hand_summary = _hand_summary(sim_hand_arr, q_ref)
    dense_hand_summary = _same_time_hand_summary(
        dense_hand_arr,
        dense_target_arr,
        alignment="dense_sim_hand_after_200hz_step_vs_dense_target",
    )
    summary: dict[str, Any] = {
        "label": "allegro_200hz_fugue_alignment",
        "checkpoint": str(policy.checkpoint_path),
        "song_key": str(song_key),
        "environment_name": str(environment),
        "demo_id": int(demo_id),
        "feature_mode": policy.sample_config.feature_mode,
        "rollout_backend": "rp1m_simulator",
        "base_controller": "fugue",
        "alignment_controller": "allegro",
        "source_hz": float(alignment.source_hz),
        "control_hz": float(alignment.control_hz),
        "substeps_per_source_step": int(substeps),
        "closed_loop_conditioning": (
            "Fugue predicts one base action at 20 Hz; Allegro executes 200 Hz residual "
            "actions from current RoboPianist hand state and online feedback-error labels."
        ),
        "alignment_target_mode": str(effective_alignment_target_mode),
        "requested_alignment_target_mode": str(alignment_target_mode),
        "alignment_target_power": float(alignment_target_power),
        "base_action_source": str(base_action_source),
        "planner_source": "held_out_rp1m_hand_joints" if dense_target_q_ref is None else "external_dense_hand_plan",
        "dense_target_source": None if dense_target_q_ref is None else str(dense_target_source or "external_dense_hand_joints"),
        "dense_target_steps_available": 0 if dense_target_q_ref is None else int(dense_target_q_ref.shape[0]),
        "dense_goal_source": dense_goal_source,
        "dense_goal_steps_available": int(dense_goals.shape[0]),
        "piano_state_policy": "not_restored_or_used_by_simulator",
        "reference_piano_state_policy": "scoring_only" if piano_ref is not None else "not_loaded",
        "initial_hand_restored": bool(effective_config.restore_initial_hand),
        "qpos_restored_count": int(qpos_restored),
        "qvel_restored_count": int(qvel_restored),
        "source_steps_requested": int(source_steps),
        "source_steps_played": int(played_arr.shape[0]),
        "dense_steps_played": int(dense_played_arr.shape[0]),
        "actions_executed": int(actions_executed),
        "terminated": bool(terminated),
        "total_reward": float(total_reward),
        "fugue_params": asdict(params),
        "alignment": {} if aligner is None else aligner.summary(),
        "rollout_config": asdict(sim_config),
        "effective_rollout_config": asdict(effective_config),
        "hand_anchor_calibration": hand_anchor_calibration,
        "post_reset_hand_anchor": post_reset_hand_anchor,
        "mujoco_options_applied": mujoco_options_applied,
        "action_control_overrides": {
            "wrist_action_policy": effective_config.wrist_action_policy,
            "indices": sorted(int(idx) for idx in control_overrides),
            "values": {str(idx): float(value) for idx, value in sorted(control_overrides.items())},
        },
        "hand_resync_policy": {
            "interval": None
            if effective_config.hand_resync_interval is None
            else int(effective_config.hand_resync_interval),
            "uses_rp1m_hand_joints": bool(effective_config.hand_resync_interval is not None),
            "resync_count": int(hand_resync_count),
        },
        "state_correction_policy": {
            "gain": float(state_correction_gain),
            "qvel_scale": float(state_correction_qvel_scale),
            "pre_step": bool(state_correction_gain > 0.0),
            "post_step": bool(post_step_state_correction and state_correction_gain > 0.0),
            "uses_rp1m_hand_joints": bool(state_correction_gain > 0.0),
            "correction_count": int(state_correction_count),
            "pre_step_count": int(state_correction_pre_count),
            "post_step_count": int(state_correction_post_count),
            "qpos_restored_count": int(state_correction_qpos_restored),
            "qvel_restored_count": int(state_correction_qvel_restored),
        },
        "load_info": load_info,
        "chunk_vote_count_mean": None if chunk_vote_arr.size == 0 else float(chunk_vote_arr.mean()),
        "chunk_vote_count_min": None if chunk_vote_arr.size == 0 else int(chunk_vote_arr.min()),
        "chunk_vote_count_max": None if chunk_vote_arr.size == 0 else int(chunk_vote_arr.max()),
        "action_summary": action_summary,
        "base_action_summary": _action_summary(base_source_arr),
        "dense_action_summary": _action_summary(adjusted_dense_arr),
        "dense_residual_summary": _action_summary(residual_dense_arr),
        "hand_qpos_l2_vs_next_reference": hand_summary,
        "dense_hand_qpos_l2_vs_target": dense_hand_summary,
        "against_goals": _score_against(goals, played_arr, threshold=float(effective_config.threshold)),
        "against_dense_goals": _score_against(
            dense_goals,
            dense_played_arr,
            threshold=float(effective_config.threshold),
        ),
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
        "alignment_diagnostics_tail": alignment_diagnostics[-20:],
    }
    summary["closed_loop_score"] = closed_loop_score(summary)
    summary_path = rp1m.save_json(output / "allegro_200hz_summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _stack_or_empty(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, int(width)), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _same_time_hand_summary(sim_hand: np.ndarray, target_hand: np.ndarray, *, alignment: str) -> dict[str, Any] | None:
    if sim_hand.size == 0 or target_hand.size == 0:
        return None
    n = min(int(sim_hand.shape[0]), int(target_hand.shape[0]))
    if n <= 0:
        return None
    width = min(int(sim_hand.shape[-1]), int(target_hand.shape[-1]))
    values = np.linalg.norm(sim_hand[:n, :width] - target_hand[:n, :width], axis=1)
    abs_error = np.abs(sim_hand[:n, :width] - target_hand[:n, :width])
    return {
        "alignment": str(alignment),
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "rmse_mean": float(np.sqrt(np.mean(np.square(abs_error)))),
        "max_abs_mean": float(np.max(abs_error, axis=1).mean()),
        "max_abs_p95": float(np.percentile(np.max(abs_error, axis=1), 95)),
    }


def _hand_qpos_to_normalized_action(
    hand_qpos: np.ndarray,
    *,
    action_spec: Any,
    tendon_mode: str = "sum",
) -> np.ndarray:
    q = np.asarray(hand_qpos, dtype=np.float32).reshape(-1)
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    if minimum.size != 39 or maximum.size != 39 or q.size < 46:
        return np.zeros((minimum.size,), dtype=np.float32)
    control = np.zeros((39,), dtype=np.float32)
    tendon_scale = 0.5 if tendon_mode == "mean" else 1.0

    def pair(a: int, b: int) -> float:
        return float(tendon_scale * (q[a] + q[b]))

    control[0] = q[0]
    control[1] = q[1]
    control[2] = q[18]
    control[3] = q[19]
    control[4] = q[20]
    control[5] = q[2]
    control[6] = q[3]
    control[7] = pair(4, 5)
    control[8] = q[6]
    control[9] = q[7]
    control[10] = pair(8, 9)
    control[11] = q[10]
    control[12] = q[11]
    control[13] = pair(12, 13)
    control[14] = q[14]
    control[15] = q[15]
    control[16] = pair(16, 17)
    control[17] = q[21]
    control[18] = q[22]

    offset = 19
    qoff = 23
    control[offset + 0] = q[qoff + 0]
    control[offset + 1] = q[qoff + 1]
    control[offset + 2] = q[qoff + 18]
    control[offset + 3] = q[qoff + 19]
    control[offset + 4] = q[qoff + 20]
    control[offset + 5] = q[qoff + 2]
    control[offset + 6] = q[qoff + 3]
    control[offset + 7] = pair(qoff + 4, qoff + 5)
    control[offset + 8] = q[qoff + 6]
    control[offset + 9] = q[qoff + 7]
    control[offset + 10] = pair(qoff + 8, qoff + 9)
    control[offset + 11] = q[qoff + 10]
    control[offset + 12] = q[qoff + 11]
    control[offset + 13] = pair(qoff + 12, qoff + 13)
    control[offset + 14] = q[qoff + 14]
    control[offset + 15] = q[qoff + 15]
    control[offset + 16] = pair(qoff + 16, qoff + 17)
    control[offset + 17] = q[qoff + 21]
    control[offset + 18] = q[qoff + 22]
    denom = np.maximum(maximum - minimum, 1e-6)
    normalized = 2.0 * (control - minimum) / denom - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)
