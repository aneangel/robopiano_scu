from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ballade.action_masks import mask_for_goal_keys
from ballade.config import CostWeights, JacobianTrackerConfig, LocalSearchConfig
from ballade.costs import total_tracking_cost
from ballade.evaluation import summarize_rollout
from ballade.features import build_feature_vector, pad_or_trim
from ballade.jacobian_tracker import OnlineJacobianTracker
from ballade.local_search import CandidateActionSearch
from ballade.models import ResidualMLPController
from ballade.online_env import BalladeOnlineEnv, BalladeOnlineEnvConfig
from ballade.replay_buffer import OnlineTeacherReplayBuffer, TeacherTransition
from ballade.targets import build_micro_targets


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_online_jacobian_rollout(
    *,
    trajectory: Any,
    output_dir: str | Path,
    source_dt: float = 0.05,
    control_dt: float = 0.005,
    max_source_steps: int | None = None,
    env_config: BalladeOnlineEnvConfig | None = None,
    tracker_config: JacobianTrackerConfig | None = None,
    search_config: LocalSearchConfig | None = None,
    weights: CostWeights | dict[str, float] | None = None,
    collect_teacher: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hand = np.asarray(trajectory.hand_joints, dtype=np.float32)
    goals = np.asarray(trajectory.goals, dtype=np.float32)[:, :88]
    fingertips = None if trajectory.hand_fingertips is None else np.asarray(trajectory.hand_fingertips, dtype=np.float32)
    source_steps = min(hand.shape[0], goals.shape[0])
    if max_source_steps is not None:
        source_steps = min(source_steps, int(max_source_steps))
    if source_steps < 2:
        raise ValueError(f"Need at least two source steps, got {source_steps}")
    target_sequence = build_micro_targets(
        hand[:source_steps],
        goals_20hz=goals[:source_steps],
        fingertips_20hz=None if fingertips is None else fingertips[:source_steps],
        source_dt=source_dt,
        control_dt=control_dt,
        method="linear",
    )
    env = BalladeOnlineEnv(trajectory=trajectory, output_dir=output, config=env_config)
    tracker = OnlineJacobianTracker(action_dim=env.action_dim, config=tracker_config)
    search = CandidateActionSearch(config=search_config, weights=weights)
    replay = OnlineTeacherReplayBuffer()
    obs = env.reset()
    previous_action = np.zeros((env.action_dim,), dtype=np.float32)
    dense_played: list[np.ndarray] = []
    dense_hand: list[np.ndarray] = []
    dense_target_hand: list[np.ndarray] = []
    dense_goal: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    search_used: list[bool] = []
    terminated = False
    try:
        for target in target_sequence:
            mask = mask_for_goal_keys(target.goal_key_mask, action_dim=env.action_dim)
            base_action = tracker.propose_action(obs, target, previous_action, active_mask=mask)
            before_cost = total_tracking_cost(obs, target, base_action, previous_action, weights)
            result = search.search(env, obs, target, base_action, previous_action, active_mask=mask)
            selected_action = result.action
            next_obs, _reward, done, _info = env.step_normalized(selected_action)
            after_cost = total_tracking_cost(next_obs, target, selected_action, previous_action, weights)
            tracker.update(obs, selected_action, next_obs)
            if collect_teacher:
                replay.append(
                    TeacherTransition(
                        obs=obs.as_dict(),
                        target=target.as_dict(),
                        base_action=base_action,
                        selected_action=selected_action,
                        next_obs=next_obs.as_dict(),
                        cost_before=float(before_cost),
                        cost_after=float(after_cost),
                        source_frame_index=int(target.source_frame_index),
                        microstep_index=int(target.microstep_index),
                        song_key=str(trajectory.song_key),
                        demo_id=int(trajectory.demo_id),
                    )
                )
            dense_played.append(next_obs.piano_activation.astype(np.float32))
            dense_hand.append(next_obs.q.astype(np.float32))
            dense_target_hand.append(target.target_q_micro.astype(np.float32))
            dense_goal.append(target.goal_key_mask.astype(np.float32))
            actions.append(selected_action.astype(np.float32))
            search_used.append(bool(result.used_search))
            previous_action = selected_action
            obs = next_obs
            if done:
                terminated = True
                break
    finally:
        env.close()

    dense_played_arr = np.stack(dense_played, axis=0) if dense_played else np.zeros((0, 88), dtype=np.float32)
    dense_hand_arr = (
        np.stack(dense_hand, axis=0) if dense_hand else np.zeros((0, hand.shape[-1]), dtype=np.float32)
    )
    dense_target_arr = (
        np.stack(dense_target_hand, axis=0)
        if dense_target_hand
        else np.zeros((0, hand.shape[-1]), dtype=np.float32)
    )
    dense_goal_arr = np.stack(dense_goal, axis=0) if dense_goal else np.zeros((0, 88), dtype=np.float32)
    actions_arr = np.stack(actions, axis=0) if actions else np.zeros((0, 0), dtype=np.float32)
    search_arr = np.asarray(search_used, dtype=bool)
    npz_path = output / "ballade_rollout.npz"
    np.savez_compressed(
        npz_path,
        dense_played_piano=dense_played_arr,
        dense_target_goals=dense_goal_arr,
        dense_hand_qpos=dense_hand_arr,
        dense_target_hand_qpos=dense_target_arr,
        actions=actions_arr,
        search_used=search_arr,
    )
    if collect_teacher and len(replay):
        replay.save_shard(output / "teacher_data")
    summary = {
        "song_key": str(trajectory.song_key),
        "demo_id": int(trajectory.demo_id),
        "source_dt": float(source_dt),
        "control_dt": float(control_dt),
        "dense_steps_played": int(dense_played_arr.shape[0]),
        "terminated": bool(terminated),
        "rollout_npz": str(npz_path),
        **summarize_rollout(
            dense_goal_mask=dense_goal_arr,
            dense_played=dense_played_arr,
            dense_hand=dense_hand_arr,
            dense_target_hand=dense_target_arr,
            actions=actions_arr,
            search_used=search_arr,
            terminated=terminated,
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_residual_checkpoint(checkpoint: str | Path, *, device: str = "cpu") -> ResidualMLPController:
    payload = torch.load(checkpoint, map_location=torch.device(device))
    model = ResidualMLPController(
        feature_dim=int(payload["feature_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        hidden_layers=int(payload.get("hidden_layers", 3)),
        residual_scale=float(payload.get("residual_scale", 0.2)),
    ).to(torch.device(device))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def run_residual_controller_rollout(
    *,
    trajectory: Any,
    checkpoint: str | Path,
    output_dir: str | Path,
    source_dt: float = 0.05,
    control_dt: float = 0.005,
    max_source_steps: int | None = None,
    env_config: BalladeOnlineEnvConfig | None = None,
    tracker_config: JacobianTrackerConfig | None = None,
    search_config: LocalSearchConfig | None = None,
    weights: CostWeights | dict[str, float] | None = None,
    search_fallback: str = "none",
    device: str = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_residual_checkpoint(checkpoint, device=device)
    hand = np.asarray(trajectory.hand_joints, dtype=np.float32)
    goals = np.asarray(trajectory.goals, dtype=np.float32)[:, :88]
    fingertips = None if trajectory.hand_fingertips is None else np.asarray(trajectory.hand_fingertips, dtype=np.float32)
    source_steps = min(hand.shape[0], goals.shape[0])
    if max_source_steps is not None:
        source_steps = min(source_steps, int(max_source_steps))
    target_sequence = build_micro_targets(
        hand[:source_steps],
        goals_20hz=goals[:source_steps],
        fingertips_20hz=None if fingertips is None else fingertips[:source_steps],
        source_dt=source_dt,
        control_dt=control_dt,
        method="linear",
    )
    env = BalladeOnlineEnv(trajectory=trajectory, output_dir=output, config=env_config)
    tracker = OnlineJacobianTracker(action_dim=env.action_dim, config=tracker_config)
    effective_search_config = search_config or LocalSearchConfig()
    effective_search_config.enabled = str(search_fallback) != "none"
    search = CandidateActionSearch(config=effective_search_config, weights=weights)
    obs = env.reset()
    previous_action = np.zeros((env.action_dim,), dtype=np.float32)
    dense_played: list[np.ndarray] = []
    dense_hand: list[np.ndarray] = []
    dense_target_hand: list[np.ndarray] = []
    dense_goal: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    search_used: list[bool] = []
    terminated = False
    try:
        for target in target_sequence:
            mask = mask_for_goal_keys(target.goal_key_mask, action_dim=env.action_dim)
            base_action = tracker.propose_action(obs, target, previous_action, active_mask=mask)
            feature = build_feature_vector(obs, target, base_action)
            feature = pad_or_trim(feature, model.feature_dim)
            with torch.no_grad():
                tensor_feature = torch.from_numpy(feature[None]).to(torch.device(device)).float()
                tensor_base = torch.from_numpy(base_action[None]).to(torch.device(device)).float()
                policy_action = model(tensor_feature, tensor_base).detach().cpu().numpy()[0].astype(np.float32)
            result = search.search(env, obs, target, policy_action, previous_action, active_mask=mask)
            selected_action = result.action
            next_obs, _reward, done, _info = env.step_normalized(selected_action)
            tracker.update(obs, selected_action, next_obs)
            dense_played.append(next_obs.piano_activation.astype(np.float32))
            dense_hand.append(next_obs.q.astype(np.float32))
            dense_target_hand.append(target.target_q_micro.astype(np.float32))
            dense_goal.append(target.goal_key_mask.astype(np.float32))
            actions.append(selected_action.astype(np.float32))
            search_used.append(bool(result.used_search))
            previous_action = selected_action
            obs = next_obs
            if done:
                terminated = True
                break
    finally:
        env.close()

    dense_played_arr = np.stack(dense_played, axis=0) if dense_played else np.zeros((0, 88), dtype=np.float32)
    dense_hand_arr = np.stack(dense_hand, axis=0) if dense_hand else np.zeros((0, hand.shape[-1]), dtype=np.float32)
    dense_target_arr = (
        np.stack(dense_target_hand, axis=0)
        if dense_target_hand
        else np.zeros((0, hand.shape[-1]), dtype=np.float32)
    )
    dense_goal_arr = np.stack(dense_goal, axis=0) if dense_goal else np.zeros((0, 88), dtype=np.float32)
    actions_arr = np.stack(actions, axis=0) if actions else np.zeros((0, 0), dtype=np.float32)
    search_arr = np.asarray(search_used, dtype=bool)
    npz_path = output / "ballade_residual_rollout.npz"
    np.savez_compressed(
        npz_path,
        dense_played_piano=dense_played_arr,
        dense_target_goals=dense_goal_arr,
        dense_hand_qpos=dense_hand_arr,
        dense_target_hand_qpos=dense_target_arr,
        actions=actions_arr,
        search_used=search_arr,
    )
    summary = {
        "song_key": str(trajectory.song_key),
        "demo_id": int(trajectory.demo_id),
        "checkpoint": str(checkpoint),
        "source_dt": float(source_dt),
        "control_dt": float(control_dt),
        "dense_steps_played": int(dense_played_arr.shape[0]),
        "terminated": bool(terminated),
        "rollout_npz": str(npz_path),
        **summarize_rollout(
            dense_goal_mask=dense_goal_arr,
            dense_played=dense_played_arr,
            dense_hand=dense_hand_arr,
            dense_target_hand=dense_target_arr,
            actions=actions_arr,
            search_used=search_arr,
            terminated=terminated,
        ),
    }
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary
