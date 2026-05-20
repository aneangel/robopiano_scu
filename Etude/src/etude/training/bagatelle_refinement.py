from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from etude.controllers.bagatelle_residual import BagatelleResidualController
from etude.controllers.residual_mlp import ResidualMLP
from etude.evaluation.event_matching import extract_key_events, match_events
from etude.evaluation.event_metrics import EventMetricsConfig, compute_event_metrics
from etude.evaluation.fingertip_metrics import compute_fingertip_assignment_metrics
from etude.evaluation.metrics import action_metrics, joint_metrics, note_metrics
from etude.evaluation.rollout import rollout_controller
from etude.evaluation.tracker_eval_utils import (
    build_fingertip_feature_spec,
    build_phase_feature_spec,
    build_tracker_controller_from_checkpoint_payload,
    resolve_device,
)
from etude.features.fingertip_phase_blocks import build_fingertip_phase_features
from etude.training.rollout_objectives import RolloutObjectiveComposer


@dataclass(slots=True)
class RefinementState:
    base_checkpoint: dict[str, Any]
    correction_state_dict: dict[str, Any] | None
    stage: str | None
    source_path: str


def load_refinement_state(path: str | Path) -> RefinementState:
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must deserialize to a dict, got {type(payload)!r}")
    if payload.get("kind") == "bagatelle_refinement":
        base_checkpoint = payload.get("base_checkpoint")
        if not isinstance(base_checkpoint, dict):
            raise ValueError("Refinement checkpoint is missing base_checkpoint")
        correction_state = payload.get("correction_model")
        if correction_state is not None and not isinstance(correction_state, dict):
            raise ValueError("Refinement checkpoint correction_model must be a state_dict")
        return RefinementState(
            base_checkpoint=base_checkpoint,
            correction_state_dict=correction_state,
            stage=str(payload.get("stage")) if payload.get("stage") is not None else None,
            source_path=str(checkpoint_path),
        )
    return RefinementState(
        base_checkpoint=payload,
        correction_state_dict=None,
        stage=None,
        source_path=str(checkpoint_path),
    )


def build_refinement_controller(
    mapping: Any,
    config: dict[str, Any],
    state: RefinementState,
    correction_model: torch.nn.Module,
) -> BagatelleResidualController:
    base_controller = build_tracker_controller_from_checkpoint_payload(mapping, state.base_checkpoint)
    controller_cfg = _as_dict(config.get("controller"))
    device = resolve_device(controller_cfg, config)
    return BagatelleResidualController(
        mapping,
        base_controller,
        correction_model,
        fingertip_spec=build_fingertip_feature_spec(config),
        phase_spec=build_phase_feature_spec(config),
        device=device,
        residual_limit=controller_cfg.get("residual_limit"),
        residual_scale=float(controller_cfg.get("residual_scale", 1.0)),
    )


def determine_feature_dim(config: dict[str, Any], rollout: dict[str, np.ndarray], metadata: dict[str, Any]) -> int:
    if rollout["q"].shape[0] == 0:
        raise ValueError("Rollout produced no steps, cannot determine feature dimension")
    return int(
        build_fingertip_phase_features(
            t=0,
            metadata=metadata,
            plan_bundle=metadata.get("plan_bundle"),
            target_keys=metadata.get("target_keys"),
            current_fingertips=np.asarray(rollout.get("fingertips", [None])[0], dtype=np.float32)
            if "fingertips" in rollout
            else None,
            desired_fingertips=_metadata_or_rollout_fingertips(rollout, {"metadata": metadata}, 0),
            fingertip_weights=_metadata_or_rollout_mask(rollout, {"metadata": metadata}, 0, key="fingertip_weights"),
            active_finger_mask=_metadata_or_rollout_mask(rollout, {"metadata": metadata}, 0, key="active_finger_mask"),
            inactive_finger_mask=_metadata_or_rollout_mask(rollout, {"metadata": metadata}, 0, key="inactive_finger_mask"),
            fingertip_spec=build_fingertip_feature_spec(config),
            phase_spec=build_phase_feature_spec(config),
        ).shape[0]
    )


def create_or_validate_correction_model(
    *,
    config: dict[str, Any],
    state: RefinementState,
    feature_dim: int,
    action_dim: int,
) -> torch.nn.Module:
    controller_cfg = _as_dict(config.get("controller"))
    model = ResidualMLP(
        input_dim=feature_dim,
        action_dim=action_dim,
        hidden_dim=int(controller_cfg.get("hidden_dim", 256)),
        num_layers=int(controller_cfg.get("num_layers", 4)),
        dropout=float(controller_cfg.get("dropout", 0.05)),
        activation=str(controller_cfg.get("activation", "gelu")),
    )
    _zero_last_layer(model)
    if state.correction_state_dict is not None:
        saved_input_dim = _read_checkpoint_dim(state, key="input_dim")
        saved_action_dim = _read_checkpoint_dim(state, key="action_dim")
        if saved_input_dim != feature_dim or saved_action_dim != action_dim:
            raise ValueError(
                "Refinement correction-head feature mismatch: "
                f"saved input_dim/action_dim = {saved_input_dim}/{saved_action_dim}, "
                f"current = {feature_dim}/{action_dim}"
            )
        model.load_state_dict(state.correction_state_dict)
    return model


def train_refinement_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    residual_l2_weight: float,
) -> float:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)),
    )
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=True)
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        predicted = model(batch_features)
        loss = torch.nn.functional.mse_loss(predicted, batch_targets)
        if residual_l2_weight > 0.0:
            loss = loss + float(residual_l2_weight) * predicted.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item()) * int(batch_features.shape[0])
        total_count += int(batch_features.shape[0])
    model.eval()
    return total_loss / max(total_count, 1)


def build_stage1_training_batch(
    controller: BagatelleResidualController,
    rollout: dict[str, np.ndarray],
    trajectory: dict[str, Any],
    action_dim: int,
    *,
    gain: float = 0.25,
    error_scale_cm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(rollout["q"].shape[0])
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    q_ref = np.asarray(trajectory["q_ref"], dtype=np.float32)
    for t in range(steps):
        obs = _rollout_obs_at_timestep(rollout, trajectory, t)
        features.append(controller.build_features(obs, t))
        desired = _metadata_or_rollout_fingertips(rollout, trajectory, t)
        current = obs.get("fingertips")
        active_mask = _metadata_or_rollout_mask(rollout, trajectory, t, key="active_finger_mask")
        fingertip_gate = _active_fingertip_error_gate(current, desired, active_mask, error_scale_cm=error_scale_cm)
        joint_error = q_ref[min(t, q_ref.shape[0] - 1), :action_dim] - np.asarray(rollout["q"][t], dtype=np.float32)[:action_dim]
        targets.append((float(gain) * fingertip_gate * np.tanh(joint_error)).astype(np.float32))
    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.float32)


def build_stage2_training_batch(
    controller: BagatelleResidualController,
    rollout: dict[str, np.ndarray],
    trajectory: dict[str, Any],
    action_dim: int,
    *,
    gain: float = 0.35,
    error_scale_cm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(rollout["q"].shape[0])
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    q_ref = np.asarray(trajectory["q_ref"], dtype=np.float32)
    target_keys = np.asarray(trajectory.get("metadata", {}).get("target_keys"), dtype=np.float32)
    predicted_keys = np.asarray(rollout.get("key_state"), dtype=np.float32) if "key_state" in rollout else None
    for t in range(steps):
        obs = _rollout_obs_at_timestep(rollout, trajectory, t)
        features.append(controller.build_features(obs, t))
        joint_error = q_ref[min(t, q_ref.shape[0] - 1), :action_dim] - np.asarray(rollout["q"][t], dtype=np.float32)[:action_dim]
        fingertip_gate = _active_fingertip_error_gate(
            obs.get("fingertips"),
            _metadata_or_rollout_fingertips(rollout, trajectory, t),
            _metadata_or_rollout_mask(rollout, trajectory, t, key="active_finger_mask"),
            error_scale_cm=error_scale_cm,
        )
        key_gate = 0.0
        if predicted_keys is not None and target_keys.size:
            pred = predicted_keys[min(t, predicted_keys.shape[0] - 1)]
            tgt = target_keys[min(t, target_keys.shape[0] - 1)]
            missed = np.clip(tgt - pred, 0.0, None).mean()
            wrong = np.clip(pred - tgt, 0.0, None).mean()
            key_gate = float(missed + 0.5 * wrong)
        total_gate = np.clip(key_gate + 0.25 * fingertip_gate, 0.0, 2.0)
        targets.append((float(gain) * total_gate * np.tanh(joint_error)).astype(np.float32))
    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.float32)


def evaluate_refinement_candidate(
    env: Any,
    controller: BagatelleResidualController,
    mapping: Any,
    trajectory: dict[str, Any],
    config: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rollout = rollout_controller(
        env,
        controller,
        mapping,
        np.asarray(trajectory["q_ref"], dtype=np.float32),
        np.asarray(trajectory["qdot_ref"], dtype=np.float32),
        metadata=trajectory.get("metadata"),
    )
    metrics = compute_refinement_metrics(rollout, trajectory, config=config, stage=stage)
    diagnostics_fn = getattr(controller, "diagnostics", None)
    if callable(diagnostics_fn):
        metrics.update({key: float(value) for key, value in diagnostics_fn().items()})
    return metrics, rollout


def compute_refinement_metrics(
    rollout: dict[str, np.ndarray],
    trajectory: dict[str, Any],
    *,
    config: dict[str, Any],
    stage: str,
) -> dict[str, float]:
    q = np.asarray(rollout["q"], dtype=np.float32)
    qdot = np.asarray(rollout["qdot"], dtype=np.float32)
    steps = q.shape[0]
    metrics = joint_metrics(
        q,
        np.asarray(trajectory["q_ref"], dtype=np.float32)[:steps],
        qdot,
        np.asarray(trajectory["qdot_ref"], dtype=np.float32)[:steps],
    )
    metrics.update(action_metrics(np.asarray(rollout["actions"], dtype=np.float32), None, None))

    current = rollout.get("fingertips")
    desired = rollout.get("desired_fingertips", trajectory.get("metadata", {}).get("desired_fingertips"))
    if current is not None and desired is not None:
        metrics.update(
            compute_fingertip_assignment_metrics(
                np.asarray(current, dtype=np.float32),
                np.asarray(desired, dtype=np.float32),
                active_finger_mask=rollout.get(
                    "active_finger_mask",
                    trajectory.get("metadata", {}).get("active_finger_mask"),
                ),
            )
        )

    predicted = rollout.get("key_state")
    target = trajectory.get("metadata", {}).get("target_keys")
    if predicted is not None and target is not None:
        predicted_keys = np.asarray(predicted, dtype=np.float32)
        target_keys = np.asarray(target, dtype=np.float32)[:predicted_keys.shape[0]]
        metrics.update(note_metrics(predicted_keys, target_keys))
        event_cfg = EventMetricsConfig(dt=float(trajectory.get("dt", 0.005)))
        metrics.update(compute_event_metrics(predicted_keys, target_keys, event_cfg))
        if stage == "keypress_temporal":
            metrics.update(compute_rollout_objective_metrics(rollout, target_keys, trajectory, config=config))
    return {key: float(value) for key, value in metrics.items()}


def compute_rollout_objective_metrics(
    rollout: dict[str, np.ndarray],
    target_keys: np.ndarray,
    trajectory: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, float]:
    composer = RolloutObjectiveComposer.from_config(config)
    if not composer.weights:
        return {}
    if "key_state" not in rollout:
        return {}
    predicted_keys = np.asarray(rollout.get("key_state"), dtype=np.float32)
    actions = np.asarray(rollout["actions"], dtype=np.float32)
    desired = rollout.get("desired_fingertips", trajectory.get("metadata", {}).get("desired_fingertips"))
    current = rollout.get("fingertips")
    steps = min(predicted_keys.shape[0], target_keys.shape[0], actions.shape[0])
    if steps == 0:
        return {}
    timing_error, release_error = _framewise_event_errors(predicted_keys[:steps], target_keys[:steps], dt=float(trajectory.get("dt", 0.005)))
    breakdown_total: dict[str, float] = {}
    total_score = 0.0
    previous_action = np.zeros_like(actions[0], dtype=np.float32)
    for t in range(steps):
        state = {
            "predicted_keys": predicted_keys[t],
            "target_keys": target_keys[t],
            "action": actions[t],
            "previous_action": previous_action,
            "residual_action": actions[t],
            "timing_error": np.asarray([timing_error[t]], dtype=np.float32),
            "release_error": np.asarray([release_error[t]], dtype=np.float32),
        }
        if current is not None and desired is not None:
            state["fingertip_positions"] = np.asarray(current[t], dtype=np.float32)
            state["target_fingertip_positions"] = np.asarray(desired[t], dtype=np.float32)
        score, breakdown = composer.compute(state, return_breakdown=True)
        total_score += float(score)
        for key, value in breakdown.items():
            breakdown_total[key] = breakdown_total.get(key, 0.0) + float(value)
        previous_action = actions[t]
    metrics = {"rollout_objective/total": float(total_score)}
    for key, value in breakdown_total.items():
        metrics[f"rollout_objective/{key}"] = float(value) / float(steps)
    return metrics


def build_refinement_checkpoint(
    *,
    stage: str,
    config: dict[str, Any],
    state: RefinementState,
    model: torch.nn.Module,
    input_dim: int,
    action_dim: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "kind": "bagatelle_refinement",
        "stage": stage,
        "config": config,
        "base_checkpoint": state.base_checkpoint,
        "base_checkpoint_source": state.source_path,
        "correction_model_module": "etude.controllers.residual_mlp:ResidualMLP",
        "correction_model": model.state_dict(),
        "input_dim": int(input_dim),
        "action_dim": int(action_dim),
        "metrics": metrics,
    }


def _rollout_obs_at_timestep(rollout: dict[str, np.ndarray], trajectory: dict[str, Any], t: int) -> dict[str, np.ndarray]:
    obs = {
        "q": np.asarray(rollout["q"][t], dtype=np.float32),
        "qdot": np.asarray(rollout["qdot"][t], dtype=np.float32),
    }
    if "fingertips" in rollout:
        obs["fingertips"] = np.asarray(rollout["fingertips"][t], dtype=np.float32)
    if "key_state" in rollout:
        obs["key_state"] = np.asarray(rollout["key_state"][t], dtype=np.float32)
    target_keys = trajectory.get("metadata", {}).get("target_keys")
    if target_keys is not None:
        obs["target_keys"] = np.asarray(target_keys, dtype=np.float32)
    return obs


def _metadata_or_rollout_fingertips(rollout: dict[str, np.ndarray], trajectory: dict[str, Any], t: int) -> np.ndarray | None:
    if "desired_fingertips" in rollout:
        return np.asarray(rollout["desired_fingertips"][t], dtype=np.float32)
    desired = trajectory.get("metadata", {}).get("desired_fingertips")
    if desired is None:
        return None
    desired_array = np.asarray(desired, dtype=np.float32)
    if desired_array.ndim == 2:
        return desired_array
    return desired_array[min(t, desired_array.shape[0] - 1)]


def _metadata_or_rollout_mask(
    rollout: dict[str, np.ndarray],
    trajectory: dict[str, Any],
    t: int,
    *,
    key: str,
) -> np.ndarray | None:
    if key in rollout:
        return np.asarray(rollout[key][t], dtype=np.float32)
    value = trajectory.get("metadata", {}).get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array
    return array[min(t, array.shape[0] - 1)]


def _active_fingertip_error_gate(
    current: np.ndarray | None,
    desired: np.ndarray | None,
    active_mask: np.ndarray | None,
    *,
    error_scale_cm: float,
) -> float:
    if current is None or desired is None:
        return 0.0
    current_array = np.asarray(current, dtype=np.float32)
    desired_array = np.asarray(desired, dtype=np.float32)
    if current_array.shape != desired_array.shape:
        return 0.0
    distances = np.linalg.norm(desired_array - current_array, axis=-1)
    finite = np.isfinite(distances)
    if active_mask is None:
        mask = finite
    else:
        mask = np.logical_and(np.asarray(active_mask, dtype=np.float32) > 0.5, finite)
    if not np.any(mask):
        return 0.0
    mean_dist_m = float(np.mean(distances[mask]))
    return float(np.clip((mean_dist_m * 100.0) / max(float(error_scale_cm), 1e-6), 0.0, 2.0))


def _framewise_event_errors(predicted_keys: np.ndarray, target_keys: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray]:
    event_cfg = EventMetricsConfig(dt=dt)
    predicted_events = extract_key_events(
        predicted_keys,
        activation_threshold=event_cfg.key_activation_threshold,
        min_press_duration_frames=1,
        merge_gap_frames=0,
    )
    target_events = extract_key_events(
        target_keys,
        activation_threshold=event_cfg.key_activation_threshold,
        min_press_duration_frames=1,
        merge_gap_frames=0,
    )
    matches, _, _ = match_events(
        predicted_events,
        target_events,
        onset_tolerance_frames=999999,
        offset_tolerance_frames=999999,
    )
    timing_error = np.zeros((predicted_keys.shape[0],), dtype=np.float32)
    release_error = np.zeros((predicted_keys.shape[0],), dtype=np.float32)
    for match in matches:
        frame = min(max(match.target_event.onset_frame, 0), predicted_keys.shape[0] - 1)
        timing_error[frame] = abs(float(match.onset_error_frames) * float(dt))
        release_frame = min(max(match.target_event.offset_frame - 1, 0), predicted_keys.shape[0] - 1)
        release_error[release_frame] = abs(float(match.offset_error_frames) * float(dt))
    return timing_error, release_error


def _read_checkpoint_dim(state: RefinementState, *, key: str) -> int:
    if state.correction_state_dict is None:
        raise ValueError("Refinement state does not contain a correction model")
    base = torch.load(Path(state.source_path), map_location="cpu")
    value = base.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Refinement checkpoint is missing integer {key}")
    return int(value)


def _zero_last_layer(model: torch.nn.Module) -> None:
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            return


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
