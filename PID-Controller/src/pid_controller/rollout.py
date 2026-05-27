from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pid_controller.controller import ControllerKind, HandPIDController, make_controller_config
from pid_controller.mapping import (
    action_signals_from_hand_state,
    build_reduced_action_mapping,
    joint_index_groups,
)


DEFAULT_IMPROMPTU_RUN_ROOTS: tuple[Path, ...] = (
    Path("/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch"),
    Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs"),
)


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metadata_for_trajectory(path: Path) -> dict[str, Any]:
    run_dir = path.parent
    metadata = _load_json(run_dir / "metadata.json")
    for candidate in (
        run_dir / "variant_result.json",
        run_dir / "impromptu_rp1m_retest_result.json",
        run_dir / "rp1m_retest" / "impromptu_rp1m_retest_result.json",
    ):
        if candidate.exists():
            metadata.update({f"result_{k}": v for k, v in _load_json(candidate).items()})
    return metadata


def rollout_rank_score(path: Path) -> float:
    metadata = _metadata_for_trajectory(path)
    score_keys = (
        "result_event_f1",
        "result_frame_f1",
        "result_rp1m_key_f1",
        "event_f1",
        "frame_f1",
        "rp1m_key_f1",
        "key_f1",
    )
    for key in score_keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def discover_trajectory_npzs(roots: list[str | Path] | tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.name == "trajectory.npz":
            paths.append(root)
            continue
        if root.is_dir():
            paths.extend(root.rglob("trajectory.npz"))
    unique = sorted(set(paths), key=lambda path: (rollout_rank_score(path), str(path)), reverse=True)
    return unique


def load_impromptu_targets(
    trajectory_npz: str | Path,
    *,
    max_source_steps: int | None = None,
    prefer_dense: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    path = Path(trajectory_npz)
    with np.load(path, allow_pickle=False) as data:
        if prefer_dense and "planned_hand_joints_dense" in data:
            hand = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
            source_dt = 0.005
            source_name = "planned_hand_joints_dense"
            if "target_keys" in data:
                control_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
                substeps = max(int(round(0.05 / source_dt)), 1)
                goals = np.repeat(control_keys, substeps, axis=0)[: hand.shape[0]]
            else:
                goals = np.zeros((hand.shape[0], 88), dtype=np.float32)
        else:
            if "planned_hand_joints" in data:
                hand = np.asarray(data["planned_hand_joints"], dtype=np.float32)
                source_name = "planned_hand_joints"
            elif "planned_hand_joints_dense" in data:
                dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
                hand = dense[::10]
                source_name = "planned_hand_joints_dense_downsampled_by_10"
            else:
                raise ValueError(f"{path} does not contain planned_hand_joints or planned_hand_joints_dense")
            source_dt = 0.05
            if "target_keys" in data:
                goals = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
            else:
                goals = np.zeros((hand.shape[0], 88), dtype=np.float32)
    steps = min(int(hand.shape[0]), int(goals.shape[0]))
    if max_source_steps is not None:
        steps = min(steps, max(int(max_source_steps), 0))
    hand = hand[:steps].astype(np.float32)
    goals = goals[:steps, :88].astype(np.float32)
    if hand.ndim != 2 or hand.shape[1] != 46:
        raise ValueError(f"Expected hand targets [T, 46], got {hand.shape} from {path}")
    return hand, goals, source_dt, {"hand_source": source_name, "trajectory_npz": str(path)}


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _l2_stats(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "scored_steps": 0,
        }
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "max": float(data.max()),
        "scored_steps": int(data.shape[0]),
    }


def _l2_series_stats(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    stats = _l2_stats(data)
    if data.size == 0:
        stats.update(
            {
                "final": 0.0,
                "first_third_mean": 0.0,
                "last_third_mean": 0.0,
                "last_minus_first": 0.0,
                "slope_per_step": 0.0,
            }
        )
        return stats
    split = max(int(data.size) // 3, 1)
    first = data[:split]
    last = data[-split:]
    slope = float(np.polyfit(np.arange(data.size, dtype=np.float64), data.astype(np.float64), deg=1)[0]) if data.size >= 2 else 0.0
    stats.update(
        {
            "final": float(data[-1]),
            "first_third_mean": float(first.mean()),
            "last_third_mean": float(last.mean()),
            "last_minus_first": float(last.mean() - first.mean()),
            "slope_per_step": slope,
        }
    )
    return stats


def _tracking_split_metrics(
    *,
    rollout_npz: str | Path | None,
    target_hand: np.ndarray,
) -> dict[str, Any]:
    if rollout_npz is None:
        return {}
    path = Path(str(rollout_npz))
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as rollout:
            actual = np.asarray(rollout["source_hand_after_step"], dtype=np.float32)
    except Exception:
        return {}
    reference = np.asarray(target_hand, dtype=np.float32)[1 : actual.shape[0] + 1]
    n = min(int(actual.shape[0]), int(reference.shape[0]))
    if n <= 0:
        return {"alignment": "after_action_t_vs_target_t_plus_1", "scored_steps": 0}
    actual = actual[:n]
    reference = reference[:n]
    mapping = build_reduced_action_mapping()
    groups = joint_index_groups(mapping)
    total_l2 = np.linalg.norm(actual - reference, axis=1)

    def qpos_group(name: str) -> dict[str, Any]:
        indices = list(groups[name])
        if not indices:
            return _l2_stats(np.asarray([], dtype=np.float32))
        return _l2_series_stats(np.linalg.norm(actual[:, indices] - reference[:, indices], axis=1))

    actual_signals = np.stack(
        [action_signals_from_hand_state(row, mapping) for row in actual],
        axis=0,
    )
    target_signals = np.stack(
        [action_signals_from_hand_state(row, mapping) for row in reference],
        axis=0,
    )
    actuator_indices = [entry.action_index for entry in mapping if entry.kind != "sustain"]
    actuator_l2 = np.linalg.norm(
        actual_signals[:, actuator_indices] - target_signals[:, actuator_indices],
        axis=1,
    )
    return {
        "alignment": "after_action_t_vs_target_t_plus_1",
        "scored_steps": int(n),
        "projection": "weighted_least_squares",
        "joint_counts": {key: int(len(value)) for key, value in groups.items()},
        "total_qpos_l2": _l2_series_stats(total_l2),
        "one_to_one_qpos_l2": qpos_group("one_to_one"),
        "coupled_qpos_l2": qpos_group("coupled"),
        "actuator_signal_l2": _l2_series_stats(actuator_l2),
    }


def run_impromptu_pid_rollout(
    *,
    trajectory_npz: str | Path,
    output_dir: str | Path,
    environment_name: str,
    controller_kind: ControllerKind = "pd",
    kp: float | None = None,
    kd: float | None = None,
    ki: float | None = None,
    integral_limit: float = 0.25,
    setpoint_policy: str = "next",
    use_target_velocity: bool = False,
    target_velocity_scale: float = 0.0,
    feedforward_scale: float = 0.0,
    lookahead_substeps: int = 10,
    sustain_value: float = 0.0,
    threshold: float = 0.5,
    seed: int = 0,
    max_source_steps: int | None = 20,
    render_mp4: bool = False,
    hand_anchor_y_offset: float | None = None,
    disable_hand_collisions: bool = False,
) -> dict[str, Any]:
    from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout

    hand, goals, source_dt, load_info = load_impromptu_targets(
        trajectory_npz,
        max_source_steps=max_source_steps,
        prefer_dense=False,
    )
    if not np.isclose(source_dt, 0.05):
        raise ValueError(f"PID rollout expects 20 Hz source hand targets, got dt={source_dt}")
    actions = np.zeros((hand.shape[0], 39), dtype=np.float32)
    trajectory = make_rp1m_trajectory_from_arrays(
        song_key=str(environment_name),
        demo_id=0,
        actions=actions,
        goals=goals,
        hand_joints=hand,
        environment_name=str(environment_name),
    )
    controller = HandPIDController(
        make_controller_config(
            controller_kind,
            kp=kp,
            kd=kd,
            ki=ki,
            integral_limit=integral_limit,
            setpoint_policy=setpoint_policy,  # type: ignore[arg-type]
            use_target_velocity=bool(use_target_velocity),
            target_velocity_scale=float(target_velocity_scale),
            feedforward_scale=float(feedforward_scale),
            lookahead_substeps=int(lookahead_substeps),
            sustain_value=float(sustain_value),
        )
    )
    config = RolloutConfig(
        mode="action",
        dataset_timestep=0.05,
        simulation_timestep=0.005,
        hand_anchor_y_offset=hand_anchor_y_offset,
        auto_hand_anchor_y_offset=False,
        action_source_scale="actuator_units",
        action_substep_policy="repeat",
        wrist_action_policy="recorded",
        restore_initial_hand=False,
        set_hand_qvel=False,
        seed=int(seed),
        threshold=float(threshold),
        max_source_steps=None,
        render_mp4=bool(render_mp4),
        render_audio=False,
        fps=200,
        disable_hand_collisions=bool(disable_hand_collisions),
    )
    output = Path(output_dir)
    summary = simulate_rp1m_rollout(trajectory, config, output, action_controller=controller)
    score: dict[str, Any] = {}
    try:
        from intermezzo.online_eval import score_rollout

        with np.load(summary["rollout_npz"], allow_pickle=False) as rollout:
            played = np.asarray(rollout["source_played_piano"], dtype=np.float32)
            target = np.asarray(rollout["goals"], dtype=np.float32)
        score = score_rollout(
            target_keys=target,
            played_keys=played,
            dt=0.05,
            threshold=float(threshold),
            timing_tolerance_s=0.15,
        )
    except Exception as exc:
        score = {"error": f"{type(exc).__name__}: {exc}"}
    result = {
        "trajectory_npz": str(trajectory_npz),
        "output_dir": str(output),
        "environment_name": str(environment_name),
        "controller_kind": str(controller_kind),
        "controller_metadata": controller.metadata(),
        "load_info": load_info,
        "max_source_steps": max_source_steps,
        "rp1m_summary_json": str(output / "summary.json"),
        "rollout_npz": str(summary.get("rollout_npz")),
        "source_steps_played": int(summary.get("source_steps_played", 0)),
        "actions_executed": int(summary.get("actions_executed", 0)),
        "hand_qpos_l2_vs_reference": summary.get("hand_qpos_l2_vs_reference"),
        "hand_tracking_split": _tracking_split_metrics(
            rollout_npz=summary.get("rollout_npz"),
            target_hand=hand,
        ),
        "rp1m_key_f1": float((summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "rp1m_key_precision": float((summary.get("against_goals") or {}).get("key_precision", 0.0)),
        "rp1m_key_recall": float((summary.get("against_goals") or {}).get("key_recall", 0.0)),
        "frame_f1": float(score.get("frame_f1", 0.0)) if "error" not in score else 0.0,
        "event_f1": float(event_f1(score)) if "error" not in score else 0.0,
        "intermezzo_score": score,
        "piano_state_policy": summary.get("piano_state_policy"),
        "hand_resync_policy": summary.get("hand_resync_policy"),
        "terminated": bool(summary.get("terminated", False)),
    }
    save_json(output / "pid_rollout_result.json", result)
    return result
