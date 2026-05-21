#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import strftime

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT,
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.kinematics import IKResult  # noqa: E402
from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.evaluation import evaluate_trajectory_payload  # noqa: E402
from impromptu.planner import plan_target_keys  # noqa: E402


OUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/test_runs")
METRIC_KEYS = (
    "assignment_rate",
    "waypoint_fingertip_error_p95",
    "exact_waypoint_sparse_error_p95",
    "exact_waypoint_anchor_error_p95",
    "exact_waypoint_anchor_success_rate_020m",
    "ik_anchor_error_weight_ge_1_p95",
    "ik_anchor_success_rate_020m_weight_ge_1",
    "ik_anchor_fingertip_distance_p95",
    "waypoint_dense_activity_rate",
    "waypoint_has_exact_anchor_rate",
    "joint_velocity_p95",
    "joint_acceleration_p95",
    "joint_jerk_p95",
)


def _row(*keys: int) -> np.ndarray:
    out = np.zeros((88,), dtype=np.float32)
    for key in keys:
        out[int(key)] = 1.0
    return out


def _target_sequences() -> dict[str, np.ndarray]:
    sequences: dict[str, np.ndarray] = {}
    single = np.zeros((20, 88), dtype=np.float32)
    single[5:9, 40] = 1.0
    sequences["single_note"] = single

    repeated = np.zeros((20, 88), dtype=np.float32)
    repeated[3:5, 44] = 1.0
    repeated[10:12, 44] = 1.0
    sequences["repeated_note"] = repeated

    chord = np.zeros((20, 88), dtype=np.float32)
    chord[6:11, [36, 40]] = 1.0
    sequences["two_note_chord"] = chord

    alternating = np.zeros((20, 88), dtype=np.float32)
    alternating[2:4, 24] = 1.0
    alternating[7:9, 62] = 1.0
    alternating[12:14, 28] = 1.0
    alternating[16:18, 67] = 1.0
    sequences["alternating_regions"] = alternating

    sparse_chord = np.zeros((20, 88), dtype=np.float32)
    sparse_chord[8:13, [30, 37, 44, 52]] = 1.0
    sequences["sparse_chord"] = sparse_chord
    return sequences


class FakeKinematics:
    def __init__(self) -> None:
        self.neutral_qpos = np.zeros((46,), dtype=np.float32)
        self.joint_lower = np.full((46,), -1.0, dtype=np.float32)
        self.joint_upper = np.full((46,), 1.0, dtype=np.float32)
        self.environment_name = "fake-env"
        self.midi_proto_path = "fake.proto"

    def close(self) -> None:
        return None

    def clip_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(qpos, dtype=np.float32), self.joint_lower, self.joint_upper)

    def fingertip_positions_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float32)
        tips = np.zeros((10, 3), dtype=np.float32)
        tips[:, 0] = np.arange(10, dtype=np.float32)
        tips[:, 1] = float(values[0])
        tips[:, 2] = float(values[1])
        return tips

    def key_contact_targets(self, keys: np.ndarray) -> np.ndarray:
        return np.asarray([[float(key % 10), float(key) * 0.001, 0.0] for key in keys], dtype=np.float32)

    def key_press_targets(self, keys: np.ndarray, press_depth: float | None = None) -> np.ndarray:
        out = self.key_contact_targets(keys)
        out[:, 2] -= 0.008 if press_depth is None else float(press_depth)
        return out

    def solve_press_pose(self, assignments, previous_qpos, neutral_qpos=None, config=None) -> IKResult:
        pose = np.asarray(previous_qpos, dtype=np.float32).copy()
        pose[0] += 0.04
        pose[1] += 0.01
        fingertips = self.fingertip_positions_for_qpos(pose)
        if assignments.count:
            fingertips[assignments.assigned_finger_indices] = assignments.target_positions
        distances = (
            np.linalg.norm(fingertips[assignments.assigned_finger_indices] - assignments.target_positions, axis=1)
            if assignments.count
            else np.zeros((0,), dtype=np.float32)
        )
        return IKResult(
            pose=pose,
            fingertip_positions=fingertips,
            assigned_distances=distances.astype(np.float32),
            residual_norm=float(np.linalg.norm(distances)),
            max_residual=float(np.max(distances)) if distances.size else 0.0,
            success=True,
            optimizer_success=True,
            optimizer_status=1,
            optimizer_message="fake",
            optimizer_cost=0.0,
            nfev=1,
            active_keys=assignments.active_keys,
            assigned_keys=assignments.assigned_keys,
            assigned_finger_indices=assignments.assigned_finger_indices,
            unassigned_keys=assignments.unassigned_keys,
        )


def _run_one(name: str, target_keys: np.ndarray, *, use_fake: bool) -> dict[str, object]:
    config = ImpromptuConfig(interpolation_substeps=4, ik_max_nfev=30, anchor_stride=2)
    kinematics = FakeKinematics() if use_fake else None
    plan = plan_target_keys(target_keys, config=config, kinematics=kinematics)
    payload = plan.npz_payload()
    payload["control_timestep"] = np.asarray([config.control_timestep], dtype=np.float32)
    metrics = evaluate_trajectory_payload(payload)
    return {
        "name": name,
        "num_target_frames": int(target_keys.shape[0]),
        "num_waypoints": int(metrics["num_waypoints"]),
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
    }


def main() -> None:
    sequences = _target_sequences()
    mode = "project_kinematics"
    fallback_reason = None
    try:
        results = [_run_one(name, target, use_fake=False) for name, target in sequences.items()]
    except Exception as exc:
        mode = "fake_kinematics"
        fallback_reason = f"{type(exc).__name__}: {exc}"
        print(f"Falling back to fake kinematics: {fallback_reason}")
        results = [_run_one(name, target, use_fake=True) for name, target in sequences.items()]

    aggregate: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [float(item["metrics"][key]) for item in results if item["metrics"].get(key) is not None]
        aggregate[f"{key}_mean"] = float(np.mean(values)) if values else 0.0

    for key in METRIC_KEYS:
        print(f"{key}={aggregate[f'{key}_mean']:.6f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"metrics_probe_{strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "fallback_reason": fallback_reason,
                "aggregate": aggregate,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote metrics probe summary: {out_path}")


if __name__ == "__main__":
    main()
