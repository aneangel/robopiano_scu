from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO = ROOT.parents[0]
for path in (SRC, REPO / "Bagatelle" / "src", REPO / "Intermezzo" / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.kinematics import IKResult  # noqa: E402
from bagatelle.planner import plan_target_keys as plan_bagatelle_target_keys  # noqa: E402
from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.planner import _bagatelle_config  # noqa: E402
from impromptu.planner import plan_target_keys  # noqa: E402
from impromptu.trajectory import REQUIRED_TRAJECTORY_FIELDS, ImpromptuTrajectory  # noqa: E402


def _row(*keys: int) -> np.ndarray:
    out = np.zeros((88,), dtype=np.float32)
    for key in keys:
        out[int(key)] = 1.0
    return out


class FakeKinematics:
    def __init__(self) -> None:
        self.neutral_qpos = np.zeros((46,), dtype=np.float32)
        self.joint_lower = np.full((46,), -1.0, dtype=np.float32)
        self.joint_upper = np.full((46,), 1.0, dtype=np.float32)
        self.environment_name = "fake-env"
        self.midi_proto_path = "fake.proto"

    def close(self) -> None:
        raise AssertionError("caller-owned kinematics should not be closed")

    def clip_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(qpos, dtype=np.float32), self.joint_lower, self.joint_upper)

    def fingertip_positions_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        tips = np.zeros((10, 3), dtype=np.float32)
        tips[:, 0] = np.arange(10, dtype=np.float32)
        tips[:, 1] = float(np.asarray(qpos, dtype=np.float32)[0])
        return tips

    def key_contact_targets(self, keys: np.ndarray) -> np.ndarray:
        return np.asarray([[float(key % 10), float(key) * 0.001, 0.0] for key in keys], dtype=np.float32)

    def key_press_targets(self, keys: np.ndarray, press_depth: float | None = None) -> np.ndarray:
        out = self.key_contact_targets(keys)
        out[:, 2] -= 0.008 if press_depth is None else float(press_depth)
        return out

    def solve_press_pose(self, assignments, previous_qpos, neutral_qpos=None, config=None) -> IKResult:
        pose = np.asarray(previous_qpos, dtype=np.float32).copy()
        pose[0] += 0.05
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


def test_plan_target_keys_schema_with_fake_kinematics() -> None:
    target = np.stack([_row(), _row(10), _row(10), _row(12), _row()], axis=0)
    config = ImpromptuConfig(interpolation_substeps=3, ik_max_nfev=2)

    plan = plan_target_keys(target, config=config, kinematics=FakeKinematics())

    assert isinstance(plan, ImpromptuTrajectory)
    payload = plan.npz_payload()
    assert set(REQUIRED_TRAJECTORY_FIELDS).issubset(payload)
    assert plan.planned_hand_joints.shape == (target.shape[0], 46)
    assert plan.planned_hand_joints_dense.shape == (target.shape[0] * config.interpolation_substeps, 46)
    assert plan.ik_anchor_qpos.shape == (plan.ik_anchor_frames_dense.shape[0], 46)


def test_plan_starts_from_bagatelle_control_qpos_when_first_press_is_later_than_frame_zero() -> None:
    target = np.stack([_row(), _row(), _row(10), _row(10), _row()], axis=0)
    config = ImpromptuConfig(interpolation_substeps=3, ik_max_nfev=2)

    impromptu = plan_target_keys(target, config=config, kinematics=FakeKinematics())
    bagatelle = plan_bagatelle_target_keys(target, config=_bagatelle_config(config), kinematics=FakeKinematics())

    np.testing.assert_allclose(impromptu.planned_hand_joints_dense[0], bagatelle.planned_hand_joints[0], atol=1e-6)


def test_impromptu_delegates_assignment_and_control_qpos_to_bagatelle() -> None:
    target = np.stack([_row(), _row(10), _row(10), _row(12, 16), _row()], axis=0)
    config = ImpromptuConfig(interpolation_substeps=3, ik_max_nfev=2)

    impromptu = plan_target_keys(target, config=config, kinematics=FakeKinematics())
    bagatelle = plan_bagatelle_target_keys(target, config=_bagatelle_config(config), kinematics=FakeKinematics())

    np.testing.assert_array_equal(impromptu.assignments, bagatelle.assignments)
    np.testing.assert_allclose(impromptu.fingertip_targets, bagatelle.fingertip_targets)
    np.testing.assert_allclose(impromptu.waypoint_fingertips, bagatelle.waypoint_fingertips)
    np.testing.assert_allclose(impromptu.waypoint_hand_joints, bagatelle.waypoint_hand_joints)
    np.testing.assert_allclose(impromptu.planned_hand_joints, bagatelle.planned_hand_joints)
    np.testing.assert_allclose(impromptu.planned_hand_velocities, bagatelle.planned_hand_velocities)
    assert impromptu.metadata["ik_source"] == "Bagatelle.plan_target_keys"
    assert impromptu.metadata["assignment_source"] == "Bagatelle.plan_target_keys"
