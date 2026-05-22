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
from impromptu.joint_space_trajectory import ALL_FINGER_JOINT_INDICES, TRAJECTORY_MODE_DENSE_FINGERTIP_IK  # noqa: E402
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


class CurledFakeKinematics(FakeKinematics):
    def solve_press_pose(self, assignments, previous_qpos, neutral_qpos=None, config=None) -> IKResult:
        result = super().solve_press_pose(assignments, previous_qpos, neutral_qpos=neutral_qpos, config=config)
        pose = result.pose.astype(np.float32, copy=True)
        for finger in result.assigned_finger_indices.astype(np.int64):
            if 0 <= int(finger) < 10:
                pose[ALL_FINGER_JOINT_INDICES] = 0.6
        return IKResult(
            pose=pose,
            fingertip_positions=result.fingertip_positions,
            assigned_distances=result.assigned_distances,
            residual_norm=result.residual_norm,
            max_residual=result.max_residual,
            success=result.success,
            optimizer_success=result.optimizer_success,
            optimizer_status=result.optimizer_status,
            optimizer_message=result.optimizer_message,
            optimizer_cost=result.optimizer_cost,
            nfev=result.nfev,
            active_keys=result.active_keys,
            assigned_keys=result.assigned_keys,
            assigned_finger_indices=result.assigned_finger_indices,
            unassigned_keys=result.unassigned_keys,
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


def test_impromptu_assignment_knobs_forward_to_bagatelle_config() -> None:
    config = ImpromptuConfig(
        assignment_distance_weight=2.5,
        same_finger_bonus=0.3,
        reassignment_penalty=0.4,
        finger_crossing_penalty=0.7,
        wrong_hand_penalty=1.2,
        large_jump_penalty=1.8,
        same_key_same_finger_bonus=0.6,
        key_press_depth=0.011,
        ik_fingertip_weight=3.0,
        ik_smoothness_weight=0.12,
        ik_neutral_weight=0.04,
        ik_max_nfev=17,
        residual_success_threshold=0.013,
    )

    bag = _bagatelle_config(config)

    assert bag.distance_weight == config.assignment_distance_weight
    assert bag.assignment_distance_weight == config.assignment_distance_weight
    assert bag.same_finger_bonus == config.same_finger_bonus
    assert bag.reassignment_penalty == config.reassignment_penalty
    assert bag.finger_crossing_penalty == config.finger_crossing_penalty
    assert bag.assignment_crossing_weight == config.finger_crossing_penalty
    assert bag.wrong_hand_penalty == config.wrong_hand_penalty
    assert bag.assignment_wrong_hand_penalty == config.wrong_hand_penalty
    assert bag.large_jump_penalty == config.large_jump_penalty
    assert bag.same_key_same_finger_bonus == config.same_key_same_finger_bonus
    assert bag.key_press_depth == config.key_press_depth
    assert bag.ik_fingertip_weight == config.ik_fingertip_weight
    assert bag.ik_smoothness_weight == config.ik_smoothness_weight
    assert bag.ik_neutral_weight == config.ik_neutral_weight
    assert bag.ik_max_nfev == config.ik_max_nfev
    assert bag.residual_success_threshold == config.residual_success_threshold


def test_plan_starts_from_bagatelle_control_qpos_when_first_press_is_later_than_frame_zero() -> None:
    target = np.stack([_row(), _row(), _row(10), _row(10), _row()], axis=0)
    config = ImpromptuConfig(interpolation_substeps=3, ik_max_nfev=2)

    impromptu = plan_target_keys(target, config=config, kinematics=FakeKinematics())
    bagatelle = plan_bagatelle_target_keys(target, config=_bagatelle_config(config), kinematics=FakeKinematics())

    np.testing.assert_allclose(impromptu.planned_hand_joints_dense[0], bagatelle.planned_hand_joints[0], atol=1e-6)


def test_impromptu_delegates_assignment_and_press_seeds_to_bagatelle() -> None:
    target = np.stack([_row(), _row(10), _row(10), _row(12, 16), _row()], axis=0)
    config = ImpromptuConfig(interpolation_substeps=3, ik_max_nfev=2)

    impromptu = plan_target_keys(target, config=config, kinematics=FakeKinematics())
    bagatelle = plan_bagatelle_target_keys(target, config=_bagatelle_config(config), kinematics=FakeKinematics())

    np.testing.assert_array_equal(impromptu.assignments, bagatelle.assignments)
    np.testing.assert_allclose(impromptu.fingertip_targets, bagatelle.fingertip_targets)
    np.testing.assert_allclose(impromptu.waypoint_fingertips, bagatelle.waypoint_fingertips)
    np.testing.assert_allclose(impromptu.waypoint_hand_joints, bagatelle.waypoint_hand_joints)
    assert impromptu.metadata["trajectory_mode"] == "joint_space_straighten"
    assert impromptu.metadata["ik_source"] == "Bagatelle.solve_press_pose_sparse_only"
    assert impromptu.metadata["assignment_source"] == "Bagatelle.plan_target_keys"
    assert impromptu.metadata["planned_hand_joints_source"] == "downsampled_from_joint_space_straight_finger_trajectory"
    assert impromptu.metadata["dense_ik_seed_source"] == "not_used_joint_space_straighten_mode"


def test_default_joint_space_mode_inserts_straight_finger_anchors() -> None:
    target = np.stack([_row(), _row(10), _row(), _row(12), _row()], axis=0)
    config = ImpromptuConfig(interpolation_substeps=4, ik_max_nfev=2)

    plan = plan_target_keys(target, config=config, kinematics=CurledFakeKinematics())

    straight_frames = plan.metadata["joint_space_trajectory"]["joint_space_straight_anchor_frames_dense"]
    assert straight_frames
    assert float(np.max(np.abs(plan.waypoint_hand_joints[:, ALL_FINGER_JOINT_INDICES]))) > 0.5
    for frame in straight_frames:
        rows = np.flatnonzero(plan.ik_anchor_frames_dense == int(frame))
        assert rows.size == 1
        np.testing.assert_allclose(plan.ik_anchor_qpos[int(rows[0]), ALL_FINGER_JOINT_INDICES], 0.0, atol=1e-6)


def test_anchor_config_changes_planner_anchor_output() -> None:
    target = np.stack([_row(), _row(10), _row(10), _row(12), _row(), _row(16), _row()], axis=0)
    dense_config = ImpromptuConfig(
        trajectory_mode=TRAJECTORY_MODE_DENSE_FINGERTIP_IK,
        interpolation_substeps=4,
        anchor_stride=1,
        solve_contact_window_only=False,
        include_midpoint_anchors=True,
        ik_max_nfev=2,
    )
    sparse_config = ImpromptuConfig(
        trajectory_mode=TRAJECTORY_MODE_DENSE_FINGERTIP_IK,
        interpolation_substeps=4,
        anchor_stride=8,
        solve_contact_window_only=True,
        include_midpoint_anchors=False,
        ik_max_nfev=2,
    )

    dense = plan_target_keys(target, config=dense_config, kinematics=FakeKinematics())
    sparse = plan_target_keys(target, config=sparse_config, kinematics=FakeKinematics())

    assert dense.ik_anchor_frames_dense.size > sparse.ik_anchor_frames_dense.size
    assert dense.metadata["selected_anchor_config"]["anchor_stride"] == 1
    assert sparse.metadata["selected_anchor_config"]["anchor_stride"] == 8
    assert sparse.metadata["selected_anchor_config"]["solve_contact_window_only"] is True
