from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from impromptu.config import ImpromptuConfig
from impromptu.ik_solver import interpolate_anchor_qpos
from impromptu.paths import ensure_repo_paths

ensure_repo_paths()
from bagatelle.kinematics import HAND_STATE_DIM  # noqa: E402
from intermezzo.constants import FINGER_JOINT_INDICES, LEFT_FOREARM_TY_INDEX, RIGHT_FOREARM_TY_INDEX  # noqa: E402


TRAJECTORY_MODE_JOINT_SPACE_STRAIGHTEN = "joint_space_straighten"
TRAJECTORY_MODE_DENSE_FINGERTIP_IK = "dense_fingertip_ik"
TRAJECTORY_MODES = (TRAJECTORY_MODE_JOINT_SPACE_STRAIGHTEN, TRAJECTORY_MODE_DENSE_FINGERTIP_IK)

FINGER_JOINT_INDEX_ROWS = tuple(tuple(int(index) for index in row) for row in FINGER_JOINT_INDICES)
BAGATELLE_FINGER_JOINT_INDEX_ROWS = FINGER_JOINT_INDEX_ROWS[5:] + FINGER_JOINT_INDEX_ROWS[:5]
ALL_FINGER_JOINT_INDICES = np.unique(
    np.asarray([index for row in FINGER_JOINT_INDEX_ROWS for index in row], dtype=np.int64)
).astype(np.int64)
BAGATELLE_LEFT_FINGERS = range(0, 5)
BAGATELLE_RIGHT_FINGERS = range(5, 10)


@dataclass(frozen=True)
class JointSpaceTrajectory:
    dense_qpos: np.ndarray
    segment_ids: np.ndarray
    anchor_frames: np.ndarray
    anchor_qpos: np.ndarray
    waypoint_qpos: np.ndarray
    straight_anchor_mask: np.ndarray
    metadata: dict[str, object]


def _clip_qpos(kin: Any | None, qpos: np.ndarray) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if values.shape != (HAND_STATE_DIM,):
        raise ValueError(f"qpos must have shape [{HAND_STATE_DIM}], got {values.shape}")
    if kin is not None and hasattr(kin, "clip_qpos"):
        return np.asarray(kin.clip_qpos(values), dtype=np.float32)
    lower = np.asarray(getattr(kin, "joint_lower", np.full((HAND_STATE_DIM,), -np.inf)), dtype=np.float32)
    upper = np.asarray(getattr(kin, "joint_upper", np.full((HAND_STATE_DIM,), np.inf)), dtype=np.float32)
    return np.clip(values, lower, upper).astype(np.float32)


def _finger_joint_indices_for_assignments(
    assignments: np.ndarray,
    *,
    previous_row: int,
    next_row: int,
    preserve_sustained: bool,
    allow_sustained_hold: bool = True,
) -> np.ndarray:
    indices: list[int] = []
    for finger in range(10):
        previous_key = int(assignments[previous_row, finger]) if 0 <= previous_row < assignments.shape[0] else -1
        next_key = int(assignments[next_row, finger]) if 0 <= next_row < assignments.shape[0] else -1
        if bool(preserve_sustained) and bool(allow_sustained_hold) and previous_key >= 0 and previous_key == next_key:
            continue
        indices.extend(BAGATELLE_FINGER_JOINT_INDEX_ROWS[finger])
    if not indices:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.asarray(indices, dtype=np.int64)).astype(np.int64)


def _straight_finger_values(kin: Any | None, config: ImpromptuConfig) -> np.ndarray:
    candidate = np.zeros((HAND_STATE_DIM,), dtype=np.float32)
    candidate[ALL_FINGER_JOINT_INDICES] = np.float32(float(config.joint_space_straight_value))
    return _clip_qpos(kin, candidate)[ALL_FINGER_JOINT_INDICES]


def _straightened_pose(
    *,
    base: np.ndarray,
    finger_joint_indices: np.ndarray,
    straight_values_by_all_finger_joint: np.ndarray,
    kin: Any | None,
) -> np.ndarray:
    out = np.asarray(base, dtype=np.float32).copy()
    if finger_joint_indices.size:
        positions = np.searchsorted(ALL_FINGER_JOINT_INDICES, finger_joint_indices)
        out[finger_joint_indices] = straight_values_by_all_finger_joint[positions]
    return _clip_qpos(kin, out)


def _sustained_hands(
    assignments: np.ndarray,
    previous_row: int,
    next_row: int,
    *,
    allow_sustained_hold: bool = True,
) -> set[str]:
    sustained: set[str] = set()
    if not bool(allow_sustained_hold):
        return sustained
    if previous_row < 0 or next_row < 0 or previous_row >= assignments.shape[0] or next_row >= assignments.shape[0]:
        return sustained
    for hand, fingers in (("left", BAGATELLE_LEFT_FINGERS), ("right", BAGATELLE_RIGHT_FINGERS)):
        for finger in fingers:
            previous_key = int(assignments[previous_row, finger])
            next_key = int(assignments[next_row, finger])
            if previous_key >= 0 and previous_key == next_key:
                sustained.add(hand)
                break
    return sustained


def _lifted_straight_anchor_base(
    *,
    base: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    sustained_hands: set[str],
    config: ImpromptuConfig,
    kin: Any | None,
) -> np.ndarray:
    out = np.asarray(base, dtype=np.float32).copy()
    if not bool(config.joint_space_lift_straight_anchors):
        return _clip_qpos(kin, out)
    lift = float(config.joint_space_straight_lift_height)
    if lift <= 0.0:
        return _clip_qpos(kin, out)
    for hand, index in (("right", RIGHT_FOREARM_TY_INDEX), ("left", LEFT_FOREARM_TY_INDEX)):
        if hand in sustained_hands:
            continue
        out[index] = max(float(out[index]), float(max(q0[index], q1[index])) + lift)
    return _clip_qpos(kin, out)


def _press_pose_with_idle_fingers_straight(
    *,
    pose: np.ndarray,
    assignment_row: np.ndarray,
    straight_values_by_all_finger_joint: np.ndarray,
    config: ImpromptuConfig,
    kin: Any | None,
) -> np.ndarray:
    qpos = _clip_qpos(kin, pose)
    if not bool(config.joint_space_straighten_idle_fingers_at_waypoints):
        return qpos
    assigned = np.asarray(assignment_row, dtype=np.int32).reshape(-1)
    finger_joints: list[int] = []
    for finger in range(10):
        if finger >= assigned.size or int(assigned[finger]) < 0:
            finger_joints.extend(BAGATELLE_FINGER_JOINT_INDEX_ROWS[finger])
    if not finger_joints:
        return qpos
    return _straightened_pose(
        base=qpos,
        finger_joint_indices=np.unique(np.asarray(finger_joints, dtype=np.int64)).astype(np.int64),
        straight_values_by_all_finger_joint=straight_values_by_all_finger_joint,
        kin=kin,
    )


def _dedupe_anchors(
    frames: list[int],
    qpos_rows: list[np.ndarray],
    straight_flags: list[bool],
    *,
    dense_total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not frames:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    ordered = sorted(
        (
            (int(np.clip(frame, 0, max(int(dense_total) - 1, 0))), order, np.asarray(qpos, dtype=np.float32), bool(flag))
            for order, (frame, qpos, flag) in enumerate(zip(frames, qpos_rows, straight_flags))
        ),
        key=lambda item: (item[0], item[1]),
    )
    by_frame: dict[int, tuple[np.ndarray, bool]] = {}
    for frame, _order, qpos, flag in ordered:
        by_frame[int(frame)] = (qpos, bool(flag))
    out_frames = np.asarray(sorted(by_frame), dtype=np.int64)
    out_qpos = np.stack([by_frame[int(frame)][0] for frame in out_frames], axis=0).astype(np.float32)
    out_flags = np.asarray([by_frame[int(frame)][1] for frame in out_frames], dtype=bool)
    return out_frames, out_qpos, out_flags


def _frame_count(seconds: float, dense_dt: float) -> int:
    if float(seconds) <= 0.0:
        return 0
    return max(int(np.ceil(float(seconds) / max(float(dense_dt), 1e-8))), 1)


def _boundary_finger_joint_indices() -> np.ndarray:
    # Boundary anchors move out of or back into a rest pose, so all fingers are
    # allowed to straighten. Sustained-key preservation only applies between
    # adjacent press waypoints.
    return ALL_FINGER_JOINT_INDICES


def _interpolate_pose(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    u = float(np.clip(alpha, 0.0, 1.0))
    return (
        np.asarray(q0, dtype=np.float32) + u * (np.asarray(q1, dtype=np.float32) - np.asarray(q0, dtype=np.float32))
    ).astype(np.float32)


def build_joint_space_straightened_trajectory(
    *,
    total_steps: int,
    waypoint_frames: np.ndarray,
    waypoint_release_frames: np.ndarray | None = None,
    waypoint_qpos: np.ndarray,
    assignments: np.ndarray,
    neutral_qpos: np.ndarray,
    config: ImpromptuConfig,
    kinematics: Any | None = None,
) -> JointSpaceTrajectory:
    """Interpolate sparse press poses through explicit straight-finger anchors."""
    total = max(int(total_steps), 0)
    substeps = max(int(config.interpolation_substeps), 1)
    dense_total = total * substeps
    neutral = _clip_qpos(kinematics, neutral_qpos)
    if dense_total == 0:
        return JointSpaceTrajectory(
            dense_qpos=np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            segment_ids=np.zeros((0,), dtype=np.int32),
            anchor_frames=np.zeros((0,), dtype=np.int64),
            anchor_qpos=np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            waypoint_qpos=np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            straight_anchor_mask=np.zeros((0,), dtype=bool),
            metadata={"joint_space_straight_anchor_count": 0},
        )

    frames_control = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    qpos = np.asarray(waypoint_qpos, dtype=np.float32)
    assigns = np.asarray(assignments, dtype=np.int32)
    if frames_control.size == 0:
        dense = np.repeat(neutral.reshape(1, -1), dense_total, axis=0).astype(np.float32)
        return JointSpaceTrajectory(
            dense_qpos=dense,
            segment_ids=np.full((dense_total,), -1, dtype=np.int32),
            anchor_frames=np.zeros((0,), dtype=np.int64),
            anchor_qpos=np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            waypoint_qpos=np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
            straight_anchor_mask=np.zeros((0,), dtype=bool),
            metadata={"joint_space_straight_anchor_count": 0},
        )
    if qpos.shape != (frames_control.size, HAND_STATE_DIM):
        raise ValueError(f"waypoint_qpos must have shape [{frames_control.size}, {HAND_STATE_DIM}], got {qpos.shape}")
    if assigns.shape[0] != frames_control.size or assigns.shape[1] != 10:
        raise ValueError(f"assignments must have shape [{frames_control.size}, 10], got {assigns.shape}")

    waypoint_dense = np.clip(frames_control * substeps, 0, max(dense_total - 1, 0)).astype(np.int64)
    if waypoint_release_frames is None:
        release_control = frames_control
    else:
        release_control = np.asarray(waypoint_release_frames, dtype=np.int64).reshape(-1)
        if release_control.shape != frames_control.shape:
            raise ValueError(
                f"waypoint_release_frames must have shape [{frames_control.size}], got {release_control.shape}"
            )
        release_control = np.maximum(release_control, frames_control)
    hold_end_dense = np.clip(
        release_control * substeps + max(substeps - 1, 0),
        0,
        max(dense_total - 1, 0),
    ).astype(np.int64)
    straight_values = _straight_finger_values(kinematics, config)
    release_fraction = float(np.clip(float(config.joint_space_release_fraction), 0.0, 1.0))
    approach_fraction = float(np.clip(float(config.joint_space_approach_fraction), 0.0, 1.0))
    dense_dt = float(config.control_timestep) / float(substeps)
    approach_frames = _frame_count(float(config.approach_s), dense_dt)
    release_frames = _frame_count(float(config.release_s), dense_dt)
    same_key_gap_count = 0
    continuous_sustained_finger_count = 0

    press_qpos = np.stack(
        [
            _press_pose_with_idle_fingers_straight(
                pose=qpos[index],
                assignment_row=assigns[index],
                straight_values_by_all_finger_joint=straight_values,
                config=config,
                kin=kinematics,
            )
            for index in range(frames_control.size)
        ],
        axis=0,
    ).astype(np.float32)

    anchor_frames: list[int] = []
    anchor_qpos: list[np.ndarray] = []
    straight_flags: list[bool] = []
    boundary_straight_count = 0
    boundary_neutral_count = 0

    first_press = int(waypoint_dense[0])
    if first_press > 0:
        anchor_frames.append(0)
        anchor_qpos.append(neutral)
        straight_flags.append(False)
        boundary_neutral_count += 1
        approach_frame = max(0, first_press - approach_frames)
        if 0 < approach_frame < first_press:
            span = max(first_press, 1)
            base = _interpolate_pose(neutral, press_qpos[0], float(approach_frame) / float(span))
            base = _lifted_straight_anchor_base(
                base=base,
                q0=neutral,
                q1=press_qpos[0],
                sustained_hands=set(),
                config=config,
                kin=kinematics,
            )
            anchor_frames.append(int(approach_frame))
            anchor_qpos.append(
                _straightened_pose(
                    base=base,
                    finger_joint_indices=_boundary_finger_joint_indices(),
                    straight_values_by_all_finger_joint=straight_values,
                    kin=kinematics,
                )
            )
            straight_flags.append(True)
            boundary_straight_count += 1
    anchor_frames.append(first_press)
    anchor_qpos.append(press_qpos[0])
    straight_flags.append(False)
    hold_anchor_count = 0

    for index in range(frames_control.size - 1):
        press_start = int(waypoint_dense[index])
        hold_end = int(hold_end_dense[index])
        if hold_end > press_start:
            anchor_frames.append(hold_end)
            anchor_qpos.append(press_qpos[index])
            straight_flags.append(False)
            hold_anchor_count += 1
        start = max(press_start, hold_end)
        end = int(waypoint_dense[index + 1])
        q0 = press_qpos[index]
        q1 = press_qpos[index + 1]
        if end <= start:
            anchor_frames.append(end)
            anchor_qpos.append(q1)
            straight_flags.append(False)
            continue
        span = end - start
        continuous_hold_to_next = bool(int(release_control[index]) >= int(frames_control[index + 1]) - 1)
        release_repeated_key_gaps = bool(getattr(config, "joint_space_release_repeated_keys_across_gaps", True))
        allow_sustained_preservation = bool(continuous_hold_to_next or not release_repeated_key_gaps)
        same_key_fingers = [
            finger
            for finger in range(10)
            if int(assigns[index, finger]) >= 0 and int(assigns[index, finger]) == int(assigns[index + 1, finger])
        ]
        if continuous_hold_to_next:
            continuous_sustained_finger_count += len(same_key_fingers)
        else:
            same_key_gap_count += len(same_key_fingers)
        if bool(config.joint_space_straighten_all_fingers):
            finger_joints = ALL_FINGER_JOINT_INDICES
        else:
            finger_joints = _finger_joint_indices_for_assignments(
                assigns,
                previous_row=index,
                next_row=index + 1,
                preserve_sustained=bool(config.joint_space_preserve_sustained_fingers),
                allow_sustained_hold=allow_sustained_preservation,
            )
        sustained_hands = (
            _sustained_hands(assigns, index, index + 1, allow_sustained_hold=allow_sustained_preservation)
            if bool(config.joint_space_preserve_sustained_fingers)
            else set()
        )
        release_frame = start + max(int(round(span * release_fraction)), 1)
        approach_frame = end - max(int(round(span * approach_fraction)), 1)
        if release_frame >= end:
            release_frame = start + max(span // 2, 1)
        if approach_frame <= start:
            approach_frame = end - max(span // 2, 1)
        if approach_frame < release_frame:
            mid = start + max(span // 2, 1)
            release_frame = mid
            approach_frame = mid
        for frame in sorted({release_frame, approach_frame}):
            if not (start < int(frame) < end):
                continue
            alpha = (float(frame) - float(start)) / float(span)
            base = (q0 + alpha * (q1 - q0)).astype(np.float32)
            base = _lifted_straight_anchor_base(
                base=base,
                q0=q0,
                q1=q1,
                sustained_hands=sustained_hands,
                config=config,
                kin=kinematics,
            )
            anchor_frames.append(int(frame))
            anchor_qpos.append(
                _straightened_pose(
                    base=base,
                    finger_joint_indices=finger_joints,
                    straight_values_by_all_finger_joint=straight_values,
                    kin=kinematics,
                )
            )
            straight_flags.append(True)
        anchor_frames.append(end)
        anchor_qpos.append(q1)
        straight_flags.append(False)

    last_press = int(waypoint_dense[-1])
    last_hold_end = int(hold_end_dense[-1])
    if last_hold_end > last_press:
        anchor_frames.append(last_hold_end)
        anchor_qpos.append(press_qpos[-1])
        straight_flags.append(False)
        hold_anchor_count += 1
    final_frame = max(dense_total - 1, 0)
    release_start = max(last_press, last_hold_end)
    if release_start < final_frame:
        release_frame = min(final_frame, release_start + release_frames)
        if release_start < release_frame < final_frame:
            span = max(final_frame - release_start, 1)
            base = _interpolate_pose(press_qpos[-1], neutral, float(release_frame - release_start) / float(span))
            base = _lifted_straight_anchor_base(
                base=base,
                q0=press_qpos[-1],
                q1=neutral,
                sustained_hands=set(),
                config=config,
                kin=kinematics,
            )
            anchor_frames.append(int(release_frame))
            anchor_qpos.append(
                _straightened_pose(
                    base=base,
                    finger_joint_indices=_boundary_finger_joint_indices(),
                    straight_values_by_all_finger_joint=straight_values,
                    kin=kinematics,
                )
            )
            straight_flags.append(True)
            boundary_straight_count += 1
        anchor_frames.append(final_frame)
        anchor_qpos.append(neutral)
        straight_flags.append(False)
        boundary_neutral_count += 1

    frames, anchors, flags = _dedupe_anchors(anchor_frames, anchor_qpos, straight_flags, dense_total=dense_total)
    dense, segment_ids = interpolate_anchor_qpos(anchor_frames=frames, anchor_qpos=anchors, dense_total=dense_total)
    return JointSpaceTrajectory(
        dense_qpos=dense.astype(np.float32),
        segment_ids=segment_ids.astype(np.int32),
        anchor_frames=frames,
        anchor_qpos=anchors.astype(np.float32),
        waypoint_qpos=press_qpos.astype(np.float32),
        straight_anchor_mask=flags.astype(bool),
        metadata={
            "joint_space_straight_anchor_count": int(np.count_nonzero(flags)),
            "joint_space_straight_anchor_frames_dense": frames[flags].astype(int).tolist(),
            "joint_space_finger_joint_indices": ALL_FINGER_JOINT_INDICES.astype(int).tolist(),
            "joint_space_release_fraction": release_fraction,
            "joint_space_approach_fraction": approach_fraction,
            "joint_space_boundary_straight_anchor_count": int(boundary_straight_count),
            "joint_space_boundary_neutral_anchor_count": int(boundary_neutral_count),
            "joint_space_hold_anchor_count": int(hold_anchor_count),
            "joint_space_same_key_gap_finger_count": int(same_key_gap_count),
            "joint_space_continuous_sustained_finger_count": int(continuous_sustained_finger_count),
            "joint_space_approach_frames": int(approach_frames),
            "joint_space_release_frames": int(release_frames),
            "joint_space_waypoint_release_frames_control": release_control.astype(int).tolist(),
            "joint_space_waypoint_hold_end_frames_dense": hold_end_dense.astype(int).tolist(),
            "joint_space_straight_value": float(config.joint_space_straight_value),
            "joint_space_straighten_all_fingers": bool(config.joint_space_straighten_all_fingers),
            "joint_space_preserve_sustained_fingers": bool(config.joint_space_preserve_sustained_fingers),
            "joint_space_release_repeated_keys_across_gaps": bool(
                getattr(config, "joint_space_release_repeated_keys_across_gaps", True)
            ),
            "joint_space_straighten_idle_fingers_at_waypoints": bool(
                config.joint_space_straighten_idle_fingers_at_waypoints
            ),
            "joint_space_lift_straight_anchors": bool(config.joint_space_lift_straight_anchors),
            "joint_space_straight_lift_height": float(config.joint_space_straight_lift_height),
            "joint_space_bagatelle_finger_to_joint_indices": [
                [int(index) for index in row] for row in BAGATELLE_FINGER_JOINT_INDEX_ROWS
            ],
        },
    )
