from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np


HAND_STATE_DIM = 46
HAND_STATE_DIM_PER_HAND = 23
REDUCED_ACTION_DIM = 39
REDUCED_ACTION_DIM_PER_HAND = 19
SUSTAIN_ACTION_INDEX = 38
ProjectionMode = Literal["weighted_least_squares", "legacy_sum"]

RIGHT_HAND_SLICE = slice(0, HAND_STATE_DIM_PER_HAND)
LEFT_HAND_SLICE = slice(HAND_STATE_DIM_PER_HAND, HAND_STATE_DIM)

# RoboPianist's reduced hand-joint order is the exact order used by
# task.right_hand.joints and task.left_hand.joints. Forearm sliders are appended
# after the XML hand joints.
NOMINAL_REDUCED_HAND_JOINT_NAMES: tuple[str, ...] = (
    "WRJ2",
    "WRJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "THJ4",
    "THJ3",
    "THJ2",
    "forearm_tx",
    "forearm_ty",
)

# RoboPianist's reduced action space removes THJ5, THJ1, and LFJ5. Four distal
# finger actions drive fixed tendons, not a single qpos channel.
NOMINAL_REDUCED_HAND_ACTUATOR_NAMES: tuple[str, ...] = (
    "A_WRJ2",
    "A_WRJ1",
    "A_THJ4",
    "A_THJ3",
    "A_THJ2",
    "A_FFJ4",
    "A_FFJ3",
    "A_FFJ0",
    "A_MFJ4",
    "A_MFJ3",
    "A_MFJ0",
    "A_RFJ4",
    "A_RFJ3",
    "A_RFJ0",
    "A_LFJ4",
    "A_LFJ3",
    "A_LFJ0",
    "forearm_tx",
    "forearm_ty",
)

TENDON_JOINTS: dict[str, tuple[str, str]] = {
    "FFJ0": ("FFJ2", "FFJ1"),
    "MFJ0": ("MFJ2", "MFJ1"),
    "RFJ0": ("RFJ2", "RFJ1"),
    "LFJ0": ("LFJ2", "LFJ1"),
}


@dataclass(frozen=True, slots=True)
class ActionJointMapEntry:
    action_index: int
    hand: str
    actuator_name: str
    joint_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    weights: tuple[float, ...]
    kind: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _strip_namespace(name: str) -> str:
    value = str(name)
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value


def normalize_joint_name(name: str) -> str:
    value = _strip_namespace(name)
    for prefix in ("rh_", "lh_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value


def normalize_actuator_name(name: str) -> str:
    value = normalize_joint_name(name)
    if value.startswith("A_"):
        value = value[2:]
    return value


def _nominal_joint_names() -> tuple[str, ...]:
    return tuple(f"rh_{name}" for name in NOMINAL_REDUCED_HAND_JOINT_NAMES) + tuple(
        f"lh_{name}" for name in NOMINAL_REDUCED_HAND_JOINT_NAMES
    )


def _nominal_actuator_names() -> tuple[str, ...]:
    return tuple(f"rh_{name}" for name in NOMINAL_REDUCED_HAND_ACTUATOR_NAMES) + tuple(
        f"lh_{name}" for name in NOMINAL_REDUCED_HAND_ACTUATOR_NAMES
    ) + ("sustain",)


def _resolve_hand_entries(
    *,
    action_offset: int,
    joint_offset: int,
    hand: str,
    hand_actuator_names: Sequence[str],
    hand_joint_names: Sequence[str],
) -> list[ActionJointMapEntry]:
    normalized_to_index = {
        normalize_joint_name(name): joint_offset + index for index, name in enumerate(hand_joint_names)
    }
    entries: list[ActionJointMapEntry] = []
    for local_action_index, actuator_name in enumerate(hand_actuator_names):
        actuator_key = normalize_actuator_name(actuator_name)
        if actuator_key in TENDON_JOINTS:
            local_joint_names = TENDON_JOINTS[actuator_key]
            kind = "fixed_tendon"
        else:
            local_joint_names = (actuator_key,)
            kind = "joint"
        missing = [name for name in local_joint_names if name not in normalized_to_index]
        if missing:
            raise ValueError(
                f"Cannot map {hand} action {local_action_index} ({actuator_name!r}); "
                f"missing joint(s): {missing}"
            )
        joint_indices = tuple(int(normalized_to_index[name]) for name in local_joint_names)
        entries.append(
            ActionJointMapEntry(
                action_index=int(action_offset + local_action_index),
                hand=hand,
                actuator_name=str(actuator_name),
                joint_indices=joint_indices,
                joint_names=tuple(str(hand_joint_names[index - joint_offset]) for index in joint_indices),
                weights=tuple(1.0 for _ in joint_indices),
                kind=kind,
            )
        )
    return entries


def build_reduced_action_mapping(
    *,
    action_dim: int = REDUCED_ACTION_DIM,
    hand_joint_names: Sequence[str] | None = None,
    actuator_names: Sequence[str] | None = None,
) -> tuple[ActionJointMapEntry, ...]:
    """Build the 39D reduced action to 46D hand-state map.

    The preferred path is to pass names captured from the live RoboPianist task.
    When names are unavailable, the function falls back to the reduced
    RoboPianist order used by Bagatelle/Impromptu hand-state rollouts.
    """
    if int(action_dim) != REDUCED_ACTION_DIM:
        raise ValueError(f"Only reduced 39D actions are supported, got action_dim={action_dim}")
    joint_names = tuple(hand_joint_names) if hand_joint_names is not None else _nominal_joint_names()
    if len(joint_names) != HAND_STATE_DIM:
        raise ValueError(f"Expected {HAND_STATE_DIM} hand joint names, got {len(joint_names)}")
    acts = tuple(actuator_names) if actuator_names is not None else _nominal_actuator_names()
    if len(acts) not in {REDUCED_ACTION_DIM - 1, REDUCED_ACTION_DIM}:
        raise ValueError(f"Expected 38 hand actuators plus optional sustain, got {len(acts)}")
    hand_acts = acts[: REDUCED_ACTION_DIM - 1]
    right_entries = _resolve_hand_entries(
        action_offset=0,
        joint_offset=0,
        hand="right",
        hand_actuator_names=hand_acts[:REDUCED_ACTION_DIM_PER_HAND],
        hand_joint_names=joint_names[RIGHT_HAND_SLICE],
    )
    left_entries = _resolve_hand_entries(
        action_offset=REDUCED_ACTION_DIM_PER_HAND,
        joint_offset=HAND_STATE_DIM_PER_HAND,
        hand="left",
        hand_actuator_names=hand_acts[REDUCED_ACTION_DIM_PER_HAND:],
        hand_joint_names=joint_names[LEFT_HAND_SLICE],
    )
    return tuple(
        [
            *right_entries,
            *left_entries,
            ActionJointMapEntry(
                action_index=SUSTAIN_ACTION_INDEX,
                hand="pedal",
                actuator_name=str(acts[SUSTAIN_ACTION_INDEX]) if len(acts) > SUSTAIN_ACTION_INDEX else "sustain",
                joint_indices=(),
                joint_names=(),
                weights=(),
                kind="sustain",
            ),
        ]
    )


def action_signals_from_hand_state(
    hand_qpos: np.ndarray,
    mapping: Sequence[ActionJointMapEntry],
    *,
    projection: ProjectionMode = "weighted_least_squares",
) -> np.ndarray:
    """Project a 46D hand qpos row into 39D actuator-position targets.

    The source trajectory is a hand-qpos target, not an actuator-space target.
    One-to-one actuators copy their matching joint coordinate. Fixed-tendon
    actuators project their coupled joint targets onto a single actuator scalar.
    The default projection is weighted least squares: for coupled target joints
    y and actuator coupling vector a, solve min_x ||a*x - y||_W.
    """
    qpos = np.asarray(hand_qpos, dtype=np.float32).reshape(-1)
    if qpos.shape != (HAND_STATE_DIM,):
        raise ValueError(f"hand_qpos must have shape [{HAND_STATE_DIM}], got {qpos.shape}")
    if projection not in {"weighted_least_squares", "legacy_sum"}:
        raise ValueError(f"Unknown projection mode: {projection!r}")
    signals = np.zeros((REDUCED_ACTION_DIM,), dtype=np.float32)
    for entry in mapping:
        if entry.kind == "sustain":
            continue
        weights = np.asarray(entry.weights, dtype=np.float32)
        values = qpos[list(entry.joint_indices)]
        if entry.kind == "fixed_tendon" and projection == "weighted_least_squares":
            numerator = float(np.dot(weights, values))
            denominator = float(np.dot(weights, weights))
            signals[entry.action_index] = np.float32(numerator / max(denominator, 1e-8))
        else:
            signals[entry.action_index] = np.float32(np.dot(values, weights))
    return signals


def joint_index_groups(
    mapping: Sequence[ActionJointMapEntry],
) -> dict[str, tuple[int, ...]]:
    one_to_one: set[int] = set()
    coupled: set[int] = set()
    actuated: set[int] = set()
    for entry in mapping:
        if entry.kind == "sustain":
            continue
        indices = set(int(index) for index in entry.joint_indices)
        actuated.update(indices)
        if entry.kind == "fixed_tendon":
            coupled.update(indices)
        elif len(indices) == 1:
            one_to_one.update(indices)
    return {
        "one_to_one": tuple(sorted(one_to_one - coupled)),
        "coupled": tuple(sorted(coupled)),
        "actuated": tuple(sorted(actuated)),
    }


def mapping_to_jsonable(mapping: Sequence[ActionJointMapEntry]) -> list[dict[str, object]]:
    return [entry.to_dict() for entry in mapping]
