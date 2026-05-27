from __future__ import annotations

from dataclasses import asdict, dataclass


SHADOW_HAND_XML_JOINTS_PER_HAND = 24
SHADOW_HAND_XML_ACTUATORS_PER_HAND = 20
DEFAULT_FOREARM_DOFS_PER_HAND = 2
REDUCED_EXCLUDED_DOFS_PER_HAND = 3
SUSTAIN_ACTION_DIM = 1
TENDON_ACTIONS_PER_HAND = 4

FULL_HAND_JOINT_DIM_PER_HAND = SHADOW_HAND_XML_JOINTS_PER_HAND + DEFAULT_FOREARM_DOFS_PER_HAND
FULL_HAND_ACTION_DIM_PER_HAND = SHADOW_HAND_XML_ACTUATORS_PER_HAND + DEFAULT_FOREARM_DOFS_PER_HAND
REDUCED_HAND_JOINT_DIM_PER_HAND = FULL_HAND_JOINT_DIM_PER_HAND - REDUCED_EXCLUDED_DOFS_PER_HAND
REDUCED_HAND_ACTION_DIM_PER_HAND = FULL_HAND_ACTION_DIM_PER_HAND - REDUCED_EXCLUDED_DOFS_PER_HAND

FULL_HAND_STATE_DIM = 2 * FULL_HAND_JOINT_DIM_PER_HAND
FULL_HAND_ACTION_DIM = 2 * FULL_HAND_ACTION_DIM_PER_HAND
FULL_ENV_ACTION_DIM = FULL_HAND_ACTION_DIM + SUSTAIN_ACTION_DIM

REDUCED_HAND_STATE_DIM = 2 * REDUCED_HAND_JOINT_DIM_PER_HAND
REDUCED_HAND_ACTION_DIM = 2 * REDUCED_HAND_ACTION_DIM_PER_HAND
REDUCED_ENV_ACTION_DIM = REDUCED_HAND_ACTION_DIM + SUSTAIN_ACTION_DIM

REDUCED_EXCLUDED_DOFS = ("A_THJ5", "A_THJ1", "A_LFJ5")
TENDON_ACTIONS = ("A_FFJ0", "A_MFJ0", "A_RFJ0", "A_LFJ0")


@dataclass(frozen=True, slots=True)
class ActionSpaceSummary:
    name: str
    reduced_action_space: bool
    hand_action_dim_per_hand: int
    hand_joint_dim_per_hand: int
    hand_action_dim: int
    hand_state_dim: int
    env_action_dim: int
    sustain_action_dim: int
    excluded_dofs_per_hand: int
    tendon_actions_per_hand: int

    @property
    def has_46_hand_actions_for_46_hand_states(self) -> bool:
        return self.hand_action_dim == 46 and self.hand_state_dim == 46

    @property
    def has_46_env_actions_for_46_hand_states(self) -> bool:
        return self.env_action_dim == 46 and self.hand_state_dim == 46

    @property
    def hand_action_state_dims_match(self) -> bool:
        return self.hand_action_dim == self.hand_state_dim

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["has_46_hand_actions_for_46_hand_states"] = (
            self.has_46_hand_actions_for_46_hand_states
        )
        payload["has_46_env_actions_for_46_hand_states"] = (
            self.has_46_env_actions_for_46_hand_states
        )
        payload["hand_action_state_dims_match"] = self.hand_action_state_dims_match
        excluded_dofs = list(REDUCED_EXCLUDED_DOFS) if self.reduced_action_space else []
        payload["reduced_excluded_dofs"] = excluded_dofs
        payload["tendon_actions"] = list(TENDON_ACTIONS)
        return payload


def expected_action_space_summary(*, reduced_action_space: bool) -> ActionSpaceSummary:
    if reduced_action_space:
        return ActionSpaceSummary(
            name="reduced",
            reduced_action_space=True,
            hand_action_dim_per_hand=REDUCED_HAND_ACTION_DIM_PER_HAND,
            hand_joint_dim_per_hand=REDUCED_HAND_JOINT_DIM_PER_HAND,
            hand_action_dim=REDUCED_HAND_ACTION_DIM,
            hand_state_dim=REDUCED_HAND_STATE_DIM,
            env_action_dim=REDUCED_ENV_ACTION_DIM,
            sustain_action_dim=SUSTAIN_ACTION_DIM,
            excluded_dofs_per_hand=REDUCED_EXCLUDED_DOFS_PER_HAND,
            tendon_actions_per_hand=TENDON_ACTIONS_PER_HAND,
        )
    return ActionSpaceSummary(
        name="full",
        reduced_action_space=False,
        hand_action_dim_per_hand=FULL_HAND_ACTION_DIM_PER_HAND,
        hand_joint_dim_per_hand=FULL_HAND_JOINT_DIM_PER_HAND,
        hand_action_dim=FULL_HAND_ACTION_DIM,
        hand_state_dim=FULL_HAND_STATE_DIM,
        env_action_dim=FULL_ENV_ACTION_DIM,
        sustain_action_dim=SUSTAIN_ACTION_DIM,
        excluded_dofs_per_hand=0,
        tendon_actions_per_hand=TENDON_ACTIONS_PER_HAND,
    )


def expected_action_space_summaries() -> tuple[ActionSpaceSummary, ActionSpaceSummary]:
    return (
        expected_action_space_summary(reduced_action_space=True),
        expected_action_space_summary(reduced_action_space=False),
    )
