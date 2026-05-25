from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from rhapsody.constants import FINGERTIP_DIM, HAND_STATE_DIM, NUM_FINGERS


@dataclass(frozen=True, slots=True)
class RhapsodyConfig:
    """Training and model parameters for the reward-trained IK policy."""

    qpos_dim: int = HAND_STATE_DIM
    num_fingers: int = NUM_FINGERS
    fingertip_dim: int = FINGERTIP_DIM
    policy_hidden_dims: tuple[int, ...] = (256, 256)
    fk_hidden_dims: tuple[int, ...] = (256, 256)
    policy_action_scale: float = 2.0
    policy_exploration_std: float = 0.35
    reward_active_weight: float = 1.0
    reward_max_error_weight: float = 0.25
    reward_smoothness_weight: float = 0.02
    imitation_weight: float = 0.05
    fk_learning_rate: float = 3.0e-4
    policy_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 256
    policy_samples_per_state: int = 4
    validation_fraction: float = 0.2
    gradient_clip_norm: float = 10.0
    seed: int = 524

    def with_overrides(self, **kwargs: Any) -> "RhapsodyConfig":
        return replace(self, **kwargs)
