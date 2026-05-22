from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NoteEvent:
    event_index: int
    onset_frame: int
    end_frame: int
    keys: tuple[int, ...]

    @property
    def chord_size(self) -> int:
        return len(self.keys)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["keys"] = list(self.keys)
        return data


@dataclass(frozen=True, slots=True)
class SegmentCandidate:
    segment_id: int
    event_index: int
    demo_id: int
    window_start: int
    window_end: int
    local_score: float
    frame_f1: float
    frame_precision: float
    frame_recall: float
    event_precision: float
    event_recall: float
    missed_keys: int
    wrong_keys: int
    timing_abs_error_frames: float
    action_smoothness: float
    joint_velocity: float
    joint_acceleration: float
    fingertip_jerk: float
    hit_keys: int = 0
    target_keys: int = 0
    avoidable_missed_keys: int = 0
    unavoidable_missed_keys: int = 0
    avoidable_wrong_keys: int = 0
    all_demo_wrong_keys: int = 0
    interval_missed_keys: int = 0
    interval_wrong_keys: int = 0
    avoidable_interval_missed_keys: int = 0
    unavoidable_interval_missed_keys: int = 0
    avoidable_interval_wrong_keys: int = 0
    all_demo_interval_wrong_keys: int = 0
    clean_hit: bool = False
    interval_clean: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransitionWeights:
    joint_position: float = 25.0
    joint_velocity: float = 5.0
    fingertip_position: float = 35.0
    action: float = 2.0
    source_switch: float = 0.05
    contact_multiplier: float = 2.0
    hard_joint_jump: float = 5.0
    hard_fingertip_jump: float = 0.35


@dataclass(frozen=True, slots=True)
class StitchConfig:
    dt: float = 0.05
    window_frames: int = 64
    pre_frames: int = 24
    chord_tolerance_frames: int = 1
    event_tolerance_frames: int = 3
    seam_blend_radius: int = 4
    threshold: float = 0.5
    objective_mode: str = "strict"
    repair_enabled: bool = True
    repair_max_passes: int = 2
    repair_transition_margin: float = 250.0
    adaptive_seam_enabled: bool = True
    seam_search_margin_frames: int = 3
    transition_interpolation_enabled: bool = True
    avoidable_miss_cost: float = 100000.0
    avoidable_wrong_cost: float = 50000.0
    unavoidable_miss_cost: float = 1200.0
    all_demo_wrong_cost: float = 400.0

    @property
    def post_frames(self) -> int:
        return max(int(self.window_frames) - int(self.pre_frames), 1)
