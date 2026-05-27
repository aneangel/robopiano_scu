from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal


SelectionMetric = Literal[
    "objective",
    "event_f1",
    "frame_f1",
    "rp1m_key_f1",
    "hand_l2_mean",
    "hand_l2_final",
    "hand_l2_last_third_mean",
    "one_to_one_l2_mean",
    "one_to_one_l2_final",
    "one_to_one_l2_last_third_mean",
    "coupled_l2_mean",
    "actuator_l2_mean",
]


@dataclass(frozen=True, slots=True)
class GainCandidate:
    controller_kind: str
    kp: float
    kd: float
    ki: float
    integral_limit: float
    setpoint_policy: str
    use_target_velocity: bool
    target_velocity_scale: float
    feedforward_scale: float = 0.0
    lookahead_substeps: int = 10
    sustain_value: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def label(self, index: int | None = None) -> str:
        prefix = f"c{int(index):03d}_" if index is not None else ""
        velocity = "tv" if self.use_target_velocity else "notv"
        return (
            f"{prefix}{self.controller_kind}_kp{self.kp:g}_kd{self.kd:g}_ki{self.ki:g}_"
            f"il{self.integral_limit:g}_{self.setpoint_policy}_{velocity}"
            f"{self.target_velocity_scale:g}_ff{self.feedforward_scale:g}_"
            f"h{self.lookahead_substeps:g}_sus{self.sustain_value:g}"
        ).replace("-", "m").replace(".", "p")


def _as_floats(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def make_gain_candidates(
    *,
    controllers: Iterable[str],
    kp_values: Iterable[float],
    kd_values: Iterable[float],
    ki_values: Iterable[float],
    integral_limits: Iterable[float],
    setpoint_policies: Iterable[str],
    target_velocity_scales: Iterable[float],
    feedforward_scales: Iterable[float] = (0.0,),
    lookahead_substeps_values: Iterable[int] = (10,),
    sustain_values: Iterable[float] = (0.0,),
) -> list[GainCandidate]:
    kp_grid = _as_floats(kp_values)
    kd_grid = _as_floats(kd_values)
    ki_grid = _as_floats(ki_values)
    integral_grid = _as_floats(integral_limits)
    velocity_grid = _as_floats(target_velocity_scales)
    feedforward_grid = _as_floats(feedforward_scales)
    lookahead_grid = tuple(max(int(value), 1) for value in lookahead_substeps_values)
    sustain_grid = _as_floats(sustain_values)
    candidates: list[GainCandidate] = []
    for controller in controllers:
        kind = str(controller).lower()
        if kind not in {"p", "pd", "pid"}:
            raise ValueError(f"Unknown controller kind: {controller!r}")
        active_kd_values = (0.0,) if kind == "p" else kd_grid
        active_ki_values = (0.0,) if kind != "pid" else ki_grid
        active_integral_limits = (0.0,) if kind != "pid" else integral_grid
        for kp in kp_grid:
            for kd in active_kd_values:
                for ki in active_ki_values:
                    for integral_limit in active_integral_limits:
                        for setpoint_policy in setpoint_policies:
                            policy = str(setpoint_policy)
                            if policy not in {"next", "linear", "minimum_jerk"}:
                                raise ValueError(f"Unknown setpoint policy: {setpoint_policy!r}")
                            for target_velocity_scale in velocity_grid:
                                for feedforward_scale in feedforward_grid:
                                    for lookahead_substeps in lookahead_grid:
                                        for sustain_value in sustain_grid:
                                            candidates.append(
                                                GainCandidate(
                                                    controller_kind=kind,
                                                    kp=kp,
                                                    kd=kd,
                                                    ki=ki,
                                                    integral_limit=integral_limit,
                                                    setpoint_policy=policy,
                                                    use_target_velocity=target_velocity_scale != 0.0,
                                                    target_velocity_scale=target_velocity_scale,
                                                    feedforward_scale=feedforward_scale,
                                                    lookahead_substeps=lookahead_substeps,
                                                    sustain_value=sustain_value,
                                                )
                                            )
    return candidates


def score_pid_result(
    result: dict[str, object],
    *,
    event_weight: float = 3.0,
    frame_weight: float = 1.0,
    key_weight: float = 1.0,
    hand_l2_weight: float = 0.05,
    final_l2_weight: float = 0.1,
    late_l2_weight: float = 0.1,
    max_l2_weight: float = 0.05,
    termination_penalty: float = 1.0,
) -> float:
    hand_l2 = result.get("hand_qpos_l2_vs_reference") or {}
    hand_l2_mean = 0.0
    hand_l2_max = 0.0
    if isinstance(hand_l2, dict):
        hand_l2_mean = float(hand_l2.get("mean") or 0.0)
        hand_l2_max = float(hand_l2.get("max") or 0.0)
    split = result.get("hand_tracking_split") or {}
    total = {}
    if isinstance(split, dict):
        total = split.get("total_qpos_l2") or {}
    hand_l2_final = float(total.get("final") or 0.0) if isinstance(total, dict) else 0.0
    hand_l2_late = float(total.get("last_third_mean") or 0.0) if isinstance(total, dict) else 0.0
    objective = (
        float(event_weight) * float(result.get("event_f1") or 0.0)
        + float(frame_weight) * float(result.get("frame_f1") or 0.0)
        + float(key_weight) * float(result.get("rp1m_key_f1") or 0.0)
        - float(hand_l2_weight) * hand_l2_mean
        - float(final_l2_weight) * hand_l2_final
        - float(late_l2_weight) * hand_l2_late
        - float(max_l2_weight) * hand_l2_max
    )
    if bool(result.get("terminated")):
        objective -= float(termination_penalty)
    return float(objective)


def metric_from_row(row: dict[str, object], metric: SelectionMetric) -> float:
    if metric == "objective":
        return float(row.get("mean_objective") or row.get("objective") or 0.0)
    if metric == "hand_l2_mean":
        return -float(
            row.get("mean_hand_l2_mean")
            or row.get("mean_hand_l2")
            or row.get("hand_l2_mean")
            or 0.0
        )
    if metric == "hand_l2_final":
        return -float(row.get("mean_hand_l2_final") or row.get("hand_l2_final") or 0.0)
    if metric == "hand_l2_last_third_mean":
        return -float(row.get("mean_hand_l2_last_third_mean") or row.get("hand_l2_last_third_mean") or 0.0)
    if metric == "one_to_one_l2_mean":
        return -float(row.get("mean_one_to_one_l2_mean") or row.get("one_to_one_l2_mean") or 0.0)
    if metric == "one_to_one_l2_final":
        return -float(row.get("mean_one_to_one_l2_final") or row.get("one_to_one_l2_final") or 0.0)
    if metric == "one_to_one_l2_last_third_mean":
        return -float(
            row.get("mean_one_to_one_l2_last_third_mean")
            or row.get("one_to_one_l2_last_third_mean")
            or 0.0
        )
    if metric == "coupled_l2_mean":
        return -float(row.get("mean_coupled_l2_mean") or row.get("coupled_l2_mean") or 0.0)
    if metric == "actuator_l2_mean":
        return -float(row.get("mean_actuator_l2_mean") or row.get("actuator_l2_mean") or 0.0)
    return float(row.get(f"mean_{metric}") or row.get(metric) or 0.0)
