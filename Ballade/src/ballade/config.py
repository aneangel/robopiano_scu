from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ballade.constants import CONTROL_DT, SOURCE_DT


@dataclass(slots=True)
class CostWeights:
    keypress: float = 8.0
    non_target_keypress: float = 5.0
    hand_endpoint_q: float = 4.0
    fingertip: float = 3.0
    hand_q: float = 1.0
    hand_qvel: float = 0.25
    action_smoothness: float = 0.05
    action_saturation: float = 0.02
    residual_size: float = 0.01


@dataclass(slots=True)
class JacobianTrackerConfig:
    ridge: float = 1e-3
    damping: float = 5e-2
    history_size: int = 512
    min_samples: int = 8
    max_action_delta: float = 0.15
    proportional_gain: float = 0.25
    action_low: float = -1.0
    action_high: float = 1.0


@dataclass(slots=True)
class LocalSearchConfig:
    enabled: bool = True
    mode: str = "event_triggered"
    candidate_count: int = 16
    horizon_microsteps: int = 1
    action_sigma: float = 0.08
    q_error_trigger: float = 0.75
    fingertip_error_trigger: float = 0.03
    wrong_key_trigger: bool = True
    seed: int = 0


@dataclass(slots=True)
class ResidualModelConfig:
    residual_scale: float = 0.2
    hidden_dim: int = 256
    hidden_layers: int = 3
    dropout: float = 0.0


@dataclass(slots=True)
class BalladeRunConfig:
    run_name: str = "ballade_run"
    source_dt: float = SOURCE_DT
    control_dt: float = CONTROL_DT
    max_demos: int = 3
    max_source_steps: int | None = None
    render: bool = False
    weights: CostWeights = field(default_factory=CostWeights)
    tracker: JacobianTrackerConfig = field(default_factory=JacobianTrackerConfig)
    local_search: LocalSearchConfig = field(default_factory=LocalSearchConfig)
    residual_model: ResidualModelConfig = field(default_factory=ResidualModelConfig)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is present on WAVE.
        raise RuntimeError("PyYAML is required to load Ballade YAML configs") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return data
