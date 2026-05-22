from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    config_path = _resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def _resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    if config_path.exists():
        return config_path
    parts = config_path.parts
    if "configs" in parts:
        config_index = parts.index("configs")
        etude_root = Path(__file__).resolve().parents[2]
        candidate = etude_root.joinpath(*parts[config_index:])
        if candidate.exists():
            return candidate
    return config_path


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of base recursively updated with overrides."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
