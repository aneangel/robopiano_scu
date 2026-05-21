from __future__ import annotations

from typing import Any

import numpy as np

from etude.robopianist.state_mapping import StateMapping


def extract_tracking_observation(
    source: Any,
    mapping: StateMapping,
    *,
    include_key_state: bool = True,
) -> dict[str, np.ndarray]:
    obs = {
        "q": mapping.extract_q(source),
        "qdot": mapping.extract_qdot(source),
    }
    fingertips = mapping.extract_fingertips(source)
    if fingertips is not None:
        obs["fingertips"] = fingertips
    key_state = mapping.extract_key_state(source) if include_key_state else None
    if key_state is not None:
        obs["key_state"] = key_state
    return obs


def extract_raw_key_state(source: Any) -> np.ndarray | None:
    data = _find_array(source, ("key_state", "keys", "piano_keys", "piano/state"))
    if data is None:
        return None
    return np.asarray(data, dtype=np.float32).reshape(-1)


def _find_array(source: Any, names: tuple[str, ...]) -> Any:
    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
        for value in source.values():
            found = _find_array(value, names)
            if found is not None:
                return found
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    if hasattr(source, "observation"):
        return _find_array(source.observation, names)
    return None
