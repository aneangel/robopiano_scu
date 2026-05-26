from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class TeacherTransition:
    obs: Any
    target: Any
    base_action: np.ndarray
    selected_action: np.ndarray
    next_obs: Any
    cost_before: float
    cost_after: float
    source_frame_index: int
    microstep_index: int
    song_key: str
    demo_id: int

    def to_json(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class OnlineTeacherReplayBuffer:
    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = None if capacity is None else int(capacity)
        self.transitions: list[TeacherTransition] = []

    def append(self, transition: TeacherTransition | dict[str, Any]) -> None:
        if isinstance(transition, TeacherTransition):
            item = transition
        else:
            item = TeacherTransition(
                obs=transition["obs"],
                target=transition["target"],
                base_action=np.asarray(transition["base_action"], dtype=np.float32),
                selected_action=np.asarray(transition["selected_action"], dtype=np.float32),
                next_obs=transition["next_obs"],
                cost_before=float(transition["cost_before"]),
                cost_after=float(transition["cost_after"]),
                source_frame_index=int(transition["source_frame_index"]),
                microstep_index=int(transition["microstep_index"]),
                song_key=str(transition["song_key"]),
                demo_id=int(transition["demo_id"]),
            )
        self.transitions.append(item)
        if self.capacity is not None:
            overflow = len(self.transitions) - self.capacity
            if overflow > 0:
                del self.transitions[:overflow]

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def save_shard(self, output_root: str | Path, *, shard_index: int = 0) -> Path:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"teacher_{int(shard_index):06d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for transition in self.transitions:
                handle.write(json.dumps(transition.to_json(), sort_keys=True) + "\n")
        return path

    @classmethod
    def load_shards(cls, root: str | Path) -> "OnlineTeacherReplayBuffer":
        buffer = cls()
        for path in sorted(Path(root).rglob("teacher_*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    buffer.append(payload)
        return buffer

    def action_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.transitions:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        base = np.stack([np.asarray(t.base_action, dtype=np.float32).reshape(-1) for t in self.transitions], axis=0)
        selected = np.stack(
            [np.asarray(t.selected_action, dtype=np.float32).reshape(-1) for t in self.transitions],
            axis=0,
        )
        return base.astype(np.float32), selected.astype(np.float32)
