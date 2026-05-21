#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from sweep_common import (
    ASSIGNMENT_DEFAULTS,
    ASSIGNMENT_GRID,
    DEFAULT_CODE_ROOT,
    DEFAULT_FIXED_ARGS,
    DEFAULT_MIDI_PATH,
    DEFAULT_SWEEP_ROOT,
    MAGNET_GRID,
    MAGNET_DEFAULTS,
    PARAMETER_GRID,
    dedupe_configs,
    json_dump,
    run_name_from_config,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense Impromptu sweep configs.")
    parser.add_argument("--output-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--midi-path", default=str(DEFAULT_MIDI_PATH))
    parser.add_argument("--random-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=524)
    return parser.parse_args()


def _sample_lhs(grid: dict[str, list[float]], count: int, rng: random.Random) -> list[dict[str, float]]:
    keys = list(grid)
    slots = {key: list(range(count)) for key in keys}
    for key in keys:
        rng.shuffle(slots[key])
    rows: list[dict[str, float]] = []
    for index in range(count):
        row = {}
        for key in keys:
            values = grid[key]
            bucket = slots[key][index]
            row[key] = float(values[bucket % len(values)])
        rows.append(row)
    return rows


def _handpicked_configs() -> list[dict[str, float]]:
    base = {**MAGNET_DEFAULTS, **ASSIGNMENT_DEFAULTS}
    return [
        base,
        {**base, "active_press_weight": 1.0, "hover_weight": 0.05, "travel_weight": 0.00, "press_lead_s": 0.000},
        {**base, "active_press_weight": 1.4, "hover_weight": 0.10, "travel_weight": 0.03, "wrong_key_avoid_weight": 1.0},
        {**base, "active_press_weight": 1.2, "release_end_weight": 0.10, "inactive_clearance_weight": 1.5, "wrong_hand_penalty": 0.50},
        {**base, "active_press_weight": 1.6, "key_press_depth": 0.0075, "clearance_height": 0.050, "large_jump_penalty": 0.10},
    ]
def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    sampled = _sample_lhs(PARAMETER_GRID, max(int(args.random_count), 1), rng)
    configs: list[dict[str, Any]] = []
    for row in sampled + _handpicked_configs():
        merged = {
            **DEFAULT_FIXED_ARGS,
            **MAGNET_DEFAULTS,
            **ASSIGNMENT_DEFAULTS,
            **row,
            "code_root": str(Path(args.code_root).expanduser()),
            "midi_path": str(Path(args.midi_path).expanduser()),
        }
        configs.append(merged)
    configs = dedupe_configs(configs)
    for index, config in enumerate(configs):
        config["run_name"] = run_name_from_config(index, config)
        config["config_index"] = index
    manifest = write_manifest(output_root, configs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
