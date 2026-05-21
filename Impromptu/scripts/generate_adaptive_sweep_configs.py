#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any

from sweep_common import (
    ASSIGNMENT_DEFAULTS,
    DEFAULT_CODE_ROOT,
    DEFAULT_FIXED_ARGS,
    DEFAULT_MIDI_PATH,
    DEFAULT_SWEEP_ROOT,
    MAGNET_DEFAULTS,
    PARAMETER_GRID,
    adaptive_objective,
    config_signature,
    dedupe_configs,
    is_balanced_target_hit,
    json_load,
    run_name_from_config,
    stage_output_root,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adaptive stage-2 dense sweep configs.")
    parser.add_argument("--base-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--stage-name", default="stage2_adaptive")
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--midi-path", default=str(DEFAULT_MIDI_PATH))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--proposals", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1524)
    return parser.parse_args()


def _coerce_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _read_rows(base_root: Path) -> list[dict[str, Any]]:
    csv_path = base_root / "sweep_results.csv"
    if csv_path.is_file():
        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cooked = dict(row)
                for key, value in list(cooked.items()):
                    maybe = _coerce_float(value)
                    if maybe is not None:
                        cooked[key] = maybe
                cooked["adaptive_objective"] = adaptive_objective(cooked)
                cooked["meets_balanced_target"] = is_balanced_target_hit(cooked)
                rows.append(cooked)
        return rows

    rows = []
    for run_dir in base_root.iterdir():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        score_path = run_dir / "score.json"
        if not (config_path.is_file() and score_path.is_file()):
            continue
        config = json_load(config_path)
        score = json_load(score_path)
        row = {
            "run_name": run_dir.name,
            "TP": score.get("frame_true_positives"),
            "FP": score.get("frame_false_positives"),
            "FN": score.get("frame_false_negatives"),
            "frame_f1": score.get("frame_f1"),
            "frame_precision": score.get("frame_precision"),
            "frame_recall": score.get("frame_recall"),
            "matched_press_events": score.get("matched_press_events"),
            "mispresses": score.get("mispresses"),
            "missed_key_presses": score.get("missed_key_presses"),
            "timing_abs_error_p95_s": score.get("timing_abs_error_p95_s"),
        }
        for key in PARAMETER_GRID:
            row[key] = config.get(key)
        row["adaptive_objective"] = adaptive_objective(row)
        row["meets_balanced_target"] = is_balanced_target_hit(row)
        rows.append(row)
    return rows


def _base_config_from_row(row: dict[str, Any], *, code_root: Path, midi_path: Path) -> dict[str, Any]:
    merged = {
        **DEFAULT_FIXED_ARGS,
        **MAGNET_DEFAULTS,
        **ASSIGNMENT_DEFAULTS,
        "code_root": str(code_root),
        "midi_path": str(midi_path),
    }
    for key in PARAMETER_GRID:
        if row.get(key) is not None:
            merged[key] = float(row[key])
    return merged


def _neighbor_values(key: str, value: float) -> list[float]:
    grid = PARAMETER_GRID[key]
    if value not in grid:
        nearest = min(range(len(grid)), key=lambda idx: abs(grid[idx] - value))
    else:
        nearest = grid.index(value)
    values = [grid[nearest]]
    if nearest - 1 >= 0:
        values.append(grid[nearest - 1])
    if nearest + 1 < len(grid):
        values.append(grid[nearest + 1])
    return sorted(set(float(v) for v in values))


def _propose_local_neighbors(parent: dict[str, Any], rng: random.Random, count: int) -> list[dict[str, Any]]:
    keys = list(PARAMETER_GRID)
    proposals: list[dict[str, Any]] = []
    for _ in range(max(count, 1) * 6):
        proposal = dict(parent)
        mutate_count = 2 if rng.random() < 0.65 else 3
        for key in rng.sample(keys, k=mutate_count):
            choices = [value for value in _neighbor_values(key, float(parent[key])) if value != float(parent[key])]
            if choices:
                proposal[key] = float(rng.choice(choices))
        proposals.append(proposal)
        if len(proposals) >= count:
            break
    return proposals


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root).expanduser().resolve()
    output_root = stage_output_root(base_root, str(args.stage_name))
    code_root = Path(args.code_root).expanduser().resolve()
    midi_path = Path(args.midi_path).expanduser().resolve()
    rows = _read_rows(base_root)
    if not rows:
        raise RuntimeError(f"No completed sweep results found under {base_root}")
    rows.sort(key=lambda row: float(row.get("adaptive_objective") or -1e9), reverse=True)

    prior_signatures: set[str] = set()
    for config_path in base_root.glob("configs/*.json"):
        prior_signatures.add(config_signature(json_load(config_path)))

    parents = rows[: max(int(args.top_k), 1)]
    rng = random.Random(int(args.seed))
    proposals: list[dict[str, Any]] = []
    per_parent = max(int(args.proposals) // max(len(parents), 1), 1)
    for parent_index, row in enumerate(parents):
        parent = _base_config_from_row(row, code_root=code_root, midi_path=midi_path)
        local = _propose_local_neighbors(parent, rng, per_parent + 2)
        if is_balanced_target_hit(row):
            local.insert(0, parent)
        for proposal in local:
            proposal["adaptive_parent_run_name"] = str(row.get("run_name"))
            proposal["adaptive_parent_objective"] = float(row.get("adaptive_objective") or 0.0)
            proposal["adaptive_parent_rank"] = int(parent_index)
            proposals.append(proposal)

    unique = []
    seen = set(prior_signatures)
    for proposal in dedupe_configs(proposals):
        digest = config_signature(proposal)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(proposal)
        if len(unique) >= int(args.proposals):
            break
    if not unique:
        raise RuntimeError("Adaptive proposal set is empty after dedupe; widen the neighborhood or reduce top-k.")

    for index, config in enumerate(unique):
        config["run_name"] = run_name_from_config(index, config).replace("imp_sweep_", "imp_adapt_")
        config["config_index"] = index
        config["adaptive_stage"] = str(args.stage_name)
    manifest = write_manifest(output_root, unique, manifest_name="adaptive_manifest.json")
    manifest["parent_rows"] = [
        {
            "run_name": row.get("run_name"),
            "adaptive_objective": float(row.get("adaptive_objective") or 0.0),
            "meets_balanced_target": bool(row.get("meets_balanced_target")),
        }
        for row in parents
    ]
    (output_root / "adaptive_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
