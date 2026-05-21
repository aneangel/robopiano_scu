from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODE_ROOT = Path("/WAVE/projects/ECEN-524-Wi26/robopiano")
DEFAULT_SWEEP_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/sweeps")
DEFAULT_MIDI_PATH = DEFAULT_CODE_ROOT / "robopianist/music/data/rousseau/twinkle-twinkle-trimmed.mid"
DEFAULT_ENVIRONMENT = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
TARGET_THRESHOLDS: dict[str, float] = {
    "frame_f1": 0.60,
    "frame_precision": 0.60,
    "frame_recall": 0.55,
    "FP_max": 1400.0,
    "TP_min": 1550.0,
    "matched_press_events_min": 14.0,
}

DEFAULT_FIXED_ARGS: dict[str, Any] = {
    "control_timestep": 0.05,
    "threshold": 0.5,
    "seed": 0,
    "environment_name": DEFAULT_ENVIRONMENT,
    "active_window_last_s": 10.0,
    "active_window_preroll_s": 1.0,
    "active_window_postroll_s": 0.5,
    "interpolation_substeps": 10,
    "anchor_stride": 1,
    "solve_all_stride_anchors": True,
    "ik_max_nfev": 100,
    "fps": 200,
    "width": 640,
    "height": 480,
    "timing_tolerance_s": 0.15,
}

MAGNET_DEFAULTS: dict[str, Any] = {
    "active_press_weight": 1.2,
    "hover_weight": 0.10,
    "travel_weight": 0.03,
    "release_end_weight": 0.05,
    "inactive_clearance_weight": 1.0,
    "wrong_key_avoid_weight": 0.5,
    "key_press_depth": 0.0065,
    "clearance_height": 0.040,
    "press_lead_s": 0.010,
}

ASSIGNMENT_DEFAULTS: dict[str, Any] = {
    "distance_weight": 1.0,
    "same_finger_bonus": 0.02,
    "reassignment_penalty": 0.05,
    "finger_crossing_penalty": 0.10,
    "wrong_hand_penalty": 0.25,
    "large_jump_penalty": 0.05,
    "same_key_same_finger_bonus": 0.02,
}

MAGNET_GRID: dict[str, list[float]] = {
    "active_press_weight": [1.0, 1.2, 1.4, 1.6],
    "hover_weight": [0.05, 0.10, 0.15],
    "travel_weight": [0.00, 0.03, 0.06],
    "release_end_weight": [0.00, 0.05, 0.10],
    "inactive_clearance_weight": [0.5, 1.0, 1.5],
    "wrong_key_avoid_weight": [0.0, 0.5, 1.0],
    "key_press_depth": [0.0055, 0.0065, 0.0075],
    "clearance_height": [0.035, 0.040, 0.050],
    "press_lead_s": [0.000, 0.010, 0.020],
}

ASSIGNMENT_GRID: dict[str, list[float]] = {
    "distance_weight": [0.8, 1.0, 1.2],
    "same_finger_bonus": [0.0, 0.02, 0.05],
    "reassignment_penalty": [0.0, 0.05, 0.10],
    "finger_crossing_penalty": [0.0, 0.10, 0.25],
    "wrong_hand_penalty": [0.0, 0.25, 0.50],
    "large_jump_penalty": [0.0, 0.05, 0.10],
    "same_key_same_finger_bonus": [0.0, 0.02, 0.05],
}

PARAMETER_GRID: dict[str, list[float]] = {**MAGNET_GRID, **ASSIGNMENT_GRID}

SWEEP_KEYS = (
    "active_press_weight",
    "hover_weight",
    "travel_weight",
    "release_end_weight",
    "inactive_clearance_weight",
    "wrong_key_avoid_weight",
    "key_press_depth",
    "clearance_height",
    "press_lead_s",
    "distance_weight",
    "same_finger_bonus",
    "reassignment_penalty",
    "finger_crossing_penalty",
    "wrong_hand_penalty",
    "large_jump_penalty",
    "same_key_same_finger_bonus",
)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_signature(config: dict[str, Any]) -> str:
    key = json.dumps({name: config[name] for name in SWEEP_KEYS}, sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def dedupe_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for config in configs:
        digest = config_signature(config)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(config)
    return unique


def adaptive_objective(row: dict[str, Any]) -> float:
    f1 = float(row.get("frame_f1") or 0.0)
    precision = float(row.get("frame_precision") or 0.0)
    recall = float(row.get("frame_recall") or 0.0)
    tp = float(row.get("TP") or 0.0)
    fp = float(row.get("FP") or 0.0)
    matched = float(row.get("matched_press_events") or 0.0)
    timing_p95 = float(row.get("timing_abs_error_p95_s") or 0.15)

    score = 0.0
    score += 6.0 * f1
    score += 4.0 * precision
    score += 3.0 * recall
    score += 0.003 * tp
    score += 0.08 * matched
    score -= 0.0015 * fp
    score -= 0.5 * timing_p95

    if precision < TARGET_THRESHOLDS["frame_precision"]:
        score -= 3.0 * (TARGET_THRESHOLDS["frame_precision"] - precision)
    if recall < TARGET_THRESHOLDS["frame_recall"]:
        score -= 2.0 * (TARGET_THRESHOLDS["frame_recall"] - recall)
    if fp > TARGET_THRESHOLDS["FP_max"]:
        score -= 0.002 * (fp - TARGET_THRESHOLDS["FP_max"])
    if tp < TARGET_THRESHOLDS["TP_min"]:
        score -= 0.003 * (TARGET_THRESHOLDS["TP_min"] - tp)
    if matched < TARGET_THRESHOLDS["matched_press_events_min"]:
        score -= 0.2 * (TARGET_THRESHOLDS["matched_press_events_min"] - matched)
    return float(score)


def is_balanced_target_hit(row: dict[str, Any]) -> bool:
    return (
        float(row.get("frame_f1") or 0.0) >= TARGET_THRESHOLDS["frame_f1"]
        and float(row.get("frame_precision") or 0.0) >= TARGET_THRESHOLDS["frame_precision"]
        and float(row.get("frame_recall") or 0.0) >= TARGET_THRESHOLDS["frame_recall"]
        and float(row.get("FP") or 1e9) < TARGET_THRESHOLDS["FP_max"]
        and float(row.get("TP") or 0.0) > TARGET_THRESHOLDS["TP_min"]
        and float(row.get("matched_press_events") or 0.0) > TARGET_THRESHOLDS["matched_press_events_min"]
    )


def stage_output_root(base_root: str | Path, stage_name: str) -> Path:
    return Path(base_root).expanduser().resolve() / stage_name


def write_manifest(output_root: Path, configs: list[dict[str, Any]], *, manifest_name: str = "sweep_manifest.json") -> dict[str, Any]:
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for index, config in enumerate(configs):
        json_dump(config_root / f"{index:03d}.json", config)
    manifest = {
        "output_root": str(output_root),
        "config_root": str(config_root),
        "config_count": len(configs),
        "configs": [str(config_root / f"{index:03d}.json") for index in range(len(configs))],
    }
    json_dump(output_root / manifest_name, manifest)
    return manifest


def run_name_from_config(index: int, config: dict[str, Any]) -> str:
    parts = [
        f"{index:03d}",
        f"apw{config['active_press_weight']:.2f}".replace(".", "p"),
        f"hov{config['hover_weight']:.2f}".replace(".", "p"),
        f"trv{config['travel_weight']:.2f}".replace(".", "p"),
        f"rel{config['release_end_weight']:.2f}".replace(".", "p"),
        f"icl{config['inactive_clearance_weight']:.1f}".replace(".", "p"),
        f"wka{config['wrong_key_avoid_weight']:.1f}".replace(".", "p"),
        f"kpd{config['key_press_depth']:.4f}".replace(".", "p"),
        f"clr{config['clearance_height']:.3f}".replace(".", "p"),
        f"lead{config['press_lead_s']:.3f}".replace(".", "p"),
        f"df{config['distance_weight']:.2f}".replace(".", "p"),
        f"sfb{config['same_finger_bonus']:.2f}".replace(".", "p"),
        f"rep{config['reassignment_penalty']:.2f}".replace(".", "p"),
        f"fcp{config['finger_crossing_penalty']:.2f}".replace(".", "p"),
        f"whp{config['wrong_hand_penalty']:.2f}".replace(".", "p"),
        f"ljp{config['large_jump_penalty']:.2f}".replace(".", "p"),
    ]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "imp_sweep_" + "_".join(parts))
