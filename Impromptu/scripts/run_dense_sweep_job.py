#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from impromptu.active_window import crop_active_window  # noqa: E402
from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.evaluation import evaluate_trajectory_payload  # noqa: E402
from impromptu.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
from render_hand_state_playback import render_dense_playback  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one dense Impromptu sweep config end to end.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = _load_config(config_path)
    run_root = config_path.parent.parent / payload["run_name"]
    render_dir = run_root / "render_dense"
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_save_json(run_root / "config.json", payload)

    target_keys, midi_meta = load_target_keys_from_midi(
        payload["midi_path"],
        control_timestep=float(payload["control_timestep"]),
    )
    crop = crop_active_window(
        target_keys,
        dt=float(payload["control_timestep"]),
        threshold=float(payload["threshold"]),
        active_window_last_s=float(payload["active_window_last_s"]),
        active_window_preroll_s=float(payload["active_window_preroll_s"]),
        active_window_postroll_s=float(payload["active_window_postroll_s"]),
    )
    plan_config = ImpromptuConfig(
        control_timestep=float(payload["control_timestep"]),
        threshold=float(payload["threshold"]),
        seed=int(payload["seed"]),
        environment_name=str(payload["environment_name"]),
        interpolation_substeps=int(payload["interpolation_substeps"]),
        clearance_height=float(payload["clearance_height"]),
        key_press_depth=float(payload["key_press_depth"]),
        inactive_clearance_weight=float(payload["inactive_clearance_weight"]),
        hover_weight=float(payload["hover_weight"]),
        release_end_weight=float(payload["release_end_weight"]),
        travel_weight=float(payload["travel_weight"]),
        press_weight=float(payload["active_press_weight"]),
        press_lead_s=float(payload["press_lead_s"]),
        wrong_key_avoid_weight=float(payload["wrong_key_avoid_weight"]),
        anchor_stride=int(payload["anchor_stride"]),
        solve_contact_window_only=False,
        include_midpoint_anchors=True,
        ik_max_nfev=int(payload["ik_max_nfev"]),
        output_root=str(run_root.parent),
        assignment_distance_weight=float(payload["distance_weight"]),
        same_finger_bonus=float(payload["same_finger_bonus"]),
        reassignment_penalty=float(payload["reassignment_penalty"]),
        finger_crossing_penalty=float(payload["finger_crossing_penalty"]),
        wrong_hand_penalty=float(payload["wrong_hand_penalty"]),
        large_jump_penalty=float(payload["large_jump_penalty"]),
        same_key_same_finger_bonus=float(payload["same_key_same_finger_bonus"]),
    )
    plan = plan_target_keys(crop.target_keys, config=plan_config)
    npz_payload = plan.npz_payload()
    npz_payload.update(
        active_window_crop_start_frame=crop.start_frame,
        active_window_crop_end_frame=crop.end_frame,
        active_window_original_steps=crop.metadata["original_steps"],
        active_window_cropped_steps=crop.metadata["cropped_steps"],
    )
    trajectory_path = run_root / "trajectory.npz"
    atomic_save_npz(trajectory_path, **npz_payload)
    metadata = {
        "run_dir": str(run_root),
        "trajectory_npz": str(trajectory_path),
        "source_type": "midi",
        "midi_path": str(Path(payload["midi_path"]).expanduser().resolve()),
        "midi": midi_meta,
        "active_window": crop.metadata,
        **plan.metadata,
    }
    atomic_save_json(run_root / "metadata.json", metadata)
    atomic_save_json(run_root / "active_window_summary.json", crop.metadata)

    metrics = evaluate_trajectory_payload(npz_payload)
    atomic_save_json(run_root / "metrics.json", metrics)
    render_summary = render_dense_playback(
        trajectory_npz=trajectory_path,
        output_dir=render_dir,
        environment_name=str(payload["environment_name"]),
        control_timestep=float(payload["control_timestep"]),
        interpolation_substeps=int(payload["interpolation_substeps"]),
        fps=int(payload["fps"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        seed=int(payload["seed"]),
        threshold=float(payload["threshold"]),
        active_window_last_s=None,
    )
    for name in ("render_summary.json", "score.json", "lag_sweep.json", "threshold_sweep.json", "fp_by_key.csv", "fn_by_key.csv"):
        _copy_if_exists(render_dir / name, run_root / name)
    print(json.dumps({
        "run_dir": str(run_root),
        "score": render_summary["score"],
        "timing_abs_error_p95_s": render_summary["score"].get("timing_abs_error_p95_s"),
    }, indent=2))


if __name__ == "__main__":
    main()
