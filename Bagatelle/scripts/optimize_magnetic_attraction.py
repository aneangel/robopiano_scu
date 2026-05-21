#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "Bagatelle" / "src", REPO_ROOT / "Bagatelle" / "scripts", REPO_ROOT / "Intermezzo" / "src", REPO_ROOT / "partita" / "src", REPO_ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from intermezzo.io import atomic_save_json, create_unique_run_dir  # noqa: E402
from evaluate_magnetic_bagatelle import load_payload, magnetic_plan, rollout_and_render  # noqa: E402


@dataclass(frozen=True)
class MagnetParams:
    magnet_radius: float
    magnet_sigma: float
    magnet_gain: float
    magnet_max_xy_step: float
    magnet_start_fraction: float
    magnet_power: float
    ik_damping: float
    ik_max_delta_q: float
    ik_iterations_per_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Intermezzo magnetic attraction parameters on a Bagatelle trajectory.")
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default="magnetic_attraction_opt")
    parser.add_argument("--label-prefix", default="candidate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--timing-tolerance-s", type=float, default=0.15)
    parser.add_argument("--settle-steps", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--render-best", action="store_true")
    parser.add_argument("--render-every", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def candidate_params() -> list[MagnetParams]:
    base = MagnetParams(0.12, 0.06, 1.5, 0.01, 0.40, 1.0, 1e-3, 0.03, 1)
    return [
        MagnetParams(0.0, 0.06, 0.0, 0.0, 0.40, 1.0, 1e-3, 0.03, 1),
        MagnetParams(0.08, 0.04, 0.4, 0.003, 0.55, 1.5, 1e-3, 0.015, 1),
        MagnetParams(0.10, 0.05, 0.7, 0.005, 0.50, 1.3, 1e-3, 0.020, 1),
        MagnetParams(0.12, 0.06, 1.0, 0.006, 0.45, 1.2, 1e-3, 0.025, 1),
        base,
        MagnetParams(0.14, 0.07, 1.2, 0.008, 0.45, 1.0, 2e-3, 0.020, 1),
        MagnetParams(0.10, 0.05, 1.2, 0.004, 0.60, 2.0, 1e-3, 0.015, 1),
        MagnetParams(0.16, 0.08, 0.8, 0.006, 0.55, 1.5, 2e-3, 0.020, 2),
        MagnetParams(0.08, 0.04, 1.5, 0.003, 0.65, 2.0, 1e-3, 0.012, 2),
        MagnetParams(0.12, 0.04, 0.6, 0.004, 0.70, 2.5, 1e-3, 0.015, 2),
    ]


def namespace_for(args: argparse.Namespace, params: MagnetParams, *, label: str, render: bool) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(asdict(params))
    values.update({"label": label, "render": render, "use_dense": True, "fps": 20})
    return argparse.Namespace(**values)


def objective(rollout: dict[str, Any]) -> float:
    score = rollout["score"]
    matched = float(score.get("matched_press_events") or 0.0)
    mispresses = float(score.get("mispresses") or 0.0)
    missed = float(score.get("missed_key_presses") or 0.0)
    f1 = float(score.get("frame_f1") or 0.0)
    dist = rollout.get("selected_fingertip_key_distance", {}).get("xy_distance", {})
    xy_mean = float(dist.get("mean_m") or 0.0)
    return matched + 2.0 * f1 - 0.12 * mispresses - 0.05 * missed - 15.0 * xy_mean


def main() -> None:
    args = parse_args()
    payload = load_payload(args.trajectory_npz)
    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=args.run_name, prefix="magnet_opt")
    history: list[dict[str, Any]] = []
    params_list = candidate_params()[: max(int(args.max_candidates), 1)]
    for index, params in enumerate(params_list):
        label = f"{args.label_prefix}_{index:02d}"
        cand_dir = run_dir / label
        cand_args = namespace_for(args, params, label=label, render=False)
        magnetic = magnetic_plan(payload, cand_args, cand_dir)
        dt = float(np.asarray(magnetic["dense_control_timestep"]))
        rollout = rollout_and_render(magnetic["planned_hand_joints_dense"], magnetic["target_keys_dense"], cand_dir, cand_args, control_timestep=dt)
        row = {"label": label, "params": asdict(params), "objective": objective(rollout), "rollout": rollout, "run_dir": str(cand_dir)}
        history.append(row)
        atomic_save_json(run_dir / "history.json", {"history": history})
        s = rollout["score"]
        d = rollout.get("selected_fingertip_key_distance", {}).get("xy_distance", {})
        print(f"{label} obj={row['objective']:.4f} matched={s.get('matched_press_events')}/{s.get('target_press_events')} missed={s.get('missed_key_presses')} mispress={s.get('mispresses')} f1={s.get('frame_f1')} xy_mean={d.get('mean_m')}", flush=True)
    history.sort(key=lambda row: float(row["objective"]), reverse=True)
    best = history[0]
    best_render = None
    if bool(args.render_best):
        params = MagnetParams(**best["params"])
        label = f"{best['label']}_best_render"
        cand_dir = run_dir / label
        cand_args = namespace_for(args, params, label=label, render=True)
        cand_args.render_every = int(args.render_every)
        cand_args.width = int(args.width)
        cand_args.height = int(args.height)
        magnetic = magnetic_plan(payload, cand_args, cand_dir)
        dt = float(np.asarray(magnetic["dense_control_timestep"]))
        best_render = rollout_and_render(magnetic["planned_hand_joints_dense"], magnetic["target_keys_dense"], cand_dir, cand_args, control_timestep=dt)
    summary = {"run_dir": str(run_dir), "best": best, "best_render": best_render, "history": history}
    atomic_save_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), "best": {"label": best["label"], "objective": best["objective"], "params": best["params"]}, "video_path": None if best_render is None else best_render.get("video_path")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
