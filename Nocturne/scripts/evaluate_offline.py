from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import numpy as np  # noqa: E402

from nocturne.io import save_json  # noqa: E402
from nocturne.offline_eval import evaluate_rollout  # noqa: E402
from nocturne.seams import seam_jump_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Nocturne stitched trajectory offline.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    path = Path(args.trajectory)
    with np.load(path, allow_pickle=False) as data:
        goals = np.asarray(data["goals"], dtype=np.float32)
        piano_states = np.asarray(data["piano_states"], dtype=np.float32)
        dt = float(np.asarray(data["dt"]).reshape(())) if "dt" in data else 0.05
        source_demo = np.asarray(data["source_demo_id_per_frame"], dtype=np.int64) if "source_demo_id_per_frame" in data else None
        payload = {name: np.asarray(data[name]) for name in data.files if name in {"hand_joints", "hand_fingertips", "actions"}}
    metrics = evaluate_rollout(goals, piano_states, dt=dt, threshold=float(args.threshold))
    if source_demo is not None and source_demo.size > 1:
        seams = np.flatnonzero(source_demo[1:] != source_demo[:-1]).astype(int).tolist()
        metrics.update(seam_jump_metrics(payload, [index + 1 for index in seams]))
    out = Path(args.output) if args.output else path.with_name("offline_eval.json")
    save_json(out, metrics)
    print(__import__("json").dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
