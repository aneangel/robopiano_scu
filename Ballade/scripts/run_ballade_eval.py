from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from ballade.config import load_yaml_config  # noqa: E402
from ballade.online_env import BalladeOnlineEnvConfig  # noqa: E402
from ballade.rollout import run_online_jacobian_rollout  # noqa: E402
from ballade.train import train_residual_controller  # noqa: E402
from rp1m_simulator.simulator import find_high_f1_examples, load_rp1m_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    iterations = args.iterations or int(config.get("iterations", 1))
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    max_demos = int(config.get("max_demos", 8))
    max_source_steps = int(config.get("max_source_steps", 500))
    source_dt = float(config.get("source_dt", 0.05))
    control_dt = float(config.get("control_dt", 0.005))
    examples = find_high_f1_examples(args.rp1m_root, max_songs=max(max_demos, 8), examples=max_demos, min_recorded_f1=0.0)
    summaries = []
    for iteration in range(int(iterations)):
        iteration_root = output / f"iteration_{iteration + 1:02d}"
        collection_root = iteration_root / "teacher_data_collection"
        collection_root.mkdir(parents=True, exist_ok=True)
        rollout_summaries = []
        for demo_index, (song_key, demo_id, _score) in enumerate(examples[:max_demos]):
            traj = load_rp1m_trajectory(args.rp1m_root, song_key, demo_id, include_reference_piano_states=True)
            rollout_summaries.append(
                run_online_jacobian_rollout(
                    trajectory=traj,
                    output_dir=collection_root / f"demo_{demo_index:03d}",
                    source_dt=source_dt,
                    control_dt=control_dt,
                    max_source_steps=max_source_steps,
                    env_config=BalladeOnlineEnvConfig(source_dt=source_dt, control_dt=control_dt),
                    collect_teacher=True,
                )
            )
        train_summary = train_residual_controller(
            teacher_data=collection_root,
            output_root=iteration_root / "residual_model",
            epochs=int((config.get("training") or {}).get("epochs", 10)),
            batch_size=int((config.get("training") or {}).get("batch_size", 2048)),
            learning_rate=float((config.get("training") or {}).get("learning_rate", 3e-4)),
        )
        summaries.append(
            {
                "iteration": iteration + 1,
                "rollouts": rollout_summaries,
                "train_summary": train_summary,
            }
        )
        print(json.dumps(summaries[-1], sort_keys=True))
    (output / "run_c_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
