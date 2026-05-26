from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO = ROOT.parent
for path in (SRC, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ballade.online_env import BalladeOnlineEnvConfig  # noqa: E402
from ballade.rollout import run_online_jacobian_rollout  # noqa: E402
from rp1m_simulator.simulator import find_high_f1_examples, load_rp1m_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-demos", type=int, default=20)
    parser.add_argument("--max-source-steps", type=int, default=500)
    parser.add_argument("--source-dt", type=float, default=0.05)
    parser.add_argument("--control-dt", type=float, default=0.005)
    parser.add_argument("--search-mode", default="event_triggered")
    parser.add_argument("--scan-songs", type=int, default=20)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    examples = find_high_f1_examples(args.rp1m_root, max_songs=args.scan_songs, examples=args.max_demos, min_recorded_f1=0.0)
    summaries = []
    for index, (song_key, demo_id, _score) in enumerate(examples[: args.max_demos]):
        traj = load_rp1m_trajectory(args.rp1m_root, song_key, demo_id, include_reference_piano_states=True)
        summary = run_online_jacobian_rollout(
            trajectory=traj,
            output_dir=output_root / f"demo_{index:03d}",
            source_dt=args.source_dt,
            control_dt=args.control_dt,
            max_source_steps=args.max_source_steps,
            env_config=BalladeOnlineEnvConfig(source_dt=args.source_dt, control_dt=args.control_dt),
            collect_teacher=True,
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))
    (output_root / "teacher_collection_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
