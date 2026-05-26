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

from ballade.online_env import BalladeOnlineEnvConfig  # noqa: E402
from ballade.rollout import run_residual_controller_rollout  # noqa: E402
from rp1m_simulator.simulator import find_high_f1_examples, load_rp1m_trajectory  # noqa: E402


def _examples(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.example:
        out = []
        for item in args.example:
            song_key, demo = item.rsplit(":", 1)
            out.append((song_key, int(demo)))
        return out[: args.max_demos]
    found = find_high_f1_examples(args.rp1m_root, max_songs=args.scan_songs, examples=args.max_demos, min_recorded_f1=0.0)
    return [(song_key, demo_id) for song_key, demo_id, _score in found[: args.max_demos]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--search-fallback", default="none")
    parser.add_argument("--max-demos", type=int, default=5)
    parser.add_argument("--max-source-steps", type=int, default=500)
    parser.add_argument("--scan-songs", type=int, default=8)
    parser.add_argument("--example", action="append", default=[])
    parser.add_argument("--source-dt", type=float, default=0.05)
    parser.add_argument("--control-dt", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, (song_key, demo_id) in enumerate(_examples(args)):
        traj = load_rp1m_trajectory(args.rp1m_root, song_key, demo_id, include_reference_piano_states=True)
        summary = run_residual_controller_rollout(
            trajectory=traj,
            checkpoint=args.checkpoint,
            output_dir=output_root / f"demo_{index:03d}",
            source_dt=args.source_dt,
            control_dt=args.control_dt,
            max_source_steps=args.max_source_steps,
            env_config=BalladeOnlineEnvConfig(source_dt=args.source_dt, control_dt=args.control_dt),
            search_fallback=args.search_fallback,
            device=args.device,
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))
    (output_root / "rollout_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
