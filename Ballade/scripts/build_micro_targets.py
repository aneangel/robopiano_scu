from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO = ROOT.parent
for path in (SRC, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ballade.targets import build_micro_targets  # noqa: E402
from rp1m_simulator import load_rp1m_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--song-key", required=True)
    parser.add_argument("--demo-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-dt", type=float, default=0.05)
    parser.add_argument("--control-dt", type=float, default=0.005)
    parser.add_argument("--max-source-steps", type=int, default=None)
    args = parser.parse_args()
    traj = load_rp1m_trajectory(args.rp1m_root, args.song_key, args.demo_id)
    hand = traj.hand_joints if args.max_source_steps is None else traj.hand_joints[: args.max_source_steps]
    goals = traj.goals if args.max_source_steps is None else traj.goals[: args.max_source_steps]
    tips = traj.hand_fingertips
    if tips is not None and args.max_source_steps is not None:
        tips = tips[: args.max_source_steps]
    targets = build_micro_targets(
        hand,
        goals_20hz=goals,
        fingertips_20hz=tips,
        source_dt=args.source_dt,
        control_dt=args.control_dt,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target_q_micro=targets.target_q_micro,
        target_qvel_micro=targets.target_qvel_micro,
        goal_key_mask=targets.goal_key_mask,
        source_dt=np.asarray(args.source_dt, dtype=np.float32),
        control_dt=np.asarray(args.control_dt, dtype=np.float32),
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
