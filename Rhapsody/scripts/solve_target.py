from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Rhapsody" / "src"))

from rhapsody.solver import RhapsodyIKSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve one fingertip IK target with Rhapsody.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-npy", type=Path, required=True)
    parser.add_argument("--active-mask-npy", type=Path, default=None)
    parser.add_argument("--previous-qpos-npy", type=Path, default=None)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = np.load(args.target_npy)
    active_mask = np.load(args.active_mask_npy) if args.active_mask_npy is not None else None
    previous_qpos = np.load(args.previous_qpos_npy) if args.previous_qpos_npy is not None else None
    solver = RhapsodyIKSolver.from_checkpoint(args.checkpoint, device=args.device)
    solution = solver.solve(
        target,
        active_mask=active_mask,
        previous_qpos=previous_qpos,
        refinement_steps=args.refinement_steps,
    )
    print(json.dumps(solution.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
