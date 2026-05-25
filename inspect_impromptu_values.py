from pathlib import Path
import sys

import numpy as np


def main() -> None:
    path = Path(sys.argv[1])
    with np.load(path, allow_pickle=True) as data:
        assignments = data["assignments"]
        targets = data["fingertip_targets"]
        way_tips = data["waypoint_fingertips"]
        frames = data["waypoint_frames"]
        print("assignments min/max", int(assignments.min()), int(assignments.max()))
        print("targets finite by row first10", np.isfinite(targets[:10]).all(axis=2).sum(axis=1).tolist())
        print("targets nan count", int(np.isnan(targets).sum()))
        print("way_tips nan count", int(np.isnan(way_tips).sum()))
        for idx in range(min(8, assignments.shape[0])):
            active = np.flatnonzero(assignments[idx] >= 0).tolist()
            print("row", idx, "frame", int(frames[idx]), "active", active, "keys", assignments[idx, active].tolist())
            print("target", np.round(targets[idx, active], 5).tolist())
            print("tips", np.round(way_tips[idx, active], 5).tolist())


if __name__ == "__main__":
    main()
