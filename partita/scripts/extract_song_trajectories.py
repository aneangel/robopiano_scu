from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import numpy as np

from partita.data.rp1m_loader import available_arrays, open_rp1m_root, read_trajectories, read_trajectory
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected RP1M trajectories into compact NPZ files.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = ensure_dir(output_root(config) / "data" / experiment_name(config))
    selection = load_json(data_dir / "selection.json")
    train_ids = [int(x) for x in load_json(data_dir / "train_trajectory_ids.json")]
    target = load_json(data_dir / "reconstruction_target.json")
    source_song_name = selection.get("source_song_name", selection["song_name"])
    target_song_name = selection.get("target_song_name", target.get("song_name", source_song_name))

    root = open_rp1m_root(selection["rp1m_root"])
    source_group = root[source_song_name]
    target_group = root[target_song_name]
    keep_arrays = {"actions", "goals", "piano_states", "hand_joints", "hand_fingertips"}
    source_arrays = [a for a in available_arrays(source_group) if a in keep_arrays]
    target_arrays = [a for a in available_arrays(target_group) if a in keep_arrays]
    if "actions" not in source_arrays:
        raise RuntimeError(f"Source song {source_song_name} does not contain required actions array.")
    if "actions" not in target_arrays:
        raise RuntimeError(f"Target song {target_song_name} does not contain required actions array.")

    selected = read_trajectories(source_group, train_ids, arrays=source_arrays)
    target_data = read_trajectory(target_group, int(target["trajectory_id"]), arrays=target_arrays)
    np.savez_compressed(data_dir / "selected_trajectories.npz", **selected)
    np.savez_compressed(data_dir / "target_trajectory.npz", **target_data)
    summary = {
        "song_name": source_song_name,
        "source_song_name": source_song_name,
        "target_song_name": target_song_name,
        "is_cross_song": source_song_name != target_song_name,
        "source_available_arrays": source_arrays,
        "target_available_arrays": target_arrays,
        "available_arrays": source_arrays,
        "num_training_trajectories": len(train_ids),
        "train_trajectory_ids": train_ids,
        "target_trajectory_id": int(target["trajectory_id"]),
        "selected_shapes": {k: list(v.shape) for k, v in selected.items() if hasattr(v, "shape")},
        "target_shapes": {k: list(v.shape) for k, v in target_data.items() if hasattr(v, "shape")},
    }
    save_json(data_dir / "song_summary.json", summary)
    print(f"Saved selected and target trajectory NPZ files to {data_dir}")


if __name__ == "__main__":
    main()
