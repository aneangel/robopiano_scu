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

from ballade.online_env import BalladeOnlineEnvConfig, BalladeOnlineEnv  # noqa: E402


def _first_example(rp1m_root: str) -> tuple[str, int]:
    import zarr

    root = zarr.open(rp1m_root, mode="r")
    song_key = sorted(root.keys())[0]
    return str(song_key), 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--song-key", default=None)
    parser.add_argument("--demo-id", type=int, default=0)
    parser.add_argument("--output-dir", default="/tmp/ballade_inspect")
    args = parser.parse_args()
    song_key, demo_id = (args.song_key, args.demo_id) if args.song_key else _first_example(args.rp1m_root)
    env = BalladeOnlineEnv.from_rp1m(
        rp1m_root=args.rp1m_root,
        song_key=song_key,
        demo_id=demo_id,
        output_dir=args.output_dir,
        config=BalladeOnlineEnvConfig(),
    )
    try:
        obs = env.reset()
        spec = env.action_spec()
        payload = {
            "song_key": song_key,
            "demo_id": demo_id,
            "action_shape": list(spec.shape),
            "action_min": spec.minimum.tolist(),
            "action_max": spec.maximum.tolist(),
            "q_dim": int(obs.q.size),
            "qvel_dim": int(obs.qvel.size),
            "fingertip_dim": 0 if obs.fingertips is None else int(obs.fingertips.size),
            "piano_activation_dim": int(obs.piano_activation.size),
            "load_info": env.load_info,
            "hand_anchor_calibration": env.hand_anchor_calibration,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
