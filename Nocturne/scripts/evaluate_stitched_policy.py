from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from nocturne.rollout import evaluate_policy_online  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Nocturne action policy online in RoboPianist.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--song-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_policy_online(
        checkpoint=args.checkpoint,
        trajectory=args.trajectory,
        output_root=args.output_root,
        song_name=args.song_name,
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
