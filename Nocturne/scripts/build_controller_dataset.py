from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from nocturne.controller_dataset import build_controller_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Nocturne controller dataset from a stitched trajectory.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--press-window-radius", type=int, default=2)
    parser.add_argument("--press-weight", type=float, default=5.0)
    args = parser.parse_args()
    summary = build_controller_dataset(
        args.trajectory,
        args.output_root,
        press_window_radius=int(args.press_window_radius),
        press_weight=float(args.press_weight),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
