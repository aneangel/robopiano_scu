#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fugue.constants import DEFAULT_OUTPUT_ROOT, DEFAULT_RP1M_ROOT, DEFAULT_SONG_KEY  # noqa: E402
from fugue.data import write_song_audit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit one RP1M song and create demo-level Fugue splits.")
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--song-key", default=DEFAULT_SONG_KEY)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "audit"))
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_song_audit(
        dataset_root=args.rp1m_root,
        output_root=args.output_root,
        song_key=args.song_key,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
