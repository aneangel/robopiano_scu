#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fugue.constants import DEFAULT_RP1M_ROOT  # noqa: E402
from fugue.evaluation import evaluate_checkpoint  # noqa: E402
from fugue.plots import plot_per_dim_mse  # noqa: E402
from fugue.training import save_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Fugue checkpoint on one split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--dataset-artifact-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_root=args.rp1m_root,
        dataset_artifact_root=args.dataset_artifact_root,
        split=args.split,
        batch_size=int(args.batch_size),
        device=args.device,
    )
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parents[1] / f"eval_{args.split}"
    save_json(out_dir / "metrics.json", metrics)
    plot_per_dim_mse(metrics, out_dir / "per_dim_mse.png")
    print(f"wrote_metrics={out_dir / 'metrics.json'}")
    print(f"action_mse={metrics['action_mse']:.6f} action_l1={metrics['action_l1']:.6f}")


if __name__ == "__main__":
    main()
