from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from nocturne.controller_model import train_mlp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Nocturne stitched-controller baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--wandb-project", default="robopianist")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default="Nocturne")
    parser.add_argument("--wandb-notes", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline"], help="Defaults to WANDB_MODE or online. Nocturne training does not support disabled W&B.")
    parser.add_argument("--wandb-tags", default="nocturne,stitched-controller")
    args = parser.parse_args()
    summary = train_mlp(
        Path(args.dataset_root) / "dataset.npz",
        args.output_root,
        epochs=int(args.epochs),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        device=str(args.device),
        wandb_project=str(args.wandb_project),
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_tags=[item.strip() for item in str(args.wandb_tags).split(",") if item.strip()],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
