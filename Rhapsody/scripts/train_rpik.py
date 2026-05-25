from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Rhapsody" / "src"))

import torch

from rhapsody.checkpoint import save_checkpoint
from rhapsody.config import RhapsodyConfig
from rhapsody.data import load_rp1m_ik_pairs
from rhapsody.evaluation import evaluate_policy
from rhapsody.trainer import forward_model_metrics, train_rhapsody


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rhapsody RP1M fingertip IK policy.")
    parser.add_argument("--rp1m-root", type=Path, required=True)
    parser.add_argument("--song", action="append", dest="songs", default=None)
    parser.add_argument("--max-songs", type=int, default=1)
    parser.add_argument("--random-songs", action="store_true")
    parser.add_argument("--num-demos", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=1024)
    parser.add_argument(
        "--previous-mode",
        choices=("adjacent", "random", "mean", "zero"),
        default="adjacent",
    )
    parser.add_argument("--fk-epochs", type=int, default=20)
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--policy-epochs", type=int, default=20)
    parser.add_argument("--heldout-songs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--policy-action-scale", type=float, default=2.0)
    parser.add_argument("--policy-exploration-std", type=float, default=0.35)
    parser.add_argument("--imitation-weight", type=float, default=0.05)
    parser.add_argument("--reward-smoothness-weight", type=float, default=0.02)
    parser.add_argument("--eval-refinement-steps", type=int, default=0)
    parser.add_argument("--eval-refinement-lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=524)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = RhapsodyConfig(
        batch_size=args.batch_size,
        policy_action_scale=args.policy_action_scale,
        policy_exploration_std=args.policy_exploration_std,
        imitation_weight=args.imitation_weight,
        reward_smoothness_weight=args.reward_smoothness_weight,
        seed=args.seed,
    )
    arrays = load_rp1m_ik_pairs(
        args.rp1m_root,
        song_names=args.songs,
        max_songs=args.max_songs if args.songs is None else None,
        random_songs=bool(args.random_songs),
        num_demos=args.num_demos,
        frame_stride=args.frame_stride,
        max_pairs=args.max_pairs,
        previous_mode=args.previous_mode,
        seed=args.seed,
    )
    result = train_rhapsody(
        arrays,
        config,
        fk_epochs=args.fk_epochs,
        bc_epochs=args.bc_epochs,
        policy_epochs=args.policy_epochs,
        validation_song_count=args.heldout_songs,
        device=torch.device(args.device),
    )
    fk_metrics = forward_model_metrics(
        result.fk_model,
        result.normalizer,
        arrays,
        config,
        device=torch.device(args.device),
    )
    refined_validation_metrics = None
    if int(args.eval_refinement_steps) > 0:
        refined_validation_metrics = evaluate_policy(
            result.policy,
            result.fk_model,
            result.normalizer,
            arrays.take(
                [
                    index
                    for index, song_id in enumerate(arrays.source_song_ids.tolist())
                    if arrays.song_names[song_id] in set(result.validation_song_names)
                ]
            ),
            config,
            device=torch.device(args.device),
            refinement_steps=int(args.eval_refinement_steps),
            refinement_lr=float(args.eval_refinement_lr),
        )
    checkpoint = save_checkpoint(
        args.output_dir / "rhapsody_rpik.pt",
        result,
        config,
        metadata={
            "rp1m_root": str(args.rp1m_root),
            "song_names": list(arrays.song_names),
            "examples": len(arrays),
            "previous_mode": args.previous_mode,
            "fk_epochs": int(args.fk_epochs),
            "bc_epochs": int(args.bc_epochs),
            "policy_epochs": int(args.policy_epochs),
            "heldout_songs": int(args.heldout_songs),
        },
    )
    summary = {
        "checkpoint": str(checkpoint),
        "config": asdict(config),
        "examples": len(arrays),
        "songs": list(arrays.song_names),
        "train_songs": list(result.train_song_names),
        "validation_songs": list(result.validation_song_names),
        "fk_metrics_all_pairs": fk_metrics.to_dict(),
        "train_metrics": result.train_metrics.to_dict(),
        "validation_metrics": result.validation_metrics.to_dict(),
        "refined_validation_metrics": refined_validation_metrics.to_dict()
        if refined_validation_metrics is not None
        else None,
        "history": result.history,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
