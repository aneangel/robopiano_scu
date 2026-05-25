#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT / "partita" / "src", REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from fugue.constants import DEFAULT_ENVIRONMENT_NAME, DEFAULT_RP1M_ROOT  # noqa: E402
from fugue.evaluation import load_checkpoint_model, predict_demo_actions, save_npz_prediction  # noqa: E402
from fugue.training import save_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export predicted Fugue actions for one held-out demo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--dataset-artifact-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--demo-id", type=int, default=None)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-rollout", action="store_true")
    parser.add_argument("--environment-name", default=DEFAULT_ENVIRONMENT_NAME)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--chunk-aggregation", default="uniform", choices=("uniform", "first", "temporal_aggregate"))
    parser.add_argument("--temporal-agg-decay", type=float, default=0.7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(Path(args.dataset_artifact_root) / "manifest.csv")
    split_rows = manifest[manifest["split"].astype(str) == str(args.split)].sort_values("demo_id")
    if split_rows.empty:
        raise ValueError(f"No demos found for split={args.split!r}")
    demo_id = int(args.demo_id) if args.demo_id is not None else int(split_rows.iloc[int(args.demo_index)]["demo_id"])
    model, checkpoint = load_checkpoint_model(args.checkpoint, device=args.device)
    prediction = predict_demo_actions(
        model=model,
        checkpoint=checkpoint,
        dataset_root=args.rp1m_root,
        demo_id=demo_id,
        device=args.device,
        chunk_aggregation=args.chunk_aggregation,
        temporal_agg_decay=float(args.temporal_agg_decay),
    )
    prediction.update({"checkpoint": str(args.checkpoint), "split": str(args.split)})
    npz_path = save_npz_prediction(output_dir / "predicted_actions.npz", prediction)
    summary = {"predicted_actions_npz": str(npz_path), "demo_id": demo_id, "split": str(args.split)}
    if args.run_rollout:
        from partita.evaluation.rollout import rollout_reconstructed_actions_with_robopianist

        rollout = rollout_reconstructed_actions_with_robopianist(
            actions=prediction["predicted_actions"],
            goals=prediction["goals"],
            song_name=str(args.environment_name),
            output_dir=output_dir / "rollout",
            label="fugue_predicted_actions",
            control_timestep=float(checkpoint.get("dt", 0.05)),
            max_steps=args.max_steps,
            reduced_action_space=True,
            action_source_scale="normalized_minus_one_to_one",
            reference_hand_joints=prediction["hand_joints"],
            reference_piano_states=prediction.get("piano_states"),
            restore_initial_state=True,
        )
        summary["rollout"] = rollout
    save_json(output_dir / "export_summary.json", summary)
    print(f"wrote_npz={npz_path}")
    print(f"demo_id={demo_id}")


if __name__ == "__main__":
    main()
