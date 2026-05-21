from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import math
import numpy as np
import pandas as pd

from partita.evaluation.metrics import action_metrics
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, load_json, save_json


def _entropy(counts) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _prefix_metrics(prefix: str, values: dict) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _load_rollout_validation(root: Path, exp: str) -> dict[str, object]:
    summary_path = root / "rollout" / exp / "rollout_summary.json"
    if not summary_path.exists():
        return {
            "scoring_validation_status": "rollout_missing",
            "rollout_diagnosis_category": "rollout_not_run",
            "scoring_validation_note": (
                "Run partita/scripts/simulate_rollout.py --which reconstructed through Slurm to score "
                "the learned primitive reconstruction from RoboPianist key presses."
            ),
        }
    summary = load_json(summary_path)
    results = [item for item in summary.get("results", []) if isinstance(item, dict)]
    by_label = {str(item.get("label")): item for item in results}
    reconstructed = by_label.get("reconstructed")
    original = by_label.get("original_target")
    metrics: dict[str, object] = {
        "scoring_validation_status": "rollout_scored",
        "scoring_validation_note": (
            "Primary key metrics come from RoboPianist piano activation produced by replayed reconstructed actions."
        ),
        "original_target_rollout_note": (
            "Original-target rollout metrics are replay-system diagnostics, not Partita model performance."
        ),
    }
    keep_keys = [
        "rollout_key_precision",
        "rollout_key_recall",
        "rollout_key_f1",
        "rollout_mispress_rate",
        "rollout_scored_steps",
        "rollout_scoring_source",
        "total_reward",
        "actions_executed",
        "terminated",
        "render_error",
        "video_path",
        "audio_warning",
        "audio_midi_note_event_count",
        "input_actions_min",
        "input_actions_max",
        "input_actions_mean",
        "input_actions_std",
        "input_actions_abs_p95",
        "env_action_spec_min",
        "env_action_spec_max",
        "env_action_dim",
        "input_action_dim",
        "action_dim_width_used",
        "action_source_scale",
    ]
    if reconstructed is not None:
        metrics.update(_prefix_metrics("reconstructed", {key: reconstructed.get(key) for key in keep_keys if key in reconstructed}))
    else:
        metrics["scoring_validation_status"] = "reconstructed_rollout_missing"
        metrics["scoring_validation_note"] = "Rollout summary exists, but it does not include the reconstructed action replay."
    if original is not None:
        metrics.update(_prefix_metrics("original_target", {key: original.get(key) for key in keep_keys if key in original}))
        recon_f1 = reconstructed.get("rollout_key_f1") if reconstructed is not None else None
        orig_f1 = original.get("rollout_key_f1")
        if isinstance(recon_f1, (int, float)) and isinstance(orig_f1, (int, float)) and math.isfinite(float(orig_f1)):
            metrics["reconstructed_vs_original_rollout_key_f1_ratio"] = float(recon_f1) / max(float(orig_f1), 1e-12)
    for label in ["original_state", "reconstructed_state"]:
        item = by_label.get(label)
        if item is None:
            continue
        metrics[f"{label}_status"] = item.get("status", "scored")
        goals = item.get("against_goals") or {}
        states = item.get("against_rp1m_piano_states") or {}
        if "key_f1" in goals:
            metrics[f"{label}_against_goals_key_f1"] = goals.get("key_f1")
        if "key_f1" in states:
            metrics[f"{label}_against_rp1m_piano_states_key_f1"] = states.get("key_f1")

    def _f1(item: dict[str, object] | None, key: str) -> float | None:
        if item is None:
            return None
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def _state_f1(item: dict[str, object] | None, block: str) -> float | None:
        if item is None or item.get("status") == "missing_files":
            return None
        value = (item.get(block) or {}).get("key_f1")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    original_state = by_label.get("original_state")
    reconstructed_state = by_label.get("reconstructed_state")
    original_state_f1 = _state_f1(original_state, "against_rp1m_piano_states")
    reconstructed_state_f1 = _state_f1(reconstructed_state, "against_rp1m_piano_states")
    original_action_f1 = _f1(original, "rollout_key_f1")
    reconstructed_action_f1 = _f1(reconstructed, "rollout_key_f1")
    if not results:
        diagnosis = "rollout_not_run"
    elif original_state is not None and (original_state_f1 is None or original_state_f1 < 0.95):
        diagnosis = "original_state_failed"
    elif reconstructed_state is not None and (reconstructed_state_f1 is None or reconstructed_state_f1 < 0.95):
        diagnosis = "reconstructed_state_failed"
    elif original_state_f1 is not None and original_state_f1 >= 0.95 and (
        original_action_f1 is None or original_action_f1 < 0.50
    ):
        diagnosis = "original_state_ok_action_replay_failed"
    elif original_action_f1 is not None and original_action_f1 >= 0.50 and (
        reconstructed_action_f1 is None or reconstructed_action_f1 < 0.50
    ):
        diagnosis = "original_action_ok_reconstructed_failed"
    else:
        diagnosis = "unclear"
    metrics["rollout_diagnosis_category"] = diagnosis
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Partita reconstruction quality.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    exp = experiment_name(config)
    root = output_root(config)
    data_dir = root / "data" / exp
    prim_dir = root / "primitives" / exp
    recon_dir = root / "reconstruction" / exp
    eval_dir = ensure_dir(root / "evaluation" / exp)

    original = np.load(recon_dir / "original_actions.npy")
    reconstructed = np.load(recon_dir / "reconstructed_actions.npy")
    metrics = action_metrics(original, reconstructed)

    selection = load_json(data_dir / "selection.json")
    target_assignments = pd.read_csv(recon_dir / "target_primitive_assignments.csv")
    primitive_summary = pd.read_csv(prim_dir / "primitive_summary.csv")
    counts = target_assignments["primitive_id"].value_counts().sort_index().values
    primitive_counts = target_assignments["primitive_id"].value_counts().sort_index()
    num_primitives = int(len(primitive_summary))
    metrics.update({
        "source_song_name": selection.get("source_song_name", selection.get("song_name")),
        "target_song_name": selection.get("target_song_name", selection.get("song_name")),
        "is_cross_song": bool(selection.get("is_cross_song", False)),
        "num_training_trajectories": int(selection.get("num_training_trajectories", 0)),
        "num_target_segments": int(len(target_assignments)),
        "num_primitives": num_primitives,
        "primitive_entropy": _entropy(counts),
        "mean_primitive_coverage_across_trajectories": float(primitive_summary["trajectory_coverage_fraction"].mean()) if len(primitive_summary) else 0.0,
        "max_primitive_usage_fraction": float(counts.max() / max(counts.sum(), 1)) if len(counts) else 0.0,
        "target_unique_primitives_used": int(primitive_counts.shape[0]),
        "target_fraction_primitives_used": float(primitive_counts.shape[0] / max(num_primitives, 1)),
        "target_primitive_usage_counts": {str(int(k)): int(v) for k, v in primitive_counts.items()},
    })
    metrics.update(_load_rollout_validation(root, exp))
    save_json(eval_dir / "metrics.json", metrics)

    stale_plot = eval_dir / "pianoroll_comparison.png"
    if stale_plot.exists():
        stale_plot.unlink()
    lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    (eval_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved evaluation metrics to {eval_dir / 'metrics.json'}")
    print("Key metrics:")
    for key in [
        "action_mse",
        "action_l1",
        "reconstructed_rollout_key_f1",
        "scoring_validation_status",
        "primitive_entropy",
        "mean_primitive_coverage_across_trajectories",
        "max_primitive_usage_fraction",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
