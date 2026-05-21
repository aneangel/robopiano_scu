from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Variations" / "src",
    REPO_ROOT / "Variations",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.evaluation import fingertip_summary_from_trajectory  # noqa: E402
from bagatelle.planner import plan_target_keys  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402


def payload_from_plan(plan) -> dict[str, np.ndarray]:
    payload = plan.npz_payload()
    payload["waypoint_fingertips"] = np.asarray(plan.waypoint_fingertips, dtype=np.float32)
    payload["fingertip_targets"] = np.asarray(plan.fingertip_targets, dtype=np.float32)
    return payload


def summarize(strategy: str, cfg: BagatelleConfig, target_keys: np.ndarray) -> dict[str, object]:
    plan = plan_target_keys(target_keys, config=cfg)
    fingertips = fingertip_summary_from_trajectory(
        payload_from_plan(plan),
        success_threshold=float(cfg.residual_success_threshold),
    )
    ik_metrics = np.asarray(plan.ik_metrics, dtype=np.float32)
    return {
        "strategy": strategy,
        "waypoints": int(plan.waypoint_frames.size),
        "ik_success_count": int(plan.metadata.get("ik_success_count", 0)),
        "ik_optimizer_success_count": int(plan.metadata.get("ik_optimizer_success_count", 0)),
        "ik_max_residual_mean": float(plan.metadata.get("ik_max_residual_mean", 0.0)),
        "ik_max_residual_p95": float(plan.metadata.get("ik_max_residual_p95", 0.0)),
        "assignment_candidate_rank_mean": plan.metadata.get("assignment_candidate_rank_mean"),
        "assignment_candidate_score_mean": plan.metadata.get("assignment_candidate_score_mean"),
        "fingertip_distance_mean": fingertips.get("fingertip_distance_mean"),
        "fingertip_distance_p95": fingertips.get("fingertip_distance_p95"),
        "fingertip_distance_max": fingertips.get("fingertip_distance_max"),
        "fingertip_success_rate": fingertips.get("fingertip_success_rate"),
        "fingertip_width_distance_mean": fingertips.get("fingertip_width_distance_mean"),
        "residual_norm_mean": float(np.mean(ik_metrics[:, 6])) if ik_metrics.size else 0.0,
        "residual_norm_max": float(np.max(ik_metrics[:, 6])) if ik_metrics.size else 0.0,
    }


def main() -> None:
    midi_path = REPO_ROOT / "robopianist" / "music" / "data" / "rousseau" / "twinkle-twinkle-trimmed.mid"
    target_keys, midi_meta = load_target_keys_from_midi(
        midi_path,
        control_timestep=0.05,
        max_steps=None,
        max_duration_s=None,
    )

    configs = {
        "legacy_previous_pose": BagatelleConfig(
            assignment_strategy="legacy_previous_pose",
        ),
        "composite_cost": BagatelleConfig(
            assignment_strategy="composite_cost",
            assignment_distance_weight=1.0,
            assignment_hand_zone_weight=0.25,
            assignment_finger_zone_weight=0.10,
            assignment_hold_weight=2.0,
            assignment_reach_weight=0.50,
            assignment_black_key_weight=0.05,
            assignment_wrong_hand_penalty=2.0,
        ),
        "ik_aware_topk": BagatelleConfig(
            assignment_strategy="ik_aware_topk",
            assignment_distance_weight=1.0,
            assignment_hand_zone_weight=0.25,
            assignment_finger_zone_weight=0.10,
            assignment_crossing_weight=0.50,
            assignment_hold_weight=2.0,
            assignment_reach_weight=0.50,
            assignment_black_key_weight=0.05,
            assignment_wrong_hand_penalty=2.0,
            assignment_top_k=8,
            assignment_ik_residual_weight=5.0,
            assignment_ik_max_residual_weight=5.0,
            assignment_motion_weight=0.05,
            assignment_ik_failure_penalty=10.0,
        ),
        "sequence_beam": BagatelleConfig(
            assignment_strategy="sequence_beam",
            assignment_distance_weight=1.0,
            assignment_hand_zone_weight=0.25,
            assignment_finger_zone_weight=0.10,
            assignment_crossing_weight=0.50,
            assignment_hold_weight=2.0,
            assignment_reach_weight=0.50,
            assignment_black_key_weight=0.05,
            assignment_wrong_hand_penalty=2.0,
            assignment_top_k=8,
            assignment_beam_width=4,
            assignment_candidates_per_step=8,
            assignment_ik_residual_weight=5.0,
            assignment_ik_max_residual_weight=5.0,
            assignment_motion_weight=0.05,
            assignment_ik_failure_penalty=10.0,
            assignment_unassigned_penalty=25.0,
        ),
    }

    results = {
        "midi_path": str(midi_path),
        "midi_meta": midi_meta,
        "target_keys_shape": list(np.asarray(target_keys).shape),
        "results": [summarize(name, cfg, target_keys) for name, cfg in configs.items()],
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
