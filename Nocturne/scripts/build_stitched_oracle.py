from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from nocturne.events import extract_note_events, event_intervals, goals_are_compatible, protected_press_mask  # noqa: E402
from nocturne.io import ensure_dir, save_json, save_npz, write_table  # noqa: E402
from nocturne.offline_eval import evaluate_demo_baselines, evaluate_rollout  # noqa: E402
from nocturne.repair_selected_path import repair_selected_path  # noqa: E402
from nocturne.rp1m import load_song_demos  # noqa: E402
from nocturne.schema import StitchConfig, TransitionWeights  # noqa: E402
from nocturne.scoring import build_candidates, filter_candidates_for_selection, flatten_candidates  # noqa: E402
from nocturne.seams import (  # noqa: E402
    adaptive_min_distance_intervals,
    joint_bounds_from_demos,
    seam_frames_from_intervals,
    seam_jump_metrics,
    smooth_stitched_payload,
    stitch_selected,
)
from nocturne.viterbi import transition_cost, viterbi_select  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Nocturne stitched oracle trajectory.")
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--song-name", required=True)
    parser.add_argument("--num-demos", type=int, default=50)
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--pre-frames", type=int, default=24)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chord-tolerance-frames", type=int, default=1)
    parser.add_argument("--event-tolerance-frames", type=int, default=3)
    parser.add_argument("--seam-blend-radius", type=int, default=4)
    parser.add_argument("--objective-mode", choices=("strict", "correctness", "legacy"), default="strict")
    parser.add_argument("--disable-repair", action="store_true")
    parser.add_argument("--repair-max-passes", type=int, default=2)
    parser.add_argument("--repair-transition-margin", type=float, default=250.0)
    parser.add_argument("--disable-adaptive-seams", action="store_true")
    parser.add_argument("--seam-search-margin-frames", type=int, default=3)
    parser.add_argument("--disable-transition-interpolation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StitchConfig(
        dt=float(args.dt),
        window_frames=int(args.window_frames),
        pre_frames=int(args.pre_frames),
        chord_tolerance_frames=int(args.chord_tolerance_frames),
        event_tolerance_frames=int(args.event_tolerance_frames),
        seam_blend_radius=int(args.seam_blend_radius),
        threshold=float(args.threshold),
        objective_mode=str(args.objective_mode),
        repair_enabled=bool(str(args.objective_mode) != "legacy" and not args.disable_repair),
        repair_max_passes=int(args.repair_max_passes),
        repair_transition_margin=float(args.repair_transition_margin),
        adaptive_seam_enabled=bool(not args.disable_adaptive_seams),
        seam_search_margin_frames=int(args.seam_search_margin_frames),
        transition_interpolation_enabled=bool(not args.disable_transition_interpolation),
    )
    song_dir = ensure_dir(Path(args.output_root) / "stitcher" / str(args.song_name))
    demos = load_song_demos(args.rp1m_root, args.song_name, num_demos=int(args.num_demos))
    incompatible = [
        int(demo_id)
        for row, demo_id in enumerate(demos.demo_ids.tolist())
        if not goals_are_compatible(demos.goals[0], demos.goals[row], threshold=config.threshold)
    ]
    events = extract_note_events(
        demos.goals[0],
        threshold=config.threshold,
        chord_tolerance_frames=config.chord_tolerance_frames,
    )
    if not events:
        raise RuntimeError(f"No note events found for {args.song_name}")
    intervals = event_intervals(events, demos.num_frames)
    candidates_by_event = build_candidates(demos, events, config, intervals=intervals)
    candidate_rows = [candidate.as_dict() for candidate in flatten_candidates(candidates_by_event)]
    table_paths = write_table(pd.DataFrame(candidate_rows), song_dir / "segment_scores")
    selection_candidates_by_event, selection_filter_report = filter_candidates_for_selection(candidates_by_event, config)
    contact_mask = protected_press_mask(demos.goals[0], np.asarray([event.onset_frame for event in events]), radius=1)
    transition_weights = TransitionWeights()
    selected, transition_costs, viterbi_report = viterbi_select(
        selection_candidates_by_event,
        demos,
        intervals,
        weights=transition_weights,
        contact_mask=contact_mask,
    )
    selected, transition_costs, repair_report = repair_selected_path(
        selected,
        candidates_by_event,
        demos,
        intervals,
        config=config,
        weights=transition_weights,
        contact_mask=contact_mask,
    )
    stitch_intervals, seam_refinement_report = adaptive_min_distance_intervals(
        demos,
        selected,
        events,
        intervals,
        config=config,
    )
    transition_costs = _selected_transition_costs(
        selected,
        demos,
        stitch_intervals,
        weights=transition_weights,
        contact_mask=contact_mask,
    )
    raw_payload = stitch_selected(demos, selected, events, stitch_intervals, transition_costs, config=config)
    raw_path = save_npz(song_dir / "stitched_trajectory.npz", **raw_payload)
    seam_frames = seam_frames_from_intervals(stitch_intervals)
    lower, upper = joint_bounds_from_demos(demos)
    smooth_payload = smooth_stitched_payload(
        raw_payload,
        seam_frames=seam_frames,
        press_frames=raw_payload["press_frame_indices"],
        blend_radius=config.seam_blend_radius if config.transition_interpolation_enabled else 0,
        threshold=config.threshold,
        joint_lower=lower,
        joint_upper=upper,
    )
    smooth_path = save_npz(song_dir / "smoothed_trajectory.npz", **smooth_payload)
    raw_eval = evaluate_rollout(raw_payload["goals"], raw_payload["piano_states"], dt=config.dt, threshold=config.threshold)
    smooth_eval = evaluate_rollout(smooth_payload["goals"], smooth_payload["piano_states"], dt=config.dt, threshold=config.threshold)
    baselines = evaluate_demo_baselines(demos, dt=config.dt, threshold=config.threshold)
    offline_eval = {
        "song_name": str(args.song_name),
        "raw_stitched": raw_eval,
        "smoothed_stitched": smooth_eval,
        "baselines": baselines,
        "seam_metrics_raw": seam_jump_metrics(raw_payload, seam_frames),
        "seam_metrics_smoothed": seam_jump_metrics(smooth_payload, seam_frames),
    }
    save_json(song_dir / "offline_eval.json", offline_eval)
    selected_path = {
        "song_name": str(args.song_name),
        "selected": [candidate.as_dict() for candidate in selected],
        "transition_costs": transition_costs,
        "viterbi": viterbi_report,
        "repair": repair_report,
        "selection_filter": selection_filter_report,
        "seam_refinement": seam_refinement_report,
    }
    save_json(song_dir / "selected_path.json", selected_path)
    report = {
        "song_name": str(args.song_name),
        "rp1m_root": str(args.rp1m_root),
        "num_demos": int(demos.num_demos),
        "demo_ids": [int(value) for value in demos.demo_ids.tolist()],
        "incompatible_goal_demo_ids": incompatible,
        "num_events": int(len(events)),
        "num_candidates": int(len(candidate_rows)),
        "num_selection_candidates": int(sum(len(group) for group in selection_candidates_by_event)),
        "objective_mode": str(config.objective_mode),
        "repair_enabled": bool(config.repair_enabled),
        "repair_num_swaps": int(repair_report.get("num_swaps", 0)),
        "adaptive_seam_enabled": bool(config.adaptive_seam_enabled),
        "adaptive_seam_num_changed": int(seam_refinement_report.get("num_changed", 0)),
        "transition_interpolation_enabled": bool(config.transition_interpolation_enabled),
        "seam_refinement": seam_refinement_report,
        "selection_filter": selection_filter_report,
        "segment_score_paths": table_paths,
        "stitched_trajectory": str(raw_path),
        "smoothed_trajectory": str(smooth_path),
        "offline_eval": str(song_dir / "offline_eval.json"),
        "best_demo_frame_f1": None if baselines["best_demo"] is None else baselines["best_demo"]["frame_f1"],
        "best_demo_event_f1": None if baselines["best_demo"] is None else baselines["best_demo"]["event_f1"],
        "stitched_frame_f1": smooth_eval["frame_f1"],
        "stitched_event_f1": smooth_eval["event_f1"],
    }
    save_json(song_dir / "stitch_report.json", report)
    print(f"Wrote Nocturne stitcher outputs to {song_dir}")
    print(f"best_demo_event_f1={report['best_demo_event_f1']} stitched_event_f1={report['stitched_event_f1']}")


def _selected_transition_costs(selected, demos, intervals, *, weights, contact_mask):
    costs = []
    for index in range(1, len(selected)):
        cost = transition_cost(selected[index - 1], selected[index], demos, intervals, weights=weights, contact_mask=contact_mask)
        costs.append(float(cost if np.isfinite(cost) else 0.0))
    return costs


if __name__ == "__main__":
    main()
