from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

import numpy as np  # noqa: E402

from nocturne.error_analysis import analyze_demo_error_consensus, analyze_stitched_error_context  # noqa: E402
from nocturne.io import ensure_dir, load_json, save_json, write_table  # noqa: E402
from nocturne.rp1m import list_songs, load_song_demos  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nocturne stitched-oracle evaluation over multiple songs.")
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--song-name", action="append", default=None, help="Song to include. Repeat for multiple songs.")
    parser.add_argument("--song-list", default=None, help="Optional text file with one song name per line.")
    parser.add_argument("--num-songs", type=int, default=10)
    parser.add_argument("--num-demos", type=int, default=50)
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--pre-frames", type=int, default=24)
    parser.add_argument("--event-tolerance-frames", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-s", type=float, default=0.15)
    parser.add_argument("--objective-mode", choices=("strict", "correctness", "legacy"), default="strict")
    parser.add_argument("--disable-repair", action="store_true")
    parser.add_argument("--repair-max-passes", type=int, default=2)
    parser.add_argument("--repair-transition-margin", type=float, default=250.0)
    parser.add_argument("--disable-adaptive-seams", action="store_true")
    parser.add_argument("--seam-search-margin-frames", type=int, default=3)
    parser.add_argument("--disable-transition-interpolation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    batch_dir = ensure_dir(output_root / "batch_eval")
    songs = _resolve_songs(args)
    save_json(
        batch_dir / "song_list.json",
        {
            "songs": songs,
            "num_songs": len(songs),
            "objective_mode": str(args.objective_mode),
            "repair_enabled": bool(str(args.objective_mode) != "legacy" and not args.disable_repair),
            "adaptive_seam_enabled": bool(not args.disable_adaptive_seams),
            "transition_interpolation_enabled": bool(not args.disable_transition_interpolation),
        },
    )

    song_rows: list[dict[str, Any]] = []
    all_demo_missed: list[dict[str, Any]] = []
    all_demo_mispresses: list[dict[str, Any]] = []
    all_stitched_missed: list[dict[str, Any]] = []
    all_stitched_mispresses: list[dict[str, Any]] = []

    for song_index, song_name in enumerate(songs):
        print(f"[{song_index + 1}/{len(songs)}] {song_name}", flush=True)
        song_dir = output_root / "stitcher" / song_name
        try:
            if args.overwrite or not (song_dir / "smoothed_trajectory.npz").exists():
                _run_single_song(args, song_name)
            row, demo_consensus, stitched_context = _analyze_song(args, song_name, song_dir)
            song_rows.append(row)
            all_demo_missed.extend(demo_consensus["missed_target_events"])
            all_demo_mispresses.extend(demo_consensus["mispress_buckets"])
            all_stitched_missed.extend(stitched_context["stitched_missed_events"])
            all_stitched_mispresses.extend(stitched_context["stitched_mispress_events"])
        except Exception as exc:
            failed = {"song_name": song_name, "status": "failed", "error": repr(exc)}
            song_rows.append(failed)
            print(f"FAILED {song_name}: {exc!r}", flush=True)
            if not args.continue_on_error:
                raise

    _write_outputs(
        batch_dir=batch_dir,
        song_rows=song_rows,
        demo_missed=all_demo_missed,
        demo_mispresses=all_demo_mispresses,
        stitched_missed=all_stitched_missed,
        stitched_mispresses=all_stitched_mispresses,
    )
    print(f"Wrote batch evaluation to {batch_dir}", flush=True)


def _resolve_songs(args: argparse.Namespace) -> list[str]:
    songs: list[str] = []
    if args.song_name:
        songs.extend(str(value).strip() for value in args.song_name if str(value).strip())
    if args.song_list:
        songs.extend(
            line.strip()
            for line in Path(args.song_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not songs:
        songs = list_songs(args.rp1m_root, limit=int(args.num_songs))
    deduped = []
    seen = set()
    for song in songs:
        if song in seen:
            continue
        seen.add(song)
        deduped.append(song)
        if len(deduped) >= int(args.num_songs):
            break
    return deduped


def _run_single_song(args: argparse.Namespace, song_name: str) -> None:
    script = Path(__file__).with_name("build_stitched_oracle.py")
    cmd = [
        sys.executable,
        str(script),
        "--rp1m-root",
        str(args.rp1m_root),
        "--song-name",
        song_name,
        "--num-demos",
        str(args.num_demos),
        "--window-frames",
        str(args.window_frames),
        "--pre-frames",
        str(args.pre_frames),
        "--event-tolerance-frames",
        str(args.event_tolerance_frames),
        "--output-root",
        str(args.output_root),
        "--dt",
        str(args.dt),
        "--threshold",
        str(args.threshold),
        "--objective-mode",
        str(args.objective_mode),
        "--repair-max-passes",
        str(args.repair_max_passes),
        "--repair-transition-margin",
        str(args.repair_transition_margin),
        "--seam-search-margin-frames",
        str(args.seam_search_margin_frames),
    ]
    if args.disable_repair:
        cmd.append("--disable-repair")
    if args.disable_adaptive_seams:
        cmd.append("--disable-adaptive-seams")
    if args.disable_transition_interpolation:
        cmd.append("--disable-transition-interpolation")
    subprocess.run(cmd, check=True)


def _analyze_song(
    args: argparse.Namespace,
    song_name: str,
    song_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    offline_path = song_dir / "offline_eval.json"
    report_path = song_dir / "stitch_report.json"
    trajectory_path = song_dir / "smoothed_trajectory.npz"
    offline = load_json(offline_path)
    report = load_json(report_path)
    demos = load_song_demos(args.rp1m_root, song_name, num_demos=int(args.num_demos))
    demo_consensus = analyze_demo_error_consensus(
        demos,
        song_name=song_name,
        dt=float(args.dt),
        threshold=float(args.threshold),
        tolerance_s=float(args.tolerance_s),
    )
    with np.load(trajectory_path, allow_pickle=False) as data:
        stitched_goals = np.asarray(data["goals"], dtype=np.float32)
        stitched_piano = np.asarray(data["piano_states"], dtype=np.float32)
        dt = float(np.asarray(data["dt"]).reshape(())) if "dt" in data else float(args.dt)
    stitched_context = analyze_stitched_error_context(
        demos,
        song_name=song_name,
        stitched_goals=stitched_goals,
        stitched_piano_states=stitched_piano,
        dt=dt,
        threshold=float(args.threshold),
        tolerance_s=float(args.tolerance_s),
    )

    best = offline["baselines"]["best_demo"]
    stitched = offline["smoothed_stitched"]
    seam = offline.get("seam_metrics_smoothed", {})
    avoidable_missed = sum(1 for row in stitched_context["stitched_missed_events"] if row["avoidable_from_raw_demos"])
    unavoidable_missed = sum(1 for row in stitched_context["stitched_missed_events"] if row["missed_by_all_raw_demos"])
    seen_mispress = sum(1 for row in stitched_context["stitched_mispress_events"] if row["seen_in_raw_demos"])
    novel_mispress = len(stitched_context["stitched_mispress_events"]) - seen_mispress
    common_demo_misses = sum(1 for row in demo_consensus["missed_target_events"] if row["missed_by_all_demos"])
    common_demo_mispresses = sum(1 for row in demo_consensus["mispress_buckets"] if row["seen_in_all_demos"])
    row = {
        "song_name": song_name,
        "status": "ok",
        "num_demos": int(report["num_demos"]),
        "num_events": int(report["num_events"]),
        "num_candidates": int(report.get("num_candidates", 0)),
        "num_selection_candidates": int(report.get("num_selection_candidates", 0)),
        "selection_filter_removed": int(report.get("selection_filter", {}).get("num_removed", 0)),
        "selection_events_filtered": int(report.get("selection_filter", {}).get("num_events_filtered", 0)),
        "objective_mode": str(report.get("objective_mode", args.objective_mode)),
        "repair_enabled": bool(report.get("repair_enabled", False)),
        "repair_num_swaps": int(report.get("repair_num_swaps", 0)),
        "adaptive_seam_enabled": bool(report.get("adaptive_seam_enabled", False)),
        "adaptive_seam_num_changed": int(report.get("adaptive_seam_num_changed", 0)),
        "transition_interpolation_enabled": bool(report.get("transition_interpolation_enabled", False)),
        "best_demo_id": None if best is None else int(best["demo_id"]),
        "best_demo_frame_f1": None if best is None else float(best["frame_f1"]),
        "best_demo_event_f1": None if best is None else float(best["event_f1"]),
        "mean_demo_frame_f1": float(offline["baselines"]["mean_frame_f1"]),
        "mean_demo_event_f1": float(offline["baselines"]["mean_event_f1"]),
        "stitched_frame_f1": float(stitched["frame_f1"]),
        "stitched_event_f1": float(stitched["event_f1"]),
        "delta_frame_f1_vs_best": None if best is None else float(stitched["frame_f1"] - best["frame_f1"]),
        "delta_event_f1_vs_best": None if best is None else float(stitched["event_f1"] - best["event_f1"]),
        "best_demo_missed_key_presses": None if best is None else int(best["missed_key_presses"]),
        "best_demo_mispresses": None if best is None else int(best["mispresses"]),
        "stitched_missed_key_presses": int(stitched["missed_key_presses"]),
        "stitched_mispresses": int(stitched["mispresses"]),
        "stitched_avoidable_missed_key_presses": int(avoidable_missed),
        "stitched_unavoidable_missed_key_presses": int(unavoidable_missed),
        "stitched_seen_raw_mispresses": int(seen_mispress),
        "stitched_novel_mispresses": int(novel_mispress),
        "demo_events_missed_by_all": int(common_demo_misses),
        "demo_mispress_buckets_seen_in_all": int(common_demo_mispresses),
        "seam_hand_joints_jump_p95": seam.get("seam/hand_joints_jump_p95"),
        "seam_hand_joints_adjacent_p95": seam.get("seam/hand_joints_within_demo_adjacent_p95_proxy"),
        "seam_actions_jump_p95": seam.get("seam/actions_jump_p95"),
        "seam_actions_adjacent_p95": seam.get("seam/actions_within_demo_adjacent_p95_proxy"),
    }
    return row, demo_consensus, stitched_context


def _write_outputs(
    *,
    batch_dir: Path,
    song_rows: list[dict[str, Any]],
    demo_missed: list[dict[str, Any]],
    demo_mispresses: list[dict[str, Any]],
    stitched_missed: list[dict[str, Any]],
    stitched_mispresses: list[dict[str, Any]],
) -> None:
    summary = pd.DataFrame(song_rows)
    write_table(summary, batch_dir / "song_summary")
    write_table(pd.DataFrame(demo_missed), batch_dir / "demo_missed_target_events")
    write_table(pd.DataFrame(demo_mispresses), batch_dir / "demo_mispress_buckets")
    write_table(pd.DataFrame(stitched_missed), batch_dir / "stitched_missed_context")
    write_table(pd.DataFrame(stitched_mispresses), batch_dir / "stitched_mispress_context")

    ok = summary[summary["status"] == "ok"] if "status" in summary else summary
    improved = ok["delta_event_f1_vs_best"].dropna() > 0 if "delta_event_f1_vs_best" in ok else []
    report = {
        "num_songs": int(len(summary)),
        "num_successful": int(len(ok)),
        "num_event_f1_improved": int(improved.sum()) if len(ok) else 0,
        "mean_delta_event_f1_vs_best": _safe_mean(ok.get("delta_event_f1_vs_best", [])),
        "mean_delta_frame_f1_vs_best": _safe_mean(ok.get("delta_frame_f1_vs_best", [])),
        "total_best_demo_missed_key_presses": _safe_sum(ok.get("best_demo_missed_key_presses", [])),
        "total_stitched_missed_key_presses": _safe_sum(ok.get("stitched_missed_key_presses", [])),
        "total_best_demo_mispresses": _safe_sum(ok.get("best_demo_mispresses", [])),
        "total_stitched_mispresses": _safe_sum(ok.get("stitched_mispresses", [])),
        "total_stitched_avoidable_missed_key_presses": _safe_sum(ok.get("stitched_avoidable_missed_key_presses", [])),
        "total_stitched_unavoidable_missed_key_presses": _safe_sum(ok.get("stitched_unavoidable_missed_key_presses", [])),
        "total_stitched_novel_mispresses": _safe_sum(ok.get("stitched_novel_mispresses", [])),
        "output_files": {
            "song_summary_csv": str(batch_dir / "song_summary.csv"),
            "demo_missed_target_events_csv": str(batch_dir / "demo_missed_target_events.csv"),
            "demo_mispress_buckets_csv": str(batch_dir / "demo_mispress_buckets.csv"),
            "stitched_missed_context_csv": str(batch_dir / "stitched_missed_context.csv"),
            "stitched_mispress_context_csv": str(batch_dir / "stitched_mispress_context.csv"),
        },
    }
    save_json(batch_dir / "batch_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _safe_mean(values: Any) -> float | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return None if arr.empty else float(arr.mean())


def _safe_sum(values: Any) -> int:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return int(arr.sum()) if not arr.empty else 0


if __name__ == "__main__":
    main()
