#!/usr/bin/env python3
"""Full-dataset planner + rollout evaluation harness.

Runs the winning planner config (see ``benchmark/results/ROLLOUT_F1_RESULT.md``)
followed by ``retest_impromptu_rp1m_simulator.py`` for every song listed in a
MAESTRO-style manifest CSV. Per-song work is fanned out across a
``ProcessPoolExecutor`` and results are appended to a CSV as each song
completes (crash-safe).

The script never executes itself when imported, and is intentionally
stdlib + numpy only.

Usage:
    python benchmark/run_full_evaluation.py \\
        --manifest /tmp/maestro_test_177.csv \\
        --output-root /tmp/maestro_eval \\
        --parallelism 6 \\
        --csv-out /tmp/maestro_eval/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Default to the repo containing this file (works on any host).
REPO_ROOT = Path(__file__).resolve().parent.parent
# Allow override via env var for unusual layouts.
import os as _os
REPO_ROOT = Path(_os.environ.get("ROBOPIANO_REPO_ROOT", str(REPO_ROOT))).resolve()
PLAN_SCRIPT = REPO_ROOT / "Impromptu" / "scripts" / "plan_trajectory.py"
ROLLOUT_SCRIPT = REPO_ROOT / "retest_impromptu_rp1m_simulator.py"

ENV_NAME = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"

# Winning planner CLI from benchmark/results/ROLLOUT_F1_RESULT.md.
# Per-song values (--midi-path, --output-root, --run-name,
# --max-duration-s, --active-window-last-s, --seed) are injected
# dynamically in ``build_plan_command``.
BASELINE_PLAN_FLAGS: list[str] = [
    "--environment-name", ENV_NAME,
    "--trajectory-mode", "joint_space_straighten",
    "--disable-adaptive-complex-song-defaults",
    "--key-press-depth", "0.002",
    "--wrong-hand-penalty", "4.0",
    "--wrong-hand-split-key", "48",
    "--assignment-dynamic-hand-split",
    "--assignment-fail-if-unassigned",
    "--assignment-strategy", "ik_aware_topk",
    "--assignment-top-k", "2",
    "--anchor-stride", "1",
    "--ik-max-nfev", "60",
    "--residual-success-threshold", "0.02",
    "--ik-smoothness-weight", "0.02",
    "--ik-static-contact-validation",
    "--ik-static-contact-settle-steps", "1",
    "--disable-ik-multistart-on-failure",
    "--ik-cache-mode", "exact_and_warm_start",
    "--ik-unassigned-fingertip-strategy", "avoid_mispresses",
    "--ik-unassigned-fingertip-avoidance-weight", "64",
    "--ik-unassigned-fingertip-avoidance-radius", "0.06",
    "--enable-trajectory-refinement",
]

CSV_COLUMNS = [
    "song",
    "midi_path",
    "plan_seconds",
    "rollout_seconds",
    "total_seconds",
    "static_f1",
    "rollout_frame_f1",
    "event_f1",
    "matched",
    "missed",
    "mispresses",
    "target",
    "error",
]

F1_TARGET = 0.668


# ---------------------------------------------------------------------------
# Dataclass for one song's outcome.
# ---------------------------------------------------------------------------
@dataclass
class SongResult:
    song: str
    midi_path: str
    run_name: str
    plan_seconds: float = float("nan")
    rollout_seconds: float = float("nan")
    total_seconds: float = float("nan")
    static_f1: float = float("nan")
    rollout_frame_f1: float = float("nan")
    event_f1: float = float("nan")
    matched: float = float("nan")
    missed: float = float("nan")
    mispresses: float = float("nan")
    target: float = float("nan")
    error: Optional[str] = None

    def to_row(self) -> dict[str, str]:
        def _fmt(v: float, prec: int = 6) -> str:
            if v is None:
                return ""
            if isinstance(v, float) and math.isnan(v):
                return ""
            if isinstance(v, float):
                return f"{v:.{prec}f}"
            return str(v)

        return {
            "song": self.song,
            "midi_path": self.midi_path,
            "plan_seconds": _fmt(self.plan_seconds, 3),
            "rollout_seconds": _fmt(self.rollout_seconds, 3),
            "total_seconds": _fmt(self.total_seconds, 3),
            "static_f1": _fmt(self.static_f1, 6),
            "rollout_frame_f1": _fmt(self.rollout_frame_f1, 6),
            "event_f1": _fmt(self.event_f1, 6),
            "matched": _fmt(self.matched, 0),
            "missed": _fmt(self.missed, 0),
            "mispresses": _fmt(self.mispresses, 0),
            "target": _fmt(self.target, 0),
            "error": self.error or "",
        }


# ---------------------------------------------------------------------------
# Manifest parsing.
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(name: str) -> str:
    stem = Path(name).stem
    slug = _SLUG_RE.sub("_", stem).strip("_")
    return slug or "song"


@dataclass
class ManifestEntry:
    song: str
    midi_path: Path
    run_name: str


def load_manifest(
    manifest_path: Path,
    manifest_root: Optional[Path],
    filter_split: Optional[str],
) -> list[ManifestEntry]:
    if manifest_root is None:
        manifest_root = manifest_path.parent
    manifest_root = manifest_root.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries: list[ManifestEntry] = []
    seen_run_names: set[str] = set()

    with manifest_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "midi_filename" not in reader.fieldnames:
            raise ValueError(
                f"Manifest {manifest_path} missing required column 'midi_filename'"
            )

        for row_idx, row in enumerate(reader):
            midi_filename = (row.get("midi_filename") or "").strip()
            if not midi_filename:
                print(
                    f"[manifest] row {row_idx}: missing midi_filename, skipping",
                    file=sys.stderr,
                )
                continue
            if filter_split is not None:
                split = (row.get("split") or "").strip()
                if split != filter_split:
                    continue

            midi_path = Path(midi_filename)
            if not midi_path.is_absolute():
                midi_path = (manifest_root / midi_path).resolve()
            else:
                midi_path = midi_path.resolve()

            song_label = midi_filename
            composer = (row.get("canonical_composer") or "").strip()
            title = (row.get("canonical_title") or "").strip()
            if composer or title:
                song_label = " - ".join([s for s in (composer, title) if s])

            base = slugify(midi_filename)
            run_name = base
            suffix = 1
            while run_name in seen_run_names:
                suffix += 1
                run_name = f"{base}_{suffix}"
            seen_run_names.add(run_name)

            entries.append(
                ManifestEntry(
                    song=song_label,
                    midi_path=midi_path,
                    run_name=run_name,
                )
            )

    return entries


# ---------------------------------------------------------------------------
# Command builders.
# ---------------------------------------------------------------------------
def build_plan_command(
    midi_path: Path,
    runs_root: Path,
    run_name: str,
    max_duration_s: float,
    active_window_last_s: float,
    seed: int,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(PLAN_SCRIPT),
        "--midi-path", str(midi_path),
        "--output-root", str(runs_root),
        "--run-name", run_name,
        "--seed", str(int(seed)),
    ]
    cmd.extend(BASELINE_PLAN_FLAGS)
    if max_duration_s and max_duration_s > 0:
        cmd.extend(["--max-duration-s", f"{max_duration_s}"])
    if active_window_last_s and active_window_last_s > 0:
        cmd.extend(["--active-window-last-s", f"{active_window_last_s}"])
    return cmd


def build_rollout_command(
    runs_root: Path,
    rollout_root: Path,
    run_name: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--run-root", str(runs_root),
        "--output-root", str(rollout_root),
        "--only-run", run_name,
        "--environment-name", ENV_NAME,
        "--threshold", "0.5",
        "--set-hand-qvel",
    ]


# ---------------------------------------------------------------------------
# JSON / metric parsing helpers.
# ---------------------------------------------------------------------------
def _safe_float(v) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return float("nan")


def _stderr_tail(stderr: str) -> str:
    if not stderr:
        return ""
    tail = stderr.strip()
    return tail[-200:].replace("\n", " | ")


def compute_static_f1(meta: dict) -> float:
    tp = _safe_float(meta.get("static_contact_hits_total"))
    fn = _safe_float(meta.get("static_contact_misses_total"))
    fp = _safe_float(meta.get("static_contact_wrongs_total"))
    if any(math.isnan(v) for v in (tp, fp, fn)):
        return float("nan")
    denom = tp + 0.5 * (fp + fn)
    if denom <= 0:
        return float("nan")
    return tp / denom


def parse_rollout_result(path: Path, result: SongResult) -> None:
    try:
        with path.open("r") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        result.error = f"rollout_result_parse_error: {exc}"
        return
    result.rollout_frame_f1 = _safe_float(data.get("frame_f1"))
    result.event_f1 = _safe_float(data.get("event_f1"))
    result.matched = _safe_float(data.get("matched"))
    result.missed = _safe_float(data.get("missed"))
    result.mispresses = _safe_float(data.get("mispresses"))
    result.target = _safe_float(data.get("target"))


# ---------------------------------------------------------------------------
# Per-song worker.
# ---------------------------------------------------------------------------
def evaluate_song(
    song: str,
    midi_path_str: str,
    run_name: str,
    runs_root_str: str,
    rollout_root_str: str,
    max_duration_s: float,
    active_window_last_s: float,
    seed: int,
) -> SongResult:
    midi_path = Path(midi_path_str)
    runs_root = Path(runs_root_str)
    rollout_root = Path(rollout_root_str)
    result = SongResult(song=song, midi_path=str(midi_path), run_name=run_name)

    if not midi_path.exists():
        result.error = f"midi_missing: {midi_path}"
        return result

    runs_root.mkdir(parents=True, exist_ok=True)
    rollout_root.mkdir(parents=True, exist_ok=True)

    # ------- Plan -------
    plan_cmd = build_plan_command(
        midi_path=midi_path,
        runs_root=runs_root,
        run_name=run_name,
        max_duration_s=max_duration_s,
        active_window_last_s=active_window_last_s,
        seed=seed,
    )
    t0 = time.perf_counter()
    try:
        plan_proc = subprocess.run(
            plan_cmd,
            text=True,
            capture_output=True,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:  # noqa: BLE001
        result.plan_seconds = time.perf_counter() - t0
        result.error = f"plan_exception: {exc}"
        return result
    result.plan_seconds = time.perf_counter() - t0

    if plan_proc.returncode != 0:
        result.error = f"plan_failed: {_stderr_tail(plan_proc.stderr)}"
        return result

    meta_path = runs_root / run_name / "metadata.json"
    if not meta_path.exists():
        result.error = "metadata_missing"
        return result
    try:
        with meta_path.open("r") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        result.error = f"metadata_parse_error: {exc}"
        return result
    result.static_f1 = compute_static_f1(meta)

    # ------- Rollout -------
    rollout_cmd = build_rollout_command(
        runs_root=runs_root,
        rollout_root=rollout_root,
        run_name=run_name,
    )
    t1 = time.perf_counter()
    try:
        rollout_proc = subprocess.run(
            rollout_cmd,
            text=True,
            capture_output=True,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:  # noqa: BLE001
        result.rollout_seconds = time.perf_counter() - t1
        result.total_seconds = result.plan_seconds + result.rollout_seconds
        result.error = f"rollout_exception: {exc}"
        return result
    result.rollout_seconds = time.perf_counter() - t1
    result.total_seconds = result.plan_seconds + result.rollout_seconds

    if rollout_proc.returncode != 0:
        result.error = f"rollout_failed: {_stderr_tail(rollout_proc.stderr)}"
        return result

    retest_path = rollout_root / run_name / "impromptu_rp1m_retest_result.json"
    if not retest_path.exists():
        result.error = "rollout_result_missing"
        return result
    parse_rollout_result(retest_path, result)
    return result


def harvest_existing(
    song: str,
    midi_path: Path,
    run_name: str,
    runs_root: Path,
    rollout_root: Path,
) -> SongResult:
    """Build a SongResult from on-disk artifacts (resume mode)."""
    result = SongResult(song=song, midi_path=str(midi_path), run_name=run_name)
    meta_path = runs_root / run_name / "metadata.json"
    if meta_path.exists():
        try:
            with meta_path.open("r") as fh:
                meta = json.load(fh)
            result.static_f1 = compute_static_f1(meta)
        except (OSError, json.JSONDecodeError):
            pass
    retest_path = rollout_root / run_name / "impromptu_rp1m_retest_result.json"
    if not retest_path.exists():
        result.error = "rollout_result_missing"
        return result
    parse_rollout_result(retest_path, result)
    result.error = (result.error or None)
    if result.error is None:
        # Mark resumed rows by leaving plan/rollout timings as NaN — the
        # CSV writer emits empty strings for NaN, which is the desired
        # signal for "this row was harvested, not freshly produced".
        pass
    return result


# ---------------------------------------------------------------------------
# Driver: pool, CSV, progress logging, summary footer.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, default=None)
    p.add_argument("--filter-split", default=None,
                   help="If set, only run rows whose 'split' equals this value (e.g. 'test').")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--csv-out", type=Path, required=True)
    p.add_argument("--parallelism", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-duration-s", type=float, default=30.0,
                   help="Clip MIDI to this many seconds (0 disables).")
    p.add_argument("--active-window-last-s", type=float, default=28.0,
                   help="Planner active window length.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="Print per-song commands for the first 3 songs; don't execute.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    csv_out = args.csv_out.resolve()
    runs_root = output_root / "runs"
    rollout_root = output_root / "rollout"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    rollout_root.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path, args.manifest_root, args.filter_split)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        print("No manifest entries to process.", file=sys.stderr)
        return 1

    n_total = len(entries)
    print(f"[driver] {n_total} songs from {manifest_path}", file=sys.stderr)

    if args.dry_run:
        for entry in entries[:3]:
            plan_cmd = build_plan_command(
                midi_path=entry.midi_path,
                runs_root=runs_root,
                run_name=entry.run_name,
                max_duration_s=args.max_duration_s,
                active_window_last_s=args.active_window_last_s,
                seed=args.seed,
            )
            rollout_cmd = build_rollout_command(
                runs_root=runs_root,
                rollout_root=rollout_root,
                run_name=entry.run_name,
            )
            print(f"# {entry.song}")
            print(" ".join(shlex.quote(c) for c in plan_cmd))
            print(" ".join(shlex.quote(c) for c in rollout_cmd))
            print()
        return 0

    # Partition resumable songs.
    pending: list[ManifestEntry] = []
    harvested: list[SongResult] = []
    if args.resume:
        for entry in entries:
            retest_path = rollout_root / entry.run_name / "impromptu_rp1m_retest_result.json"
            if retest_path.exists():
                harvested.append(
                    harvest_existing(
                        song=entry.song,
                        midi_path=entry.midi_path,
                        run_name=entry.run_name,
                        runs_root=runs_root,
                        rollout_root=rollout_root,
                    )
                )
            else:
                pending.append(entry)
    else:
        pending = list(entries)

    csv_mode = "a" if (args.resume and csv_out.exists()) else "w"
    csv_fh = csv_out.open(csv_mode, newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    if csv_mode == "w":
        writer.writeheader()
        csv_fh.flush()

    all_results: list[SongResult] = []

    # Emit harvested rows immediately.
    for h in harvested:
        writer.writerow(h.to_row())
        csv_fh.flush()
        all_results.append(h)
        idx = len(all_results)
        f1 = h.rollout_frame_f1
        print(
            f"[{idx}/{n_total}] {h.run_name} resumed F1={f1 if not math.isnan(f1) else float('nan'):.3f}",
            file=sys.stderr,
        )

    wall_t0 = time.perf_counter()

    if pending:
        # Use 'spawn' start method so the pool does not inherit half-initialized
        # MuJoCo/EGL state on Linux (the default 'fork' causes 16-way workers to
        # hang on the RTX 5080 server). Matches macOS default behaviour.
        _ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, int(args.parallelism)), mp_context=_ctx) as pool:
            future_to_entry = {}
            for entry in pending:
                fut = pool.submit(
                    evaluate_song,
                    entry.song,
                    str(entry.midi_path),
                    entry.run_name,
                    str(runs_root),
                    str(rollout_root),
                    float(args.max_duration_s),
                    float(args.active_window_last_s),
                    int(args.seed),
                )
                future_to_entry[fut] = entry

            for fut in as_completed(future_to_entry):
                entry = future_to_entry[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    result = SongResult(
                        song=entry.song,
                        midi_path=str(entry.midi_path),
                        run_name=entry.run_name,
                        error=f"worker_exception: {exc}",
                    )
                writer.writerow(result.to_row())
                csv_fh.flush()
                all_results.append(result)
                idx = len(all_results)
                tot = result.total_seconds
                tot_str = f"{tot:.1f}" if not math.isnan(tot) else "nan"
                f1 = result.rollout_frame_f1
                f1_str = f"{f1:.3f}" if not math.isnan(f1) else "nan"
                tag = "FAIL" if result.error else "done"
                print(
                    f"[{idx}/{n_total}] {entry.run_name} {tag} in {tot_str} sec  F1={f1_str}",
                    file=sys.stderr,
                )

    wall_elapsed = time.perf_counter() - wall_t0
    csv_fh.close()

    # ----- Summary footer -----
    n = len(all_results)
    succeeded = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    def _vals(attr: str, source: Iterable[SongResult]) -> list[float]:
        out: list[float] = []
        for r in source:
            v = getattr(r, attr)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            out.append(float(v))
        return out

    def _mean(vals: list[float]) -> float:
        return statistics.fmean(vals) if vals else float("nan")

    def _median(vals: list[float]) -> float:
        return statistics.median(vals) if vals else float("nan")

    plan_vals = _vals("plan_seconds", succeeded)
    rollout_vals = _vals("rollout_seconds", succeeded)
    total_vals = _vals("total_seconds", succeeded)
    f1_vals = _vals("rollout_frame_f1", succeeded)

    mean_plan = _mean(plan_vals)
    mean_rollout = _mean(rollout_vals)
    mean_total = _mean(total_vals)
    median_total = _median(total_vals)
    mean_f1 = _mean(f1_vals)
    median_f1 = _median(f1_vals)
    above = sum(1 for v in f1_vals if v >= F1_TARGET)
    above_pct = (above / len(f1_vals) * 100.0) if f1_vals else 0.0
    sum_total_h = (sum(total_vals) / 3600.0) if total_vals else 0.0
    wall_par_h = wall_elapsed / 3600.0

    def _fmt(v: float, prec: int = 1) -> str:
        return f"{v:.{prec}f}" if not math.isnan(v) else "nan"

    print("========================================")
    print(f"Songs total:     {n}")
    print(f"Songs succeeded: {len(succeeded)}")
    print(f"Songs failed:    {len(failed)}")
    print(f"Mean plan time:        {_fmt(mean_plan)} s")
    print(f"Mean rollout time:     {_fmt(mean_rollout)} s")
    print(f"Mean total per song:   {_fmt(mean_total)} s")
    print(f"Median per song:       {_fmt(median_total)} s")
    print(f"Mean rollout frame F1: {_fmt(mean_f1, 3)}")
    print(f"Median rollout F1:     {_fmt(median_f1, 3)}")
    if f1_vals:
        print(
            f"F1 >= {F1_TARGET:.3f}:           {above} / {len(f1_vals)} "
            f"({above_pct:.0f}%)"
        )
    else:
        print(f"F1 >= {F1_TARGET:.3f}:           0 / 0 (n/a)")
    print(f"Total wall (sum):      {sum_total_h:.1f} h")
    print(
        f"Wall with parallelism: {wall_par_h:.1f} h "
        f"(parallelism={args.parallelism})"
    )
    print("========================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
