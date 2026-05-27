"""Select a representative subset of planned trajectories for inclusion in the repo.

Reads a results CSV (from run_full_evaluation.py) and copies a stratified sample of
trajectory artifacts (trajectory.npz + metadata.json + rollout result) into a
target directory. Useful for sharing example outputs on GitHub without committing
the entire MAESTRO-eval output tree.

Selection picks:
  - top-K by F1 (best-case samples)
  - bottom-K by F1 (failure modes for analysis)
  - K representative songs spanning the duration range
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path,
                    help="output-root used by run_full_evaluation.py (contains runs/ and rollout/)")
    ap.add_argument("--target-dir", required=True, type=Path,
                    help="directory to copy the sampled trajectories into")
    ap.add_argument("--top-k", type=int, default=3, help="highest-F1 samples to copy")
    ap.add_argument("--bottom-k", type=int, default=2, help="lowest-F1 samples to copy")
    ap.add_argument("--stratified-k", type=int, default=3,
                    help="number of samples chosen to span duration range")
    args = ap.parse_args()

    if not args.results_csv.exists():
        print(f"results csv not found: {args.results_csv}", file=sys.stderr)
        return 2

    with args.results_csv.open() as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]
    if not rows:
        print("no successful rows in csv", file=sys.stderr)
        return 2

    for r in rows:
        try:
            r["_f1"] = float(r["rollout_frame_f1"])
        except (ValueError, KeyError):
            r["_f1"] = float("nan")
        try:
            r["_total"] = float(r["total_seconds"])
        except (ValueError, KeyError):
            r["_total"] = float("nan")

    rows = [r for r in rows if r["_f1"] == r["_f1"]]  # drop NaN F1
    rows_by_f1 = sorted(rows, key=lambda r: r["_f1"], reverse=True)
    selected: dict[str, dict] = {}

    def add(row: dict, tag: str) -> None:
        key = row.get("run_name") or Path(row["midi_path"]).stem
        if key not in selected:
            row = dict(row)
            row["_tag"] = tag
            selected[key] = row

    # Top-K and bottom-K by F1
    for r in rows_by_f1[: args.top_k]:
        add(r, "top")
    for r in rows_by_f1[-args.bottom_k :] if args.bottom_k > 0 else []:
        add(r, "bottom")

    # Stratified by duration if `_total` is available
    rows_by_dur = sorted(
        [r for r in rows if r["_total"] == r["_total"]],
        key=lambda r: r["_total"],
    )
    if rows_by_dur and args.stratified_k > 0:
        n = len(rows_by_dur)
        step = max(1, n // args.stratified_k)
        for i in range(0, n, step):
            add(rows_by_dur[i], "spanning")
            if sum(1 for r in selected.values() if r["_tag"] == "spanning") >= args.stratified_k:
                break

    args.target_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for key, row in selected.items():
        run_name = row.get("run_name") or key
        run_src = args.runs_root / "runs" / run_name
        rollout_src = args.runs_root / "rollout" / run_name
        dest = args.target_dir / row["_tag"] / run_name
        dest.mkdir(parents=True, exist_ok=True)

        copied = []
        for fname in ("trajectory.npz", "metadata.json", "active_window_summary.json"):
            src = run_src / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
                copied.append(fname)
        for fname in ("impromptu_rp1m_retest_result.json", "summary.json",
                      "intermezzo_score.json"):
            src = rollout_src / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
                copied.append(fname)

        manifest_rows.append({
            "run_name": run_name,
            "tag": row["_tag"],
            "song": row.get("song", ""),
            "midi_path": row.get("midi_path", ""),
            "rollout_frame_f1": row["_f1"],
            "total_seconds": row["_total"],
            "matched": row.get("matched", ""),
            "missed": row.get("missed", ""),
            "mispresses": row.get("mispresses", ""),
            "target": row.get("target", ""),
            "files_copied": ";".join(copied),
        })
        print(f"  [{row['_tag']:8}] F1={row['_f1']:.3f}  {row.get('song', run_name)[:60]}")

    manifest_path = args.target_dir / "MANIFEST.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nCopied {len(manifest_rows)} trajectories into {args.target_dir}")
    print(f"Manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
