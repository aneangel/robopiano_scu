"""Analyze 25-song eval results and recommend the best architecture combo
that scales to the full 177/1276 MAESTRO datasets.

Reads:
  /tmp/maestro_eval_25/results.csv  (from run_full_evaluation.py)
  /tmp/maestro/eval_25.csv          (manifest with duration + composer)
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

RESULTS = Path("/tmp/maestro_eval_25/results.csv")
MANIFEST = Path("/tmp/maestro/eval_25.csv")


def parse_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    if not RESULTS.exists():
        print(f"results CSV not found: {RESULTS}")
        return

    # Load manifest (duration + composer per song)
    by_midi = {}
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            by_midi[row["midi_filename"]] = {
                "duration": float(row["duration"]),
                "composer": row["canonical_composer"],
                "title": row["canonical_title"],
            }

    # Load results
    rows = []
    with RESULTS.open() as f:
        for r in csv.DictReader(f):
            mp = r.get("midi_path", "")
            # Strip the leading /private/tmp/maestro/maestro-v3.0.0/ or similar
            key = None
            for stub in by_midi:
                if stub in mp:
                    key = stub
                    break
            meta = by_midi.get(key, {})
            r["_dur"] = meta.get("duration", float("nan"))
            r["_composer"] = meta.get("composer", "?")
            r["_title"] = meta.get("title", "?")
            r["_f1"] = parse_float(r.get("rollout_frame_f1", "nan"))
            r["_event_f1"] = parse_float(r.get("event_f1", "nan"))
            r["_total"] = parse_float(r.get("total_seconds", "nan"))
            r["_plan"] = parse_float(r.get("plan_seconds", "nan"))
            r["_roll"] = parse_float(r.get("rollout_seconds", "nan"))
            r["_err"] = r.get("error", "")
            rows.append(r)

    ok = [r for r in rows if not r["_err"] and r["_f1"] == r["_f1"]]
    fail = [r for r in rows if r["_err"]]
    print(f"Songs total: {len(rows)}")
    print(f"Successful:  {len(ok)}")
    print(f"Failed:      {len(fail)}")
    for r in fail:
        print(f"  FAILED {r.get('song','?')[:60]}: {r['_err'][:120]}")
    print()

    if not ok:
        return

    # Overall stats
    f1s = sorted(r["_f1"] for r in ok)
    totals = sorted(r["_total"] for r in ok)
    plans = sorted(r["_plan"] for r in ok)
    rolls = sorted(r["_roll"] for r in ok)

    def pct(lst, p):
        if not lst:
            return float("nan")
        i = int(len(lst) * p / 100)
        i = max(0, min(len(lst) - 1, i))
        return lst[i]

    print("======= F1 distribution =======")
    print(f"  mean   = {statistics.mean(f1s):.3f}")
    print(f"  median = {statistics.median(f1s):.3f}")
    print(f"  p25/p75 = {pct(f1s,25):.3f} / {pct(f1s,75):.3f}")
    print(f"  min/max = {min(f1s):.3f} / {max(f1s):.3f}")
    print(f"  >= 0.60: {sum(1 for x in f1s if x >= 0.60)}/{len(f1s)}  ({100*sum(1 for x in f1s if x >= 0.60)/len(f1s):.0f}%)")
    print(f"  >= 0.65: {sum(1 for x in f1s if x >= 0.65)}/{len(f1s)}  ({100*sum(1 for x in f1s if x >= 0.65)/len(f1s):.0f}%)")
    print(f"  >= 0.70: {sum(1 for x in f1s if x >= 0.70)}/{len(f1s)}  ({100*sum(1 for x in f1s if x >= 0.70)/len(f1s):.0f}%)")
    print(f"  >= 0.75: {sum(1 for x in f1s if x >= 0.75)}/{len(f1s)}  ({100*sum(1 for x in f1s if x >= 0.75)/len(f1s):.0f}%)")
    print()

    print("======= Wall time distribution =======")
    print(f"  mean total: {statistics.mean(totals):.0f}s ({statistics.mean(totals)/60:.1f} min)")
    print(f"  median:     {statistics.median(totals):.0f}s ({statistics.median(totals)/60:.1f} min)")
    print(f"  p25/p75:    {pct(totals,25):.0f}s / {pct(totals,75):.0f}s")
    print(f"  min/max:    {min(totals):.0f}s / {max(totals):.0f}s")
    print(f"  <= 60s:  {sum(1 for x in totals if x <= 60)}/{len(totals)}")
    print(f"  <= 120s: {sum(1 for x in totals if x <= 120)}/{len(totals)}")
    print(f"  <= 180s: {sum(1 for x in totals if x <= 180)}/{len(totals)}")
    print()
    print(f"  mean plan: {statistics.mean(plans):.0f}s    mean rollout: {statistics.mean(rolls):.0f}s")
    print(f"  rollout / total: {100*statistics.mean(rolls)/statistics.mean(totals):.0f}%  (sequential bottleneck)")
    print()

    # F1 vs duration bins
    print("======= F1 binned by song duration =======")
    bin_edges = [(0, 180), (180, 300), (300, 500), (500, 9999)]
    print(f"  {'bin':<14}{'count':<8}{'mean F1':<10}{'med F1':<10}{'mean total':<12}{'med total':<12}")
    for lo, hi in bin_edges:
        sub = [r for r in ok if lo <= r["_dur"] < hi]
        if not sub:
            continue
        bf1 = [r["_f1"] for r in sub]
        bt = [r["_total"] for r in sub]
        print(f"  [{lo:>3}-{hi:>4}s] {len(sub):<8}{statistics.mean(bf1):<10.3f}{statistics.median(bf1):<10.3f}"
              f"{statistics.mean(bt):<12.0f}{statistics.median(bt):<12.0f}")
    print()

    # Per-song breakdown sorted by F1
    print("======= Per-song results (sorted by F1) =======")
    ok_sorted = sorted(ok, key=lambda r: r["_f1"], reverse=True)
    print(f"  {'F1':>6}{'event':>8}{'wall':>8}{'dur':>6}  {'composer':<22}{'title'}")
    for r in ok_sorted:
        print(f"  {r['_f1']:>6.3f}{r['_event_f1']:>8.3f}{r['_total']:>7.0f}s{r['_dur']:>5.0f}s  "
              f"{r['_composer'][:22]:<22}{r['_title'][:60]}")
    print()

    # Scaling projections
    print("======= Scaling projections =======")
    test_177_total_music_s = 71928  # from earlier analysis
    full_1276_total_music_s = 519000  # estimate ~7 min avg × 1276
    wall_per_music_s = statistics.mean(r["_total"] / r["_dur"] for r in ok if r["_dur"] > 0)
    print(f"  Wall-per-second-of-music ratio: {wall_per_music_s:.2f}")
    for label, total_s in [("177 test split", test_177_total_music_s),
                            ("1276 full", full_1276_total_music_s)]:
        total_wall = total_s * wall_per_music_s
        for cores in (1, 6, 32):
            hours = total_wall / cores / 3600
            print(f"  {label:18}  {cores:>3}-way: {hours:.1f} h")


if __name__ == "__main__":
    main()
