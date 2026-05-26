"""Translate the cProfile output into a stage-by-stage bottleneck report.

Reads ``benchmark/results/profile.cprof`` and:
  * sums the time spent inside specific keyword groups
    (least_squares, mj_forward/physics.forward, linear_sum_assignment,
    activation_for_qpos / contact validation),
  * computes each group's share of total cProfile-measured time,
  * extracts the top 10 user-code entries by cumulative time
    (excluding builtins and the stdlib),
  * writes the result to ``benchmark/results/bottleneck_report.md``.

If the wall-time file from ``run_smoke.sh`` is present, the report also
shows the externally measured wall time for context.
"""

from __future__ import annotations

import os
import pstats
import re
import sys
from typing import Iterable

RESULTS_DIR = "/Users/aangeles/robopiano/benchmark/results"
CPROF_PATH = os.path.join(RESULTS_DIR, "profile.cprof")
WALL_TIME_PATH = os.path.join(RESULTS_DIR, "smoke_wall_time.txt")
REPORT_PATH = os.path.join(RESULTS_DIR, "bottleneck_report.md")

# Each stage maps to one or more substring patterns matched against the
# "filename:lineno(funcname)" key cProfile records. We use substrings rather
# than regexes for clarity; entries are case-insensitive.
STAGES = (
    ("IK solver (least_squares)", ("least_squares",)),
    ("MuJoCo FK (physics.forward)", ("mj_forward", "physics.forward")),
    ("Hungarian assignment", ("linear_sum_assignment",)),
    ("Contact validation", ("activation_for_qpos", "contact_valid",
                            "static_contact")),
)


STDLIB_HINTS = (
    "/lib/python",
    "/python3",
    "{built-in",
    "<built-in",
    "<frozen",
    "<string>",
    "site-packages/_pytest",
)


def _is_user_code(label: str) -> bool:
    lowered = label.lower()
    if any(hint in lowered for hint in STDLIB_HINTS):
        return False
    # Anything inside the repo or our key third-party libs counts as user code.
    return ("robopiano" in lowered or "bagatelle" in lowered
            or "impromptu" in lowered or "robopianist" in lowered
            or "site-packages" in lowered)


def _label_for(func_tuple: tuple) -> str:
    filename, lineno, funcname = func_tuple
    return f"{filename}:{lineno}({funcname})"


def _stage_totals(stats: pstats.Stats, total_tt: float) -> list[tuple[str, float, float]]:
    """Return (stage_name, cumulative_seconds, fraction_of_total) per stage."""
    results: list[tuple[str, float, float]] = []
    accounted = 0.0
    for stage_name, patterns in STAGES:
        cum = 0.0
        for func, record in stats.stats.items():  # type: ignore[attr-defined]
            label = _label_for(func).lower()
            if any(p.lower() in label for p in patterns):
                # record == (cc, nc, tt, ct, callers); ct = cumulative time.
                cum += record[3]
        accounted += cum
        fraction = (cum / total_tt) if total_tt > 0 else float("nan")
        results.append((stage_name, cum, fraction))

    other_cum = max(total_tt - accounted, 0.0)
    other_frac = (other_cum / total_tt) if total_tt > 0 else float("nan")
    results.append(("Other", other_cum, other_frac))
    return results


def _top_user_entries(stats: pstats.Stats, n: int = 10) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for func, record in stats.stats.items():  # type: ignore[attr-defined]
        label = _label_for(func)
        if not _is_user_code(label):
            continue
        # record fields: (cc, nc, tt, ct, callers)
        rows.append((label, record[3], record[2]))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:n]


def _read_wall_time() -> str | None:
    if not os.path.exists(WALL_TIME_PATH):
        return None
    try:
        with open(WALL_TIME_PATH, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    match = re.search(r"^real\s+([0-9.]+)", text, re.MULTILINE)
    if match:
        return f"{match.group(1)} seconds (from /usr/bin/time -p real)"
    return text.strip() or None


def _verdict(stage_results: Iterable[tuple[str, float, float]]) -> str:
    ranked = sorted(stage_results, key=lambda item: item[1], reverse=True)
    top_name, _, top_frac = ranked[0]
    if top_name == "Other":
        # Fall back to the next-largest named stage for a useful verdict.
        for entry in ranked[1:]:
            if entry[0] != "Other":
                top_name, _, top_frac = entry
                break
    pct = (top_frac * 100.0) if top_frac == top_frac else 0.0  # NaN-safe
    return (f"Largest accounted stage is **{top_name}** at "
            f"{pct:.1f}% of cProfile-measured time.")


def main() -> int:
    if not os.path.exists(CPROF_PATH):
        print(f"[analyze_profile] missing {CPROF_PATH}", file=sys.stderr)
        return 1

    stats = pstats.Stats(CPROF_PATH)
    total_tt = float(stats.total_tt)  # type: ignore[attr-defined]

    stage_results = _stage_totals(stats, total_tt)
    top_entries = _top_user_entries(stats, n=10)
    wall_time = _read_wall_time() or "unknown (smoke_wall_time.txt not found)"

    lines: list[str] = []
    lines.append("# Bottleneck Report")
    lines.append("")
    lines.append(f"**Total wall time:** {wall_time}")
    lines.append(f"**cProfile total accounted time:** {total_tt:.3f} seconds")
    lines.append("")
    lines.append("**Top hotspots:**")
    if top_entries:
        for label, ct, tt in top_entries:
            frac = (ct / total_tt * 100.0) if total_tt > 0 else float("nan")
            lines.append(f"- `{label}` cum={ct:.3f}s tt={tt:.3f}s "
                         f"({frac:.1f}% of total)")
    else:
        lines.append("- (no user-code entries discovered)")
    lines.append("")
    lines.append("**By stage:**")
    for stage_name, cum, fraction in stage_results:
        pct = (fraction * 100.0) if fraction == fraction else float("nan")
        lines.append(f"- {stage_name}: {pct:.1f}% of time ({cum:.3f}s)")
    lines.append("")
    lines.append(f"**Verdict:** {_verdict(stage_results)}")
    lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
