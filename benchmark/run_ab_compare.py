#!/usr/bin/env python3
"""A/B comparison harness for plan_trajectory.py planner variants.

Runs N trials of each variant, captures wall time via /usr/bin/time -p and
quality metrics from metadata.json, and emits a markdown comparison table.

Usage:
    python benchmark/run_ab_compare.py \\
        --variant baseline \\
        --variant keyset_cache --extra-args "--ik-cache-mode exact_and_warm_start" \\
        --trials 3 \\
        --output benchmark/results/ab_compare.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path("/Users/aangeles/robopiano").resolve()
PLAN_SCRIPT = REPO_ROOT / "Impromptu" / "scripts" / "plan_trajectory.py"
MIDI_GEN_SCRIPT = REPO_ROOT / "benchmark" / "gen_test_midi.py"
DEFAULT_MIDI = Path("/tmp/dense_test.mid")
OUTPUT_ROOT = Path("/tmp/ab_compare_run")

# Baseline flags appended to every variant invocation. The variant's
# --extra-args are appended on top. Per-run flags (--midi-path,
# --output-root, --run-name) are injected dynamically.
BASELINE_FLAGS = [
    "--environment-name", "RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    "--trajectory-mode", "joint_space_straighten",
    "--disable-adaptive-complex-song-defaults",
    "--max-duration-s", "17.0",
    "--active-window-last-s", "15.0",
    "--key-press-depth", "0.006",
    "--wrong-hand-penalty", "4.0",
    "--wrong-hand-split-key", "48",
    "--assignment-dynamic-hand-split",
    "--assignment-strategy", "legacy_previous_pose",
    "--assignment-fail-if-unassigned",
    "--anchor-stride", "1",
    "--ik-max-nfev", "80",
    "--residual-success-threshold", "0.02",
    "--ik-static-contact-validation",
    "--ik-static-contact-settle-steps", "1",
    "--disable-ik-multistart-on-failure",
]


@dataclass
class TrialResult:
    variant: str
    trial: int
    wall_time_s: float = float("nan")
    ik_success_count: float = float("nan")
    num_ik_anchor_frames: float = float("nan")
    ik_residual_mean: float = float("nan")
    ik_residual_p95: float = float("nan")
    static_hits: float = float("nan")
    static_misses: float = float("nan")
    static_wrongs: float = float("nan")
    unassigned_keys_total: float = float("nan")
    num_waypoints: float = float("nan")
    error: Optional[str] = None


@dataclass
class VariantAgg:
    variant: str
    trials: list[TrialResult] = field(default_factory=list)
    wall_time_median: float = float("nan")
    wall_time_iqr: float = float("nan")
    ik_success_fraction: float = float("nan")
    ik_residual_mean: float = float("nan")
    ik_residual_p95: float = float("nan")
    static_contact_f1: float = float("nan")
    unassigned_keys_total: float = float("nan")
    num_waypoints: float = float("nan")
    error_summary: Optional[str] = None


def parse_real_time(stderr: str) -> float:
    """Parse `real X.XX` from /usr/bin/time -p stderr."""
    for line in stderr.splitlines():
        m = re.match(r"\s*real\s+([0-9]+\.?[0-9]*)\s*$", line)
        if m:
            return float(m.group(1))
    return float("nan")


def build_command(variant_extra: list[str], midi_path: Path, run_dir_name: str) -> list[str]:
    cmd = [
        "/usr/bin/time", "-p",
        sys.executable, str(PLAN_SCRIPT),
        "--midi-path", str(midi_path),
        "--output-root", str(OUTPUT_ROOT),
        "--run-name", run_dir_name,
    ]
    cmd.extend(BASELINE_FLAGS)
    cmd.extend(variant_extra)
    return cmd


def regenerate_midi() -> None:
    print(f"[ab_compare] regenerating MIDI via {MIDI_GEN_SCRIPT}", flush=True)
    rc = subprocess.run(
        [sys.executable, str(MIDI_GEN_SCRIPT)],
        text=True, capture_output=True,
    )
    if rc.returncode != 0:
        raise RuntimeError(f"MIDI generation failed:\n{rc.stderr}")


def parse_metadata(meta_path: Path, result: TrialResult) -> None:
    try:
        meta = json.loads(meta_path.read_text())
    except FileNotFoundError:
        result.error = (result.error or "") + " metadata.json missing"
        return
    except json.JSONDecodeError as exc:
        result.error = (result.error or "") + f" metadata parse error: {exc}"
        return

    def _get(key: str) -> float:
        v = meta.get(key)
        return float(v) if isinstance(v, (int, float)) else float("nan")

    result.ik_success_count = _get("ik_success_count")
    result.num_ik_anchor_frames = _get("num_ik_anchor_frames")
    result.ik_residual_mean = _get("ik_max_residual_mean")
    result.ik_residual_p95 = _get("ik_max_residual_p95")
    result.num_waypoints = _get("num_waypoints")
    result.static_hits = _get("static_contact_hits_total")
    result.static_misses = _get("static_contact_misses_total")
    result.static_wrongs = _get("static_contact_wrongs_total")
    result.unassigned_keys_total = _get("unassigned_keys_total")


def run_trial(variant: str, trial_idx: int, extra: list[str],
              midi_path: Path, keep_artifacts: bool) -> TrialResult:
    run_name = f"{variant}_trial{trial_idx}"
    run_dir = OUTPUT_ROOT / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    cmd = build_command(extra, midi_path, run_name)
    print(f"[ab_compare] running {variant} trial {trial_idx}: "
          f"{' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(REPO_ROOT))

    result = TrialResult(variant=variant, trial=trial_idx)
    result.wall_time_s = parse_real_time(proc.stderr)

    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        result.error = "rc=%d: %s" % (proc.returncode, " | ".join(tail))
    else:
        parse_metadata(run_dir / "metadata.json", result)

    if not keep_artifacts and run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    return result


def aggregate(variant: str, trials: list[TrialResult]) -> VariantAgg:
    agg = VariantAgg(variant=variant, trials=trials)
    wall = np.array([t.wall_time_s for t in trials if not math.isnan(t.wall_time_s)])
    if wall.size > 0:
        agg.wall_time_median = float(np.median(wall))
        q25, q75 = np.percentile(wall, [25, 75])
        agg.wall_time_iqr = float(q75 - q25)

    def _mean(attr: str) -> float:
        arr = np.array([getattr(t, attr) for t in trials], dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(arr.mean()) if arr.size > 0 else float("nan")

    successes = np.array([t.ik_success_count for t in trials], dtype=float)
    anchors = np.array([t.num_ik_anchor_frames for t in trials], dtype=float)
    fracs = []
    for s, a in zip(successes, anchors):
        if not math.isnan(s) and not math.isnan(a) and a > 0:
            fracs.append(s / a)
    agg.ik_success_fraction = float(np.mean(fracs)) if fracs else float("nan")

    agg.ik_residual_mean = _mean("ik_residual_mean")
    agg.ik_residual_p95 = _mean("ik_residual_p95")
    agg.unassigned_keys_total = _mean("unassigned_keys_total")
    agg.num_waypoints = _mean("num_waypoints")

    f1s = []
    for t in trials:
        tp, fn, fp = t.static_hits, t.static_misses, t.static_wrongs
        if any(math.isnan(v) for v in (tp, fn, fp)):
            continue
        denom = tp + 0.5 * (fp + fn)
        if denom <= 0:
            continue
        f1s.append(tp / denom)
    agg.static_contact_f1 = float(np.mean(f1s)) if f1s else float("nan")

    errors = [t.error for t in trials if t.error]
    if errors:
        agg.error_summary = errors[0]
    return agg


def _fmt(x: float, spec: str = ".3f") -> str:
    return "NaN" if math.isnan(x) else format(x, spec)


def write_markdown(out_path: Path, aggs: list[VariantAgg], midi_path: Path,
                   trials: int, baseline: VariantAgg) -> None:
    variant_names = ", ".join(a.variant for a in aggs)
    lines: list[str] = []
    lines.append(f"# A/B Comparison: {variant_names} over N={trials} trials each")
    lines.append("")
    lines.append(f"**MIDI:** {midi_path} (16.8s duration, 132 notes, dense bimanual)")
    lines.append("**Active window:** 15.0s")
    lines.append("")
    lines.append("| Variant | Wall time (median +/- IQR) | IK success frac | "
                 "Residual mean | Residual p95 | Static F1 |")
    lines.append("|---|---|---|---|---|---|")
    for a in aggs:
        if a.error_summary and math.isnan(a.wall_time_median):
            lines.append(f"| {a.variant} | FAILED: {a.error_summary} | - | - | - | - |")
            continue
        wall_cell = f"{_fmt(a.wall_time_median, '.2f')}s +/- {_fmt(a.wall_time_iqr, '.2f')}s"
        lines.append(
            f"| {a.variant} | {wall_cell} | {_fmt(a.ik_success_fraction, '.3f')} | "
            f"{_fmt(a.ik_residual_mean, '.4f')} | {_fmt(a.ik_residual_p95, '.4f')} | "
            f"{_fmt(a.static_contact_f1, '.3f')} |"
        )
    lines.append("")

    # Speedups vs baseline.
    lines.append("**Speedups vs baseline (median wall):**")
    base_wall = baseline.wall_time_median
    for a in aggs:
        if a.variant == baseline.variant:
            continue
        if math.isnan(base_wall) or math.isnan(a.wall_time_median) or a.wall_time_median <= 0:
            lines.append(f"- {a.variant}: n/a")
            continue
        ratio = base_wall / a.wall_time_median
        pct = (a.wall_time_median - base_wall) / base_wall * 100.0
        direction = "faster" if ratio > 1.0 else "slower"
        lines.append(f"- {a.variant}: {ratio:.2f}x {direction} ({pct:+.1f}%)")
    lines.append("")

    # Quality deltas vs baseline.
    lines.append("**Quality regressions:**")
    base_f1 = baseline.static_contact_f1
    for a in aggs:
        if a.variant == baseline.variant:
            continue
        if math.isnan(base_f1) or math.isnan(a.static_contact_f1):
            lines.append(f"- {a.variant}: F1 unavailable")
            continue
        delta = a.static_contact_f1 - base_f1
        flag = ""
        if delta < -0.03:
            flag = " (REGRESSION -- review before shipping)"
        elif abs(delta) <= 0.01:
            flag = " (within noise)"
        lines.append(f"- {a.variant}: {delta:+.3f} F1{flag}")
    lines.append("")

    # Verdict paragraph.
    faster = []
    quality_ok = []
    quality_regress = []
    for a in aggs:
        if a.variant == baseline.variant:
            continue
        if not math.isnan(a.wall_time_median) and not math.isnan(base_wall) \
                and a.wall_time_median < base_wall:
            faster.append(a.variant)
        if not math.isnan(a.static_contact_f1) and not math.isnan(base_f1):
            if a.static_contact_f1 - base_f1 < -0.03:
                quality_regress.append(a.variant)
            else:
                quality_ok.append(a.variant)
    verdict_parts: list[str] = []
    if faster:
        verdict_parts.append(f"Faster than baseline: {', '.join(faster)}.")
    else:
        verdict_parts.append("No variant beat baseline on median wall time.")
    if quality_ok:
        verdict_parts.append(f"Quality preserved: {', '.join(quality_ok)}.")
    if quality_regress:
        verdict_parts.append(
            f"Quality regressed (>0.03 F1): {', '.join(quality_regress)} -- "
            "do not ship without investigation."
        )
    sanity = []
    waypoints = [a.num_waypoints for a in aggs if not math.isnan(a.num_waypoints)]
    if waypoints and (max(waypoints) - min(waypoints) > 0.5):
        sanity.append("num_waypoints differs across variants (expected identical)")
    if sanity:
        verdict_parts.append("Sanity: " + "; ".join(sanity) + ".")
    lines.append("**Verdict:**")
    lines.append(" ".join(verdict_parts))
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[ab_compare] wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", action="append", dest="variants", default=[],
                   help="Variant label (repeatable, paired with --extra-args)")
    p.add_argument("--extra-args", action="append", dest="extra_args", default=[],
                   help="Extra flags for the preceding --variant (use \"\" for baseline)")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--output", type=str,
                   default=str(REPO_ROOT / "benchmark" / "results" / "ab_compare.md"))
    p.add_argument("--keep-artifacts", action="store_true")
    p.add_argument("--midi", type=str, default=str(DEFAULT_MIDI))
    p.add_argument("--regenerate-midi", action="store_true")
    return p.parse_args()


def pair_variants(names: list[str], extras: list[str]) -> list[tuple[str, list[str]]]:
    """Pair each --variant with the --extra-args that follows it on the CLI.

    argparse with action='append' loses the relative order between two flags,
    so we walk sys.argv to reconstruct the pairing.
    """
    pairs: list[tuple[str, list[str]]] = []
    argv = sys.argv[1:]
    pending: Optional[str] = None
    for i, tok in enumerate(argv):
        if tok == "--variant":
            if pending is not None:
                pairs.append((pending, []))
            pending = argv[i + 1] if i + 1 < len(argv) else None
        elif tok == "--extra-args" and pending is not None:
            val = argv[i + 1] if i + 1 < len(argv) else ""
            pairs.append((pending, shlex.split(val)))
            pending = None
    if pending is not None:
        pairs.append((pending, []))
    if not pairs:
        # Fall back to index alignment.
        for i, name in enumerate(names):
            extra = extras[i] if i < len(extras) else ""
            pairs.append((name, shlex.split(extra)))
    return pairs


def main() -> int:
    args = parse_args()
    if not args.variants:
        print("error: at least one --variant required", file=sys.stderr)
        return 2

    midi_path = Path(args.midi).resolve()
    if args.regenerate_midi or not midi_path.exists():
        regenerate_midi()
    if not midi_path.exists():
        print(f"error: MIDI {midi_path} does not exist after regeneration", file=sys.stderr)
        return 2

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    pairs = pair_variants(args.variants, args.extra_args)
    # Ensure baseline-first ordering.
    pairs.sort(key=lambda p: (0 if p[0] == "baseline" else 1))

    aggs: list[VariantAgg] = []
    for name, extra in pairs:
        trials: list[TrialResult] = []
        for i in range(args.trials):
            try:
                trials.append(run_trial(name, i, extra, midi_path, args.keep_artifacts))
            except Exception as exc:  # noqa: BLE001
                t = TrialResult(variant=name, trial=i, error=f"harness exception: {exc}")
                trials.append(t)
        aggs.append(aggregate(name, trials))

    baseline = next((a for a in aggs if a.variant == "baseline"), aggs[0])
    write_markdown(Path(args.output).resolve(), aggs, midi_path, args.trials, baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
