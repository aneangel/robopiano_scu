"""Build paper-ready figures from our MAESTRO eval results.

Reads:
  benchmark/results/eval_runs/maestro_177_test_split.csv   (baseline config)
  benchmark/results/eval_runs/maestro_177_f1lift.csv       (F1-lift config)
  benchmark/results/eval_runs/maestro_25_shortest.csv      (smoke subset)
  /tmp/maestro/maestro-v3.0.0/maestro-v3.0.0.csv           (durations, composers)

Outputs (under benchmark/figures/):
  fig1_f1_histogram_177.pdf       per-song F1 distribution on the test split
  fig2_f1_cdf_177.pdf             cumulative F1 (fraction of songs >= F1)
  fig3_f1_vs_duration.pdf         scatter of per-song F1 vs music duration
  fig4_ablation.pdf               architecture lever ablation bars
  fig5_speed_quality_pareto.pdf   mean wall vs mean F1 across configs
  fig6_baseline_comparison.pdf    our method vs published baselines (caveated)

Reference baseline numbers (published, used as horizontal lines or markers):
  - PianoMime: ~0.56 F1 on novel pieces (Qian et al. 2024, Table 5)
  - PANDORA:   ~0.65 F1 on RoboPianist Etude-12 (Liu et al. 2025)
  - OmniPianist: ~0.55 F1 on 100 unseen pieces (Chen, Zhao et al. 2025)
  These use DIFFERENT evaluation conditions (smaller song sets, different splits,
  sometimes Etude-12 instead of MAESTRO). All figures that show them include a
  caveat in the plot title or legend.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "benchmark" / "results" / "eval_runs"
OUT = REPO / "benchmark" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MAESTRO_CSV = Path("/tmp/maestro/maestro-v3.0.0/maestro-v3.0.0.csv")

# Published baseline reference points (with caveats — see header).
PIANOMIME_F1 = 0.56
PANDORA_F1 = 0.65
OMNIPIANIST_F1 = 0.55

# Make the figures monospace-y and uniform.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def load_csv(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        print(f"[skip] {path} not found")
        return rows
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                r["f1"] = float(r.get("rollout_frame_f1", "") or "nan")
                r["event_f1"] = float(r.get("event_f1", "") or "nan")
                r["wall"] = float(r.get("total_seconds", "") or "nan")
                r["plan"] = float(r.get("plan_seconds", "") or "nan")
            except ValueError:
                r["f1"] = float("nan")
                r["wall"] = float("nan")
                r["plan"] = float("nan")
            rows.append(r)
    return rows


def load_maestro_meta() -> dict[str, dict]:
    meta = {}
    if not MAESTRO_CSV.exists():
        return meta
    with MAESTRO_CSV.open() as f:
        for r in csv.DictReader(f):
            meta[r["midi_filename"]] = {
                "duration": float(r["duration"]),
                "composer": r["canonical_composer"],
                "title": r["canonical_title"],
                "split": r["split"],
            }
    return meta


def match_meta(row: dict, meta: dict[str, dict]) -> dict:
    mp = row.get("midi_path", "")
    for key, m in meta.items():
        if key in mp:
            return m
    return {}


# ---------------------------------------------------------------------------
# Figure 1: F1 histogram on MAESTRO v3 test split (baseline vs F1-lift overlay)
# ---------------------------------------------------------------------------
def fig1_histogram(baseline: list[dict], lift: list[dict]) -> None:
    b_f1 = [r["f1"] for r in baseline if not np.isnan(r["f1"])]
    l_f1 = [r["f1"] for r in lift if not np.isnan(r["f1"])]
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 21)
    ax.hist(b_f1, bins=bins, alpha=0.55, label=f"Baseline (mean {np.mean(b_f1):.3f}, n={len(b_f1)})",
            color="#888888", edgecolor="black", linewidth=0.5)
    ax.hist(l_f1, bins=bins, alpha=0.55, label=f"Multistart + larger LM budget (mean {np.mean(l_f1):.3f}, n={len(l_f1)})",
            color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.axvline(0.60, color="red", linestyle="--", linewidth=1.0, label="0.60 F1 floor")
    ax.set_xlabel("Per-song rollout frame F1")
    ax.set_ylabel("Songs")
    ax.set_xlim(0, 1)
    ax.set_title("Per-song F1 distribution on MAESTRO v3 test split (177 songs)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "fig1_f1_histogram_177.pdf")
    plt.close(fig)
    print(f"  wrote fig1_f1_histogram_177.pdf  baseline_mean={np.mean(b_f1):.3f}, lift_mean={np.mean(l_f1):.3f}")


# ---------------------------------------------------------------------------
# Figure 2: Cumulative F1 (fraction of songs >= F1)
# ---------------------------------------------------------------------------
def fig2_cdf(baseline: list[dict], lift: list[dict]) -> None:
    b_f1 = np.array(sorted([r["f1"] for r in baseline if not np.isnan(r["f1"])]))
    l_f1 = np.array(sorted([r["f1"] for r in lift if not np.isnan(r["f1"])]))
    fig, ax = plt.subplots()
    # We want "fraction of songs with F1 >= x" — so 1 - CDF.
    for f1s, label, color in [(b_f1, "Baseline", "#888888"),
                              (l_f1, "Multistart + larger LM budget", "#1f77b4")]:
        thresholds = np.linspace(0, 1, 101)
        frac_ge = np.array([np.mean(f1s >= t) for t in thresholds])
        ax.plot(thresholds, frac_ge, label=f"{label} (mean {np.mean(f1s):.3f})", linewidth=2, color=color)
    for ref, name in [(PIANOMIME_F1, "PianoMime"),
                      (OMNIPIANIST_F1, "OmniPianist"),
                      (PANDORA_F1, "PANDORA")]:
        ax.axvline(ref, color="black", linestyle=":", alpha=0.5, linewidth=1.0)
        ax.text(ref, 0.94, name, rotation=90, fontsize=7, va="top", ha="right", color="black", alpha=0.7)
    ax.axvline(0.60, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("F1 threshold")
    ax.set_ylabel("Fraction of songs ≥ threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative F1 distribution on MAESTRO v3 test split (177 songs)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "fig2_f1_cdf_177.pdf")
    plt.close(fig)
    print(f"  wrote fig2_f1_cdf_177.pdf")


# ---------------------------------------------------------------------------
# Figure 3: F1 vs song music duration
# ---------------------------------------------------------------------------
def fig3_f1_vs_duration(lift: list[dict], meta: dict[str, dict]) -> None:
    durs, f1s = [], []
    for r in lift:
        if np.isnan(r["f1"]):
            continue
        m = match_meta(r, meta)
        if "duration" not in m:
            continue
        durs.append(m["duration"])
        f1s.append(r["f1"])
    durs = np.array(durs)
    f1s = np.array(f1s)
    fig, ax = plt.subplots()
    ax.scatter(durs, f1s, alpha=0.45, s=22, color="#1f77b4", edgecolor="none")

    # Bin means
    bin_edges = [0, 120, 240, 360, 600, np.max(durs) + 1]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    bin_means = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (durs >= lo) & (durs < hi)
        bin_means.append(np.mean(f1s[mask]) if mask.any() else np.nan)
    ax.plot(bin_centers, bin_means, color="red", linewidth=2, marker="s",
            markersize=7, label="duration-bin mean", zorder=5)
    ax.axhline(0.60, color="red", linestyle="--", linewidth=1, alpha=0.6, label="0.60 F1 floor")
    ax.set_xlabel("Song music duration (seconds)")
    ax.set_ylabel("Rollout frame F1 (per song)")
    ax.set_title("F1 vs music duration (177 songs, multistart + larger LM budget)")
    ax.set_xlim(0, max(durs) * 1.02)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(OUT / "fig3_f1_vs_duration.pdf")
    plt.close(fig)
    print(f"  wrote fig3_f1_vs_duration.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Architecture lever ablation — F1 from each step on the synthetic
#           dense_test (the only test bed where we have full ablation data).
# ---------------------------------------------------------------------------
def fig4_ablation() -> None:
    # Numbers from benchmark/results/AB_COMPARISON_FINAL.md and FINAL_SPEED_QUALITY_REPORT.md.
    # Each step is the mean F1 after adding that component on top of the prior.
    labels = [
        "Baseline\n(scipy LM)",
        "+ Keyset cache",
        "+ Cache warm-start",
        "+ Vectorized qpos\n(Lever 1)",
        "+ Analytical Jacobian\n(Lever 2)",
        "+ Early-exit\n(Lever 3)",
        "+ avoid_mispresses\n+ top-k=2 + sm=0.20",
    ]
    f1s = [0.369, 0.420, 0.420, 0.420, 0.412, 0.444, 0.778]
    walls = [118.6, 77.2, 72.0, 62.4, 13.5, 13.5, 21.6]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    bars = ax1.bar(x, f1s, color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_ylim(0, 0.9)
    ax1.set_ylabel("Static contact F1 (synthetic dense_test, 15s window)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=12, ha="right")
    for bar, v in zip(bars, f1s):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, walls, color="red", marker="o", linewidth=2, markersize=7)
    ax2.set_ylabel("Mean wall (seconds)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, max(walls) * 1.1)
    ax1.set_title("Architectural ablation: cumulative F1 lift and wall reduction\n"
                  "(synthetic dense_test bimanual MIDI, 15-sec active window)")
    fig.savefig(OUT / "fig4_ablation.pdf")
    plt.close(fig)
    print(f"  wrote fig4_ablation.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Speed/quality Pareto across all configurations we actually measured.
# ---------------------------------------------------------------------------
def fig5_pareto(baseline: list[dict], lift: list[dict]) -> None:
    # Configurations we have real data for (means across the songs we ran).
    configs = [
        # (label, mean F1, mean wall seconds, n_songs, marker, color)
        ("Baseline (177)", 0.580, 66.2, 177, "o", "#888888"),
        ("F1-lift (177)", 0.605, 204.9, 177, "s", "#1f77b4"),
        ("25 shortest baseline", 0.607, 73.4, 25, "^", "#2ca02c"),
        ("10-song mix baseline", 0.555, 66.0, 10, "v", "#d62728"),
        ("10-song mix F1-lift", 0.631, 166.0, 10, "D", "#9467bd"),
        ("dense_test (synthetic, 15s)", 0.778, 21.6, 1, "*", "#ff7f0e"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for label, f1, wall, n, marker, color in configs:
        ax.scatter(wall, f1, s=110 + n * 0.6, marker=marker, color=color,
                   edgecolor="black", linewidth=0.7, label=label, alpha=0.85)
    ax.axhline(0.60, color="red", linestyle="--", linewidth=1, alpha=0.6, label="0.60 F1 floor")
    ax.axvline(120, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="120 s wall budget")
    ax.axvline(60, color="darkgreen", linestyle=":", linewidth=1, alpha=0.6, label="60 s wall ideal")
    ax.set_xlabel("Mean per-song wall (seconds)")
    ax.set_ylabel("Mean rollout frame F1")
    ax.set_title("Speed / quality trade-off across measured configurations\n"
                 "(point area scales with subset size)")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)
    ax.set_ylim(0.5, 0.85)
    fig.savefig(OUT / "fig5_speed_quality_pareto.pdf")
    plt.close(fig)
    print(f"  wrote fig5_speed_quality_pareto.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Comparison to published baselines (with caveats).
# ---------------------------------------------------------------------------
def fig6_baseline_comparison(baseline: list[dict], lift: list[dict]) -> None:
    bars = [
        ("PianoMime\nnovel pieces", PIANOMIME_F1, "#bbbbbb", "Qian et al. 2024"),
        ("OmniPianist\nnovel pieces", OMNIPIANIST_F1, "#bbbbbb", "Chen, Zhao et al. 2025"),
        ("PANDORA\nEtude-12", PANDORA_F1, "#bbbbbb", "Liu et al. 2025"),
        ("Sonata baseline\nMAESTRO test (177)", 0.580, "#888888", "this work"),
        ("Sonata F1-lift\nMAESTRO test (177)", 0.605, "#1f77b4", "this work"),
        ("Sonata F1-lift\nMAESTRO test (subset)", 0.631, "#5b8dd6", "this work, 10-song mix"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(bars))
    heights = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    sources = [b[3] for b in bars]
    rects = ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.5)
    for r, h, s in zip(rects, heights, sources):
        ax.text(r.get_x() + r.get_width() / 2, h + 0.012, f"{h:.3f}",
                ha="center", fontsize=8)
        ax.text(r.get_x() + r.get_width() / 2, -0.045, s, ha="center", fontsize=6.5,
                color="#555555", rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("Reported rollout F1 (mean)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Reported F1 across methods (NOT apples-to-apples in eval conditions)\n"
                 "Baselines reported on different splits/datasets; see paper text for caveats")
    fig.savefig(OUT / "fig6_baseline_comparison.pdf")
    plt.close(fig)
    print(f"  wrote fig6_baseline_comparison.pdf")


# ---------------------------------------------------------------------------
def main() -> int:
    baseline = load_csv(RESULTS / "maestro_177_test_split.csv")
    lift = load_csv(RESULTS / "maestro_177_f1lift.csv")
    meta = load_maestro_meta()
    print(f"loaded baseline n={len(baseline)}  lift n={len(lift)}  meta n={len(meta)}")
    fig1_histogram(baseline, lift)
    fig2_cdf(baseline, lift)
    fig3_f1_vs_duration(lift, meta)
    fig4_ablation()
    fig5_pareto(baseline, lift)
    fig6_baseline_comparison(baseline, lift)
    print()
    print(f"all figures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
