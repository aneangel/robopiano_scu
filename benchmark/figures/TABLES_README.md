# LaTeX Tables — Copy-Paste Guide

File: `benchmark/figures/paper_tables.tex` (drop into Overleaf as-is).
Requires: `\usepackage{booktabs}` in preamble.

Every number below comes from an actual eval run committed under
`benchmark/results/eval_runs/`. Numbers are reproducible from the
committed CSVs.

## Apples-to-apples coverage

These tables compare configs of OUR method on the SAME songs under the
SAME evaluation conditions:

| Table | Comparison | Apples-to-apples? |
|---|---|---|
| 1 (main) | Baseline vs lifted on the same 177 test songs | ✅ Yes |
| 2 (per-composer) | Same 177-song run sliced by composer | ✅ Yes |
| 3 (ablation) | Each lever on the same synthetic stress MIDI | ✅ Yes |
| 4 (scoreboard) | All configurations we measured (different subsets noted) | ✅ Yes, within own work |
| 5 (external) | Reference numbers from PianoMime, PANDORA, OmniPianist | ⚠️ **NOT** — different splits, made explicit in caption |
| 6 (headline) | Same as Table 1, compact form for teaser/abstract | ✅ Yes |

## What each table claims

### Table 1 — Main result on 177 MAESTRO test split
- Baseline: 0.580 F1, 66 s wall
- Lifted: **0.605 F1**, 205 s wall
- Same 177 songs, same 30 s active window, same rollout pipeline
- The only diff: LM budget (multistart seeds + nfev)
- 0 failures both rows

### Table 2 — Per-composer breakdown
- Best: Scarlatti 0.714, Mozart 0.704, Bach 0.694, Haydn 0.691
- Worst: Liszt 0.532, Rachmaninoff 0.551, Schumann 0.554, Beethoven 0.567
- Composers with ≥5 songs only
- Shows the architecture handles baroque/classical polyphony well, struggles with romantic dramatic pieces

### Table 3 — Architectural ablation
- 7 rows, each adds one component on top of the prior
- Static contact F1 on synthetic stress MIDI (15 s window)
- Wall drops from 119 s → 22 s across the ablation
- The big F1 jump (0.444 → 0.778) is at the final row (tuning), not any single lever
- Honest: the levers bought us **speed**; the tuning bought us **F1** on top

### Table 4 — Speed/quality scoreboard
- 6 rows covering every distinct config × subset we ran
- Goes from synthetic 1-song (0.778 F1, 22 s) → full 177 (0.605 F1, 205 s)
- Shows the F1 ceiling depends heavily on song mix

### Table 5 — External baselines (with caveats)
- PianoMime 0.56, OmniPianist 0.55, PANDORA 0.65, Sonata 0.605
- Caption explicitly notes different eval splits
- Sonata lands in the same operating range as published learned baselines

### Table 6 — Headline summary
- Compact version of Table 1 with crash count

## Honest qualifiers that MUST stay in the paper text

1. We evaluate on a **30 s active window** of each MAESTRO song, not the full song.
2. F1 floor of 0.60 is cleared by **0.005 margin** — honest but small.
3. Per-song wall is **205 s mean** with the lifted config — over the 2 min budget for most songs.
4. **No direct head-to-head** with PianoMime/PANDORA/OmniPianist — they use different evaluation splits. Frame as "Sonata lands in the same operating range."

## To verify the exact numbers in Tables 1–4

```bash
python - <<'PY'
import csv, statistics
for label, p in [
    ('baseline_177', 'benchmark/results/eval_runs/maestro_177_test_split.csv'),
    ('lift_177',     'benchmark/results/eval_runs/maestro_177_f1lift.csv'),
]:
    f1s, walls = [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                f1s.append(float(r['rollout_frame_f1']))
                walls.append(float(r['total_seconds']))
            except (ValueError, KeyError):
                pass
    print(f"{label}: n={len(f1s)}  meanF1={statistics.mean(f1s):.3f}  medianF1={statistics.median(f1s):.3f}"
          f"  meanWall={statistics.mean(walls):.0f}s"
          f"  >=0.60: {sum(1 for x in f1s if x>=0.60)}"
          f"  >=0.70: {sum(1 for x in f1s if x>=0.70)}")
PY
```

Expected output:
```
baseline_177: n=177  meanF1=0.580  medianF1=0.588  meanWall=66s  >=0.60: 79  >=0.70: 21
lift_177:     n=177  meanF1=0.605  medianF1=0.610  meanWall=205s  >=0.60: 98  >=0.70: 33
```
