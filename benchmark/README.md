# Planner benchmark scripts

This directory holds a small smoke + profiling harness for the Impromptu
planner pipeline. The goal is to confirm where time is being spent so we
can target IK / FK / assignment bottlenecks without guessing.

## What each script does

| Script | Purpose |
| --- | --- |
| `activate_env.sh` | Sources the conda env and exports `PYTHONPATH` for the subprojects. Provisioned by a separate process; the other scripts simply `source` it. |
| `gen_test_midi.py` | Writes `/tmp/twinkle.mid` using `robopianist.music.library.twinkle_twinkle_rousseau()`. Falls back to a minimal `pretty_midi` Twinkle around middle C if RoboPianist is unavailable. |
| `run_smoke.sh` | Runs the canonical short-song `plan_trajectory.py` smoke command, captures stdout/stderr, and records wall-clock time via `/usr/bin/time -p`. |
| `run_profile.sh` | Re-runs the smoke command under cProfile, scalene, and py-spy. Each profiler is wrapped so a single failure does not kill the run. |
| `analyze_ik_metrics.py` | Loads `ik_metrics.npy` from the smoke output and emits per-column statistics (mean / median / p95 / p99 / max) plus derived totals. |
| `analyze_profile.py` | Parses `profile.cprof` with `pstats` and writes a stage-by-stage bottleneck report at `results/bottleneck_report.md`. |

## Order to run

```bash
# One-time activation in your shell session
source benchmark/activate_env.sh

# 1. Sanity check: does the planner actually succeed on the smoke input?
benchmark/run_smoke.sh

# 2. Per-waypoint IK metric summary (cheap, reads the .npy from step 1).
python benchmark/analyze_ik_metrics.py

# 3. Profile under three profilers. This re-runs the planner multiple
#    times; expect several minutes on a Mac.
benchmark/run_profile.sh

# 4. Translate cProfile output into a bottleneck report.
python benchmark/analyze_profile.py
```

`run_smoke.sh` and `run_profile.sh` both call `gen_test_midi.py`
internally, so the MIDI file is regenerated automatically if it is
missing.

## Where results land

All artifacts go to `benchmark/results/`:

- `smoke_stdout.log`, `smoke_stderr.log` — planner output from the smoke run
- `smoke_wall_time.txt` — `real`/`user`/`sys` block from `/usr/bin/time -p`
- `profile.cprof`, `profile_top40.txt` — cProfile binary + cumulative top-40
- `scalene_profile.html` — line-level scalene HTML report
- `py_spy_flame.svg` — py-spy flame graph (if py-spy could attach)
- `cprofile_stdout.log` / `cprofile_stderr.log` etc. — per-profiler logs
- `profile_log.txt` — start/end timestamps for each profiler attempt
- `ik_metrics_summary.txt` — per-column IK statistics
- `bottleneck_report.md` — stage-by-stage summary (see below)

Planner artifacts themselves land in `/tmp/maestroso_smoke/twinkle_smoke/`.

## Interpreting `bottleneck_report.md`

The report has four sections:

1. **Total wall time** — from `/usr/bin/time -p` if available, otherwise
   `unknown`. cProfile underestimates wall time slightly because it
   misses time spent in C code that does not release the GIL.
2. **Top hotspots** — the top 10 user-code entries by cumulative time
   (cProfile's `ct` field). Builtins, the stdlib, and pytest internals
   are filtered out. This is where to look for surprises.
3. **By stage** — buckets of interest with their share of cProfile total
   time:
     - `IK solver (least_squares)` matches `scipy.optimize.least_squares`
       (the bottleneck identified at `Bagatelle/src/bagatelle/kinematics.py:674-682`).
     - `MuJoCo FK (physics.forward)` matches MuJoCo's `mj_forward` and
       the `physics.forward` wrapper (`kinematics.py:264-269`).
     - `Hungarian assignment` matches `scipy.optimize.linear_sum_assignment`.
     - `Contact validation` matches `activation_for_qpos` and related
       helpers invoked from the per-waypoint loop
       (`planner.py:550-671`).
     - `Other` is whatever total time the named stages did not cover.
4. **Verdict** — single-sentence call-out of the largest named stage.

Stage percentages can sum to slightly more than 100% if a stage's
matched functions are called from inside another (e.g. `least_squares`
calling `mj_forward`); both will be counted because cumulative time
includes children. Use the **Top hotspots** list for tie-breaking when
that happens.

## Mac-specific gotchas

- **py-spy** requires the target process to either run as root or be
  attached via a codesigned `py-spy` binary. `run_profile.sh` deliberately
  tries py-spy without `sudo` first and logs the failure if attach fails,
  so the rest of the profile pass still produces useful artifacts. If
  you actually need py-spy locally, codesign the binary
  (`codesign -s - -f $(which py-spy)`) or rerun under `sudo`.
- **MuJoCo rendering** uses `MUJOCO_GL`. On macOS the default `glfw`
  backend works for offscreen render; if a profiler run hangs at startup
  on a headless Mac, set `export MUJOCO_GL=egl` (Linux) or
  `MUJOCO_GL=osmesa` (no GPU) before sourcing `activate_env.sh`. The
  planner itself does not require rendering for the smoke command.
- **Bash 3.2** is Apple's default `/bin/bash`. The scripts here avoid
  associative arrays and other bash 4+ features so they run as-is from
  any shell.
