# From planned trajectory to rollout F1

How a `trajectory.npz` becomes a key-press F1 number in this repo, and the
minimum work to anchor our static-contact F1 (0.778 at 21.63 s on the
dense_test MIDI) against the "true" rollout F1.

## 1. The canonical rollout evaluator

`retest_impromptu_rp1m_simulator.py:1-213` is the only script that turns an
Impromptu/Bagatelle `trajectory.npz` into a rollout F1.

CLI signature (`retest_impromptu_rp1m_simulator.py:149-185`):

```bash
python retest_impromptu_rp1m_simulator.py \
  --run-root <PARENT_OF_RUN_DIRS> \
  --output-root <OUTPUT_ROOT> \
  --only-run <RUN_DIR_NAME> \
  --environment-name <ENV_NAME> \
  --threshold 0.5 \
  --set-hand-qvel \
  [--seed 0] [--render-mp4] [--max-source-steps N] [--use-rp1m-anchor-offset]
```

What it does, per run dir:
1. `_dense_impromptu_arrays(...)` (`retest_impromptu_rp1m_simulator.py:42-55`)
   loads `planned_hand_joints_dense` and `target_keys` from `trajectory.npz`,
   infers `substeps = dense_T / control_T` and `dense_dt = 0.05 / substeps`.
2. `make_rp1m_trajectory_from_arrays(...)` (`rp1m_simulator/simulator.py:170-213`)
   wraps actions (zeros, 39D), dense goals, and dense hand joints into an
   `RP1MTrajectory`.
3. `simulate_rp1m_rollout(...)` (`rp1m_simulator/simulator.py:857-1100`) loads
   RoboPianist headless (`MUJOCO_GL=egl`), restores `hand_joints[0]` (and
   optionally qvel) once, then for each source step linearly interpolates qpos
   across `substeps` MuJoCo control steps and writes that qpos at every
   substep (`simulator.py:1010-1023`). Goals are baked into a `target_goals.proto`
   MIDI proto via `write_goals_proto(...)` (`simulator.py:286-312`); no real
   MIDI is read.
4. Captures piano activations into `source_played_piano`
   (`simulator.py:1093-1099`) at the source rate.
5. Scores in two places:
   - `key_metrics(...)` (`simulator.py:245-266`) called by `_score_rollout`
     (`simulator.py:811-837`) gives `against_goals.{key_precision,key_recall,
     key_f1,mispress_rate}` written into `summary.json`.
   - `score_rollout(...)` (`Intermezzo/src/intermezzo/online_eval.py:198-252`)
     produces the post-rollout frame and event metrics
     (`frame_f1`, `frame_precision`, `matched_press_events`, timing percentiles)
     written into `intermezzo_score.json`.

Outputs per run (`retest_impromptu_rp1m_simulator.py:114-140`):
- `<output>/<run>/rollout.npz` — `source_played_piano`, `goals`, hand qpos
- `<output>/<run>/summary.json` — `against_goals.key_f1`, config, runtime
- `<output>/<run>/intermezzo_score.json` — `frame_f1`, `event_f1`, timing
- `<output>/<run>/target_goals.proto` — synthesized note sequence
- `<output>/<run>/impromptu_rp1m_retest_result.json` — flat row with all
  three F1 numbers + counts + flags

`retest_impromptu_rp1m_simulator.py:166-209` aggregates rows from `--run-root`
into `<output_root>/summary.json` with mean/median F1 across runs. It scans
directories matching `maestro_*` (`retest_impromptu_rp1m_simulator.py:143-146`),
or uses `--only-run` for a single dir. Audio is off by default
(`retest_impromptu_rp1m_simulator.py:96-99`).

### How `run_accuracy_recovery_sweep.py` drives it

`Maestroso/scripts/run_accuracy_recovery_sweep.py:614-640` shells out to the
same script as a subprocess per `(song, variant)` pair, fixing `--threshold
0.5 --set-hand-qvel` and using `ENV_NAME = "RoboPianist-debug-NocturneRousseau-v0"`
(`run_accuracy_recovery_sweep.py:47`). It pulls songs from MAESTRO via
`read_maestro(...)` (`run_accuracy_recovery_sweep.py:87-115`) which reads
`maestro-v3.0.0.csv` and respects `--songs N` to cap the count.

### Is there a batch CSV-writer?

No. `retest_impromptu_rp1m_simulator.py` already iterates a directory of
`maestro_*` run dirs and writes `summary.json` with one row per run
(`retest_impromptu_rp1m_simulator.py:166-208`). It does not emit a CSV and it
does not record wall time per song. Minimum patch to make it CSV-friendly:
add `--csv` and around `retest_impromptu_rp1m_simulator.py:189` write
`run,environment_name,frame_f1,event_f1,rp1m_key_f1,target,matched,played,
mispresses,seconds` to disk. Wall time would need a `t = time.time()` around
`run_one(...)` at `retest_impromptu_rp1m_simulator.py:175-185`. Roughly 20
lines of code.

## 2. The "177 test songs" subset

There is no committed file in this repo that names 177 songs. The MAESTRO v3
metadata (`maestro-v3.0.0/maestro-v3.0.0.csv`, lives at the WAVE path
`MAESTRO_ROOT` in `Maestroso/scripts/run_accuracy_recovery_sweep.py:41`) has a
`split` column with values `train`, `validation`, `test`. The published
MAESTRO v3 paper documents the test split as **177 piano performances**, so
the "177 songs" almost certainly refers to filtering that CSV with
`split == "test"`.

Where the CSV is read:
- `Maestroso/scripts/run_accuracy_recovery_sweep.py:87-115` — reads the full
  CSV, no split filter.
- `Impromptu/scripts/run_maestro_refinement_sweep.py:173-196` — reads the
  CSV and keeps `row["split"]` on each `SongSpec`.
- `Intermezzo/scripts/optimize_etude_bagatelle_online_rollout.py:200-228` —
  reads it and propagates `split`.

No script in the repo currently filters on `split == "test"`. The test split
is not pre-materialized.

Songs are identified by `midi_filename` (relative to `MAESTRO_ROOT`) and
optionally by `canonical_composer`/`canonical_title`. The
`--environment-name` flag is always
`RoboPianist-debug-NocturneRousseau-v0` for Maestroso runs
(`run_accuracy_recovery_sweep.py:47`); the env name does not encode the song
because `midi_file=...` overrides the env's bundled MIDI
(`robopianist/suite/__init__.py:75-83`).

## 3. The 1276 full dataset

The MAESTRO v3 metadata has **1276 performances** total (train 962 +
validation 137 + test 177). That matches the user's number exactly. RP1M is a
different beast — `HowToRun.md:57-61` points at
`/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr` (training-only
demonstrations, not a planning eval set). The PIG/PianoFingering subset is
150 songs (`robopianist/music/__init__.py:36`,
`robopianist/suite/__init__.py:27-29`) and is unrelated to the 1276 number.

The MIDI files do **not** live in this repo. `robopianist/music/data/` is not
committed locally (it would hold the PIG `.proto` files generated by
`robopianist preprocess`). The MAESTRO root is `WAVE`-only. Locally we only
have `/tmp/twinkle.mid`, `/tmp/dense_test.mid`, and the bundled
`twinkle_twinkle_rousseau` proto inside
`robopianist/music/library.py:337-348`.

The 177 test songs are a subset of the 1276 (by construction of the MAESTRO
v3 split).

## 4. Per-song rollout cost

For a `--max-duration-s 17.0 --active-window-last-s 15.0` plan like our
dense_test winner:
- `target_keys.shape = [T_control=336, 88]` at 20 Hz (`control_timestep=0.05`).
- Dense `planned_hand_joints_dense.shape = [T_dense=3360, 46]`, so
  `substeps=10` → `dense_dt=0.005` (200 Hz).
- `simulate_rp1m_rollout` does **3360 MuJoCo control steps** in
  `simulator.py:988-1055`, each with one `_set_hand_qpos` + `physics.forward()`
  + `env.step()` + activation capture.

The control loop is single-process, single-env, sequential. There is **no
batching across songs and no parallel env instances** in
`rp1m_simulator/simulator.py` or `retest_impromptu_rp1m_simulator.py`. The
only way to parallelize today is to launch N OS processes, one per song
(which `run_accuracy_recovery_sweep.py:919-936` does *not* — it runs serially).

Per-substep wall cost on Mac CPU (MuJoCo 3.8.1, Apple Silicon, glfw):
empirically dominated by `physics.forward()` plus `env.step()`, roughly
5-15 ms/substep based on the bottleneck profile
(`benchmark/results/bottleneck_report.md:3`, `BENCHMARK_REPORT.md:5`). For
3360 substeps that gives **17-50 s per song** wall, plus ~2-4 s env load.
The earlier estimate in `benchmark/design/f1_measurement.md:50` is 30-60 s
per song; we treat 30-60 s as the working figure for a 15 s active window.

The substep rate is set by `simulation_timestep=dense_dt=0.005`, i.e.
**200 Hz simulator control** with **20 Hz source key-press goals** in this
hand-state mode (`simulator.py:882-907`).

## 5. On-Mac feasibility

Local rollout already works on Mac CPU:
- `rp1m_simulator/simulator.py:863` sets `MUJOCO_GL=egl` by default but
  `benchmark/activate_env.sh:13` overrides to `glfw` for Apple Silicon. Both
  work; egl needs the conda libstdc++.
- `Bagatelle/tests`/`Impromptu/tests` exercise the same env-load path locally
  and pass.
- No CUDA features are required for `mode=hand_state` — actions are zero,
  the only GPU code path would be Rhapsody, which is off in retest.

If we had a real MAESTRO MIDI locally at `/tmp/song.mid` and an Impromptu run
already at `/tmp/run/song_test`, the smoke rollout command would be:

```bash
python retest_impromptu_rp1m_simulator.py \
  --run-root /tmp/run --only-run song_test \
  --output-root /tmp/run/rollout \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --threshold 0.5 --set-hand-qvel
```

The env name only matters for the env's bundled MIDI; since
`simulate_rp1m_rollout` synthesizes its own MIDI proto from `goals`
(`simulator.py:909-913`), any debug env name works as a shell.

## 6. The smallest validation we can do TODAY

Recommended single test:

1. Re-plan the winning config on `/tmp/dense_test.mid` (we already have it)
   into a fresh run dir under `/tmp/rollout_anchor/winner`.
2. Run `retest_impromptu_rp1m_simulator.py` on that single run.
3. Compare `impromptu_rp1m_retest_result.json:frame_f1` and
   `rp1m_key_f1` against the metadata's `static_contact_*_total` aggregated
   into a static F1 (using the formula in `benchmark/run_ab_compare.py:202-211`).

`/tmp/dense_test.mid` is the right test bed because (a) we already have the
0.778 static F1 number for it under three trials, and (b) its 16.8 s length
gives a single ~40 s rollout — fast enough to iterate. `/tmp/twinkle.mid` is
too short and too sparse to stress the rollout; the synthetic dense MIDI
remains the harder, more informative case.

Exact command sequence (after planning the winner once into
`/tmp/rollout_anchor/winner/`):

```bash
source /Users/aangeles/robopiano/benchmark/activate_env.sh
cd /Users/aangeles/robopiano

# 1. Plan the winner (~22 s on Mac per FINAL_SPEED_QUALITY_REPORT.md:14)
python Impromptu/scripts/plan_trajectory.py \
  --midi-path /tmp/dense_test.mid \
  --output-root /tmp/rollout_anchor --run-name winner \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --trajectory-mode joint_space_straighten \
  --disable-adaptive-complex-song-defaults \
  --max-duration-s 17.0 --active-window-last-s 15.0 \
  --key-press-depth 0.002 --wrong-hand-penalty 4.0 --wrong-hand-split-key 48 \
  --assignment-dynamic-hand-split --assignment-fail-if-unassigned \
  --assignment-strategy ik_aware_topk --assignment-top-k 2 \
  --anchor-stride 1 \
  --ik-max-nfev 80 --residual-success-threshold 0.02 \
  --ik-smoothness-weight 0.20 \
  --ik-static-contact-validation --ik-static-contact-settle-steps 1 \
  --disable-ik-multistart-on-failure \
  --ik-cache-mode exact_and_warm_start \
  --ik-unassigned-fingertip-strategy avoid_mispresses \
  --ik-unassigned-fingertip-avoidance-weight 64 \
  --ik-unassigned-fingertip-avoidance-radius 0.06

# 2. Rollout (~30-60 s on Mac)
python retest_impromptu_rp1m_simulator.py \
  --run-root /tmp/rollout_anchor \
  --output-root /tmp/rollout_anchor/rollout \
  --only-run winner \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --threshold 0.5 --set-hand-qvel

# 3. Read out
python -c "import json; r=json.load(open('/tmp/rollout_anchor/rollout/winner/impromptu_rp1m_retest_result.json')); print('rp1m_key_f1=', r['rp1m_key_f1'], 'frame_f1=', r['frame_f1'], 'event_f1=', r['event_f1'])"
python -c "import json; m=json.load(open('/tmp/rollout_anchor/winner/metadata.json')); h=m.get('static_contact_hits_total',0); w=m.get('static_contact_wrongs_total',0); n=m.get('static_contact_misses_total',0); d=h+0.5*(w+n); print('static_f1=', h/d if d else 0)"
```

Total wall: ~52-82 s end-to-end. Single deliverable: one line comparing
static F1 (0.778 expected) to rollout `frame_f1`.

## 7. Static vs rollout F1 correlation

Not measured in this repo. The only references are:
- `benchmark/design/f1_measurement.md:62` — predicts static F1 will be
  "slightly higher than rollout F1 for the same plan" due to transitions,
  inertia, and inter-anchor finger interference being invisible to the
  per-waypoint settle-steps=1 check.
- `benchmark/design/f1_measurement.md:136` — proposes treating
  `|static_f1 - rollout_f1| > 0.05` as a drift investigation, not a planning
  gate.
- `benchmark/results/FINAL_SPEED_QUALITY_REPORT.md:84` — flags "run rollout
  F1 once" as recommended next step. Never done.

There is no committed A/B that measured both numbers on the same plan. §6 is
the first such measurement. To turn it into a real correlation curve we
would need ≥10 (variant, MIDI) pairs covering the static F1 range 0.4-0.9
and plot rollout vs static; a single anchor point only tells us the offset
at one operating point.

## 8. Next-step command sequence

See §6. Single command block, ~80 s, produces both static and rollout F1 on
the same plan we already characterized at static F1 = 0.778.

## 9. First-cut wall-time estimate for the full splits

Per-song budget on Mac CPU (15 s active window, sequential, single env):
- planning: 22 s
- rollout: 45 s (mid-range of 30-60 s)
- IO + env load: 5 s
- **total: ~72 s/song**

| Subset | Songs | Mac CPU serial | WAVE cmp (16 cores, embarrassingly parallel) |
|---|---|---|---|
| 177 test | 177 | 177 × 72 s = **3.5 h** | 177 × 72 s / 16 ≈ **13 min** |
| 1276 full | 1276 | 1276 × 72 s ≈ **25.5 h** | 1276 × 72 s / 16 ≈ **96 min** |

WAVE numbers assume one process per CPU and no contention; in practice
expect 1.5-2x the optimistic figure because MuJoCo `forward()` is not
perfectly parallel and the rollout script reloads the env per song.
Pessimistic WAVE numbers: 20-30 min for 177 songs, 2.5-3.5 h for 1276.

For per-iteration A/B harness use, even the 177-song subset is too slow on
Mac (3.5 h is half a workday). 1-song rollout (§6) is the right granularity
for local iteration; 177-song rollout belongs on WAVE.

## 10. Open questions and risks

- **No CSV writer today.** A 20-line patch to
  `retest_impromptu_rp1m_simulator.py` adds `--csv` + per-song wall time.
  Without it, large-batch analysis means parsing N JSON files.
- **Real MAESTRO MIDIs are WAVE-only.** Anything beyond `dense_test.mid` and
  `twinkle.mid` requires either copying a handful of MIDIs to the Mac
  (cheapest, ~MB per song) or running the rollout itself on WAVE.
- **`run_accuracy_recovery_sweep.py:919-936` is sequential.** Parallelizing
  the rollout pass at the (song, variant) level would cut a 1276-song run
  from 25 h to ~1.5 h on WAVE. The Maestroso sweep already separates planning
  and rollout, so wrapping the rollout loop in a `ProcessPoolExecutor` is
  ~30 lines.
- **No committed `test_split.txt`.** Anyone reproducing the "177" number must
  re-filter the MAESTRO CSV in code. Materializing
  `Maestroso/scripts/maestro_test_split.json` (one-time output of
  `pandas.read_csv(...).query("split=='test'")`) removes ambiguity.
- **Env name vs MIDI separation.** `--environment-name` is misleading
  because `midi_file` overrides the bundled MIDI; multiple Maestroso runs
  with different MIDIs all label themselves NocturneRousseau in their JSON.
  Future risk for any per-song analysis keyed on env name.
- **Static-vs-rollout offset is unmeasured.** Until §6 lands, we cannot
  defend the static F1 number as a usable proxy for the F1 ≥ 0.65 goal. The
  rollout `key_metrics` uses the same `threshold=0.5` and 88-key piano roll,
  so the metrics are mathematically the same — the gap will be entirely
  attributable to MuJoCo dynamics across anchors, not metric mismatch.
- **Mode is `hand_state`, not `action`.** Rollout F1 here is *not* the same
  as the action-policy F1 RP1M papers report. It's the F1 you'd get if you
  could perfectly track the planned joint angles, which is the right number
  for *planning* but an upper bound for any RL controller that has to follow
  the plan.
