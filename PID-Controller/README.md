# PID-Controller

Action-only P/PD/PID tracking for Impromptu/Maestroso hand-state rollouts.

Latest Twinkle full-song local results and reproduction commands are in
`results/twinkle_rhapsody_repair_report.md`.

The controller reads 20 Hz target hand states from an Impromptu `trajectory.npz`
and emits reduced 39D RoboPianist actions at 200 Hz. Each 20 Hz target interval
gets 10 simulator action steps. The source target is a 46D hand-qpos state, not
an actuator-space command. The controller projects that hand state into reduced
actuator space, performs minimum-jerk actuator target interpolation, and applies
PD control in actuator units. Minimum jerk is anchored to the fixed start of each
control window, not the live state at every substep. The rollout path uses
`rp1m_simulator` action mode and does not restore hand qpos or piano key states
during simulation.

## What Is Controlled

Reduced RoboPianist actions are position-actuator targets:

- right hand: 19 actions
- left hand: 19 actions
- sustain: 1 action

The target hand rollout is 46D:

- right hand joints: 23 qpos values
- left hand joints: 23 qpos values

Most actions map to one joint. Distal finger tendon actions affect coupled
joint pairs:

- `FFJ0 -> FFJ2 + FFJ1`
- `MFJ0 -> MFJ2 + MFJ1`
- `RFJ0 -> RFJ2 + RFJ1`
- `LFJ0 -> LFJ2 + LFJ1`

The nominal reduced order is tested in `tests/test_mapping.py`. One-to-one
joints are optimized and reported separately from coupled tendon joints. Coupled
targets are projected with weighted least squares: for coupled target joints `y`
and actuator coupling vector `a`, solve `min_x ||a*x - y||_W`. The empirical
probe below validates the live environment by pulsing each action input and
recording changed hand qpos channels.

## Full Action Space Check

The uncompressed RoboPianist hand action space is not a direct 46 action to 46
hand-state map:

| Space | Env action dim | Hand actions | Hand qpos states | Direct map |
| --- | ---: | ---: | ---: | --- |
| reduced | 39 = 38 hand + 1 sustain | 38 | 46 | no |
| full | 45 = 44 hand + 1 sustain | 44 | 52 | no |

Full action space adds back `A_THJ5`, `A_THJ1`, and `A_LFJ5` per hand, but it
also adds back the matching hand qpos joints. The fixed tendon actions remain:
`A_FFJ0`, `A_MFJ0`, `A_RFJ0`, and `A_LFJ0` each control a pair of joints. For
46D Impromptu hand-state rollouts, keep the reduced 39D controller mapping.

Reproduce the static source-derived check locally:

```bash
python PID-Controller/scripts/check_action_space_dimensions.py
```

On WAVE, where the live RoboPianist stack is available, also verify the actual
`rp1m_simulator` environment specs:

```bash
python PID-Controller/scripts/check_action_space_dimensions.py \
  --live \
  --output-json /WAVE/datasets/ccoelho_lab-jlanders/PIDController/action_space_dimensions.json
```

## Environment

Use the same environment as other RoboPianist rollouts:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
source /WAVE/apps/el8/conda/envs/Python/20240305/etc/profile.d/conda.sh
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
```

## Step 1: Probe Action Mapping

```bash
python PID-Controller/scripts/probe_action_mapping.py \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/PIDController/probe \
  --environment-name RoboPianist-debug-NocturneRousseau-v0
```

Outputs:

- `action_mapping_probe.json`
- `action_mapping_probe.csv`

Each row lists the expected mapped joint(s) and the largest qpos deltas after
pulsing one action. Confirm single-joint actions mainly move their named joint
and tendon actions mainly move their coupled pair.

## Strong Hand-State Rollout Baseline

The imported `benchmark/ik-bottleneck-investigation` materials provide a good
hand-state rollout target before testing action tracking:

- `benchmark/results/ROLLOUT_F1_RESULT.md` documents the `sm18` recipe:
  rollout frame F1 `0.760`, median plan+rollout wall `29.7 s`, and event F1
  `0.681` on the dense synthetic test.
- `benchmark/design/f1_measurement.md` explains the canonical F1 path from an
  Impromptu `trajectory.npz` with `planned_hand_joints_dense` into
  `retest_impromptu_rp1m_simulator.py`.
- `benchmark/design/rollout_eval_plan.md` documents the 200 Hz hand-state
  rollout mechanics and the smallest validation command sequence.

Use the `sm18` command in `benchmark/results/ROLLOUT_F1_RESULT.md` to generate
a strong `trajectory.npz`, then run the existing hand-state retest:

```bash
python retest_impromptu_rp1m_simulator.py \
  --run-root <out> \
  --output-root <out>/rollout \
  --only-run <name> \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --threshold 0.5 \
  --set-hand-qvel
```

Once that hand-state rollout is near the documented F1, use the same
`trajectory.npz` as the input to the P/PD/PID action-only rollout below.

## Step 2: Prepare A Small Test Set

Prepare a manifest before running action-control tests. Start with one known
strong planned trajectory:

```bash
python PID-Controller/scripts/prepare_pid_trajectories.py \
  --run-root /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch \
  --output-json /WAVE/datasets/ccoelho_lab-jlanders/PIDController/prepared/sample_manifest.json \
  --limit 1
```

The manifest records the selected `trajectory.npz`, environment name, ranking
score, hand target source, and target shapes. This avoids silently changing the
test case while tuning the controller.

## Step 3: Simple P/PD Test On 20 States

Run the simplest action-only test first: P and lightly damped PD, both on only
20 hand states:

```bash
python PID-Controller/scripts/test_prepared_pid_trajectories.py \
  --trajectory-manifest /WAVE/datasets/ccoelho_lab-jlanders/PIDController/prepared/sample_manifest.json \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/simple_sample20 \
  --limit 1 \
  --controllers p,pd \
  --max-source-steps 20 \
  --resume
```

Check `simple_test_summary.csv` before any broader sweep. The first pass should
answer whether action-only control is stable, whether `hand_qpos_l2_vs_reference`
is moving in the right direction, and whether the rollout terminates early.
Controller result JSON also includes `hand_tracking_split.one_to_one_qpos_l2`,
`hand_tracking_split.coupled_qpos_l2`, and
`hand_tracking_split.actuator_signal_l2`.

## Hand-State Validity And Drift

Before tuning against F1, separate bad targets from controller error. This
script first replays the planned hand state directly at 200 Hz, then runs the
same sample through the action-only controller and scores hand qpos error at
each 20 Hz target boundary:

```bash
python PID-Controller/scripts/evaluate_hand_tracking.py \
  --trajectory-manifest /WAVE/datasets/ccoelho_lab-jlanders/PIDController/prepared/sample_manifest.json \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/hand_tracking_eval \
  --limit 3 \
  --max-source-steps 20 \
  --controller pd \
  --kp 2.8 \
  --kd 0.06 \
  --setpoint-policy minimum_jerk \
  --target-velocity-scale 0 \
  --resume
```

Outputs:

- `tracking_aggregate.csv/json` with direct hand-state F1, action F1, mean L2,
  first-third vs last-third L2, final L2, peak L2 step, and L2 slope per second
- one-to-one and coupled joint L2 metrics, so coupled tendon joints can be
  diagnosed without hiding one-to-one tracking quality
- per-run `action_tracking_by_20hz_step.csv` for degradation over the sample
- per-run `hand_state_dense/summary.json` for the direct hand-state baseline

If direct hand-state replay terminates or has poor piano F1, fix or reject that
trajectory before using it to tune action control.

## Step 4: P Controller On 20 States

Start with only 20 hand states:

```bash
python PID-Controller/scripts/validate_pid_rollouts.py \
  --trajectory-manifest /WAVE/datasets/ccoelho_lab-jlanders/PIDController/prepared/sample_manifest.json \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/p_sample20 \
  --limit 1 \
  --controller p \
  --max-source-steps 20
```

Main files:

- `summary.json`
- `summary.csv`
- per-run `pid_rollout_result.json`
- per-run `summary.json` from `rp1m_simulator`
- per-run `rollout.npz`

Check `hand_qpos_l2_vs_reference.mean`, `rp1m_key_f1`, `event_f1`, and
`terminated`.

## Step 5: PD Controller On 20 States

```bash
python PID-Controller/scripts/validate_pid_rollouts.py \
  --trajectory-manifest /WAVE/datasets/ccoelho_lab-jlanders/PIDController/prepared/sample_manifest.json \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/pd_sample20 \
  --limit 1 \
  --controller pd \
  --kp 1.0 \
  --kd 0.02 \
  --max-source-steps 20
```

Use `--setpoint-policy minimum_jerk` for the current actuator-space PD path.
For position-like RoboPianist controls, test feedforward residual control with
`--feedforward-scale 1.0`: the emitted action is the reference actuator target
plus PD residual correction. Add `--use-target-velocity
--target-velocity-scale <value>` to track the analytical minimum-jerk velocity
reference. `--lookahead-substeps` controls the preview horizon in 200 Hz control
steps; `10` is 50 ms, `20` is 100 ms, and `40` is 200 ms. If PD overshoots,
lower `--kd` first. If it lags, increase `--kp` while watching
`one_to_one_l2_mean`, `final_l2`, `last_third_l2`, and `coupled_l2_mean`
separately.

## Optimize Gains On Planned Trajectories

Run a gain search on the best planned `trajectory.npz` files before committing
to full-song validation:

```bash
python PID-Controller/scripts/optimize_pid_controller.py \
  --run-root /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/optimized_sample20 \
  --trajectory-limit 1 \
  --controllers pd \
  --kp-grid 0.7,1.0,1.3,1.6 \
  --kd-grid 0,0.005,0.01,0.02 \
  --target-velocity-scales 0,0.5,1.0 \
  --feedforward-scales 0.0,1.0 \
  --lookahead-substeps-grid 10,20,40 \
  --setpoint-policies minimum_jerk \
  --max-source-steps 20 \
  --selection-metric hand_l2_last_third_mean \
  --scale-best \
  --scale-limit 1 \
  --resume
```

The optimizer writes:

- `candidate_results.csv/json` for every candidate rollout
- `candidate_aggregates.csv` ranked by objective
- `optimization_summary.json` with the selected gains
- `scale_best_results.csv` if `--scale-best` is enabled

The objective rewards event F1, frame F1, and key F1 while penalizing mean,
max, final, and last-third hand qpos tracking error plus early termination. Use
the selected gains in `validate_pid_rollouts.py` for the full-song and top-10
runs.

## Step 6: PID Only If Needed

Only add integral when P/PD has stable tracking but steady-state error remains:

```bash
python PID-Controller/scripts/validate_pid_rollouts.py \
  --trajectory-manifest /WAVE/datasets/ccoelho_lab-jlanders/PIDController/prepared/sample_manifest.json \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/pid_sample20 \
  --limit 1 \
  --controller pid \
  --kp 1.0 \
  --kd 0.02 \
  --ki 0.02 \
  --integral-limit 0.10 \
  --max-source-steps 20
```

## Scale To Full Song

After 20-state tracking is reliable, rerun the same controller on one full song:

```bash
python PID-Controller/scripts/validate_pid_rollouts.py \
  --run-root /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/pd_fullsong_one \
  --limit 1 \
  --controller pd \
  --kp 1.0 \
  --kd 0.02 \
  --full-song
```

Then scale to 10 best rollouts:

```bash
python PID-Controller/scripts/validate_pid_rollouts.py \
  --run-root /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/pd_best10 \
  --limit 10 \
  --controller pd \
  --kp 1.0 \
  --kd 0.02 \
  --full-song
```

`validate_pid_rollouts.py` ranks discovered `trajectory.npz` files by available
prior `event_f1`, `frame_f1`, or `rp1m_key_f1` metadata. You can bypass
discovery with repeated `--trajectory-npz <path>` arguments.

Slurm wrappers are available:

```bash
sbatch PID-Controller/slurm/run_pid_controller_simple_test_cmp.slurm
sbatch PID-Controller/slurm/run_pid_controller_sample20_cmp.slurm
sbatch PID-Controller/slurm/run_pid_controller_optimize_cmp.slurm
sbatch PID-Controller/slurm/run_pid_controller_best10_cmp.slurm
```

## Local Tests

These tests do not require MuJoCo:

```bash
python -m pytest PID-Controller/tests
```
