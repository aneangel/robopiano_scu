# Twinkle Rhapsody Repair PID Tracking Report

This report summarizes the local PID-Controller tests run against:

```text
D:\Codex\Maestroso\ImpromptuOutptus\impromptu_twinkle_latest_default_active10_rhapsody_repair\trajectory.npz
```

Environment:

```text
RoboPianist-debug-TwinkleTwinkleLittleStar-v0
```

The trajectory has 187 hand-state targets at 20 Hz and dense hand-state targets
at 200 Hz. Action rollouts run in `rp1m_simulator` action mode with no hand
state or piano state restoration during the rollout.

## Controller Changes Tested

- Minimum-jerk interpolation is anchored to the fixed start of each control
  window, not the live current state at every substep.
- The PD derivative term can track the analytical minimum-jerk reference
  velocity.
- Position-like action feedforward is available with `--feedforward-scale`.
- Preview horizon is available with `--lookahead-substeps`.
- Optimization scoring now includes mean, max, final, and last-third hand L2.

## Full-Song Results

All action rollouts below terminated at 1781 executed 200 Hz action steps,
which corresponds to 179 scored 20 Hz source transitions. The direct hand-state
replay also terminated at 1781 dense steps, so this trajectory is not fully
stable in simulator even before action control.

| Run | Kp | Kd | velocity scale | feedforward | horizon | event F1 | key F1 | mean L2 | final L2 | last-third L2 | max L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual PD baseline | 2.8 | 0.06 | 0.0 | 0.0 | 10 | 0.222 | 0.227 | 0.369 | 1.263 | 0.614 | 1.263 |
| Residual PD + velocity | 2.8 | 0.06 | 1.0 | 0.0 | 10 | 0.222 | 0.254 | 0.379 | 1.243 | 0.698 | 1.243 |
| Residual PD + 100 ms lookahead | 2.8 | 0.06 | 1.0 | 0.0 | 20 | 0.105 | 0.142 | 0.380 | 1.185 | 0.596 | 1.185 |
| Residual PD + 150 ms lookahead | 2.8 | 0.06 | 1.0 | 0.0 | 30 | 0.033 | 0.158 | 0.454 | 1.188 | 0.743 | 1.188 |
| Residual PD + 200 ms lookahead | 2.8 | 0.06 | 1.0 | 0.0 | 40 | 0.097 | 0.148 | 0.494 | 1.239 | 0.803 | 1.352 |
| Best feedforward residual PD | 1.0 | 0.02 | 1.0 | 1.0 | 10 | 0.333 | 0.280 | 0.343 | 0.950 | 0.539 | 0.984 |

The best tested controller was:

```text
pd kp=1.0 kd=0.02 setpoint=minimum_jerk target_velocity_scale=1.0 feedforward_scale=1.0 lookahead_substeps=10
```

It improved event F1 and substantially reduced late tracking error relative to
the residual-only PD baseline, but it did not prevent late termination.

## Window Reliability

Reliability definitions used for 20-step online windows:

- strict: full coverage, `mean_one_to_one_l2 <= 0.25`, `max_l2 <= 0.75`,
  `final_l2 <= 0.75`
- usable: full coverage, `mean_one_to_one_l2 <= 0.35`, `max_l2 <= 1.0`,
  `final_l2 <= 1.0`, `mean_l2 <= 0.55`

Best feedforward residual PD:

| Window | scored steps | strict | usable | mean L2 | final L2 | mean one-to-one L2 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 0-20 | 20 | yes | yes | 0.075 | 0.080 | 0.071 |
| 20-40 | 20 | yes | yes | 0.140 | 0.108 | 0.105 |
| 40-60 | 20 | yes | yes | 0.193 | 0.328 | 0.166 |
| 60-80 | 20 | no | yes | 0.320 | 0.685 | 0.308 |
| 80-100 | 20 | no | no | 0.371 | 0.754 | 0.363 |
| 100-120 | 20 | no | no | 0.378 | 0.214 | 0.352 |
| 120-140 | 20 | no | no | 0.531 | 0.360 | 0.489 |
| 140-160 | 20 | no | no | 0.560 | 0.412 | 0.505 |
| 160-180 | 19 | no | no | 0.525 | 0.950 | 0.446 |
| 180-187 | 0 | no | no | n/a | n/a | n/a |

Summary:

- all windows: 3 strict, 4 usable out of 10
- non-empty note windows: 2 strict, 3 usable out of 8

The window count did not improve over the previous residual PD controller, but
late collapse improved: final L2 dropped from about 1.26 to 0.95.

## How To Reproduce Locally

Use the local `sonata` environment:

```powershell
$py='D:\miniconda3\envs\sonata\python.exe'
$env:PATH='D:\miniconda3\envs\sonata;D:\miniconda3\envs\sonata\Library\bin;D:\miniconda3\envs\sonata\Scripts;' + $env:PATH
$env:MUJOCO_GL='glfw'
Remove-Item Env:PYOPENGL_PLATFORM -ErrorAction SilentlyContinue
```

Prepare a manifest for the trajectory:

```powershell
& $py PID-Controller\scripts\prepare_pid_trajectories.py `
  --trajectory-npz D:\Codex\Maestroso\ImpromptuOutptus\impromptu_twinkle_latest_default_active10_rhapsody_repair\trajectory.npz `
  --output-json D:\Codex\Maestroso\runs\pid_controller_local\prepared_twinkle_rhapsody_repair\manifest.json `
  --limit 1 `
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0
```

Run the best feedforward residual PD controller through the hand-tracking
evaluator:

```powershell
& $py PID-Controller\scripts\evaluate_hand_tracking.py `
  --trajectory-manifest D:\Codex\Maestroso\runs\pid_controller_local\prepared_twinkle_rhapsody_repair\manifest.json `
  --output-root D:\Codex\Maestroso\runs\pid_controller_local\hand_tracking_eval_twinkle_best_feedforward `
  --limit 1 `
  --max-source-steps 187 `
  --controller pd `
  --kp 1.0 `
  --kd 0.02 `
  --setpoint-policy minimum_jerk `
  --use-target-velocity `
  --target-velocity-scale 1.0 `
  --feedforward-scale 1.0 `
  --lookahead-substeps 10 `
  --resume
```

Run a single full-song action rollout directly:

```powershell
& $py PID-Controller\scripts\run_pid_rollout.py `
  --trajectory-npz D:\Codex\Maestroso\ImpromptuOutptus\impromptu_twinkle_latest_default_active10_rhapsody_repair\trajectory.npz `
  --output-dir D:\Codex\Maestroso\runs\pid_controller_local\twinkle_best_feedforward_single `
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 `
  --controller pd `
  --kp 1.0 `
  --kd 0.02 `
  --setpoint-policy minimum_jerk `
  --use-target-velocity `
  --target-velocity-scale 1.0 `
  --feedforward-scale 1.0 `
  --lookahead-substeps 10 `
  --full-song
```

Run a broader gain search with the new feedforward and lookahead knobs:

```powershell
& $py PID-Controller\scripts\optimize_pid_controller.py `
  --trajectory-npz D:\Codex\Maestroso\ImpromptuOutptus\impromptu_twinkle_latest_default_active10_rhapsody_repair\trajectory.npz `
  --output-root D:\Codex\Maestroso\runs\pid_controller_local\pid_optimized_twinkle `
  --trajectory-limit 1 `
  --controllers pd `
  --kp-grid 0,0.25,0.5,1.0 `
  --kd-grid 0,0.005,0.02 `
  --target-velocity-scales 0,0.5,1.0 `
  --feedforward-scales 0.0,1.0 `
  --lookahead-substeps-grid 10,20,40 `
  --setpoint-policies minimum_jerk `
  --max-source-steps 20 `
  --selection-metric hand_l2_last_third_mean `
  --scale-best `
  --scale-limit 1 `
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 `
  --resume
```

Note: `optimize_pid_controller.py` uses short sample windows for candidate
selection, then scales the best candidate when `--scale-best` is set. For full
song exhaustive sweeps, use repeated `run_pid_rollout.py` invocations with the
grid values of interest.
