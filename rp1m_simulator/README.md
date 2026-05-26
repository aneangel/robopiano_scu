# rp1m_simulator

`rp1m_simulator` replays RP1M demonstrations through RoboPianist without loading
RP1M piano states into the simulator. Piano states, when present in `rp1m.zarr`,
are scoring references only.

The main action-only diagnostic is: can the simulator reproduce the RP1M key F1
from the same RP1M action inputs while starting from the normal RoboPianist reset
hand state? Action-only runs must not be helped by recorded hand joints except
for the optional one-time hand-anchor calibration.

## Run On WAVE

Follow the project `HowToRun.md` pattern and run rollouts on a `cmp` node, not on
the login node:

```bash
tmux new -s rp1m_action_rollout

srun -p cmp \
  --cpus-per-task=16 \
  --mem=128G \
  --time=0-08:00:00 \
  --pty bash

cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export RP1M_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr
export SONG_KEY=<rp1m_song_key>
export DEMO_ID=<demo_id>
export RUN_NAME=rp1m_action_only_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Fugue/runs/rp1m_simulator/$RUN_NAME
mkdir -p "$OUTPUT_ROOT"
```

Use `python -m rp1m_simulator rollout --help` on the compute node for the full
flag list.

## Action-Only Contract

For action-only RP1M reconstruction:

- Use `--mode action`.
- Use the 39D compressed RP1M action array in the reduced RoboPianist action
  space. Do not pass `--full-action-space` for `rp1m.zarr` actions.
- Use `--action-source-scale normalized_minus_one_to_one` for RP1M actions.
- Start from the RoboPianist default reset hand state, not `hand_joints[0]`.
  Pass `--no-restore-initial-hand --no-set-hand-qvel` explicitly. Action mode
  also enforces these defaults at runtime.
- Do not set `--hand-resync-interval`; action-only rollout must not periodically
  restore recorded RP1M hand joints.
- Treat `piano_states` as scoring-only data. They must never be restored into
  the simulator.
- The one permitted hand-state-derived calibration is the startup hand-anchor
  offset. Keep the default auto-calibration when reproducing the current RP1M
  results; disable it only for a strict no-anchor ablation.

## Most Important Run: Dense 200 Hz Controller Input

If the controller already emits actions at 200 Hz, run those actions as 200 Hz
source inputs. Do not use RP1M zero padding, and do not label a 20 Hz RP1M action
array as 200 Hz. The source timestep and simulator timestep should both be 5 ms:

```text
dataset_timestep = 0.005
simulation_timestep = 0.005
substeps_per_source_step = 1
```

With one source action per simulator step, `--action-substep-policy repeat` is
the clearest no-padding setting: there are no extra substeps to repeat across,
and no zero-padded slots are created.

For a command-line run, the input Zarr must contain dense 200 Hz actions, goals,
and scoring references at the same source rate:

```bash
python -m rp1m_simulator rollout \
  --rp1m-root "$DENSE_200HZ_ROOT" \
  --song-key "$SONG_KEY" \
  --demo-id "$DEMO_ID" \
  --output-dir "$OUTPUT_ROOT/action_200hz_dense_controller" \
  --mode action \
  --dataset-timestep 0.005 \
  --simulation-timestep 0.005 \
  --action-source-scale normalized_minus_one_to_one \
  --action-substep-policy repeat \
  --wrist-action-policy hold_initial \
  --no-restore-initial-hand \
  --no-set-hand-qvel
```

For generated actions that are not stored in Zarr, call the simulator API with a
trajectory whose arrays are already dense at 200 Hz. `actions`, `goals`, and
`hand_joints` must all have 200 Hz source rows because rollout length is bounded
by the shortest source array. In action mode those hand joints are not restored
when `restore_initial_hand=False`; they only support anchor calibration and
summary comparison.

```python
from pathlib import Path

from rp1m_simulator.simulator import (
    RolloutConfig,
    make_rp1m_trajectory_from_arrays,
    simulate_rp1m_rollout,
)

trajectory = make_rp1m_trajectory_from_arrays(
    song_key=song_key,
    environment_name=environment_name,
    demo_id=demo_id,
    actions=actions_200hz,          # [T_200hz, 39]
    goals=goals_200hz,              # [T_200hz, 89] or at least [:, :88]
    hand_joints=hand_joints_200hz,  # [T_200hz, 46], not restored in action mode
    reference_piano_states=reference_piano_states_200hz,
)

config = RolloutConfig(
    mode="action",
    dataset_timestep=0.005,
    simulation_timestep=0.005,
    action_source_scale="normalized_minus_one_to_one",
    action_substep_policy="repeat",
    wrist_action_policy="hold_initial",
    restore_initial_hand=False,
    set_hand_qvel=False,
)

simulate_rp1m_rollout(trajectory, config, Path(output_dir))
```

This is the preferred path for a true 200 Hz controller. The RP1M zero-padding
path below exists only for replaying RP1M's native 20 Hz action stream inside a
200 Hz simulator loop.

## RP1M 20 Hz Actions At 200 Hz Control

RP1M actions are source-rate 20 Hz commands. When the simulator runs with 200 Hz
control, do not repeat each 20 Hz source action on all ten 5 ms control substeps.
Use `zero_pad_hold`, which applies the RP1M source action on the first substep of
each 50 ms interval and zero-pads the remaining source-command slots while
holding the previous actuator target between sparse commands.

This is the recommended RP1M reproduction setting:

```bash
python -m rp1m_simulator rollout \
  --rp1m-root "$RP1M_ROOT" \
  --song-key "$SONG_KEY" \
  --demo-id "$DEMO_ID" \
  --output-dir "$OUTPUT_ROOT/action_200hz_zero_pad_hold" \
  --mode action \
  --dataset-timestep 0.05 \
  --simulation-timestep 0.005 \
  --action-source-scale normalized_minus_one_to_one \
  --action-substep-policy zero_pad_hold \
  --wrist-action-policy hold_initial \
  --no-restore-initial-hand \
  --no-set-hand-qvel
```

Do not use `--action-substep-policy repeat` with `--simulation-timestep 0.005`
as an RP1M reproduction setting. That runs the 20 Hz RP1M controls as repeated
200 Hz controls and changes the input signal.

## Action-Only Without Zero Padding

For a no-zero-padding action-only run, run the simulator at the RP1M source rate
instead of running 20 Hz commands at 200 Hz. Set the simulation timestep equal to
the dataset timestep, so there is exactly one control step per RP1M action:

```bash
python -m rp1m_simulator rollout \
  --rp1m-root "$RP1M_ROOT" \
  --song-key "$SONG_KEY" \
  --demo-id "$DEMO_ID" \
  --output-dir "$OUTPUT_ROOT/action_20hz_source_rate" \
  --mode action \
  --dataset-timestep 0.05 \
  --simulation-timestep 0.05 \
  --action-source-scale normalized_minus_one_to_one \
  --action-substep-policy repeat \
  --wrist-action-policy hold_initial \
  --no-restore-initial-hand \
  --no-set-hand-qvel
```

With `--simulation-timestep 0.05`, `repeat` does not actually repeat across
extra substeps; it is a single source action per simulator control step. This is
the clean comparison point for action-only rollout without the 200 Hz zero-pad
path.

## Verify No State Leakage

After any action-only run, inspect `summary.json`. These checks should pass
for a valid action-only rollout:

```bash
python - "$OUTPUT_ROOT/action_200hz_dense_controller" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads((Path(sys.argv[1]) / "summary.json").read_text())
checks = {
    "mode_is_action": summary["mode"] == "action",
    "compressed_39d_actions": summary["runtime_policy"]["action_input_format"] == "compressed_39d",
    "default_initial_hand": summary["initial_hand_policy"]["state_source"] == "robopianist_reset_default",
    "same_source_and_control_rate_or_zero_pad": (
        summary["substeps_per_source_step"] == 1
        or summary["effective_config"]["action_substep_policy"] == "zero_pad_hold"
    ),
    "no_initial_qpos_restore": summary["qpos_restored_count"] == 0,
    "no_initial_qvel_restore": summary["qvel_restored_count"] == 0,
    "no_hand_resync": not summary["hand_resync_policy"]["uses_rp1m_hand_joints"],
    "no_piano_state_restore": summary["piano_state_policy"] == "not_restored_or_used_by_simulator",
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"failed checks: {failed}")
print("against_goals.key_f1 =", summary["against_goals"]["key_f1"])
print("recorded_reference_against_goals.key_f1 =", summary["recorded_reference_against_goals"]["key_f1"])
print("action_substep_policy =", summary["effective_config"]["action_substep_policy"])
PY
```

For the RP1M zero-pad and source-rate runs, replace the path with
`$OUTPUT_ROOT/action_200hz_zero_pad_hold` or
`$OUTPUT_ROOT/action_20hz_source_rate`. A successful action-only reconstruction
should make `against_goals.key_f1` track the recorded-reference F1 closely
without any hand-state or piano-state leakage. If it does not, debug action
timing, action scaling, and action mapping before using hand-state rollout as a
comparison.
