# rp1m_simulato

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

export RP1M_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zar
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
- The one permitted hand-state-derived calibration is the startup hand-ancho
  offset. Keep the default auto-calibration when reproducing the current RP1M
  results; disable it only for a strict no-anchor ablation.

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

After either run, inspect `summary.json`. These checks should pass for a valid
action-only rollout:

```bash
python - "$OUTPUT_ROOT/action_200hz_zero_pad_hold" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads((Path(sys.argv[1]) / "summary.json").read_text())
checks = {
    "mode_is_action": summary["mode"] == "action",
    "compressed_39d_actions": summary["runtime_policy"]["action_input_format"] == "compressed_39d",
    "default_initial_hand": summary["initial_hand_policy"]["state_source"] == "robopianist_reset_default",
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

For the source-rate run, replace the path with
`$OUTPUT_ROOT/action_20hz_source_rate`. A successful RP1M action-only
reconstruction should make `against_goals.key_f1` track the recorded-reference
F1 closely without any hand-state or piano-state leakage. If it does not, debug
action timing, action scaling, and action mapping before using hand-state rollout
as a comparison.
