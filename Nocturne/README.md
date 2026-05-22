# Nocturne

Nocturne is a first-pass RP1M local-moment stitcher for RoboPianist. It mines
multiple demonstrations of one song, scores local event-centered windows,
selects a transition-aware sequence with Viterbi, saves a stitched oracle
trajectory, and trains a small action policy baseline.

The module is intentionally standalone. It reads from RP1M and writes experiment
outputs under `/WAVE/datasets/ccoelho_lab-jlanders/Nocturne`.

## First experiment

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Nocturne/runs/french_sarabande_v1
export PYTHONPATH=$PWD/Nocturne/src:$PWD/Sonata/src:$PWD/Etude/src:$PWD/partita/src:$PWD/Impromptu/src:$PWD:$PYTHONPATH

python Nocturne/scripts/build_stitched_oracle.py \
  --rp1m-root $RP1M_300_ROOT \
  --song-name RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0_0 \
  --num-demos 50 \
  --window-frames 64 \
  --pre-frames 24 \
  --output-root $OUTPUT_ROOT

python Nocturne/scripts/evaluate_offline.py \
  --trajectory $OUTPUT_ROOT/stitcher/RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0_0/smoothed_trajectory.npz
```

New oracle builds default to the `strict` objective with a 3-frame event
tolerance, matching the default 0.15s offline event metric at `dt=0.05`. It adds
cross-demo event priors, scores wrong press onsets over the full stitched
interval, and filters each event to candidates with the minimum local
miss/mispress tuple before transition-aware Viterbi runs. Use
`--objective-mode legacy --event-tolerance-frames 2 --disable-repair` to
reproduce the original V1 scalar score behavior if the stricter objective
regresses a song.

After candidate selection, Nocturne can refine connections between adjacent
events. By default it moves each seam to the between-press frame where the
selected demos' hand joints/fingertips/actions are closest, then interpolates
`hand_joints`, `hand_fingertips`, and `actions` around the seam while protecting
press/contact frames. Use `--disable-adaptive-seams` or
`--disable-transition-interpolation` to turn those connection refinements off.

Controller baseline:

```bash
python Nocturne/scripts/build_controller_dataset.py \
  --trajectory $OUTPUT_ROOT/stitcher/RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0_0/smoothed_trajectory.npz \
  --output-root $OUTPUT_ROOT/controller_dataset

python Nocturne/scripts/train_stitched_controller.py \
  --dataset-root $OUTPUT_ROOT/controller_dataset \
  --output-root $OUTPUT_ROOT/controller_train \
  --epochs 100 \
  --device cuda

MUJOCO_GL=egl python Nocturne/scripts/evaluate_stitched_policy.py \
  --checkpoint $OUTPUT_ROOT/controller_train/checkpoints/best.pt \
  --trajectory $OUTPUT_ROOT/stitcher/RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0_0/smoothed_trajectory.npz \
  --output-root $OUTPUT_ROOT/online_rollout_eval
```

## Notes

- `goals[:, :88]` are score targets.
- `piano_states[:, :88]` are realized key states for offline scoring.
- Index 88 is preserved as sustain metadata when present.
- Direct hand or piano state injection is only a debug oracle technique. The
  online policy evaluator emits RoboPianist actions through `env.step(action)`.
- Controller training always starts a W&B run. By default it logs to project
  `robopianist` in online mode. Set `WANDB_ENTITY` and log in on WAVE before
  launching long training jobs; use `WANDB_MODE=offline` only for network
  outages, then sync the run later.
