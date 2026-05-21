# Partita

Partita is a small primitive-learning pipeline for RP1M. It is intentionally narrower than Sonata: instead of trying to discover primitives across the full RP1M repertoire, Partita learns an unsupervised primitive library from many successful trajectories of one song, then reconstructs a held-out or selected trajectory from that same song.

The current version is a goal-conditioned nearest-primitive action reconstructor. Primitive assignment uses MIDI-derived goal/timing features only, and primitives output action trajectories. It does not train a transformer, diffusion model, or policy.

## Why One Song

RP1M is large, and cross-song primitive learning is hard to debug. Same-song primitive learning keeps the musical structure fixed while varying the demonstrations. If primitive reuse is meaningful here, it gives a clearer starting point for later Sonata-scale experiments.

French Suite No. 5 Sarabande is the preferred first target because the RP1M paper uses it as a clean exemplar trajectory. The default selector searches for related group names before falling back to other known RP1M examples or a small scan of available songs.

## First Experiment

The intended debug run is:

1. Inspect the RP1M Zarr root.
2. Select one preferred song.
3. Score all trajectories from that song.
4. Use the top successful trajectories for primitive learning.
5. Hold out the best trajectory by default as the reconstruction target.
6. Segment selected trajectories into short chunks.
7. Cluster goal/timing segment features into a shared primitive library.
8. Reconstruct the target action trajectory from nearest primitive centers.
9. Evaluate action similarity and, after rollout, simulated key-press performance.
10. Save compact reports and plots.

## Commands

From the repo root on WAVE, use the existing `sonata` conda environment:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

python partita/scripts/inspect_rp1m.py \
  --rp1m-root /WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr \
  --max-songs 5

python partita/scripts/run_partita_debug.py \
  --config partita/configs/debug.yaml
```

Every stage is also runnable independently with `--config partita/configs/debug.yaml`.


## Online Rollout Video

After running the debug pipeline, render RoboPianist rollout videos with:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

python partita/scripts/simulate_rollout.py   --config partita/configs/debug.yaml   --which both   --width 640   --height 480
```

This writes videos and rollout JSON reports under `partita/outputs/rollout/<experiment_name>/`.
When `rollout.prefer_canonical_midi: true` (default) and the song is in `robopianist.suite.ALL`, the canonical PIG MIDI is loaded directly. Otherwise a synthesized `.proto` is built from the RP1M `goals` pianoroll. If the local environment exposes more action dimensions than RP1M, the replay pads the missing controls with zeros.

## Action Rollout Fidelity

`simulate_rollout.py` reproduces RP1M's recorded states when stepping recorded RP1M actions only if the simulator starts from RP1M's t=0 state, the action interpretation matches collection, and the task config matches the MJCF / dynamics RP1M was collected against. Three knobs together drive the fidelity:

1. `rollout.restore_initial_state: true` — after `env.reset()`, the rollout copies `hand_joints[0]`, `piano_states[0]`, sustain, and zeros all hand+piano `qvel` from `target_trajectory.npz` into MuJoCo before the first step.
2. `rollout.action_source_scale` and `rollout.action_mapping` — how RP1M's stored vector is mapped to the env's action spec. If unsure, run the calibration sweep:

   ```bash
   python partita/scripts/simulate_rollout.py \
     --config partita/configs/debug.yaml \
     --calibrate-action-scale \
     --calibration-probe-steps 5
   ```

   This sweeps `(normalized_minus_one_to_one, actuator_units) x (as_is, swap_hands, zero_sustain, invert_sustain, swap_hands_zero_sustain)` over the first frames of `original_actions.npy` and writes the winner to `partita/outputs/rollout/<exp>/calibration.json`. Subsequent runs auto-load it.

3. `rollout.task_kwargs` — explicit RP1M-matching values for `n_steps_lookahead`, `trim_silence`, `wrong_press_termination`, `randomize_hand_positions`, `gravity_compensation`, `primitive_fingertip_collisions`, etc.

To validate end-to-end:

```bash
python partita/scripts/simulate_rollout.py \
  --config partita/configs/debug.yaml \
  --validate-fidelity
```

On Slurm, the same calibration plus validation sequence is packaged as:

```bash
sbatch /WAVE/projects/ECEN-524-Wi26/robopiano/partita/scripts/run_rollout_fidelity.slurm
```

Optional: `PARTITA_SKIP_PIPELINE=1` if outputs already exist; `PARTITA_CALIBRATE=0` to reuse `calibration.json`; override config with `PARTITA_CONFIG=...`.

This forces `restore_initial_state=True`, runs only the `original_target` job, writes `<exp>/fidelity_summary.json`, and exits non-zero if any of the success criteria fail:

- Mean per-step `||hand_qpos - rp1m_hand_joints||_2` over the first 50 steps `< 1e-3`.
- Piano-state F1 vs `rp1m_piano_states` over the full clip `> 0.95`.
- F1 vs `goals` for `original_target` `> 0.5` (sanity floor; the replay should at minimum approach the recorded-state-playback ceiling).

Per-step diagnostics live in `<label>_fidelity_frames.csv` (`step, hand_qpos_l2, piano_state_iou, sim_keys, ref_keys, goal_keys`). Use these to see which step error explodes; sudden spikes early are MJCF / task-flag mismatches, slow drift is missing `qvel` or solver determinism.

### One-step dynamics diagnostic

To **remove trajectory accumulation** and test whether `(state_t, action_t) → state_{t+1}` matches RP1M **one control step at a time**:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

python partita/scripts/debug_one_step_rp1m.py --config partita/configs/debug.yaml
```

Each iteration: `env.reset()`, restore RP1M `hand_joints[t]` and `piano_states[t]`, apply `actions[t]`, compare simulator pose to RP1M `hand_joints[t+1]` and `piano_states[t+1]`. Outputs:

- `partita/outputs/rollout/<experiment>/rp1m_one_step_consistency.csv`
- `partita/outputs/rollout/<experiment>/rp1m_one_step_consistency_summary.json`

Use `--max-pairs 50` for a quick smoke test; `--ignore-calibration` to force YAML scale/mapping only. If **mean one-step hand error is already large**, the bug is not long-run drift—it is action semantics, timestep/MJCF mismatch, or RP1M indexing vs this convention.

## Outputs

Outputs are written under `partita/outputs`:

- `inspection/rp1m_inspection.json`
- `data/<experiment_name>/trajectory_scores.csv`
- `data/<experiment_name>/selection.json`
- `data/<experiment_name>/selected_trajectories.npz`
- `primitives/<experiment_name>/segments.csv`
- `primitives/<experiment_name>/primitive_library.pkl`
- `primitives/<experiment_name>/primitive_summary.csv`
- `reconstruction/<experiment_name>/reconstructed_actions.npy`
- `evaluation/<experiment_name>/metrics.json`

## What To Look At

Useful first diagnostics:

- `key_f1`, `mispress_rate`, and `action_smoothness` in `trajectory_scores.csv`.
- Segment durations in `segment_duration_histogram.png`.
- Primitive counts and trajectory coverage in `primitive_summary.csv`.
- Primitive reuse in `primitive_usage_by_trajectory.csv` and `primitive_timeline_by_trajectory.png`.
- Reconstruction action MSE/L1 in `metrics.json`.
- Online rollout key F1, precision, recall, and mispress rate in `metrics.json` after running `simulate_rollout.py`.
- Fidelity block in `<label>_rollout.json` (`hand_qpos_l2_mean`, `hand_qpos_l2_mean_first_n`, `piano_state_iou_mean`, `against_rp1m_piano_states.key_f1`) and the per-step `<label>_fidelity_frames.csv`. The original-target rollout fidelity is the ceiling; reconstructed-rollout fidelity is what the Partita model achieves.

A good first sign is that multiple primitives appear across many trajectories and no single primitive dominates the timeline. A bad sign is collapse into one primitive, mostly single-trajectory primitives, or extremely short segments.

## Current Limitations

- One song only.
- KMeans primitives only.
- Nearest-center reconstruction only.
- Partita no longer reconstructs or scores piano states offline; musical key metrics come from RoboPianist rollout activation.
- Metrics depend on RP1M arrays exposing `actions` and `goals`.
