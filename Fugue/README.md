# Fugue

Fugue is a self-contained RP1M action-reconstruction module for RoboPianist. It trains
controllers that map hand-state observations and optional temporal/goal context to reduced
RoboPianist action vectors.

The first experiment is deliberately narrow: one song, many demonstrations, demo-level
train/validation/test splits, and offline action reconstruction on held-out demonstrations.

## Repository Rules

- Fugue code lives only under `Fugue/`.
- RP1M is read from `RP1M_300_ROOT`, normally:

```bash
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
```

- Large outputs should be written under:

```bash
/WAVE/datasets/ccoelho_lab-jlanders/Fugue
```

- Follow the project-level `HowToRun.md` for Slurm, tmux, and the `sonata` Conda environment.

## Install And Smoke Test

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
pip install -e Fugue
pytest Fugue/tests
```

The tests build tiny synthetic Zarr archives and do not require the full RP1M dataset.

## Data Contract

Fugue assumes the RP1M Zarr layout used by the rest of this repository:

- `hand_joints`: `[num_demos, T, 46]`
- `actions`: `[num_demos, T, 39]`
- `goals`: `[num_demos, T, 89]`; Fugue uses `goals[..., :88]` by default
- `hand_fingertips`: optional `[num_demos, T, 30]`
- `joint_velocities`: optional `[num_demos, T, 46]`; if missing, Fugue finite-differences `hand_joints`

Splits are by demonstration, never by timestep.

## Audit One Song And Build Splits

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export RUN_NAME=fugue_audit_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Fugue/$RUN_NAME

python Fugue/scripts/audit_song.py \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$OUTPUT_ROOT/dataset"
```

Default song:

```text
RoboPianist-debug-TwinkleTwinkleLittleStar-v0_0
```

Audit outputs:

- `manifest.csv`: one row per demonstration, including split
- `splits.csv`: `demo_id,split`
- `dataset_summary.json`: array shapes and split counts
- `normalization.json`: train-only normalization stats, written by training if missing

## Train Model A

```bash
python Fugue/scripts/train_action_model.py \
  --config Fugue/configs/model_a_stateless.yaml \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$OUTPUT_ROOT/model_a" \
  --alignment-sweep
```

This trains `delta=-1,0,+1` and writes each run under `delta_m1`, `delta_0`, and `delta_p1`.
The selected validation alignment is written to `alignment_summary.json`.
W&B is enabled by default in the checked-in training configs, following the Sonata pattern.
Use `--no-wandb` to disable it, or `--wandb-mode offline` to buffer locally.

## Train Model B

```bash
python Fugue/scripts/train_action_model.py \
  --config Fugue/configs/model_b_history.yaml \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$OUTPUT_ROOT/model_b" \
  --alignment-sweep
```

Model B uses:

- `H=8` frames of qpos/qvel history
- `H=8` previous actions
- `K=16` future score-goal frames

## Validate Configs Before Queueing

Use the validation script on `cmp` before submitting GPU jobs for new model families:

```bash
python Fugue/scripts/validate_configs.py \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Fugue/validation/latest \
  --batch-size 16 \
  --device cpu
```

It builds each configured dataset, runs one forward/backward optimizer step, checks validation
forward shapes, and writes `validation_report.json`.

## Oracle Inverse Dynamics

```bash
python Fugue/scripts/train_action_model.py \
  --config Fugue/configs/model_c_oracle_inverse.yaml \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$OUTPUT_ROOT/model_c_oracle" \
  --alignment-sweep
```

This uses true future held-out hand states as desired states and must be reported as an oracle
diagnostic, not as a deployable controller.

## Later Sequence Models

Model D uses `feature_mode: sequence` with token inputs `[B, H + K, D_token]` when future goals
are enabled:

```bash
python Fugue/scripts/train_action_model.py \
  --config Fugue/configs/model_d_transformer.yaml \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$OUTPUT_ROOT/model_d_transformer" \
  --alignment-sweep
```

Available later-model configs:

- `model_d_transformer.yaml`: causal transformer, single-step action prediction.
- `model_d_tcn.yaml`: causal temporal convolution, single-step action prediction.
- `model_e_action_chunk_transformer.yaml`: causal transformer, `C=4` action chunks.
- `model_e_chunk_history.yaml`: flattened history MLP, `C=4` action chunks.

## Planner-Next Controller

Model F is the first deployable planner-conditioned inverse-dynamics controller. It learns from a
short target hand trajectory window, not just one target pose:

```text
f(q_sim_or_demo[t],
  dq_sim_or_demo[t],
  a_history[t-H:t],
  q_target[t+1:t+K],
  dq_target[t+1:t+K],
  q_target[t+1:t+K] - q_sim_or_demo[t],
  score_goals[t:t+K]) -> action[t]
```

During training, `q_sim_or_demo[t]` is the recorded RP1M hand state. During online rollout,
`q_sim_or_demo[t]` is captured from RoboPianist after the previous action, while
`q_target[t+1:t+K]` comes from a planner. The first diagnostic planner is an unseen
demonstration's hand-state trajectory. The default config uses `H=8` and `K=16`.

Train:

```bash
python Fugue/scripts/train_action_model.py \
  --config Fugue/configs/model_f_planner_next.yaml \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$OUTPUT_ROOT/model_f_planner_next" \
  --delta 0
```

Closed-loop planner-following rollout:

```bash
python Fugue/scripts/rollout_planner_next.py \
  --checkpoint "$OUTPUT_ROOT/model_f_planner_next/checkpoints/best.pt" \
  --rp1m-root "$RP1M_300_ROOT" \
  --dataset-artifact-root "$OUTPUT_ROOT/model_f_planner_next/dataset" \
  --split test \
  --demo-index 0 \
  --output-dir "$OUTPUT_ROOT/model_f_planner_next_rollout"
```

The rollout video audio is generated from RoboPianist piano MIDI keypress events produced by
the simulated hands, not from the target score.

## Export Predicted Actions

```bash
python Fugue/scripts/export_rollout_actions.py \
  --checkpoint "$OUTPUT_ROOT/model_b/delta_0/checkpoints/best.pt" \
  --rp1m-root "$RP1M_300_ROOT" \
  --dataset-artifact-root "$OUTPUT_ROOT/model_b/dataset" \
  --split test \
  --demo-index 0 \
  --output-dir "$OUTPUT_ROOT/export_test_demo"
```

This writes `predicted_actions.npz` containing predicted actions, reference actions, goals,
hand joints, piano states when available, and metadata.

To invoke RoboPianist rollout through existing Partita utilities, add `--run-rollout`. This is
intentionally opt-in because it is slower and depends on MuJoCo/RoboPianist runtime availability.

## Metrics

Training and evaluation report:

- action MSE and L1
- press-frame MSE and L1
- smoothness MSE on action differences
- per-action-dimension MSE/L1
- train-mean and previous-action baselines where applicable

Plots are written when matplotlib is available:

- `training_curves.png`
- `per_dim_mse.png`

## Compare Runs

After Model B/C jobs finish, build a ranked comparison:

```bash
python Fugue/scripts/compare_runs.py \
  --runs-root /WAVE/datasets/ccoelho_lab-jlanders/Fugue/runs \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/Fugue/comparisons/latest \
  --evaluate-missing-test \
  --device cuda
```

Outputs:

- `comparison_table.csv`: sortable table for all discovered runs.
- `comparison_table.json`: structured records and recommendation.
- `comparison_report.md`: human-readable recommendation and ranked table.

The recommendation separates deployable models from oracle inverse-dynamics diagnostics. Model C
can win the overall score while still being treated as an upper bound rather than the controller
to deploy.

## Implementation Notes

- Normalization is fitted on train demonstrations only.
- Action outputs are normalized during training and unnormalized/clamped to `[-1, 1]` for export.
- `delta` is a validation-selected action-state alignment, not a test-tuned setting.
- Press-frame weighting uses active/onset score-goal frames dilated by `press_window`.
