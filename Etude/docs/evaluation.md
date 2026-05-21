# Etude Evaluation Pipeline

Training loss is useful for supervised preselection, but it is not the final model objective. A controller with lower behavior-cloning loss can still miss keypresses, add false notes, drift in timing, or clip actions during closed-loop RoboPianist rollout. Final model selection should therefore use held-out rollout metrics.

## Create A Deterministic Split

Create `splits.csv` next to the dataset `manifest.csv`:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano/Etude
python scripts/create_dataset_split.py \
  --dataset-root /WAVE/datasets/ccoelho_lab-jlanders/etude/data/medium \
  --seed 7
```

The default split is 80% train, 10% val, and 10% test. The output columns are `episode_id`, `path`, `source`, `timesteps`, and `split`. If source labels are reliable song identifiers, add `--group-by-source` to avoid splitting episodes from the same source across train/val/test.

## Train With Train/Val

`scripts/train_controller.py` automatically uses `splits.csv` when it exists:

```bash
python scripts/train_controller.py \
  --config configs/experiments/04_fingertip_phase.yaml \
  --output-root runs/train/04_fingertip_phase
```

It writes `checkpoints/last.pt`, `checkpoints/best_train.pt`, `checkpoints/best_val.pt`, `checkpoints/best.pt`, `training_history.csv`, and `training_summary.json`. `best.pt` aliases `best_val.pt` when validation exists, otherwise `best_train.pt` for backward compatibility.

## Evaluate Checkpoints On Test Rollouts

Evaluate all candidate checkpoints on the held-out test split:

```bash
python scripts/evaluate_checkpoints.py \
  --config configs/experiments/04_fingertip_phase.yaml \
  --checkpoint-glob "runs/train/04_fingertip_phase/checkpoints/*.pt" \
  --dataset-root /WAVE/datasets/ccoelho_lab-jlanders/etude/data/medium \
  --split test \
  --output-root runs/eval/fingertip_phase_test \
  --primary-metric piano/event_f1 \
  --selection-mode max \
  --task RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --max-episodes 20
```

Each checkpoint gets an evaluation directory containing per-episode `metrics.json`, `episode_metrics.csv`, and `aggregate_metrics.json`. Pass `--save-rollout` to store `rollout.npz` files.

## Rank Checkpoints

Rank aggregate rollout results:

```bash
python scripts/rank_checkpoints.py \
  --eval-root runs/eval/fingertip_phase_test \
  --primary-metric piano/event_f1 \
  --selection-mode max \
  --output runs/eval/fingertip_phase_test/leaderboard.csv \
  --copy-best-to runs/eval/fingertip_phase_test/best_rollout.pt
```

The default primary objective is to maximize `piano/event_f1`. Ties are broken by minimizing, in order: `piano/missed_events`, `piano/false_events`, `piano/timing_abs_error_mean_s`, `fingertip/active_l2_mean`, `tracking/joint_pos_rmse`, and `control/action_clip_rate`.

For fingertip-only refinement experiments, use:

```bash
python scripts/rank_checkpoints.py \
  --eval-root runs/eval/fingertip_refine_test \
  --primary-metric fingertip/active_l2_mean \
  --selection-mode min
```

## Recommended Metrics

Use `piano/event_f1` as the final authority for piano-event controllers. Inspect missed and false events to understand recall/precision failures, timing error for rhythmic quality, fingertip active L2 for physical contact quality, joint RMSE for trajectory tracking, and action clip rate for controller saturation.

Expected ranking outputs are `leaderboard.csv`, `best_checkpoint.json`, and optionally a copied `best_rollout.pt`.
