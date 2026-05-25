#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J barcrev
#SBATCH -a 2,4,7,8,10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/barcrev_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/barcrev_%A_%a.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
BEST=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random30_best_adaptive_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/barcarolle_reverse_legato_rondo_20260524
idx=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
run=maestro30_${idx}
SRC="$BEST/$run/legato_caprice/trajectory.npz"
RONDO="$BEST/$run/rondo/trajectory.npz"
mkdir -p "$ROOT"

"$PY" barcarolle_event_window_ensemble_gate.py \
  --source-npz "$SRC" \
  --candidate-npz rondo="$RONDO" \
  --output-dir "$ROOT/$run" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --event-types mispress \
  --scopes target_hand,full \
  --pre-dense-frames 2 \
  --post-dense-frames 10 \
  --max-events 120 \
  --max-total-evals 240 \
  --keep-if-matched-increases \
  --matched-gain-max-f1-drop 0.002 \
  --allow-equal-f1-mispress-drop
