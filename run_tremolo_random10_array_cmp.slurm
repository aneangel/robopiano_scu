#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J trem10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/trem10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/trem10_%A_%a.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_gigue_chunk_20260524
OUT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_finger_20260524
run=$(printf "maestro_%02d" "${SLURM_ARRAY_TASK_ID}")

"$PY" tremolo_nearest_finger_scrub_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$OUT/$run" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 1 \
  --post-dense-frames 8 \
  --top-fingers 2 \
  --modes neutral,hold_before,zero \
  --max-events 60 \
  --max-total-evals 180 \
  --allow-equal-f1-mispress-drop
