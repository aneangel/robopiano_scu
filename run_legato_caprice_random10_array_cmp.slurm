#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J legcap10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/legcap10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/legcap10_%A_%a.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_caprice_contact_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_legato_caprice_20260524
mkdir -p "$ROOT"

run=$(printf "maestro_%02d" "${SLURM_ARRAY_TASK_ID}")

"$PY" legato_duplicate_hold_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$ROOT/$run" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 10 \
  --post-dense-frames 8 \
  --top-fingers 2 \
  --modes onset,onset_plus,pre_50ms,pre_100ms,mid_interval \
  --max-events 80 \
  --max-total-evals 180 \
  --allow-equal-f1-mispress-drop
