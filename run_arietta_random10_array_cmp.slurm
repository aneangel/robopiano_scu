#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J arietta10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/arietta10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/arietta10_%A_%a.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_arietta_margin_20260524
RUN=$(printf "maestro_%02d" "$SLURM_ARRAY_TASK_ID")
mkdir -p "$ROOT"

SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_legato_caprice_20260524/$RUN/trajectory.npz
if [ "$RUN" = "maestro_01" ] && [ -f /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro01_20260524/$RUN/trajectory.npz ]; then
  SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro01_20260524/$RUN/trajectory.npz
fi
if [ "$RUN" = "maestro_07" ] && [ -f /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro07_20260524/$RUN/trajectory.npz ]; then
  SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro07_20260524/$RUN/trajectory.npz
fi

"$PY" arietta_lateral_margin_gate.py \
  --source-npz "$SOURCE" \
  --output-dir "$ROOT/$RUN" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 2 \
  --post-dense-frames 10 \
  --ty-deltas=-0.030,-0.024,-0.016,-0.010,-0.006,0.006,0.010,0.016,0.024,0.030 \
  --tx-deltas=0.0 \
  --active-target-only \
  --max-events 80 \
  --max-total-evals 180 \
  --allow-equal-f1-mispress-drop
