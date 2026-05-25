#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J sic10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sic10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sic10_%A_%a.err
#SBATCH -t 01:40:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_siciliana_zero_20260524
mkdir -p "$ROOT"

idx=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
run=maestro_${idx}
echo "=== Siciliana zero-fp 10-song confirmation $run ==="
"$PY" siciliana_grouped_injection_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$ROOT/$run" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pulse-width 1 \
  --pulse-gap 0 \
  --max-static-fp 0 \
  --max-candidate-played-count 0 \
  --max-pulses-per-onset 2 \
  --played-weight 0.07
