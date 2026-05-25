#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J cad10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cad10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cad10_%A_%a.err
#SBATCH -t 02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_cadenza_20260524
mkdir -p "$ROOT"

idx=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
run=maestro_${idx}
echo "=== Cadenza 10-song confirmation $run ==="

"$PY" cadenza_event_injection_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$ROOT/$run/strict" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pulse-width 2 \
  --pulse-gap 0 \
  --max-static-fp 1 \
  --min-static-f1 0.40 \
  --only-static-missed

"$PY" cadenza_event_injection_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$ROOT/$run/open" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pulse-width 2 \
  --pulse-gap 0 \
  --max-static-fp 2 \
  --min-static-f1 0.25 \
  --only-static-missed
