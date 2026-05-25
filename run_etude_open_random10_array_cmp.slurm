#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J etuopen
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etuopen_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etuopen_%A_%a.err
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
mkdir -p "$ROOT"

idx=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
run=maestro_${idx}
echo "=== etude open-recall array $run ==="
"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$ROOT/$run" \
  --samples-per-mask 180 \
  --elite-count 8 \
  --iterations 3 \
  --top-seeds 6 \
  --finger-sigma 0.050 \
  --forearm-sigma 0.035 \
  --min-mean-static-improvement 0.02 \
  --max-fp-increase 1.00 \
  --fp-rank-weight 0.05 \
  --threshold 0.5 \
  --seed 713
