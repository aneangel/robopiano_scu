#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J gigue10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/gigue10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/gigue10_%A_%a.err
#SBATCH -t 03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_fandango_hybrid_20260524
OUT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_gigue_chunk_20260524
run=$(printf "maestro_%02d" "${SLURM_ARRAY_TASK_ID}")

"$PY" gigue_sim_chunk_scrub_gate.py \
  --source-npz "$SRC/$run/hybrid/trajectory.npz" \
  --output-dir "$OUT/$run" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --chunk-control-frames 5
