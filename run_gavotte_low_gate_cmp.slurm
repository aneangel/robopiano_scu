#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J gavlow
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/gavlow_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/gavlow_%j.err
#SBATCH -t 00:20:00
#SBATCH --cpus-per-task=2

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_fandango_hybrid_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/gavotte_low_gate_20260524

for run in maestro_01 maestro_07; do
  for post in 4 8 12; do
    "$PY" gavotte_event_window_scrub_gate.py \
      --source-npz "$SRC/$run/hybrid/trajectory.npz" \
      --output-dir "$ROOT/$run/post${post}" \
      --threshold 0.5 \
      --pre-substeps 0 \
      --post-substeps "$post"
  done
done
