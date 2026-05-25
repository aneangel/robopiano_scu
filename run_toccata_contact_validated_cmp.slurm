#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J toccata_cv
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/toccata_cv_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/toccata_cv_%j.err
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
$PY toccata_contact_validated_retrieval_gate.py \
  --source-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_05/trajectory.npz \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro05_toccata_contact_validated_20260523/default \
  --max-songs 80 \
  --demos-per-song 2 \
  --stride 5 \
  --per-mask-limit 12 \
  --min-static-improvement 0.10 \
  --max-fp-increase 0 \
  --threshold 0.5 \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --seed 0