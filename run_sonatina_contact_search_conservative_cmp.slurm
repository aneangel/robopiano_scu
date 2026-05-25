#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J sonatina_c2
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sonatina_c2_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sonatina_c2_%j.err
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
$PY sonatina_contact_search_gate.py \
  --source-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_05/trajectory.npz \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro05_sonatina_contact_search_20260523/conservative \
  --samples-per-waypoint 64 \
  --sigma 0.025 \
  --min-static-improvement 0.10 \
  --max-fp-increase 0 \
  --threshold 0.5 \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --seed 525