#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J m07_lin
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/m07_lin_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/m07_lin_%j.err
#SBATCH -t 00:40:00
#SBATCH --cpus-per-task=2

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python

"$PY" dense_interp_retest_gate.py \
  --source-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523/maestro_07/trajectory.npz \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro07_etude_linear_retest_20260523 \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5
