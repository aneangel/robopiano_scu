#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J capric_c2
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/capric_c2_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/capric_c2_%j.err
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
$PY capriccio_mask_contact_gate.py \
  --source-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_05/trajectory.npz \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro05_capriccio_mask_contact_20260523/strict_fp \
  --samples-per-mask 100 \
  --sigma 0.025 \
  --top-seeds 4 \
  --min-mean-static-improvement 0.08 \
  --max-fp-increase -0.25 \
  --threshold 0.5 \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --seed 527