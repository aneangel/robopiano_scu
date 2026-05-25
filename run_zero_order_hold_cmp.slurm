#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J zoh_m10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/zoh_m10_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/zoh_m10_%j.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_10/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_zero_order_hold_20260523
rm -rf "$ROOT"
mkdir -p "$ROOT"

$PY zero_order_hold_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/baseline_zoh" \
  --threshold 0.5 \
  --environment-name RoboPianist-debug-NocturneRousseau-v0
