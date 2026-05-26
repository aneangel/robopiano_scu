#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J MaestrosoRecovery
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAccuracyRecovery/logs/recovery_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAccuracyRecovery/logs/recovery_%j.err
#SBATCH -t 0-06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano

source /WAVE/apps/el8/conda/envs/Python/20240305/etc/profile.d/conda.sh
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT=/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAccuracyRecovery
mkdir -p "$OUT/logs"

python Maestroso/scripts/run_accuracy_recovery_sweep.py \
  --output-root "$OUT" \
  --maestro-root /WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0 \
  --songs 2 \
  --max-duration-s 30 \
  --hard-window-max-duration-s 20 \
  --rhapsody-checkpoint /WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/random10_heldout2_prev_random_scale6_refine16_20260522/rhapsody_rpik.pt \
  --rhapsody-device cuda \
  --variant-timeout-s 1800 \
  --rollout-timeout-s 900 \
  --chunk-overlap-seconds 1.0
