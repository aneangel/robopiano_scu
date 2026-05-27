#!/bin/bash
#SBATCH -J pid_opt
#SBATCH -p cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/PIDController/logs/%x-%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/PIDController/logs/%x-%j.err

set -euo pipefail

source /WAVE/apps/el8/conda/envs/Python/20240305/etc/profile.d/conda.sh
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl

cd /WAVE/projects/ECEN-524-Wi26/robopiano
mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/PIDController/logs

python PID-Controller/scripts/optimize_pid_controller.py \
  --run-root /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/optimized_sample20 \
  --trajectory-limit 1 \
  --controllers pd \
  --kp-grid 0.7,1.0,1.3,1.6 \
  --kd-grid 0,0.005,0.01,0.02 \
  --target-velocity-scales 0,0.5,1.0 \
  --setpoint-policies linear,next \
  --max-source-steps 20 \
  --scale-best \
  --scale-limit 1 \
  --resume
