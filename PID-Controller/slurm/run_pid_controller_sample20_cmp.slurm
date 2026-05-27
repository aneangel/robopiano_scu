#!/bin/bash
#SBATCH -J pid_sample20
#SBATCH -p cmp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/PIDController/logs/%x-%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/PIDController/logs/%x-%j.err

set -euo pipefail

source /WAVE/apps/el8/conda/envs/Python/20240305/etc/profile.d/conda.sh
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl

cd /WAVE/projects/ECEN-524-Wi26/robopiano
mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/PIDController/logs

python PID-Controller/scripts/validate_pid_rollouts.py \
  --run-root /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/pd_sample20 \
  --limit 1 \
  --controller pd \
  --kp 1.0 \
  --kd 0.02 \
  --max-source-steps 20
