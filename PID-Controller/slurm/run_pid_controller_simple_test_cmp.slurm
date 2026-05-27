#!/bin/bash
#SBATCH -J pid_simple
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

MANIFEST=/WAVE/datasets/ccoelho_lab-jlanders/PIDController/prepared/sample_manifest.json

python PID-Controller/scripts/prepare_pid_trajectories.py \
  --run-root /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch \
  --output-json "$MANIFEST" \
  --limit 1

python PID-Controller/scripts/test_prepared_pid_trajectories.py \
  --trajectory-manifest "$MANIFEST" \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/PIDController/simple_sample20 \
  --limit 1 \
  --controllers p,pd \
  --max-source-steps 20 \
  --resume
