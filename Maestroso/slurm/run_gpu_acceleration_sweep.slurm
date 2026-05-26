#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J MaestrosoGPU
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch/logs/gpu_sweep_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch/logs/gpu_sweep_%j.err
#SBATCH -t 0-02:00:00
#SBATCH --cpus-per-task=24
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

OUT=/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch
mkdir -p "$OUT/logs"

python Maestroso/scripts/run_acceleration_sweep.py \
  --output-root "$OUT" \
  --songs 2 \
  --max-duration-s 45 \
  --variants cpu_fast,gpu_batch_proposal,gpu_batch_proposal_bagatelle_assign,gpu_rhapsody_seed,gpu_rhapsody_refine,chunked_gpu_rhapsody \
  --variant-timeout-s 900 \
  --rollout-timeout-s 600 \
  --successive-halving \
  --round1-max-duration-s 15 \
  --stage-b-keep 3 \
  --stage-c-keep 1 \
  --stage-d-render \
  --proxy-activation-frames 32 \
  --proposal-device cuda \
  --proposal-batch-size 512 \
  --proposal-passes 2 \
  --proposal-refinement-steps 0 \
  --chunk-seconds 15 \
  --chunk-overlap-seconds 0 \
  --chunk-workers 2
