#!/bin/bash
#SBATCH --job-name=impromptu_render_200hz
#SBATCH --partition=cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/slurm_logs/impromptu_render_200hz_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/slurm_logs/impromptu_render_200hz_%j.err

set -euxo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano
eval "$(conda shell.bash hook)"
conda activate sonata

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export MUJOCO_GL=egl
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

TRAJECTORY_NPZ="${TRAJECTORY_NPZ:?TRAJECTORY_NPZ is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

python Impromptu/scripts/render_hand_state_playback.py \
  --trajectory-npz "${TRAJECTORY_NPZ}" \
  --output-dir "${OUTPUT_DIR}" \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --control-timestep 0.05 \
  --interpolation-substeps 10 \
  --fps 200 \
  --width 640 \
  --height 480

echo "VIDEO=${OUTPUT_DIR}/rollout_video.mp4"
