#!/bin/bash
#SBATCH --job-name=impromptu_twinkle_200hz
#SBATCH --partition=cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/slurm_logs/impromptu_twinkle_200hz_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/slurm_logs/impromptu_twinkle_200hz_%j.err

set -euxo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano
eval "$(conda shell.bash hook)"
conda activate sonata
python -V
hostname

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export MUJOCO_GL=egl
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs
mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/evaluation
mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/slurm_logs

RUN_NAME="impromptu_twinkle_precision_safe_active10_${SLURM_JOB_ID}"
RUN_DIR="/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/${RUN_NAME}"
EVAL_DIR="/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/evaluation/${RUN_NAME}_eval"
MIDI_PATH="/WAVE/projects/ECEN-524-Wi26/robopiano/robopianist/music/data/rousseau/twinkle-twinkle-trimmed.mid"

python Impromptu/scripts/plan_impromptu.py \
  --midi-path "${MIDI_PATH}" \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs \
  --run-name "${RUN_NAME}" \
  --control-timestep 0.05 \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --interpolation-substeps 10 \
  --active-window-last-s 12.0 \
  --active-window-preroll-s 1.0 \
  --active-window-postroll-s 0.5 \
  --key-press-depth 0.0065 \
  --clearance-height 0.04 \
  --approach-s 0.070 \
  --hold-s 0.018 \
  --release-s 0.025 \
  --anchor-stride 1 \
  --solve-all-stride-anchors \
  --include-midpoint-anchors \
  --ik-fingertip-weight 1.20 \
  --ik-smoothness-weight 0.08 \
  --ik-neutral-weight 0.015 \
  --ik-max-nfev 100 \
  --residual-success-threshold 0.018

python Impromptu/scripts/evaluate_impromptu.py \
  --trajectory-npz "${RUN_DIR}/trajectory.npz" \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/evaluation \
  --run-name "${RUN_NAME}_eval"

python Impromptu/scripts/render_dense_playback.py \
  --trajectory-npz "${RUN_DIR}/trajectory.npz" \
  --output-dir "${RUN_DIR}/render_200fps_precision_safe" \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --control-timestep 0.05 \
  --interpolation-substeps 10 \
  --fps 200 \
  --width 640 \
  --height 480 \
  --threshold 0.5

echo "RUN_DIR=${RUN_DIR}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "VIDEO=${RUN_DIR}/render_200fps_precision_safe/rollout_video.mp4"
