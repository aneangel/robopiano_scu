#!/bin/bash
#SBATCH --job-name=allegro_200hz
#SBATCH --partition=cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/Allegro/slurm_logs/%x_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/Allegro/slurm_logs/%x_%j.err

set -euo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano
source /WAVE/apps/el8/conda/envs/Python/20240305/etc/profile.d/conda.sh
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

: "${RP1M_300_ROOT:=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}"
: "${FUGUE_OUTPUT_ROOT:?Set FUGUE_OUTPUT_ROOT to the existing Fugue run containing model_f_planner_next}"
: "${RUN_NAME:=allegro_$(date +%Y%m%d_%H%M%S)}"
: "${OUTPUT_ROOT:=/WAVE/datasets/ccoelho_lab-jlanders/Allegro/${RUN_NAME}}"
: "${SPLIT:=test}"
: "${DEMO_INDEX:=0}"
: "${DEVICE:=cpu}"

mkdir -p "$OUTPUT_ROOT" /WAVE/datasets/ccoelho_lab-jlanders/Allegro/slurm_logs

python Allegro/scripts/rollout_fugue_200hz.py \
  --checkpoint "$FUGUE_OUTPUT_ROOT/model_f_planner_next/checkpoints/best.pt" \
  --rp1m-root "$RP1M_300_ROOT" \
  --dataset-artifact-root "$FUGUE_OUTPUT_ROOT/model_f_planner_next/dataset" \
  --split "$SPLIT" \
  --demo-index "$DEMO_INDEX" \
  --output-dir "$OUTPUT_ROOT/allegro_200hz_rollout" \
  --device "$DEVICE"
