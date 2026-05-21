#!/bin/bash
#SBATCH -J imp_maestro_refine
#SBATCH -p cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/imp_maestro_refine_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/imp_maestro_refine_%j.err

set -euo pipefail

set +u
source ~/.bashrc
set -u
conda activate sonata

cd /WAVE/projects/ECEN-524-Wi26/robopiano

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/maestro_refinement_${RUN_STAMP}}"

mkdir -p "${OUTPUT_ROOT}"

python Impromptu/scripts/run_maestro_refinement_sweep.py \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
