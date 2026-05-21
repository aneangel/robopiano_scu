#!/bin/bash
#SBATCH --job-name=maestro_selective_z_10cm
#SBATCH --partition=cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-1
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/slurm_logs/maestro_selective_z_10cm_%A_%a.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/slurm_logs/maestro_selective_z_10cm_%A_%a.err

set -exo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano
source ~/.bashrc
conda activate sonata

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/slurm_logs

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  export MAESTRO_CONDITION_LABEL=below_key_10cm_decay
  export BAGATELLE_KEY_PRESS_DEPTH=0.005
else
  export MAESTRO_CONDITION_LABEL=above_key_10cm_decay
  export BAGATELLE_KEY_PRESS_DEPTH=-0.005
fi

python Intermezzo/scripts/test_maestro_selective_z_compare_10cm.py
