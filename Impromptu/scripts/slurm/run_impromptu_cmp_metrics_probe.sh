#!/usr/bin/env bash
#SBATCH --job-name=imp_cmp_probe
#SBATCH --partition=cmp
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/logs/imp_cmp_probe_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/logs/imp_cmp_probe_%j.err

set -euo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano

mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/logs
mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/test_runs

set +u
source ~/.bashrc || true
set -u
if command -v conda >/dev/null 2>&1; then
  conda activate sonata || conda activate robopiano || conda activate base || true
fi

export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTHONPATH="$PWD/Impromptu/src:$PWD/Bagatelle/src:$PWD/Intermezzo/src:$PWD:${PYTHONPATH:-}"

python --version
python Impromptu/scripts/debug_impromptu_metrics_probe.py
