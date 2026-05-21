#!/usr/bin/env bash
#SBATCH --job-name=imp_cmp_tests
#SBATCH --partition=cmp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/logs/imp_cmp_tests_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/logs/imp_cmp_tests_%j.err

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
python - <<'PY'
import sys
print('python executable:', sys.executable)
print('sys.path head:', sys.path[:8])
PY

echo "=== compileall ==="
python -m compileall Impromptu/src/impromptu Impromptu/scripts

echo "=== unit tests ==="
pytest -q \
  Impromptu/tests/test_evaluation_metrics.py \
  Impromptu/tests/test_waypoint_preservation.py \
  --tb=short

echo "=== optional broader Impromptu tests ==="
pytest -q Impromptu/tests --tb=short

echo "=== done ==="
