#!/bin/bash
# Partita: RP1M action rollout fidelity (calibration + validation against recorded poses).
#
# Submits like other Partita Slurm jobs:
#   sbatch /WAVE/projects/ECEN-524-Wi26/robopiano/partita/scripts/run_rollout_fidelity.slurm
#
# Environment overrides (optional):
#   PARTITA_CONFIG=/path/to/config.yaml
#   PARTITA_SKIP_PIPELINE=1        Skip run_partita_debug.py (requires existing outputs under partita/outputs/)
#   PARTITA_CALIBRATE=0            Skip --calibrate-action-scale (use rollout YAML + existing calibration.json)
#   PARTITA_CALIBRATION_PROBE_STEPS=5
#   PARTITA_IGNORE_CALIBRATION=1   Pass --ignore-calibration to simulate_rollout.py before validate
#   PARTITA_VIDEO_* / MAX_STEPS same as run_partita_validation.slurm
#
# Primary artifacts:
#   partita/outputs/rollout/<experiment>/calibration.json
#   partita/outputs/rollout/<experiment>/fidelity_summary.json
#   partita/outputs/rollout/<experiment>/original_target_fidelity_frames.csv
#
#SBATCH --job-name=partita-rollout-fid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/WAVE/projects/ECEN-524-Wi26/robopiano/partita/outputs/logs/rollout_fidelity_%j.out
#SBATCH --error=/WAVE/projects/ECEN-524-Wi26/robopiano/partita/outputs/logs/rollout_fidelity_%j.err

set -eo pipefail

source ~/.bashrc
set -u
conda activate sonata

cd /WAVE/projects/ECEN-524-Wi26/robopiano

export PYTHONPATH=$PWD/partita/src:$PWD:$PWD/robopianist:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl

CONFIG=${PARTITA_CONFIG:-partita/configs/debug.yaml}
WIDTH=${PARTITA_VIDEO_WIDTH:-640}
HEIGHT=${PARTITA_VIDEO_HEIGHT:-480}
FPS=${PARTITA_VIDEO_FPS:-20}
RENDER_EVERY=${PARTITA_RENDER_EVERY:-1}
PROBE_STEPS=${PARTITA_CALIBRATION_PROBE_STEPS:-5}
MAX_STEPS_FLAG=()
if [[ -n "${PARTITA_MAX_STEPS:-}" ]]; then
  MAX_STEPS_FLAG=(--max-steps "${PARTITA_MAX_STEPS}")
fi
IGNORE_CALIB_FLAG=()
if [[ -n "${PARTITA_IGNORE_CALIBRATION:-}" ]]; then
  IGNORE_CALIB_FLAG=(--ignore-calibration)
fi

mkdir -p partita/outputs/logs

if [[ "${PARTITA_SKIP_PIPELINE:-0}" != "1" ]]; then
  python partita/scripts/run_partita_debug.py --config "${CONFIG}"
fi

if [[ "${PARTITA_CALIBRATE:-1}" == "1" ]]; then
  echo "=== Rolling action scale x mapping calibration ==="
  python partita/scripts/simulate_rollout.py \
    --config "${CONFIG}" \
    --calibrate-action-scale \
    --calibration-probe-steps "${PROBE_STEPS}" \
    --which original \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --render-every "${RENDER_EVERY}" \
    "${MAX_STEPS_FLAG[@]}"
fi

echo "=== Validating RP1M reproduction (exit non-zero if criteria fail) ==="
python partita/scripts/simulate_rollout.py \
  --config "${CONFIG}" \
  --validate-fidelity \
  "${IGNORE_CALIB_FLAG[@]}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --render-every "${RENDER_EVERY}" \
  "${MAX_STEPS_FLAG[@]}"

echo "=== Done. Inspect fidelity_summary.json under partita/outputs/rollout/<experiment_name>/ ==="
