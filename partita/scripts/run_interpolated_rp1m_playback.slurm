#!/bin/bash
# Partita: RP1M interpolated hand-state playback in RoboPianist.
#
# Useful overrides:
#   PARTITA_RP1M_ROOT=/path/to/rp1m_repertoire.zarr
#   PARTITA_SONG_KEY=RoboPianist-...-v0_0
#   PARTITA_DEMO_ID=206
#   PARTITA_ENVIRONMENT_NAME=RoboPianist-...-v0
#   PARTITA_OUTPUT_DIR=/path/to/output_dir
#   PARTITA_MAX_SOURCE_STEPS=40
#   PARTITA_RENDER_EVERY_SOURCE_STEP=5
#
# Submit smoke test:
#   PARTITA_MAX_SOURCE_STEPS=40 PARTITA_RENDER_EVERY_SOURCE_STEP=5 \
#     sbatch partita/scripts/run_interpolated_rp1m_playback.slurm
#
#SBATCH --job-name=rp1m-interp
#SBATCH --partition=cmp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/WAVE/projects2/ECEN-524-Wi26/robopiano/partita/outputs/logs/interpolated_rp1m_%j.out
#SBATCH --error=/WAVE/projects2/ECEN-524-Wi26/robopiano/partita/outputs/logs/interpolated_rp1m_%j.err

set -eo pipefail

source ~/.bashrc
set -u
conda activate sonata

cd /WAVE/projects2/ECEN-524-Wi26/robopiano

export PYTHONPATH=$PWD/partita/src:$PWD:$PWD/robopianist:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl

RP1M_ROOT=${PARTITA_RP1M_ROOT:-/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}
SONG_KEY=${PARTITA_SONG_KEY:-RoboPianist-GP-AlberyDorothyPreludeInDFlatMajor-v0_0}
DEMO_ID=${PARTITA_DEMO_ID:-206}
LABEL=${PARTITA_LABEL:-rp1m_demo206_interpolated}
WIDTH=${PARTITA_VIDEO_WIDTH:-640}
HEIGHT=${PARTITA_VIDEO_HEIGHT:-480}
FPS=${PARTITA_VIDEO_FPS:-20}
DATASET_TIMESTEP=${PARTITA_DATASET_TIMESTEP:-0.05}
SIMULATION_TIMESTEP=${PARTITA_SIMULATION_TIMESTEP:-0.005}
RENDER_EVERY=${PARTITA_RENDER_EVERY_SOURCE_STEP:-1}
ACTION_SOURCE_SCALE=${PARTITA_ACTION_SOURCE_SCALE:-normalized_minus_one_to_one}
ACTION_MAPPING=${PARTITA_ACTION_MAPPING:-as_is}
PLAYBACK_MODE=${PARTITA_PLAYBACK_MODE:-recorded_state}
SEED=${PARTITA_SEED:-0}
THRESHOLD=${PARTITA_THRESHOLD:-0.5}

mkdir -p partita/outputs/logs

ARGS=(
  --rp1m-root "${RP1M_ROOT}"
  --song-key "${SONG_KEY}"
  --demo-id "${DEMO_ID}"
  --label "${LABEL}"
  --dataset-timestep "${DATASET_TIMESTEP}"
  --simulation-timestep "${SIMULATION_TIMESTEP}"
  --fps "${FPS}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --render-every-source-step "${RENDER_EVERY}"
  --seed "${SEED}"
  --threshold "${THRESHOLD}"
  --action-source-scale "${ACTION_SOURCE_SCALE}"
  --action-mapping "${ACTION_MAPPING}"
  --playback-mode "${PLAYBACK_MODE}"
)

if [[ -n "${PARTITA_ENVIRONMENT_NAME:-}" ]]; then
  ARGS+=(--environment-name "${PARTITA_ENVIRONMENT_NAME}")
fi
if [[ -n "${PARTITA_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${PARTITA_OUTPUT_DIR}")
fi
if [[ -n "${PARTITA_MAX_SOURCE_STEPS:-}" ]]; then
  ARGS+=(--max-source-steps "${PARTITA_MAX_SOURCE_STEPS}")
fi
if [[ "${PARTITA_FULL_ACTION_SPACE:-0}" == "1" ]]; then
  ARGS+=(--full-action-space)
fi
if [[ "${PARTITA_NO_RESTORE_INITIAL_PIANO:-0}" == "1" ]]; then
  ARGS+=(--no-restore-initial-piano)
fi
if [[ "${PARTITA_NO_SET_HAND_QVEL:-0}" == "1" ]]; then
  ARGS+=(--no-set-hand-qvel)
fi
if [[ "${PARTITA_DISABLE_HAND_COLLISIONS:-0}" == "1" ]]; then
  ARGS+=(--disable-hand-collisions)
fi
if [[ "${PARTITA_ENABLE_FINGERING_REWARD:-0}" == "1" ]]; then
  ARGS+=(--enable-fingering-reward)
fi
if [[ "${PARTITA_ENABLE_FOREARM_REWARD:-0}" == "1" ]]; then
  ARGS+=(--enable-forearm-reward)
fi
if [[ "${PARTITA_PREFER_CANONICAL_MIDI:-0}" == "1" ]]; then
  ARGS+=(--prefer-canonical-midi)
fi

python partita/scripts/playback_interpolated_rp1m.py "${ARGS[@]}"
