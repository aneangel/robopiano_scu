#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J evt_modes
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/evt_modes_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/evt_modes_%j.err
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=1

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python

"$PY" analyze_event_error_modes.py \
  --root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523 \
  --output-json /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523/event_error_modes.json

"$PY" analyze_event_error_modes.py \
  --root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_rp1m_sim_20260523 \
  --output-json /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_rp1m_sim_20260523/event_error_modes.json
