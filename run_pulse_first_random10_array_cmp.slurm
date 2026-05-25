#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J pulse10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/pulse10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/pulse10_%A_%a.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
idx=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
run=maestro_${idx}

"$PY" pulse_event_postprocess_gate.py \
  --source-npz "/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523/$run/trajectory.npz" \
  --baseline-npz "/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/$run/trajectory.npz" \
  --output-dir "/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_pulse_first_20260523/$run" \
  --post-frames 1 \
  --rest-source first \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5
