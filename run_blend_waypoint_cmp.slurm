#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J blend_wp
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/blend_wp_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/blend_wp_%j.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=2

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
BASE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_05/trajectory.npz
CAND=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro05_sonatina_contact_search_20260523/conservative/trajectory.npz
for alpha in 0.25 0.50 0.75; do
  safe=${alpha/./p}
  $PY blend_waypoint_trajectory_gate.py \
    --baseline-npz "$BASE" \
    --candidate-npz "$CAND" \
    --output-dir "/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro05_sonatina_contact_search_20260523/blend_${safe}" \
    --alpha "$alpha" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5
done