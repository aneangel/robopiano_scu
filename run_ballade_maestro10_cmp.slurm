#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J bal_m10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/bal_m10_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/bal_m10_%j.err
#SBATCH -t 05:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_10/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_ballade_selective_20260523
mkdir -p "$ROOT"

"$PY" ballade_event_selective_cem_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/event_proxy" \
  --samples-per-mask 180 \
  --elite-count 8 \
  --iterations 3 \
  --top-seeds 6 \
  --finger-sigma 0.050 \
  --forearm-sigma 0.035 \
  --min-static-improvement 0.02 \
  --max-fp-increase 1.00 \
  --fp-rank-weight 0.05 \
  --min-proxy-event-improvement 0.002 \
  --proxy-frame-weight 0.10 \
  --threshold 0.5 \
  --seed 809
