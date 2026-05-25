#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J rankfix_t
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/rankfix_t_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/rankfix_t_%j.err
#SBATCH -t 00:20:00
#SBATCH --cpus-per-task=2

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python

"$PY" -m py_compile \
  Bagatelle/src/bagatelle/config.py \
  Bagatelle/src/bagatelle/kinematics.py \
  Impromptu/src/impromptu/config.py \
  Impromptu/src/impromptu/planner.py \
  Impromptu/scripts/plan_trajectory.py
"$PY" -m pytest Bagatelle/tests/test_planner.py Impromptu/tests/test_planner_schema.py
