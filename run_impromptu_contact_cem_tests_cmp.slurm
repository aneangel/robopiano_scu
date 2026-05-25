#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J imp_cem_t
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/imp_cem_t_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/imp_cem_t_%j.err
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=2

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
export PYTHONPATH=/WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/src:/WAVE/projects/ECEN-524-Wi26/robopiano/Bagatelle/src:/WAVE/projects/ECEN-524-Wi26/robopiano/Intermezzo/src:/WAVE/projects/ECEN-524-Wi26/robopiano/partita/src:/WAVE/projects/ECEN-524-Wi26/robopiano

"$PY" -m py_compile Impromptu/src/impromptu/contact_cem.py Impromptu/src/impromptu/config.py Impromptu/src/impromptu/planner.py Impromptu/scripts/plan_trajectory.py
"$PY" -m pytest -q Impromptu/tests/test_planner_schema.py
