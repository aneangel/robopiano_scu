#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J imp_m10_ns
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/imp_m10_ns_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/imp_m10_ns_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_no_sustain_20260523
SIM=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_no_sustain_rp1m_sim_20260523
rm -rf "$ROOT" "$SIM"
mkdir -p "$ROOT" "$SIM"

MIDI=$("$PY" - <<'PY'
import json
from pathlib import Path
items = json.loads(Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_20260522/sample.json').read_text())
print(items[9])
PY
)

"$PY" Impromptu/scripts/plan_trajectory.py \
  --midi-path "$MIDI" \
  --output-root "$ROOT" \
  --run-name maestro_10 \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --control-timestep 0.05 \
  --interpolation-substeps 10 \
  --active-window-last-s 12.0 \
  --active-window-preroll-s 1.0 \
  --active-window-postroll-s 0.25 \
  --trajectory-mode joint_space_straighten \
  --key-press-depth 0.006 \
  --assignment-strategy ik_aware_topk \
  --assignment-top-k 8 \
  --ik-static-contact-validation \
  --ik-multistart-seed-count 4 \
  --ik-static-contact-wrong-key-weight 0.25 \
  --ik-static-contact-missed-key-weight 4.0 \
  --no-joint-space-preserve-sustained-fingers

"$PY" retest_impromptu_rp1m_simulator.py \
  --run-root "$ROOT" \
  --output-root "$SIM" \
  --only-run maestro_10 \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5
