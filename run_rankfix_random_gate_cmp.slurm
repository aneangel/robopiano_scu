#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J rankfix_g
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/rankfix_g_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/rankfix_g_%j.err
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SAMPLE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_20260522/sample.json
SRC_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_rankfix_source_20260523
SRC_SIM_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_rankfix_source_rp1m_sim_20260523
ETUDE_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_rankfix_etude_open_gate_20260523
IDX="${IDX:-}"
if [[ -z "$IDX" ]]; then
  IDX="$("$PY" - <<'PY'
import random
print(random.SystemRandom().randint(1, 10))
PY
)"
fi
idx=$(printf "%02d" "$IDX")
run=maestro_${idx}
midi="$("$PY" - "$SAMPLE" "$IDX" <<'PY'
import json
import sys
from pathlib import Path
sample = json.loads(Path(sys.argv[1]).read_text())
idx = int(sys.argv[2]) - 1
print(sample[idx])
PY
)"

echo "=== rankfix random gate $run $midi ==="
mkdir -p "$SRC_ROOT" "$SRC_SIM_ROOT" "$ETUDE_ROOT"

"$PY" Impromptu/scripts/plan_trajectory.py \
  --midi-path "$midi" \
  --output-root "$SRC_ROOT" \
  --run-name "$run" \
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
  --ik-static-contact-failure-weight 25.0

"$PY" retest_impromptu_rp1m_simulator.py \
  --run-root "$SRC_ROOT" \
  --output-root "$SRC_SIM_ROOT" \
  --only-run "$run" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5

"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC_ROOT/$run/trajectory.npz" \
  --output-dir "$ETUDE_ROOT/$run" \
  --samples-per-mask 180 \
  --elite-count 8 \
  --iterations 3 \
  --top-seeds 6 \
  --finger-sigma 0.050 \
  --forearm-sigma 0.035 \
  --min-mean-static-improvement 0.02 \
  --max-fp-increase 1.00 \
  --fp-rank-weight 0.05 \
  --threshold 0.5 \
  --seed 713

"$PY" - "$run" "$SRC_SIM_ROOT" "$ETUDE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
run, src_sim_root, etude_root = sys.argv[1:4]
old_etude = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523') / run / 'summary.json'
old_weighted = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_rp1m_sim_20260523') / run / 'impromptu_rp1m_retest_result.json'
new_source = Path(src_sim_root) / run / 'impromptu_rp1m_retest_result.json'
new_etude = Path(etude_root) / run / 'summary.json'
summary = {
    'run': run,
    'old_weighted': json.loads(old_weighted.read_text()) if old_weighted.exists() else None,
    'old_etude_open': json.loads(old_etude.read_text()) if old_etude.exists() else None,
    'rankfix_source': json.loads(new_source.read_text()) if new_source.exists() else None,
    'rankfix_etude_open': json.loads(new_etude.read_text()) if new_etude.exists() else None,
}
out = Path(etude_root) / f'{run}_gate_summary.json'
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
