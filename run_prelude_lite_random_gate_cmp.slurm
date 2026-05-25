#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J prel_lite
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/prel_lite_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/prel_lite_%j.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=3

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_prelude_lite_gate_20260523
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
mkdir -p "$ROOT"
echo "=== prelude-lite arpeggio random gate $run ==="

"$PY" -m py_compile prelude_lite_arpeggio_gate.py
"$PY" prelude_lite_arpeggio_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$ROOT/$run" \
  --max-keys-per-frame 2 \
  --max-shift-frames 3 \
  --samples 80 \
  --iterations 2 \
  --threshold 0.5

"$PY" - "$run" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
run, root = sys.argv[1:3]
old = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523') / run / 'summary.json'
new = Path(root) / run / 'summary.json'
summary = {
    'run': run,
    'old_etude_open': json.loads(old.read_text()) if old.exists() else None,
    'prelude_lite': json.loads(new.read_text()) if new.exists() else None,
}
out = Path(root) / f'{run}_gate_summary.json'
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
