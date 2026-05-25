#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J cap_m10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap_m10_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap_m10_%j.err
#SBATCH -t 03:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_10/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_capriccio_tight_20260523
rm -rf "$ROOT"
mkdir -p "$ROOT"
$PY capriccio_mask_contact_gate.py --source-npz "$SRC" --output-dir "$ROOT/fp_m050" --samples-per-mask 120 --sigma 0.022 --top-seeds 5 --min-mean-static-improvement 0.08 --max-fp-increase -0.50 --threshold 0.5 --environment-name RoboPianist-debug-NocturneRousseau-v0 --seed 528
$PY capriccio_mask_contact_gate.py --source-npz "$SRC" --output-dir "$ROOT/fp_m100" --samples-per-mask 120 --sigma 0.022 --top-seeds 5 --min-mean-static-improvement 0.05 --max-fp-increase -1.00 --threshold 0.5 --environment-name RoboPianist-debug-NocturneRousseau-v0 --seed 529
$PY - <<'PY'
import json
from pathlib import Path
root=Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_capriccio_tight_20260523')
rows=[]
for p in sorted(root.glob('*/summary.json')):
    d=json.loads(p.read_text()); d['variant']=p.parent.name; rows.append(d)
summary={'root':str(root),'baseline_weighted':{'event_f1':0.2476190476190476,'frame_f1':0.09269785745518146,'matched':13,'target':64,'mispresses':28,'played':41},'rows':rows}
(root/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY