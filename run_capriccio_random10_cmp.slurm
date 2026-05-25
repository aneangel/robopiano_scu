#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J cap10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap10_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap10_%j.err
#SBATCH -t 06:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_capriccio_strict_20260523
rm -rf "$ROOT"
mkdir -p "$ROOT"
for idx in $(seq -w 1 10); do
  run=maestro_${idx}
  echo "=== capriccio $run ==="
  $PY capriccio_mask_contact_gate.py \
    --source-npz "$SRC/$run/trajectory.npz" \
    --output-dir "$ROOT/$run" \
    --samples-per-mask 100 \
    --sigma 0.025 \
    --top-seeds 4 \
    --min-mean-static-improvement 0.08 \
    --max-fp-increase -0.25 \
    --threshold 0.5 \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --seed 527
done
$PY - <<'PY'
import json
from pathlib import Path
import numpy as np
root=Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_capriccio_strict_20260523')
rows=[]
for path in sorted(root.glob('maestro_*/summary.json')):
    d=json.loads(path.read_text())
    d['run']=path.parent.name
    rows.append(d)
frame=np.asarray([float(r.get('frame_f1',0)) for r in rows], dtype=np.float64)
event=np.asarray([float(r.get('event_f1',0)) for r in rows], dtype=np.float64)
summary={
    'root': str(root),
    'source_root': '/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522',
    'runs': rows,
    'mean_frame_f1': float(frame.mean()) if frame.size else 0.0,
    'median_frame_f1': float(np.median(frame)) if frame.size else 0.0,
    'mean_event_f1': float(event.mean()) if event.size else 0.0,
    'median_event_f1': float(np.median(event)) if event.size else 0.0,
    'total_matched': int(sum(int(r.get('matched',0)) for r in rows)),
    'total_target': int(sum(int(r.get('target',0)) for r in rows)),
    'total_mispresses': int(sum(int(r.get('mispresses',0)) for r in rows)),
    'total_played': int(sum(int(r.get('played',0)) for r in rows)),
}
(root/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY