#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J sar10agg
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sar10agg_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sar10agg_%j.err
#SBATCH -t 00:20:00
#SBATCH --cpus-per-task=1

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_sarabande_lift_20260524')
source = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523')
rows = []
for path in sorted(root.glob('maestro_*/summary.json')):
    row = json.loads(path.read_text())
    row['run'] = path.parent.name
    old = json.loads((source / row['run'] / 'summary.json').read_text())
    row['old_event_f1'] = float(old.get('event_f1', 0.0))
    row['old_frame_f1'] = float(old.get('frame_f1', 0.0))
    row['old_matched'] = int(old.get('matched', 0))
    row['old_target'] = int(old.get('target', 0))
    row['old_mispresses'] = int(old.get('mispresses', 0))
    row['old_played'] = int(old.get('played', 0))
    row['delta_event_f1'] = float(row.get('event_f1', 0.0)) - row['old_event_f1']
    row['delta_frame_f1'] = float(row.get('frame_f1', 0.0)) - row['old_frame_f1']
    rows.append(row)
frame = np.asarray([float(r.get('frame_f1', 0.0)) for r in rows], dtype=np.float64)
event = np.asarray([float(r.get('event_f1', 0.0)) for r in rows], dtype=np.float64)
summary = {
    'root': str(root),
    'source_root': str(source),
    'runs_completed': int(len(rows)),
    'mean_event_f1': float(event.mean()) if event.size else 0.0,
    'median_event_f1': float(np.median(event)) if event.size else 0.0,
    'mean_frame_f1': float(frame.mean()) if frame.size else 0.0,
    'median_frame_f1': float(np.median(frame)) if frame.size else 0.0,
    'total_matched': int(sum(int(r.get('matched', 0)) for r in rows)),
    'total_target': int(sum(int(r.get('target', 0)) for r in rows)),
    'total_mispresses': int(sum(int(r.get('mispresses', 0)) for r in rows)),
    'total_played': int(sum(int(r.get('played', 0)) for r in rows)),
    'old_mean_event_f1': float(np.mean([float(r.get('old_event_f1', 0.0)) for r in rows])) if rows else 0.0,
    'old_mean_frame_f1': float(np.mean([float(r.get('old_frame_f1', 0.0)) for r in rows])) if rows else 0.0,
    'old_total_matched': int(sum(int(r.get('old_matched', 0)) for r in rows)),
    'old_total_target': int(sum(int(r.get('old_target', 0)) for r in rows)),
    'old_total_mispresses': int(sum(int(r.get('old_mispresses', 0)) for r in rows)),
    'old_total_played': int(sum(int(r.get('old_played', 0)) for r in rows)),
    'runs': rows,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
