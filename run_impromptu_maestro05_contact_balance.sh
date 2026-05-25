#!/usr/bin/env bash
set -uo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY="/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python"
ROOT="/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro05_contact_balance_20260523"
SRC="/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_20260522/sample.json"
MIDI="$("$PY" - <<'PY'
import json
from pathlib import Path
sample = json.loads(Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_20260522/sample.json').read_text())
print(sample[4])
PY
)"

rm -rf "$ROOT"
mkdir -p "$ROOT"
printf '%s\n' "$MIDI" > "$ROOT/midi.txt"

run_variant() {
  local name="$1"
  local wrong="$2"
  local missed="$3"
  local xy_weight="$4"
  local xy_radius="$5"
  local depth="$6"
  echo "=== $name wrong=$wrong missed=$missed xy=$xy_weight radius=$xy_radius depth=$depth ==="

  "$PY" Impromptu/scripts/plan_trajectory.py \
    --midi-path "$MIDI" \
    --output-root "$ROOT" \
    --run-name "$name" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --control-timestep 0.05 \
    --interpolation-substeps 10 \
    --active-window-last-s 12.0 \
    --active-window-preroll-s 1.0 \
    --active-window-postroll-s 0.25 \
    --trajectory-mode joint_space_straighten \
    --disable-adaptive-complex-song-defaults \
    --key-press-depth "$depth" \
    --wrong-hand-penalty 4.0 \
    --wrong-hand-split-key 48 \
    --assignment-dynamic-hand-split \
    --assignment-dynamic-hand-split-min-span 12 \
    --assignment-dynamic-hand-split-min-keys 3 \
    --finger-crossing-penalty 1.0 \
    --same-key-same-finger-bonus 0.25 \
    --assignment-strategy ik_aware_topk \
    --assignment-top-k 8 \
    --ik-unassigned-fingertip-strategy avoid_mispresses \
    --ik-unassigned-fingertip-avoidance-weight 0.5 \
    --ik-unassigned-fingertip-avoidance-radius 0.03 \
    --ik-wrong-key-xy-avoidance-weight "$xy_weight" \
    --ik-wrong-key-xy-avoidance-radius "$xy_radius" \
    --ik-static-contact-validation \
    --ik-multistart-seed-count 4 \
    --ik-static-contact-wrong-key-weight "$wrong" \
    --ik-static-contact-missed-key-weight "$missed"

  env MUJOCO_GL=egl "$PY" retest_impromptu_rp1m_simulator.py \
    --run-root "$ROOT" \
    --output-root "$ROOT/rp1m_sim" \
    --only-run "$name" \
    --threshold 0.5 \
    --seed 0
}

run_variant "xy05_w025_m4_d0040" 0.25 4.0 0.5 0.03 0.004
run_variant "xy05_w075_m4_d0040" 0.75 4.0 0.5 0.03 0.004
run_variant "xy10_w075_m5_d0040" 0.75 5.0 1.0 0.03 0.004

"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro05_contact_balance_20260523')
rows = []
for path in sorted((root / 'rp1m_sim').glob('*/impromptu_rp1m_retest_result.json')):
    row = json.loads(path.read_text())
    rows.append(row)
summary = {
    'root': str(root),
    'midi': (root / 'midi.txt').read_text().strip(),
    'baseline_weighted_maestro05': {
        'event_f1': 0.33146067415730335,
        'frame_f1': 0.2756543415039468,
        'matched': 59,
        'target': 175,
        'mispresses': 122,
        'played': 181,
    },
    'runs': rows,
}
if rows:
    best = max(rows, key=lambda r: (float(r.get('event_f1', 0.0)), float(r.get('frame_f1', 0.0))))
    summary['best_by_event_f1'] = best
    summary['mean_event_f1'] = float(np.mean([float(r.get('event_f1', 0.0)) for r in rows]))
    summary['mean_frame_f1'] = float(np.mean([float(r.get('frame_f1', 0.0)) for r in rows]))
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
