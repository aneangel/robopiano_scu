#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J bestsel30
#SBATCH -a 1-30%5
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/bestsel30_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/bestsel30_%A_%a.err
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random30_best_adaptive_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/impromptu_best_bidirectional_barcarolle_rhapsody_30_20260524
RHAPSODY=/WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/random10_heldout2_prev_random_scale6_refine16_20260522/rhapsody_rpik.pt

idx=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
run=maestro30_${idx}
row="$ROOT/$run"
mkdir -p "$row"

base_summary="$SRC_ROOT/$run/best_summary.json"
rondo_npz="$SRC_ROOT/$run/rondo/trajectory.npz"
legato_npz="$SRC_ROOT/$run/legato_caprice/trajectory.npz"

test -f "$base_summary"
test -f "$rondo_npz"
test -f "$legato_npz"

if [ ! -f "$row/forward/summary.json" ]; then
  "$PY" barcarolle_event_window_ensemble_gate.py \
    --source-npz "$rondo_npz" \
    --candidate-npz legato="$legato_npz" \
    --output-dir "$row/forward" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --event-types missed \
    --scopes target_hand,full \
    --pre-dense-frames 2 \
    --post-dense-frames 10 \
    --max-events 120 \
    --max-total-evals 240 \
    --keep-if-matched-increases \
    --matched-gain-max-f1-drop 0.002 \
    --allow-equal-f1-mispress-drop
fi

if [ ! -f "$row/reverse/summary.json" ]; then
  "$PY" barcarolle_event_window_ensemble_gate.py \
    --source-npz "$legato_npz" \
    --candidate-npz rondo="$rondo_npz" \
    --output-dir "$row/reverse" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --event-types mispress \
    --scopes target_hand,full \
    --pre-dense-frames 2 \
    --post-dense-frames 10 \
    --max-events 120 \
    --max-total-evals 240 \
    --keep-if-matched-increases \
    --matched-gain-max-f1-drop 0.002 \
    --allow-equal-f1-mispress-drop
fi

run_rhapsody=$("$PY" - "$base_summary" <<'PY'
import json
import sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
target = int(d.get("target", 0))
f1 = float(d.get("event_f1", 0.0))
print("yes" if target <= 40 and f1 < 0.70 else "no")
PY
)

if [ "$run_rhapsody" = "yes" ] && [ ! -f "$row/rhapsody/summary.json" ]; then
  "$PY" waltz_contact_mpc_gate.py \
    --source-npz "$rondo_npz" \
    --output-dir "$row/rhapsody" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --finger-mode all \
    --scopes finger,hand,full \
    --receding \
    --max-rounds 8 \
    --rhapsody-checkpoint "$RHAPSODY" \
    --rhapsody-refinement-steps 16 \
    --pre-dense-frames 1 \
    --post-dense-frames 8 \
    --max-events 40 \
    --max-candidates-per-event 80 \
    --max-total-evals 360 \
    --keep-if-matched-increases \
    --matched-gain-max-f1-drop 0.003 \
    --allow-equal-f1-mispress-drop
fi

"$PY" - "$run" "$base_summary" "$row" <<'PY'
import json
import sys
from pathlib import Path

run = sys.argv[1]
base_path = Path(sys.argv[2])
row = Path(sys.argv[3])

base = json.loads(base_path.read_text())
choices = [
    {
        "name": f"adaptive_{base.get('choice', 'unknown')}",
        "event_f1": float(base["event_f1"]),
        "frame_f1": float(base.get("frame_f1", 0.0)),
        "matched": int(base["matched"]),
        "target": int(base["target"]),
        "played": int(base["played"]),
        "mispresses": int(base["mispresses"]),
        "summary_path": str(base_path),
    }
]

for name in ("forward", "reverse", "rhapsody"):
    p = row / name / "summary.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    score = data.get("final_score", data)
    choices.append(
        {
            "name": name,
            "event_f1": float(score["event_f1"]),
            "frame_f1": float(score.get("frame_f1", 0.0)),
            "matched": int(score["matched"]),
            "target": int(score["target"]),
            "played": int(score["played"]),
            "mispresses": int(score["mispresses"]),
            "accepted_count": int(data.get("accepted_count", 0)),
            "eval_count": int(data.get("eval_count", 0)),
            "summary_path": str(p),
        }
    )

best = max(choices, key=lambda x: (x["event_f1"], x["matched"], -x["mispresses"], x["frame_f1"]))
out = {
    "run": run,
    "selected": best["name"],
    "event_f1": best["event_f1"],
    "frame_f1": best["frame_f1"],
    "matched": best["matched"],
    "target": best["target"],
    "played": best["played"],
    "mispresses": best["mispresses"],
    "choices": choices,
}
(row / "selector_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
