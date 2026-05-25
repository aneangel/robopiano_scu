#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J best30
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/best30_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/best30_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=1-30%5

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random30_best_adaptive_20260524
MANIFEST="$ROOT/manifest.json"
IDX="$SLURM_ARRAY_TASK_ID"
RUN=$(printf "maestro30_%02d" "$IDX")
RUN_DIR="$ROOT/$RUN"
mkdir -p "$RUN_DIR"

MIDI=$("$PY" - "$MANIFEST" "$IDX" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
index = int(sys.argv[2]) - 1
print(manifest["songs"][index]["midi_path"])
PY
)

echo "=== Best adaptive Impromptu 30-song validation $RUN ==="
echo "midi=$MIDI"

if [ ! -f "$RUN_DIR/base/trajectory.npz" ]; then
  "$PY" Impromptu/scripts/plan_trajectory.py \
    --midi-path "$MIDI" \
    --output-root "$RUN_DIR" \
    --run-name base \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --active-window-last-s 12.0 \
    --active-window-preroll-s 1.0 \
    --active-window-postroll-s 0.25 \
    --assignment-strategy ik_aware_topk \
    --assignment-top-k 8 \
    --ik-static-contact-validation \
    --ik-static-contact-missed-key-weight 4.0 \
    --ik-static-contact-wrong-key-weight 0.25 \
    --ik-multistart-seed-count 4 \
    --key-press-depth 0.006 \
    --wrong-hand-penalty 4.0 \
    --wrong-hand-split-key 48 \
    --assignment-dynamic-hand-split \
    --assignment-dynamic-hand-split-min-span 12 \
    --assignment-dynamic-hand-split-min-keys 3 \
    --finger-crossing-penalty 1.0 \
    --same-key-same-finger-bonus 0.25 \
    --ik-unassigned-fingertip-strategy avoid_mispresses \
    --ik-unassigned-fingertip-avoidance-weight 0.5 \
    --ik-unassigned-fingertip-avoidance-radius 0.03
fi

if [ ! -f "$RUN_DIR/etude_open/trajectory.npz" ]; then
  "$PY" etude_contact_cem_gate.py \
    --source-npz "$RUN_DIR/base/trajectory.npz" \
    --output-dir "$RUN_DIR/etude_open" \
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
fi

if [ ! -f "$RUN_DIR/cadenza/strict/trajectory.npz" ]; then
  "$PY" cadenza_event_injection_gate.py \
    --source-npz "$RUN_DIR/etude_open/trajectory.npz" \
    --output-dir "$RUN_DIR/cadenza/strict" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pulse-width 2 \
    --pulse-gap 0 \
    --max-static-fp 1 \
    --min-static-f1 0.40 \
    --only-static-missed
fi

if [ ! -f "$RUN_DIR/cadenza/open/trajectory.npz" ]; then
  "$PY" cadenza_event_injection_gate.py \
    --source-npz "$RUN_DIR/etude_open/trajectory.npz" \
    --output-dir "$RUN_DIR/cadenza/open" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pulse-width 2 \
    --pulse-gap 0 \
    --max-static-fp 2 \
    --min-static-f1 0.25 \
    --only-static-missed
fi

if [ ! -f "$RUN_DIR/siciliana/trajectory.npz" ]; then
  "$PY" siciliana_grouped_injection_gate.py \
    --source-npz "$RUN_DIR/etude_open/trajectory.npz" \
    --output-dir "$RUN_DIR/siciliana" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pulse-width 1 \
    --pulse-gap 0 \
    --max-static-fp 0 \
    --max-candidate-played-count 0 \
    --max-pulses-per-onset 2 \
    --played-weight 0.07
fi

if [ ! -f "$RUN_DIR/fandango/siciliana_greedy/trajectory.npz" ]; then
  "$PY" fandango_sim_greedy_gate.py \
    --source-npz "$RUN_DIR/etude_open/trajectory.npz" \
    --candidate-npz "$RUN_DIR/siciliana/trajectory.npz" \
    --candidate-summary "$RUN_DIR/siciliana/summary.json" \
    --output-dir "$RUN_DIR/fandango/siciliana_greedy" \
    --threshold 0.5 \
    --min-delta-event-f1 0.000001 \
    --keep-if-matched-increases
fi

if [ ! -f "$RUN_DIR/fandango/hybrid/trajectory.npz" ]; then
  "$PY" fandango_sim_greedy_gate.py \
    --source-npz "$RUN_DIR/fandango/siciliana_greedy/trajectory.npz" \
    --candidate-npz "$RUN_DIR/cadenza/open/trajectory.npz" \
    --candidate-summary "$RUN_DIR/cadenza/open/summary.json" \
    --output-dir "$RUN_DIR/fandango/hybrid" \
    --threshold 0.5 \
    --min-delta-event-f1 0.000001 \
    --keep-if-matched-increases
fi

if [ ! -f "$RUN_DIR/fandango_neutral/siciliana_greedy/trajectory.npz" ]; then
  "$PY" fandango_sim_greedy_gate.py \
    --source-npz "$RUN_DIR/etude_open/trajectory.npz" \
    --candidate-npz "$RUN_DIR/siciliana/trajectory.npz" \
    --candidate-summary "$RUN_DIR/siciliana/summary.json" \
    --output-dir "$RUN_DIR/fandango_neutral/siciliana_greedy" \
    --threshold 0.5 \
    --base-mode neutral \
    --min-delta-event-f1 0.000001 \
    --keep-if-matched-increases
fi

if [ ! -f "$RUN_DIR/fandango_neutral/hybrid/trajectory.npz" ]; then
  "$PY" fandango_sim_greedy_gate.py \
    --source-npz "$RUN_DIR/fandango_neutral/siciliana_greedy/trajectory.npz" \
    --candidate-npz "$RUN_DIR/cadenza/open/trajectory.npz" \
    --candidate-summary "$RUN_DIR/cadenza/open/summary.json" \
    --output-dir "$RUN_DIR/fandango_neutral/hybrid" \
    --threshold 0.5 \
    --base-mode source \
    --min-delta-event-f1 0.000001 \
    --keep-if-matched-increases
fi

if [ ! -f "$RUN_DIR/gigue/trajectory.npz" ]; then
  "$PY" gigue_sim_chunk_scrub_gate.py \
    --source-npz "$RUN_DIR/fandango/hybrid/trajectory.npz" \
    --output-dir "$RUN_DIR/gigue" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --chunk-control-frames 5
fi

if [ ! -f "$RUN_DIR/tremolo/trajectory.npz" ]; then
  "$PY" tremolo_nearest_finger_scrub_gate.py \
    --source-npz "$RUN_DIR/gigue/trajectory.npz" \
    --output-dir "$RUN_DIR/tremolo" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pre-dense-frames 1 \
    --post-dense-frames 8 \
    --top-fingers 2 \
    --modes neutral,hold_before,zero \
    --max-events 60 \
    --max-total-evals 180 \
    --allow-equal-f1-mispress-drop
fi

if [ ! -f "$RUN_DIR/caprice/trajectory.npz" ]; then
  "$PY" caprice_contact_scrub_gate.py \
    --source-npz "$RUN_DIR/tremolo/trajectory.npz" \
    --output-dir "$RUN_DIR/caprice" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pre-dense-frames 1 \
    --post-dense-frames 8 \
    --top-fingers 2 \
    --modes neutral,hold_before,zero \
    --samples-per-finger 18 \
    --top-candidates 4 \
    --finger-sigma 0.035 \
    --forearm-sigma 0.016 \
    --static-fp-weight 0.30 \
    --static-fn-weight 0.18 \
    --max-events 45 \
    --max-total-evals 160 \
    --allow-equal-f1-mispress-drop \
    --seed 524
fi

if [ ! -f "$RUN_DIR/legato_caprice/trajectory.npz" ]; then
  "$PY" legato_duplicate_hold_gate.py \
    --source-npz "$RUN_DIR/caprice/trajectory.npz" \
    --output-dir "$RUN_DIR/legato_caprice" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pre-dense-frames 10 \
    --post-dense-frames 8 \
    --top-fingers 2 \
    --modes onset,onset_plus,pre_50ms,pre_100ms,mid_interval \
    --max-events 80 \
    --max-total-evals 180 \
    --allow-equal-f1-mispress-drop
fi

if [ ! -f "$RUN_DIR/rondo/trajectory.npz" ]; then
  "$PY" rondo_missed_event_inject_gate.py \
    --source-npz "$RUN_DIR/fandango_neutral/hybrid/trajectory.npz" \
    --candidate-npz "$RUN_DIR/tremolo/trajectory.npz" \
    --output-dir "$RUN_DIR/rondo" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pre-dense-frames 1 \
    --post-dense-frames 10 \
    --top-fingers 2 \
    --scopes finger,hand,full \
    --max-events 80 \
    --max-total-evals 180 \
    --keep-if-matched-increases \
    --matched-gain-max-f1-drop 0.002
fi

"$PY" - "$RUN_DIR" "$MIDI" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
midi = sys.argv[2]

def final_score(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "final_score" in data:
        return data["final_score"]
    return data

legato = final_score(run_dir / "legato_caprice" / "summary.json")
rondo = final_score(run_dir / "rondo" / "summary.json")
choice = "rondo" if float(rondo["event_f1"]) > float(legato["event_f1"]) else "legato_caprice"
score = rondo if choice == "rondo" else legato
summary = {
    "run": run_dir.name,
    "midi_path": midi,
    "choice": choice,
    "event_f1": float(score["event_f1"]),
    "frame_f1": float(score["frame_f1"]),
    "matched": int(score["matched"]),
    "target": int(score["target"]),
    "played": int(score["played"]),
    "mispresses": int(score["mispresses"]),
    "legato_caprice": {k: legato[k] for k in ("event_f1", "frame_f1", "matched", "target", "played", "mispresses")},
    "rondo": {k: rondo[k] for k in ("event_f1", "frame_f1", "matched", "target", "played", "mispresses")},
}
(run_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
