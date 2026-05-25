#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J barcarnd
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/barcarolle_random_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/barcarolle_random_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/barcarolle_random_gate_20260524
mkdir -p "$ROOT"

RUN=$("$PY" - <<'PY'
import random
runs = [f"maestro_{i:02d}" for i in range(1, 11)]
print(random.choice(runs))
PY
)
echo "$RUN" > "$ROOT/selected_run.txt"
echo "Selected random run: $RUN"

LEGATO=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_legato_caprice_20260524/$RUN/trajectory.npz
SOURCE="$LEGATO"
if [ "$RUN" = "maestro_01" ] && [ -f /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro01_20260524/$RUN/trajectory.npz ]; then
  SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro01_20260524/$RUN/trajectory.npz
fi
if [ "$RUN" = "maestro_07" ] && [ -f /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro07_20260524/$RUN/trajectory.npz ]; then
  SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro07_20260524/$RUN/trajectory.npz
fi

declare -a CANDS=()
add_candidate() {
  local name="$1"
  local path="$2"
  if [ -f "$path" ]; then
    CANDS+=(--candidate-npz "$name=$path")
  fi
}

add_candidate legato "$LEGATO"
add_candidate caprice /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_caprice_contact_20260524/$RUN/trajectory.npz
add_candidate tremolo /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_finger_20260524/$RUN/trajectory.npz
add_candidate gigue /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_gigue_chunk_20260524/$RUN/trajectory.npz
add_candidate fandango /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_fandango_hybrid_20260524/$RUN/hybrid/trajectory.npz
add_candidate siciliana /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_siciliana_zero_20260524/$RUN/trajectory.npz
add_candidate cadenza_open /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_cadenza_20260524/$RUN/open/trajectory.npz
add_candidate cadenza_strict /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_cadenza_20260524/$RUN/strict/trajectory.npz
add_candidate rondo01 /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro01_20260524/$RUN/trajectory.npz
add_candidate rondo07 /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro07_20260524/$RUN/trajectory.npz

"$PY" barcarolle_event_window_ensemble_gate.py \
  --source-npz "$SOURCE" \
  "${CANDS[@]}" \
  --output-dir "$ROOT/$RUN" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 2 \
  --post-dense-frames 10 \
  --event-types mispress,missed \
  --scopes target_hand,full \
  --max-events 80 \
  --max-total-evals 180 \
  --allow-equal-f1-mispress-drop \
  --keep-if-matched-increases \
  --matched-gain-max-f1-drop 0.002

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/barcarolle_random_gate_20260524")
run = (root / "selected_run.txt").read_text().strip()
base_path = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_adaptive_legato_caprice_rondo_20260524/summary.json")
base = json.loads(base_path.read_text())
base_row = {row["run"]: row for row in base["runs"]}[run]
row = json.loads((root / run / "summary.json").read_text())
final = row["final_score"]
out = {
    "root": str(root),
    "run": run,
    "base_summary": str(base_path),
    "base_event_f1": float(base_row["event_f1"]),
    "base_frame_f1": float(base_row["frame_f1"]),
    "base_matched": int(base_row["matched"]),
    "base_target": int(base_row["target"]),
    "base_played": int(base_row["played"]),
    "base_mispresses": int(base_row["mispresses"]),
    "event_f1": float(final["event_f1"]),
    "frame_f1": float(final["frame_f1"]),
    "matched": int(final["matched"]),
    "target": int(final["target"]),
    "played": int(final["played"]),
    "mispresses": int(final["mispresses"]),
    "accepted_count": int(row.get("accepted_count", 0)),
    "eval_count": int(row.get("eval_count", 0)),
}
out["delta_event_f1"] = float(out["event_f1"] - out["base_event_f1"])
out["delta_mispresses"] = int(out["mispresses"] - out["base_mispresses"])
(root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
