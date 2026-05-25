#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J cap10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap10_%A_%a.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
TREM=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_finger_20260524
FANDANGO_NEUTRAL=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/fandango_neutral_gate_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_caprice_contact_20260524
mkdir -p "$ROOT"

run=$(printf "maestro_%02d" "${SLURM_ARRAY_TASK_ID}")
if [[ "$run" == "maestro_07" ]]; then
  source_npz="$FANDANGO_NEUTRAL/$run/hybrid/trajectory.npz"
else
  source_npz="$TREM/$run/trajectory.npz"
fi

echo "=== Caprice contact-aware 10-song $run ==="
echo "source_npz=$source_npz"

"$PY" caprice_contact_scrub_gate.py \
  --source-npz "$source_npz" \
  --output-dir "$ROOT/$run" \
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
