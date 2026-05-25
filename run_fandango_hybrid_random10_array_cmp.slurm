#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J fanhy10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/fanhy10_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/fanhy10_%A_%a.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --array=1-10

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
SIC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_siciliana_zero_20260524
CAD=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_cadenza_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_fandango_hybrid_20260524
mkdir -p "$ROOT"

idx=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
run=maestro_${idx}
echo "=== Fandango hybrid $run ==="

"$PY" fandango_sim_greedy_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --candidate-npz "$SIC/$run/trajectory.npz" \
  --candidate-summary "$SIC/$run/summary.json" \
  --output-dir "$ROOT/$run/siciliana_greedy" \
  --threshold 0.5 \
  --min-delta-event-f1 0.000001 \
  --keep-if-matched-increases

"$PY" fandango_sim_greedy_gate.py \
  --source-npz "$ROOT/$run/siciliana_greedy/trajectory.npz" \
  --candidate-npz "$CAD/$run/open/trajectory.npz" \
  --candidate-summary "$CAD/$run/open/summary.json" \
  --output-dir "$ROOT/$run/hybrid" \
  --threshold 0.5 \
  --min-delta-event-f1 0.000001 \
  --keep-if-matched-increases
