#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J fannlow
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/fannlow_%A_%a.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/fannlow_%A_%a.err
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --array=1-2

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
SIC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_siciliana_zero_20260524
CAD=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_cadenza_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/fandango_neutral_gate_20260524

if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  run=maestro_01
else
  run=maestro_08
fi
echo "=== neutral-base Fandango $run ==="

"$PY" fandango_sim_greedy_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --candidate-npz "$SIC/$run/trajectory.npz" \
  --candidate-summary "$SIC/$run/summary.json" \
  --output-dir "$ROOT/$run/siciliana_greedy" \
  --threshold 0.5 \
  --base-mode neutral \
  --min-delta-event-f1 0.000001 \
  --keep-if-matched-increases

"$PY" fandango_sim_greedy_gate.py \
  --source-npz "$ROOT/$run/siciliana_greedy/trajectory.npz" \
  --candidate-npz "$CAD/$run/open/trajectory.npz" \
  --candidate-summary "$CAD/$run/open/summary.json" \
  --output-dir "$ROOT/$run/hybrid" \
  --threshold 0.5 \
  --base-mode source \
  --min-delta-event-f1 0.000001 \
  --keep-if-matched-increases
