#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J LocalGPUF1
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoLocalGpuSample10F1/logs/sample10_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoLocalGpuSample10F1/logs/sample10_%j.err
#SBATCH -t 0-08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano

source /WAVE/apps/el8/conda/envs/Python/20240305/etc/profile.d/conda.sh
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT=/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoLocalGpuSample10F1
PLAN_ROOT="$OUT/plans"
RETEST_ROOT="$OUT/rp1m_retest"
mkdir -p "$OUT/logs" "$PLAN_ROOT" "$RETEST_ROOT"

python Maestroso/scripts/local_gpu_plan_benchmark.py \
  --songs 10 \
  --output-root "$PLAN_ROOT" \
  --maestro-root /WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0 \
  --checkpoint /WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/random10_heldout2_prev_random_scale6_refine16_20260522/rhapsody_rpik.pt \
  --key-targets /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoLocalArtifacts/key_press_targets_006.npy \
  --sample-manifest /WAVE/datasets/ccoelho_lab-jlanders/MaestrosoLocalArtifacts/local_sample10_manifest.json \
  --device cuda \
  --batch-size 512 \
  --passes 2 \
  --refinement-steps 0

python retest_impromptu_rp1m_simulator.py \
  --run-root "$PLAN_ROOT" \
  --output-root "$RETEST_ROOT" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --set-hand-qvel

python - <<'PY'
import json
from pathlib import Path

out = Path("/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoLocalGpuSample10F1")
plan_summary = json.loads((out / "plans" / "summary.json").read_text())
rows = []
for result in sorted((out / "rp1m_retest").glob("*/impromptu_rp1m_retest_result.json")):
    row = json.loads(result.read_text())
    rows.append(row)
event_f1 = [float(row.get("event_f1") or 0.0) for row in rows]
summary = {
    "plan_summary": plan_summary,
    "retest_count": len(rows),
    "mean_event_f1": sum(event_f1) / len(event_f1) if event_f1 else 0.0,
    "min_event_f1": min(event_f1) if event_f1 else 0.0,
    "max_event_f1": max(event_f1) if event_f1 else 0.0,
    "rows": rows,
    "ready_for_maestroso_threshold": bool(event_f1 and (sum(event_f1) / len(event_f1)) > 0.6),
}
(out / "sample10_f1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps({k: summary[k] for k in ("retest_count", "mean_event_f1", "min_event_f1", "max_event_f1", "ready_for_maestroso_threshold")}, indent=2, sort_keys=True))
PY
