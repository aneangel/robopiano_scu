#!/usr/bin/env bash
#SBATCH --job-name=etude-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/etude/20260519_tracking_dataset_4a884b7_r2/logs/probe_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/etude/20260519_tracking_dataset_4a884b7_r2/logs/probe_%j.err

set -euo pipefail

cd /WAVE/projects/ECEN-524-Wi26/robopiano/Etude
source ~/.bashrc
conda activate sonata

export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"

export ETUDE_SOURCE_DATA=/WAVE/datasets/ccoelho_lab-jlanders/etude/20260519_tracking_dataset_4a884b7_r2
export ETUDE_VIEW_ROOT=$ETUDE_SOURCE_DATA/views
export ETUDE_RUN_ROOT=$ETUDE_SOURCE_DATA/runs_probe
export ETUDE_EVAL_ROOT=$ETUDE_SOURCE_DATA/eval_probe
export ETUDE_LOG_ROOT=$ETUDE_SOURCE_DATA/logs

mkdir -p "$ETUDE_VIEW_ROOT" "$ETUDE_RUN_ROOT" "$ETUDE_EVAL_ROOT" "$ETUDE_LOG_ROOT"

echo "[1] Build probe dataset views"
python - <<'PY'
from pathlib import Path
import pandas as pd

source = Path("/WAVE/datasets/ccoelho_lab-jlanders/etude/20260519_tracking_dataset_4a884b7_r2")
views = source / "views"
views.mkdir(parents=True, exist_ok=True)

manifest_path = source / "manifest.csv"
if manifest_path.exists():
    df = pd.read_csv(manifest_path)
    df["path"] = df["path"].apply(
        lambda p: str((source / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
else:
    episodes = sorted((source / "episodes").glob("*.npz"))
    if not episodes:
        raise SystemExit(f"No manifest.csv and no episodes/*.npz found under {source}")
    df = pd.DataFrame({
        "episode_id": range(len(episodes)),
        "path": [str(p.resolve()) for p in episodes],
        "source": ["recovered_from_episode_dir"] * len(episodes),
        "timesteps": [-1] * len(episodes),
    })

df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

def write_view(name, start, n):
    out = views / name
    out.mkdir(parents=True, exist_ok=True)
    sub = df.iloc[start:start+n].copy()
    if sub.empty:
        raise SystemExit(f"Cannot create empty view {name}; source has only {len(df)} episodes.")
    sub.to_csv(out / "manifest.csv", index=False)
    (out / "normalization.json").write_text("{}\n")
    print(f"Wrote {len(sub)} episodes -> {out}")

write_view("train_probe_24", 0, 24)
write_view("eval_probe_6", 24, 6)
write_view("tiny_debug_4", 0, 4)
PY

echo "[2] Dry-run configs"
for cfg in configs/experiments_probe/probe_*.yaml; do
  echo "Dry run: $cfg"
  python scripts/run_experiment.py \
    --config "$cfg" \
    --output-root "$ETUDE_RUN_ROOT" \
    --dry-run
done

echo "[3] Train probe models"
python scripts/run_experiment.py \
  --config configs/experiments_probe/probe_bc_mlp.yaml \
  --output-root "$ETUDE_RUN_ROOT"

python scripts/run_experiment.py \
  --config configs/experiments_probe/probe_temporal_gru.yaml \
  --output-root "$ETUDE_RUN_ROOT"

python scripts/run_experiment.py \
  --config configs/experiments_probe/probe_inverse_dynamics.yaml \
  --output-root "$ETUDE_RUN_ROOT"

echo "[4] Load eval trajectories"
mapfile -t TRAJS < <(python - <<'PY'
import pandas as pd
from pathlib import Path

view = Path("/WAVE/datasets/ccoelho_lab-jlanders/etude/20260519_tracking_dataset_4a884b7_r2/views/eval_probe_6")
df = pd.read_csv(view / "manifest.csv")
for p in df["path"]:
    print(p)
PY
)

echo "[5] Evaluate PD baseline"
for traj in "${TRAJS[@]}"; do
  name=$(basename "$traj" .npz)
  python scripts/evaluate_tracker.py \
    --config configs/experiments_probe/probe_pd_eval.yaml \
    --trajectory "$traj" \
    --output-root "$ETUDE_EVAL_ROOT/pd/$name"
done

echo "[6] Evaluate BC MLP"
BC_CKPT="$ETUDE_RUN_ROOT/probe_bc_mlp/checkpoints/best.pt"
for traj in "${TRAJS[@]}"; do
  name=$(basename "$traj" .npz)
  python scripts/evaluate_tracker.py \
    --config configs/experiments_probe/probe_bc_mlp.yaml \
    --checkpoint "$BC_CKPT" \
    --trajectory "$traj" \
    --output-root "$ETUDE_EVAL_ROOT/bc_mlp/$name"
done

echo "[7] Evaluate Temporal GRU"
GRU_CKPT="$ETUDE_RUN_ROOT/probe_temporal_gru/checkpoints/best.pt"
for traj in "${TRAJS[@]}"; do
  name=$(basename "$traj" .npz)
  python scripts/evaluate_tracker.py \
    --config configs/experiments_probe/probe_temporal_gru.yaml \
    --checkpoint "$GRU_CKPT" \
    --trajectory "$traj" \
    --output-root "$ETUDE_EVAL_ROOT/temporal_gru/$name"
done

echo "[8] Evaluate Inverse Dynamics"
ID_CKPT="$ETUDE_RUN_ROOT/probe_inverse_dynamics/checkpoints/best.pt"
for traj in "${TRAJS[@]}"; do
  name=$(basename "$traj" .npz)
  python scripts/evaluate_tracker.py \
    --config configs/experiments_probe/probe_inverse_dynamics.yaml \
    --checkpoint "$ID_CKPT" \
    --trajectory "$traj" \
    --output-root "$ETUDE_EVAL_ROOT/inverse_dynamics/$name"
done

echo "[9] Summarize"
python scripts/summarize_results.py \
  --root "$ETUDE_EVAL_ROOT" \
  --out "$ETUDE_EVAL_ROOT/summary.csv"

echo "Done."
echo "Summary: $ETUDE_EVAL_ROOT/summary.csv"
