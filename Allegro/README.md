# Allegro

Allegro is a 200 Hz online residual alignment module for Fugue planner-following
RoboPianist rollouts. Fugue still owns the 20 Hz planner-conditioned action prior;
Allegro runs ten 5 ms residual-control substeps inside each Fugue action interval.

The design is deliberately hybrid:

- **Residual control:** execute `u = u_fugue + delta_u`, where Fugue provides the
  low-frequency action prior and Allegro learns the residual correction.
- **Feedback-error learning:** use the closed-loop PD feedback residual as the
  supervised online target for the residual model, so the learner distills the
  correction that kept the hand closer to the planner.
- **DAgger-style distribution matching:** train from states induced by the live
  closed-loop rollout instead of only from recorded RP1M states.
- **Safety envelope:** clip, smooth, and rate-limit residuals before they reach
  the RoboPianist action space.

The approach is motivated by established literature: DAgger for online imitation
under induced state distributions (Ross, Gordon, and Bagnell, AISTATS 2011),
feedback-error learning for using feedback motor commands as teaching signals
(Kawato, Furukawa, and Suzuki, Biological Cybernetics 1987; Kawato 1990), and
residual policy learning where a learned residual augments a conventional
controller (Johannink et al., ICRA 2019).

## Install

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
pip install -e Fugue
pip install -e Allegro
pytest Allegro/tests
```

## Run A 200 Hz Fugue Rollout

Online rollouts use `rp1m_simulator` and should be launched from a Slurm
allocation, not directly on the login node. Use `cmp` by default; request a GPU
node only when the Fugue checkpoint inference is too slow on CPU or when you are
intentionally benchmarking CUDA inference.

```bash
tmux new -s allegro_rollout

srun -p cmp \
  --cpus-per-task=16 \
  --mem=128G \
  --time=0-08:00:00 \
  --pty bash

cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export FUGUE_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Fugue/<existing_fugue_run>
export RUN_NAME=allegro_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Allegro/$RUN_NAME
mkdir -p "$OUTPUT_ROOT"

python Allegro/scripts/rollout_fugue_200hz.py \
  --checkpoint "$FUGUE_OUTPUT_ROOT/model_f_planner_next/checkpoints/best.pt" \
  --rp1m-root "$RP1M_300_ROOT" \
  --dataset-artifact-root "$FUGUE_OUTPUT_ROOT/model_f_planner_next/dataset" \
  --split test \
  --demo-index 0 \
  --output-dir "$OUTPUT_ROOT/allegro_200hz_rollout" \
  --device cpu
```

For GPU inference, follow the `HowToRun.md` GPU allocation pattern and pass
`--device cuda` explicitly.

Useful tuning flags:

- `--kp`, `--kd`: feedback residual gains in normalized action coordinates.
- `--learned-residual-scale`: how much of the online residual model to add to
  the current feedback command.
- `--residual-clip-norm`, `--residual-clip-per-dim`: residual safety limits.
- `--update-every-substeps`: online update cadence; default `5`, so updates at
  40 Hz after warmup.
- `--disable-online-learning`: keep the high-frequency feedback residual but
  disable model updates for ablations.

Outputs:

- `allegro_200hz_summary.json`
- `allegro_200hz_rollout.npz`
- optional `allegro_200hz_rollout.mp4`

The NPZ stores source-rate and dense-rate actions, residuals, feedback labels,
learned residual predictions, simulated hand states, played keys, goals, and
reference arrays for scoring.

## Best RP1M Hand-State Reproduction Preset

For reproducing a full RP1M demonstration when the recorded RP1M action prior is
available, the best current CPU-tested preset uses reference actions plus a small
200 Hz Allegro PD correction. The target phase power moves the dense target a bit
ahead of linear interpolation without jumping all the way to the interval
endpoint.

```bash
python Allegro/scripts/rollout_fugue_200hz.py \
  --checkpoint "$CHECKPOINT" \
  --rp1m-root "$RP1M_300_ROOT" \
  --dataset-artifact-root "$DATASET_ARTIFACT_ROOT" \
  --split test \
  --demo-id 1 \
  --output-dir "$OUTPUT_ROOT/best_r06_clip2_full/allegro_200hz_rollout" \
  --device cpu \
  --base-action-source reference \
  --alignment-target-power 0.85 \
  --action-gain 0.88 \
  --max-action-delta 1.0 \
  --kp 0.060 \
  --kd 0.0024 \
  --feedback-residual-scale 1.0 \
  --learned-residual-scale 0.0 \
  --residual-clip-norm 0.12 \
  --residual-clip-per-dim 0.06 \
  --residual-smoothing-alpha 0.05 \
  --max-action-delta-per-substep 2.0 \
  --disable-online-learning
```

The optional `--use-shadow-action-jacobian` flag enables a literal MJCF actuator
to qpos feedback projection, but it is disabled by default because the RP1M
reduced action arrays currently track better with the empirical reduced-action
ordering used by the simulator adapter.

## References

- Ross, S., Gordon, G., and Bagnell, D. "A Reduction of Imitation Learning and
  Structured Prediction to No-Regret Online Learning." AISTATS 2011.
- Kawato, M., Furukawa, K., and Suzuki, R. "A hierarchical neural-network model
  for control and learning of voluntary movement." Biological Cybernetics, 1987.
- Kawato, M. "Feedback-Error-Learning Neural Network for Supervised Motor
  Learning." Advanced Neural Computers, 1990.
- Johannink, T. et al. "Residual Reinforcement Learning for Robot Control."
  ICRA 2019.
