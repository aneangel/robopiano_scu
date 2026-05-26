# Ballade

Ballade is a top-level module for a 200 Hz online action controller for the
RP1M/RoboPianist simulator. It is intended to consume hand/key trajectory plans
from Impromptu, interpolate the 20 Hz hand states into 200 Hz soft targets, and
choose simulator actions from online feedback.

The central rule is that interpolated hand states are references, not simulator
commands. Ballade may load the initial hand state at reset, and it may restore a
full simulator snapshot when evaluating candidate actions from the same state.
The deployed rollout path must move through `env.step(action)`, not by writing
hand qpos at every microstep.

## Why Ballade Does Not Imitate RP1M Actions First

Current online replay of RP1M recorded actions is unreliable in RoboPianist.
Ballade therefore treats recorded actions as diagnostics or optional action
priors. The main teacher is the simulator-observed action selected by the
online tracker and optional local search.

## 20 Hz To 200 Hz Targets

RP1M and Impromptu source frames arrive at 20 Hz:

```text
source_dt = 0.05
control_dt = 0.005
substeps = 10
```

`ballade.interpolation.build_micro_q_targets` expands consecutive source
states into `[source_steps - 1, substeps, q_dim]`. Ballade uses endpoint
inclusive phases:

```text
1 / substeps, 2 / substeps, ..., 1.0
```

This means the last microstep in an interval lands exactly on the next 20 Hz
waypoint. These micro targets are soft references used by costs, features, and
controllers.

## Run Types

Run A uses `OnlineJacobianTracker` and optional event-triggered local search.
It is the first milestone because it tests whether online simulator feedback can
produce useful 200 Hz actions from interpolated hand/key references.

Run B collects Run A selected actions as teacher data and trains
`ResidualMLPController` to predict a small residual over the tracker/base
action.

Run C repeats collection and training with search fallback on policy failures,
so the residual controller learns recovery behavior while search usage is
reduced over iterations.

## WAVE Runtime

Follow the repository `HowToRun.md` first. The expected setup is:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export BALLADE_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Ballade/$(date +%Y%m%d_%H%M%S)
mkdir -p "$BALLADE_OUTPUT_ROOT"
```

Large outputs, teacher shards, checkpoints, rollout media, and cached data
belong under `/WAVE/datasets/ccoelho_lab-jlanders`, not in the repo or home
directory.

## Smoke Tests

```bash
python Ballade/scripts/smoke_imports.py
pytest Ballade/tests -q
```

## Run A

```bash
python Ballade/scripts/rollout_online_jacobian.py \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$BALLADE_OUTPUT_ROOT/run_a_online_jacobian" \
  --source-dt 0.05 \
  --control-dt 0.005 \
  --max-demos 3 \
  --max-source-steps 300 \
  --no-render
```

## Run B

```bash
python Ballade/scripts/collect_online_teacher_data.py \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$BALLADE_OUTPUT_ROOT/run_b_teacher_data" \
  --max-demos 20 \
  --max-source-steps 500 \
  --search-mode event_triggered

python Ballade/scripts/train_residual_controller.py \
  --teacher-data "$BALLADE_OUTPUT_ROOT/run_b_teacher_data" \
  --output-root "$BALLADE_OUTPUT_ROOT/run_b_residual_model" \
  --epochs 50 \
  --batch-size 2048

python Ballade/scripts/rollout_residual_controller.py \
  --checkpoint "$BALLADE_OUTPUT_ROOT/run_b_residual_model/checkpoints/best.pt" \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$BALLADE_OUTPUT_ROOT/run_b_rollout_eval" \
  --search-fallback none \
  --max-demos 5
```

## Run C

```bash
python Ballade/scripts/run_ballade_eval.py \
  --config Ballade/configs/ballade_run_c_dagger_search.yaml \
  --rp1m-root "$RP1M_300_ROOT" \
  --output-root "$BALLADE_OUTPUT_ROOT/run_c_dagger" \
  --iterations 3
```

## Success Metrics

Ballade should be evaluated by simulator behavior:

- goal key precision, recall, and F1
- mispress rate and note event count
- hand qpos L2 at 20 Hz endpoints and over dense 200 Hz targets
- fingertip error near press frames when available
- action saturation fraction and action delta p95
- search usage fraction
- closed-loop termination or failure status

The minimum useful milestone is nonzero goal key F1, nonzero audio/key events,
lower hand drift than the Model F online baseline, and no immediate closed-loop
collapse.
