# Dense Sweep Workflow

From `/WAVE/projects/ECEN-524-Wi26/robopiano` on WAVE:

```bash
conda activate sonata

python Impromptu/scripts/submit_dense_sweep.py \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/sweeps \
  --partition cmp \
  --random-count 48 \
  --submit
```

This generates:

- `configs/*.json`: one config per end-to-end run
- `launch_dense_sweep_array.slurm`: Slurm job array launcher
- `submission_summary.json`: manifest plus `sbatch` output

Each array task runs:

1. Impromptu planning with the active filtered song window
2. Dense playback rendering
3. Score export
4. Root-level artifact collation

Each run directory contains:

- `config.json`
- `metrics.json`
- `render_summary.json`
- `score.json`
- `lag_sweep.json`
- `threshold_sweep.json`
- `fp_by_key.csv`
- `fn_by_key.csv`

Aggregate after jobs finish:

```bash
python Impromptu/scripts/aggregate_dense_sweep_results.py \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/sweeps
```

This writes:

- `sweep_results.csv`
- `top10_summary.json`
- `architectural_recommendation.json`

Use `partition cpu` instead of `cmp` if the node class is better suited for planning/eval and GPU is not required.

## Adaptive Stage

Build a second-stage adaptive batch from completed stage-1 results without submitting:

```bash
python Impromptu/scripts/aggregate_dense_sweep_results.py \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/sweeps

python Impromptu/scripts/submit_adaptive_dense_sweep.py \
  --base-root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs/sweeps \
  --stage-name stage2_adaptive \
  --proposals 24
```

This writes:

- `stage2_adaptive/configs/*.json`
- `stage2_adaptive/adaptive_manifest.json`
- `stage2_adaptive/launch_dense_sweep_array.slurm`
- `stage2_adaptive/submission_summary.json`

The adaptive generator does not submit unless `--submit` is provided.
