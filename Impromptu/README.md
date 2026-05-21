# Impromptu Precision-Safe Twinkle Retry

Plan:

```bash
python Impromptu/scripts/plan_impromptu.py \
  --midi-path <MIDI_OR_PROTO_INPUT> \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs \
  --run-name impromptu_twinkle_precision_safe_active10 \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --control-timestep 0.05 \
  --interpolation-substeps 10 \
  --active-window-last-s 10.0 \
  --active-window-preroll-s 0.5 \
  --active-window-postroll-s 0.25 \
  --key-press-depth 0.005 \
  --clearance-height 0.04 \
  --approach-s 0.055 \
  --hold-s 0.008 \
  --release-s 0.025 \
  --anchor-stride 1 \
  --solve-all-stride-anchors \
  --include-midpoint-anchors \
  --ik-fingertip-weight 0.9 \
  --ik-smoothness-weight 0.10 \
  --ik-neutral-weight 0.02 \
  --ik-max-nfev 80 \
  --residual-success-threshold 0.018
```

Render and score:

```bash
python Impromptu/scripts/render_dense_playback.py \
  --trajectory-npz <RUN_DIR>/trajectory.npz \
  --output-dir <RUN_DIR>/render_200fps_precision_safe \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
  --control-timestep 0.05 \
  --interpolation-substeps 10 \
  --fps 200 \
  --threshold 0.5
```
