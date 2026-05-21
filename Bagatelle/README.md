# Bagatelle: OT IK Press-Pose Planner

Bagatelle builds deterministic RoboPianist hand-pose targets from `target_keys[T, 88]`.
It avoids the Variations shortcut of learning one pose directly from a keyset. Instead it:

1. assigns fingers to active keys with RP1M-style optimal transport,
2. solves a bounded MuJoCo forward-kinematics IK problem for the reduced 46-D hand qpos,
3. interpolates between solved press waypoints and evaluates the result with RoboPianist.

Finger assignment order is left fingertips `0..4`, then right fingertips `5..9`, matching the
OT fingering reward implementation in `robopianist/suite/tasks/piano_with_shadow_hands.py`.
The emitted 46-D joint state keeps the existing RoboPianist/Intermezzo order: right hand joints,
then left hand joints.

## Commands

From the repo root on WAVE:

```bash
conda activate sonata

python Bagatelle/scripts/plan_trajectory.py \
  --target-keys-npz /path/to/target_keys.npz \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/runs

python Bagatelle/scripts/evaluate_bagatelle.py \
  --trajectory-npz /WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/runs/<run>/trajectory.npz
```

For MIDI input:

```bash
python Bagatelle/scripts/plan_trajectory.py \
  --midi-path /path/to/piece.mid \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/runs
```

## Assignment Strategies

Bagatelle supports configurable fingertip assignment strategies:

- `legacy_previous_pose`: original nearest previous fingertip Hungarian assignment.
- `composite_cost`: Hungarian assignment with configurable anatomical and musical cost terms.
- `ik_aware_topk`: generates multiple assignment candidates and selects using IK residual, motion, crossing, and assignment cost.

Default remains `legacy_previous_pose` for reproducibility.

```bash
python Bagatelle/scripts/plan_trajectory.py \
  --midi-path /path/to/piece.mid \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/runs \
  --assignment-strategy composite_cost

python Bagatelle/scripts/plan_trajectory.py \
  --midi-path /path/to/piece.mid \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/runs \
  --assignment-strategy ik_aware_topk \
  --assignment-top-k 8
```

## Smoke Tests

```bash
pytest Bagatelle/tests
python Bagatelle/scripts/plan_trajectory.py --help
```

Slow simulator tests are skipped unless explicitly enabled:

```bash
BAGATELLE_RUN_SLOW=1 pytest Bagatelle/tests/test_kinematics_slow.py
```
