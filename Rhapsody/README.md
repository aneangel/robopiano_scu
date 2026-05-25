# Rhapsody

Rhapsody is a reinforcement-learning inverse-kinematics module for RoboPianist.
It learns to map desired fingertip positions to a 46-dimensional reduced hand
state using RP1M examples where both fingertip positions and hand state are
known.

The module is intentionally separate from `Nocturne`. Nocturne remains a
comparison planner; Rhapsody is an IK policy that can be used by trajectory
planners such as Impromptu.

## Approach

Rhapsody trains two models:

1. A forward-kinematics surrogate from RP1M hand states to fingertip positions.
2. A residual IK policy from target fingertip positions, active-finger mask, and
   previous hand state to the next hand state.

The IK policy is optimized with reward-weighted policy gradients. The reward is
based on active fingertip error under the learned FK surrogate, max fingertip
error, smoothness from the previous hand state, and an optional RP1M imitation
term. This gives the policy a direct objective matching online rollout: put the
active fingertips at the requested coordinates while avoiding erratic hand-state
changes.

## Smoke Training

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python Rhapsody/scripts/train_rpik.py \
  --rp1m-root /WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr \
  --max-songs 1 \
  --num-demos 1 \
  --frame-stride 20 \
  --max-pairs 256 \
  --previous-mode random \
  --fk-epochs 2 \
  --bc-epochs 2 \
  --policy-epochs 2 \
  --output-dir /WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/smoke
```

The saved checkpoint can be loaded through `rhapsody.solver.RhapsodyIKSolver`.
