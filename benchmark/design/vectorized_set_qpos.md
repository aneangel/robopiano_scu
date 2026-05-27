# Vectorized `_set_qpos` — Design Scoping

## 1. Current implementation

`Bagatelle/src/bagatelle/kinematics.py:264-269`:

```python
def _set_qpos(self, qpos: np.ndarray) -> None:
    values = self.clip_qpos(qpos)
    for joint, value in zip(self.joint_handles, values):
        self.physics.bind(joint).qpos = float(value)
    if hasattr(self.physics, "forward"):
        self.physics.forward()
```

`self.joint_handles` is built at `kinematics.py:250-255` via `_hand_joint_handles`:
joins the MJCF joint elements from `task.right_hand.joints` then `task.left_hand.joints`,
yielding `HAND_STATE_DIM = 46` (`kinematics.py:18`) `mjcf.Element` joint objects in the
order documented by `JOINT_ORDER = "right_hand_joints_then_left_hand_joints"`
(`kinematics.py:20`).

Range mapping (`JOINT_INDEX_RANGES_BY_HAND`, `kinematics.py:21-24`):
- right_hand → [0, 23]
- left_hand  → [23, 46]

These indices are positions in the 46-element pose vector, NOT MuJoCo qpos indices.

The same per-joint pattern appears in `current_qpos` (`kinematics.py:257-262`).
A precedent for the desired mapping already exists in
`Bagatelle/scripts/debug_verify_ik_teleport.py:103-120`, which uses
`model.name2id(name, "joint")` + `model.jnt_qposadr[joint_id]`.

## 2. dm_control physics interface

`self.physics` is a `dm_control.mjcf.Physics` instance returned by `_load_env` and
`_locate_task_physics_piano` (`kinematics.py:144-161`). The mjcf Physics subclasses
`mujoco.Physics`, so:

- `physics.data.qpos`    is a numpy array of shape `(nq,)`
- `physics.model.jnt_qposadr` maps joint id → first qpos slot
  (verified: `dm_control/suite/dog.py:191-193` uses exactly this idiom)
- `physics.model.name2id(name, "joint")` resolves joint name → joint id
  (`dm_control/mujoco/wrapper/core.py:339-356`)
- `physics.forward()` calls `mj_forward` and clears the mjcf `_dirty` flag
  (`dm_control/mjcf/physics.py:510-513`)
- `physics.bind(joint).qpos = v` ends up doing `array[index] = value` then
  `mark_as_dirty()` because `qpos` triggers dirty (`mjcf/physics.py:356-369`).

So per joint the current code pays for: dict lookup in `_bindings`, attribute
descriptor walk, named-indexer build (cached), SynchronizingArrayWrapper machinery,
plus `mark_as_dirty()`. Multiplied by 46 joints × ~230k FK calls = 10.5M binds.

The hand joints have 1 DoF each (hinge / slide), so `jnt_qposadr[i+1] - jnt_qposadr[i] == 1`
and our pose vector index maps 1:1 to a single qpos slot.

## 3. Cached index design

Compute the mapping once at the end of `__init__` (`kinematics.py:165-185`), after
`self.joint_handles` and `self.physics` are populated:

```python
# new helper, e.g. kinematics.py:~250
def _resolve_joint_qpos_indices(self) -> np.ndarray:
    model = self.physics.model
    ids = self.physics.bind(self.joint_handles).element_id   # int ndarray
    addrs = np.asarray(model.jnt_qposadr, dtype=np.int64)[np.asarray(ids, dtype=np.int64)]
    return addrs.astype(np.intp, copy=False)
```

Then in `__init__` (after the bounds block at `kinematics.py:169-174`):

```python
self._joint_qpos_indices = self._resolve_joint_qpos_indices()
assert self._joint_qpos_indices.shape == (HAND_STATE_DIM,), (
    f"joint qpos index map shape mismatch: {self._joint_qpos_indices.shape}"
)
# Every hand DoF must be a 1-slot joint (hinge/slide); verify uniqueness:
assert np.unique(self._joint_qpos_indices).size == HAND_STATE_DIM
```

Using `physics.bind(...).element_id` avoids manual name resolution and is the
documented dm_control idiom (`mjcf/physics.py:297-312`). It returns a numpy
array of MuJoCo joint ids when given a list, exactly what we want.

## 4. Correctness check

- `self.joint_handles` is a list of MJCF joint elements (one per hand DoF). The
  multi-element bind at `kinematics.py:169` already proves they bind cleanly
  to namespace `"joint"`.
- Length is asserted to be 46 (`kinematics.py:163-164`), matching `HAND_STATE_DIM`.
- The piano keys, sustain pedal, and any forearm free joints live at different
  qpos slots; because we only write `data.qpos[self._joint_qpos_indices]` (46 of
  them), the rest of `qpos` is untouched — strictly equivalent to the per-joint
  binding write today.
- Joint name uniqueness inside `joint_handles` is guaranteed by the right/left
  hand MJCF namespaces; the `np.unique` assertion above is a defensive check.

## 5. Patch sketch

```python
# kinematics.py — inside __init__, after `self._repair_bounds()`:
self._joint_qpos_indices = self._resolve_joint_qpos_indices()
if self._joint_qpos_indices.shape != (HAND_STATE_DIM,):
    raise RuntimeError(
        f"qpos index map shape {self._joint_qpos_indices.shape} != ({HAND_STATE_DIM},)"
    )
if np.unique(self._joint_qpos_indices).size != HAND_STATE_DIM:
    raise RuntimeError("hand joint qpos indices are not unique")

# new helper just below _hand_joint_handles (kinematics.py:255):
def _resolve_joint_qpos_indices(self) -> np.ndarray:
    ids = np.asarray(
        self.physics.bind(self.joint_handles).element_id, dtype=np.int64
    )
    addrs = np.asarray(self.physics.model.jnt_qposadr, dtype=np.int64)[ids]
    return addrs.astype(np.intp, copy=False)

# replacement for kinematics.py:264-269:
def _set_qpos(self, qpos: np.ndarray) -> None:
    values = self.clip_qpos(qpos)
    self.physics.data.qpos[self._joint_qpos_indices] = values.astype(np.float64, copy=False)
    self.physics.mark_as_dirty()  # keep binding cache invalidation honest
    self.physics.forward()        # always run mj_forward; clears _dirty

# (optional companion) replacement for current_qpos at kinematics.py:257-262:
def current_qpos(self) -> np.ndarray:
    return np.asarray(
        self.physics.data.qpos[self._joint_qpos_indices], dtype=np.float32
    ).copy()
```

Notes:
- `self.physics.data.qpos` is the live MuJoCo numpy buffer; assignment via
  fancy indexing writes in place.
- `mark_as_dirty()` is cheap (one bool flip) and keeps the dm_control invariant
  that any binding read after a raw qpos write sees a `forward()` invalidation
  — although we immediately call `forward()` ourselves, the explicit
  `mark_as_dirty()` makes the contract explicit and survives future refactors
  that might remove the `forward()` call (e.g. for batched FK over many seeds).

## 6. Risks

- **Callbacks / type conversion in `Binding.__setattr__`** (`mjcf/physics.py:356-369`):
  the only side effects are `array[index] = value`, `mark_as_dirty()`, and zeroing
  attributes in `disable_on_write`. For joint `qpos`, the constants table
  (`mjbindings/constants.py` MJMODEL_DISABLE_ON_WRITE) does not zero anything when
  writing qpos in practice (the per-joint loop has been functioning as-is). Calling
  `mark_as_dirty()` ourselves preserves the dirty semantics. No silent unit
  conversion happens — direct float copy.
- **Duplicate joint names in `JOINT_ORDER`**: not possible because each MJCF joint
  has a unique `full_identifier`; the `np.unique` assertion guarantees safety.
- **dm_control resetting `physics.data.qpos`**: `env.reset()` is only called once
  during `__init__`. `env.step(...)` in `activation_for_qpos` (`kinematics.py:315`)
  does mutate qpos via the engine — but it runs strictly after the new `_set_qpos`,
  so the order of operations is preserved.
- **`self.physics.data.qpos` writeability**: it's a mutable numpy view; the suite
  uses `self.data.qpos[idx]` writes in `dm_control/suite/*.py`, confirming the
  pattern is supported.
- **Float dtype**: MuJoCo's `qpos` is `float64`. Current code casts to `float`
  (Python double) per element; the vectorized version casts the array to
  `float64`, matching exactly.

## 7. Validation strategy

Add `benchmark/tests/test_vectorized_set_qpos.py` (~30 lines):

```python
import numpy as np
from bagatelle.kinematics import BagatelleKinematics, HAND_STATE_DIM


def _old_set_qpos(kin, q):
    q = kin.clip_qpos(q)
    for joint, value in zip(kin.joint_handles, q):
        kin.physics.bind(joint).qpos = float(value)
    kin.physics.forward()


def test_vectorized_matches_per_joint():
    kin = BagatelleKinematics()
    rng = np.random.default_rng(0)
    lo, hi = kin.joint_bounds
    for _ in range(8):
        q = rng.uniform(lo, hi).astype(np.float32)

        _old_set_qpos(kin, q)
        qpos_old = np.asarray(kin.physics.data.qpos).copy()
        xpos_old = np.asarray(kin.physics.data.xpos).copy()
        tips_old = kin.current_fingertips().copy()

        # perturb to force a re-set, then use the new path
        _old_set_qpos(kin, kin.neutral_qpos)
        kin._set_qpos(q)  # new vectorized impl
        qpos_new = np.asarray(kin.physics.data.qpos)
        xpos_new = np.asarray(kin.physics.data.xpos)
        tips_new = kin.current_fingertips()

        np.testing.assert_allclose(qpos_new, qpos_old, atol=0, rtol=0)
        np.testing.assert_allclose(xpos_new, xpos_old, atol=1e-12, rtol=0)
        np.testing.assert_allclose(tips_new, tips_old, atol=1e-7, rtol=0)
    kin.close()
```

Acceptance: bit-exact `qpos` equality (we write the same float64 values to the same
slots) and floating-tolerance equality on `xpos` / fingertip outputs of `mj_forward`.
Run before merging; pair with the existing benchmark harness
(`benchmark/run_ab_compare.py`) to confirm the expected ~20s wall-time win.
