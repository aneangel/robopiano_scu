# Analytical Jacobian for the Bagatelle IK Residual

## 1. Summary

`scipy.optimize.least_squares` currently estimates the Jacobian of the IK
residual via forward finite differences inside `_numdiff.approx_derivative`,
perturbing each of the 46 reduced-hand joints in turn and re-running the
residual closure. Each residual call invokes `physics.forward()` through
`_set_qpos` at `Bagatelle/src/bagatelle/kinematics.py:264-269` plus a fingertip
read in `current_fingertips` at `kinematics.py:277-281`. cProfile shows that
~70% of the 223k `physics.forward()` calls (90.8 s of 131 s wall) come from
this perturbation loop. Replacing it with an analytical Jacobian computed from
MuJoCo's `mj_jacSite` removes 46 of the 47 FK calls per LM iteration and is
projected to save 50-70 s on the dense smoke test.

## 2. The Residual, term by term

The residual closure lives at `kinematics.py:676-706`. With the dense smoke
test config (`ik_inactive_fingertip_clearance_weight = 0.0`,
`ik_unassigned_fingertip_strategy = "legacy"`,
`ik_wrong_key_xy_avoidance_weight = 0.0`), only three terms are active:

| Term | Source line | Expression | Length |
|------|-------------|------------|--------|
| (a) Fingertip | `kinematics.py:679-681` | `(fingertips[finger_indices] - target_positions) * fingertip_axis_weights[None, :] * ik_fingertip_weight` | `3 * n_active` |
| (b) Smoothness | `kinematics.py:682` | `(q - previous) * ik_smoothness_weight` | 46 |
| (c) Neutral | `kinematics.py:683` | `(q - neutral) * ik_neutral_weight` | 46 |
| (d) Inactive Z clearance | `kinematics.py:685-687` | `max(clearance_z - fingertips[inactive,2], 0) * inactive_clearance_weight` | `n_inactive` (often 0) |
| (e) Avoid mispress | `kinematics.py:688-700` | inactive-vs-wrong-key distance; only if `unassigned_strategy=="avoid_mispresses"` | variable |
| (f) Wrong-key XY | `kinematics.py:701-705` | active fingertip vs wrong-key XY clearance; only if `wrong_key_weight>0` | `n_active * n_wrong` |

For the smoke benchmark we implement (a)-(c), gate the analytical path on
`(d)`, and fall back to finite differences when `(e)` or `(f)` is active.

### 2.1 Analytical gradients

Let `q in R^46`, `J_i in R^(3,46)` be the translational Jacobian of fingertip
site `i`'s world position w.r.t. the 46 hand DOFs (the rows of `mj_jacSite` for
site `i`, restricted to the columns of our 46 joints' `dofadr`). Let `W` be
the diagonal of `fingertip_axis_weights` (3-vector) scaled by
`ik_fingertip_weight`. Let `e_z` be the row picker for the Z-component.

- (a) Fingertip term: for active finger `f_k` (row `k` in `finger_indices`),
  `d(residual_a[3k:3k+3])/dq = diag(W) @ J_{f_k}`. Stacked over active fingers,
  this is `(3*n_active, 46)`.
- (b) Smoothness: `d/dq = ik_smoothness_weight * I_46`.
- (c) Neutral: `d/dq = ik_neutral_weight * I_46`.
- (d) Inactive Z clearance: for inactive finger `i`, with `z_i = fingertips[i,2]`:
  - if `z_i >= clearance_z`: row is 0 (max() inactive).
  - else: row is `-inactive_clearance_weight * J_i[2, :]` (chain rule on
    `clearance_z - z_i`).
  Stacked over inactive fingers.

`mj_jacSite` accounts for the rigid site offset (`_FINGERTIP_OFFSET = 0.026`,
`_THUMBTIP_OFFSET = 0.0275` at
`robopianist/models/hands/shadow_hand.py:81-82`) automatically, so no
additional offset bookkeeping is needed.

## 3. MuJoCo Jacobian API

Fingertips are MuJoCo **sites**, not bodies. `BagatelleKinematics` reads them
at `kinematics.py:165` (`self.task.left_hand.fingertip_sites + ...`) and
position via `physics.bind(self.fingertip_sites).xpos` at
`kinematics.py:278`. The sites are constructed at
`robopianist/models/hands/shadow_hand.py:190-207` as fixed-offset spheres
under each fingertip body, so `mj_jacSite` returns the exact Jacobian of the
positions the residual consumes.

The right call (matching `dm_control/utils/inverse_kinematics.py:179-180`) is:

```python
mjlib.mj_jacSite(physics.model.ptr, physics.data.ptr, jac_pos, jac_rot, site_id)
```

with `jac_pos` and `jac_rot` shaped `(3, nv)`, dtype `float64`, C-contiguous.
We need only translation so `jac_rot` can be a scratch buffer reused across
calls. `site_id` is obtained once at construction via
`physics.bind(site).element_id` or `physics.model.name2id(name, "site")`.

## 4. nq vs nv

All 46 reduced-hand joints are 1-DOF: 4 slide joints (forearm tx/ty per hand,
declared `joint_type="slide"` at `robopianist/models/hands/shadow_hand.py:42-54`)
and 42 hinge joints. Therefore `qposadr[joint] == dofadr[joint]` for every
joint in `self.joint_handles`, and our 46-element ordering maps directly into
both `data.qpos` and the columns of `mj_jac*`. No ball/free joints exist in
the hand model. We precompute `self._hand_dofadr = np.array([
    model.jnt_dofadr[model.name2id(name, "joint")] for name in joint_names ])`
once in `__init__` (parallel to the existing pattern at
`Bagatelle/scripts/debug_verify_ik_teleport.py:103-119`).

## 5. Patch sketch

### 5.1 New helper on `BagatelleKinematics`

```python
# __init__ additions (around kinematics.py:165, after fingertip_sites built):
self._fingertip_site_ids = np.array(
    [int(self.physics.bind(site).element_id) for site in self.fingertip_sites],
    dtype=np.int32,
)
self._hand_dofadr = np.array(
    [int(self.physics.model.jnt_dofadr[
        self.physics.model.name2id(_model_element_name(j), "joint")])
     for j in self.joint_handles],
    dtype=np.int64,
)
# Scratch buffers (avoid allocation in hot loop).
nv = int(self.physics.model.nv)
self._jac_pos_buf = np.zeros((3, nv), dtype=np.float64)
self._jac_rot_buf = np.zeros((3, nv), dtype=np.float64)

def fingertip_jacobian(self, finger_indices: np.ndarray) -> np.ndarray:
    """Return d(fingertips[finger_indices, :]) / dq with shape (n, 3, 46).

    Caller is responsible for having called self._set_qpos(q) (which runs
    physics.forward) before this method.
    """
    from dm_control.mujoco.wrapper.mjbindings import mjlib
    model_ptr = self.physics.model.ptr
    data_ptr = self.physics.data.ptr
    out = np.empty((len(finger_indices), 3, HAND_STATE_DIM), dtype=np.float64)
    for k, finger in enumerate(finger_indices):
        site_id = int(self._fingertip_site_ids[int(finger)])
        mjlib.mj_jacSite(model_ptr, data_ptr,
                         self._jac_pos_buf, self._jac_rot_buf, site_id)
        out[k] = self._jac_pos_buf[:, self._hand_dofadr]
    return out
```

### 5.2 Jacobian closure inside `solve_press_pose`

Adjacent to `residual` at `kinematics.py:676`:

```python
analytical_jac_supported = (
    inactive_clearance_weight == 0.0 or inactive_indices.size > 0  # see (d) below
) and unassigned_strategy != "avoid_mispresses" and wrong_key_weight == 0.0

def jacobian(values: np.ndarray) -> np.ndarray:
    q = self.clip_qpos(values)
    # State sync: residual() may not have been called immediately before
    # jac() (LM in scipy calls fun then jac, but we cannot rely on the
    # data buffer not being mutated by another consumer of self.physics).
    # _set_qpos calls physics.forward(), so mj_jacSite sees the right state.
    self._set_qpos(q)
    fingertips = self.current_fingertips()  # cheap read, no extra forward

    # (a) Fingertip term: (3*n_active, 46)
    jac_finger = self.fingertip_jacobian(finger_indices)  # (n_active, 3, 46)
    jac_a = (jac_finger * fingertip_axis_weights[None, :, None]).reshape(-1, HAND_STATE_DIM)
    jac_a *= float(cfg.ik_fingertip_weight)

    # (b) Smoothness identity
    jac_b = np.eye(HAND_STATE_DIM, dtype=np.float64) * float(cfg.ik_smoothness_weight)

    # (c) Neutral identity
    jac_c = np.eye(HAND_STATE_DIM, dtype=np.float64) * float(cfg.ik_neutral_weight)

    parts = [jac_a, jac_b, jac_c]

    # (d) Inactive Z clearance (piecewise)
    if inactive_clearance_weight > 0.0 and inactive_indices.size:
        jac_inactive = self.fingertip_jacobian(inactive_indices)   # (n_inactive, 3, 46)
        inactive_z = fingertips[inactive_indices, 2]
        active_mask = inactive_z < clearance_z                      # subgradient at boundary: 0
        rows = np.zeros((inactive_indices.size, HAND_STATE_DIM), dtype=np.float64)
        rows[active_mask] = -jac_inactive[active_mask, 2, :] * inactive_clearance_weight
        parts.append(rows)

    return np.concatenate(parts, axis=0)
```

### 5.3 Plugging into `least_squares`

Modify `solve_from_seed` at `kinematics.py:708-717`:

```python
opt_kwargs = dict(
    bounds=(self.joint_lower.astype(np.float64), self.joint_upper.astype(np.float64)),
    max_nfev=max(int(cfg.ik_max_nfev), 1),
    ftol=float(cfg.ik_ftol), xtol=float(cfg.ik_xtol), gtol=float(cfg.ik_gtol),
)
if analytical_jac_supported and bool(getattr(cfg, "ik_analytical_jacobian", True)):
    opt_kwargs["jac"] = jacobian
opt = least_squares(residual, self.clip_qpos(seed).astype(np.float64), **opt_kwargs)
```

Add an opt-out flag `ik_analytical_jacobian: bool = True` to `BagatelleConfig`
so the finite-difference path remains reachable for unit-test parity.

## 6. Validation strategy

A unit test at `Bagatelle/tests/test_analytical_jacobian.py` should cover:

1. **Numerical consistency at a fixed point.** Build a `BagatelleKinematics`,
   pick 5 random `q` from `[joint_lower, joint_upper]`, build the residual and
   jacobian closures using a synthetic `FingerAssignmentResult` (e.g. 4 active
   fingers on 4 piano keys). Assert
   `np.allclose(jacobian(q), approx_derivative(residual, q, method="3-point"),
   rtol=1e-3, atol=1e-5)`. Use `scipy.optimize._numdiff.approx_derivative`
   directly.
2. **End-to-end IK parity.** For the dense smoke MIDI fixture, run
   `solve_press_pose` with `ik_analytical_jacobian=False` and `=True`. Compare
   converged `pose` element-wise with `atol=1e-3` and require
   `abs(residual_norm_fd - residual_norm_an) <= 1e-4`.
3. **Performance sanity.** Same fixture, measure wall time of the analytical
   path; assert at least 30% speedup over finite differences (lower bound to
   absorb noise; the design target is ~50%).
4. **Fallback.** With `unassigned_strategy="avoid_mispresses"` or
   `ik_wrong_key_xy_avoidance_weight>0`, assert `least_squares` is called
   without a `jac` kwarg (mock and inspect). Optional but cheap.

## 7. Risks and mitigations

- **State drift.** `mj_jacSite` reads from `data` in its current state. `scipy`
  LM calls `fun(x)` then `jac(x)` with the same `x`, but we cannot assume the
  MuJoCo data buffer was not modified by another consumer between calls.
  Mitigation: `jacobian()` calls `self._set_qpos(q)` (which runs
  `physics.forward()`) at the top. This adds 1 FK per LM iteration but
  guarantees correctness, and 1 FK vs 46 FK is still a 45x reduction.
- **Cache opportunity.** If we observe that `residual(q)` was the last call
  before `jacobian(q)` (track `self._last_residual_q`), we can skip the
  `_set_qpos` re-run. Defer; the 1-FK overhead is not on the critical path
  for the projected savings.
- **Site offset correctness.** Fingertips are sphere sites at fixed local
  offset from each fingertip body. `mj_jacSite` already returns the Jacobian
  of the site's world position including the rotated offset. We verified the
  declaration at `shadow_hand.py:198-205`. No manual offset chain rule is
  required.
- **Joint vs DOF index drift.** All 46 reduced joints are 1-DOF (slide or
  hinge), so `jnt_qposadr == jnt_dofadr` per joint. We cache `_hand_dofadr`
  once in `__init__` rather than recomputing per call.
- **Non-smoothness from `clip_qpos`.** Both the residual (`kinematics.py:677`)
  and the proposed jacobian clip to joint bounds. At the boundary the
  analytical Jacobian assumes the unclipped derivative; this matches scipy's
  finite-difference behavior because LM uses `bounds=...` internally and does
  not perturb across bounds via `clip_qpos` either.
- **Optional terms (e)(f).** Their analytical gradients are mechanical
  extensions (chain rule of `norm(diff)` with a piecewise indicator) but they
  are out of scope. Until they are added, `analytical_jac_supported` gates the
  path so behavior is unchanged when those terms are enabled.

## 8. Estimated impact

cProfile baseline: 90.8 s of 131 s in `physics.forward`, ~70% of those from
the 46-perturbation loop. Replacing 46 FK calls with 1 FK + 10 `mj_jacSite`
calls per LM iteration (cheaper than FK; they reuse the already-forwarded
state) projects to:

- Saved FK calls: ~145k of 223k.
- Saved wall time: 50-70 s on the dense smoke test, on top of the
  keyset-cache and warm-start gains. Combined with prior optimizations, the
  benchmark should drop below 60 s total.
