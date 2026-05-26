# Bottleneck Report

**Total wall time:** 131.675
**cProfile total accounted time:** 131.675 seconds

**Top hotspots:**
- `Impromptu/scripts/plan_trajectory.py:1(<module>)` cum=131.681s tt=0.000s (100.0% of total)
- `Impromptu/scripts/plan_trajectory.py:214(main)` cum=130.716s tt=0.000s (99.3% of total)
- `/Users/aangeles/robopiano/Impromptu/src/impromptu/planner.py:360(plan_target_keys)` cum=130.581s tt=0.001s (99.2% of total)
- `/Users/aangeles/robopiano/Bagatelle/src/bagatelle/planner.py:474(plan_target_keys)` cum=126.622s tt=0.003s (96.2% of total)
- `/Users/aangeles/robopiano/Bagatelle/src/bagatelle/kinematics.py:585(solve_press_pose)` cum=126.517s tt=0.006s (96.1% of total)
- `/Users/aangeles/robopiano/Bagatelle/src/bagatelle/kinematics.py:673(solve_from_seed)` cum=121.915s tt=0.006s (92.6% of total)
- `/Users/aangeles/robopiano/Bagatelle/src/bagatelle/kinematics.py:641(residual)` cum=117.640s tt=1.721s (89.3% of total)
- `/Users/aangeles/robopiano/Bagatelle/src/bagatelle/kinematics.py:283(fingertip_positions_for_qpos)` cum=117.393s tt=0.128s (89.2% of total)
- `/Users/aangeles/robopiano/Bagatelle/src/bagatelle/kinematics.py:264(_set_qpos)` cum=116.218s tt=3.409s (88.3% of total)
- `/Users/aangeles/robopiano/Bagatelle/src/bagatelle/kinematics.py:108(__init__)` cum=3.593s tt=0.000s (2.7% of total)

**By stage:**
- IK solver (least_squares): 183.2% of time (241.218s)
- MuJoCo FK (physics.forward): 67.6% of time (89.049s)
- Hungarian assignment: 0.0% of time (0.002s)
- Contact validation: 4.8% of time (6.376s)
- Other: 0.0% of time (0.000s)

**Verdict:** Largest accounted stage is **IK solver (least_squares)** at 183.2% of cProfile-measured time.
