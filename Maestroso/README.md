# Maestroso

Accelerated full-song trajectory generation experiments for MAESTRO MIDI files.

The earlier Maestroso runs used Impromptu's `--active-window-last-s 12.0`
benchmark setting. That was useful for validating the key-press F1 mechanism,
but it does not generate full-song trajectories. This module targets full-song
generation by replacing the always-on repair cascade with faster variants:

- fast full-window Impromptu planning for a CPU baseline
- direct GPU Rhapsody batch proposals followed by exact CPU MuJoCo rollout
  verification
- GPU-backed Rhapsody IK seeding inside Impromptu planning
- GPU-backed Rhapsody IK with a small refinement budget
- chunked GPU-backed Rhapsody planning, which can later be distributed across
  many CPU/GPU workers and stitched into a full trajectory

For production full-song planning, Maestroso treats
`joint_space_straighten` as the default trajectory mode. The older
`dense_fingertip_ik` mode is intentionally not used for full songs because it is
too expensive at MAESTRO scale. It is available only as the explicit
`hard_window_dense_ik` sweep variant, and that variant clamps its input to a
short hard window, defaulting to 15 seconds.

Chunk-level parallelism uses process workers, not threads. Bagatelle kinematics
and MuJoCo mutate physics state during FK/rollout calls, so each worker process
must own its own simulator state.

Production planning does not run online RL, per-song policy updates, or contact
CEM inside the planner. Contact CEM is treated as an offline repair/training
tool only: planner run, collect bad windows, train or update a policy offline,
freeze a checkpoint, then rerun the planner with the frozen checkpoint.

The `gpu_batch_proposal` variant uses the GPU only as a proposal engine. It
batches sparse keyset-change fingertip targets through
`RhapsodyIKSolver.solve_batch` on CUDA, builds a `joint_space_straighten`
trajectory from those proposed hand states, and verifies each sparse proposal
with Bagatelle's CPU MuJoCo FK-backed IK before writing the rollout. It then
runs the same RP1M/MuJoCo CPU retest used by the other variants. The model
inputs are MIDI target key activations converted to key press target locations;
the simulator is only the exact verifier for the generated hand-state rollout.
The companion `gpu_batch_proposal_bagatelle_assign` variant keeps the same
batched proposal engine but lets the CPU verification pass recompute each
waypoint assignment from Bagatelle's previous-pose assignment rule.

The initial Slurm job is intentionally a short sweep. GPU nodes can have queue
waits, so the sweep runs several variants in one allocation and writes timing,
trajectory, and rollout metrics into one output tree.

## Accuracy Recovery Sweep

Speed is secondary in recovery mode. The recovery sweep disables successive
halving, runs short 15-30 second windows, validates every active waypoint with
CPU MuJoCo activation, and runs the full RP1M retest for every variant.

```bash
sbatch Maestroso/slurm/run_accuracy_recovery_sweep.slurm
```

Recovery variants:

- `impromptu_accuracy_baseline`
- `maestroso_bagatelle_assign_cpu_verify`
- `maestroso_bagatelle_assign_gpu_seed_cpu_verify`
- `maestroso_bagatelle_assign_gpu_seed_multistart`
- `maestroso_bagatelle_assign_gpu_seed_topk`
- `maestroso_hard_window_dense_repair`

The baseline calls `Impromptu/scripts/plan_trajectory.py` with
`joint_space_straighten`, Bagatelle legacy previous-pose assignment,
`key_press_depth=0.006`, `wrong_hand_penalty=4.0`, split key 48, dynamic hand
split, `avoid_mispresses`, `anchor_stride=2`, `ik_max_nfev=80`, and static
contact validation. It does not use Rhapsody or chunking.

Rhapsody recovery variants use the GPU only as a seed proposal. Final waypoint
poses must come from exact CPU MuJoCo IK and all-waypoint activation
validation; raw GPU proposals are never accepted directly.

Recovery outputs include:

- `all_waypoint_activation_summary.json`
- `all_waypoint_activation/per_waypoint_activation.json`
- `variant_result.json`
- `rp1m_retest/<run_id>/impromptu_rp1m_retest_result.json`
- `offline_training_failures.jsonl`

Required recovery metadata fields are written into each `variant_result.json`:
`dropped_key_count`, `all_waypoint_activation_f1`,
`all_waypoint_activation_tp`, `all_waypoint_activation_fp`,
`all_waypoint_activation_fn`, `repair_tier_counts`,
`accepted_raw_gpu_proposals`, `accepted_cpu_verified_waypoints`,
`failed_waypoints`, `chunk_overlap_seconds`, and `stitch_blend_enabled`.

`build_bagatelle_waypoint_targets()` in
`Maestroso/scripts/run_accuracy_recovery_sweep.py` is the production proposal
target builder. It runs Bagatelle's previous-pose assignment sequentially and
hard-fails if any key is dropped. The older split-even target builder remains
only as a debugging path in the fast proposal scripts.

The sweep uses staged successive halving:

1. Stage A runs planning only on a short cheap proxy window with a large anchor
   stride. It does not call MuJoCo activation checks or rollout. It ranks by
   planning metadata: anchor count, IK success count/rate, residual p95, nfev
   sum/mean/p95, and unassigned keys.
2. Stage B runs sparse MuJoCo activation checks only for waypoint/high-risk
   frames. It does not render video or perform dense rollout.
3. Stage C runs the full dense RP1M rollout for only the top configs, without
   video rendering.
4. Stage D renders video for final configs only.

Poor proxy windows are written to `offline_training_failures.jsonl` for offline
policy training or explicit repair jobs. The normal sweep does not run CEM.

## Short GPU Sweep

```bash
sbatch Maestroso/slurm/run_gpu_acceleration_sweep.slurm
```

The job uses the WAVE `gpu` partition:

```bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
```

Outputs go under:

```bash
/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch
```

Each variant writes:

- `trajectory.npz`
- `metadata.json`
- `variant_result.json`
- `rp1m_retest/<run_id>/impromptu_rp1m_retest_result.json`

The sweep summary is:

```bash
/WAVE/datasets/ccoelho_lab-jlanders/MaestrosoAcceleratedBatch/summary.json
```

## Full-Song Direction

For full MAESTRO generation, use the chunked Rhapsody path after validating it
on short windows. The intended production shape is:

1. Convert the full MIDI to `target_keys`.
2. Split into phrase or fixed-length chunks with at least 1.0 second overlap.
3. Plan chunks independently with GPU Rhapsody IK using
   `joint_space_straighten`.
4. Stitch chunk trajectories with smoothstep overlap blending, deduplicate
   overlap waypoints, and recompute velocities after stitching.
5. Run CPU-parallel MuJoCo rollout scoring.
6. Repair only chunks below the target F1 threshold.

Use `dense_fingertip_ik` only for localized windows that remain hard after the
production pass, not as a full-song planner.
