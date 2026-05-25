#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Rhapsody" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402


def main() -> None:
    source = Path(sys.argv[1])
    checkpoint = sys.argv[2]
    with np.load(source, allow_pickle=False) as data:
        target_keys = np.asarray(data["target_keys"], dtype=np.float32)
        sparse_targets = np.asarray(data["fingertip_targets"], dtype=np.float32)
        baseline_qpos = np.asarray(data["waypoint_hand_joints"], dtype=np.float32)
        baseline_tips = np.asarray(data["waypoint_fingertips"], dtype=np.float32)
    cfg = BagatelleConfig(
        rhapsody_ik_enabled=True,
        rhapsody_ik_checkpoint=checkpoint,
        rhapsody_ik_refinement_steps=0,
        rhapsody_ik_coordinate_transform="bagatelle_to_rp1m",
        rhapsody_ik_fill_inactive_from_previous=True,
    )
    with BagatelleKinematics(config=cfg, target_keys=target_keys, output_dir=source.parent / "debug_rhapsody_rows") as kin:
        solver = kin._load_rhapsody_solver(cfg)
        normalizer = solver.normalizer
        tip_mean = normalizer.fingertip_mean.detach().cpu().numpy().reshape(10, 3)
        tip_std = normalizer.fingertip_std.detach().cpu().numpy().reshape(10, 3)
        q_mean = normalizer.qpos_mean.detach().cpu().numpy()
        print("normalizer tip mean min/max", np.round(tip_mean.min(axis=0), 5).tolist(), np.round(tip_mean.max(axis=0), 5).tolist())
        print("normalizer tip std min/max", np.round(tip_std.min(axis=0), 5).tolist(), np.round(tip_std.max(axis=0), 5).tolist())
        print("q mean min/max", float(q_mean.min()), float(q_mean.max()))
        previous = kin.neutral_qpos.astype(np.float32)
        for idx in range(min(12, sparse_targets.shape[0])):
            target = sparse_targets[idx]
            mask = np.isfinite(target).all(axis=1).astype(np.float32)
            active = mask > 0
            prev_tips = kin.fingertip_positions_for_qpos(previous)
            dense = target.copy()
            dense[~active] = prev_tips[~active]
            transformed, transformed_mask = kin._rhapsody_target_transform(dense, np.ones((10,), dtype=np.float32), cfg)
            zscore = np.abs((transformed - tip_mean) / tip_std)
            raw_solution = solver.solve(transformed, active_mask=transformed_mask, previous_qpos=previous, refinement_steps=0)
            raw_qpos = raw_solution.qpos
            clipped_qpos = kin.clip_qpos(raw_qpos)
            raw_tips = kin.fingertip_positions_for_qpos(raw_qpos)
            clipped_tips = kin.fingertip_positions_for_qpos(clipped_qpos)
            base_err = np.linalg.norm(baseline_tips[idx, active] - target[active], axis=1)
            raw_err = np.linalg.norm(raw_tips[active] - target[active], axis=1)
            clip_err = np.linalg.norm(clipped_tips[active] - target[active], axis=1)
            print(
                "row",
                idx,
                "active",
                np.flatnonzero(active).tolist(),
                "base_mean",
                float(np.mean(base_err)) if base_err.size else 0.0,
                "sur_mean",
                raw_solution.mean_error_m,
                "raw_mean",
                float(np.mean(raw_err)) if raw_err.size else 0.0,
                "clip_mean",
                float(np.mean(clip_err)) if clip_err.size else 0.0,
                "clipped",
                int(np.count_nonzero(np.abs(raw_qpos - clipped_qpos) > 1e-5)),
                "zmax",
                float(np.max(zscore)),
                "qrange",
                (float(raw_qpos.min()), float(raw_qpos.max())),
            )
            previous = clipped_qpos


if __name__ == "__main__":
    main()
