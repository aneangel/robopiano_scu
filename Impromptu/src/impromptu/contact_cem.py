from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from impromptu.config import ImpromptuConfig
from impromptu.joint_space_trajectory import ALL_FINGER_JOINT_INDICES
from impromptu.paths import ensure_repo_paths

ensure_repo_paths()
from bagatelle.kinematics import HAND_STATE_DIM, BagatelleKinematics  # noqa: E402
from intermezzo.constants import LEFT_FOREARM_TY_INDEX, RIGHT_FOREARM_TY_INDEX  # noqa: E402


FOREARM_CONTACT_SEARCH_INDICES = np.asarray([21, 22, 44, 45], dtype=np.int64)


@dataclass(frozen=True)
class ContactCEMResult:
    control_qpos: np.ndarray
    dense_qpos: np.ndarray
    metadata: dict[str, object]


def mask_key(row: np.ndarray, threshold: float) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).tolist())


def contact_score(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, played).sum())
    fp = int(np.logical_and(~goal, played).sum())
    fn = int(np.logical_and(goal, ~played).sum())
    denom = 2 * tp + fp + fn
    return (float(2 * tp / denom) if denom else 1.0, tp, fp, fn)


def rank_score(score: tuple[float, int, int, int], *, fp_weight: float) -> float:
    f1, tp, fp, fn = score
    return float(f1) + 0.05 * float(tp) - float(fp_weight) * float(fp) - 0.01 * float(fn)


def perturb_qpos(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    finger_sigma: float,
    forearm_sigma: float,
    include_forearm: bool,
    lift_forearm_ty: bool,
) -> np.ndarray:
    cand = np.asarray(base, dtype=np.float32).copy()
    cand[ALL_FINGER_JOINT_INDICES] += rng.normal(
        0.0,
        float(finger_sigma),
        size=ALL_FINGER_JOINT_INDICES.size,
    ).astype(np.float32)
    if include_forearm:
        cand[FOREARM_CONTACT_SEARCH_INDICES] += rng.normal(
            0.0,
            float(forearm_sigma),
            size=FOREARM_CONTACT_SEARCH_INDICES.size,
        ).astype(np.float32)
    if lift_forearm_ty:
        lift = abs(float(forearm_sigma))
        cand[RIGHT_FOREARM_TY_INDEX] += np.float32(rng.normal(0.0, lift))
        cand[LEFT_FOREARM_TY_INDEX] += np.float32(rng.normal(0.0, lift))
    return cand


def dense_from_control(control_qpos: np.ndarray, *, substeps: int) -> np.ndarray:
    return np.repeat(np.asarray(control_qpos, dtype=np.float32), max(int(substeps), 1), axis=0).astype(np.float32)


def refine_contact_cem(
    *,
    kin: BagatelleKinematics,
    target_keys: np.ndarray,
    control_qpos: np.ndarray,
    substeps: int,
    config: ImpromptuConfig,
) -> ContactCEMResult:
    control = np.asarray(control_qpos, dtype=np.float32).copy()
    if control.ndim != 2 or control.shape[1] != HAND_STATE_DIM:
        raise ValueError(f"control_qpos must have shape [T, {HAND_STATE_DIM}], got {control.shape}")
    keys = np.asarray(target_keys, dtype=np.float32)[:, :88]
    if keys.shape[0] != control.shape[0]:
        raise ValueError(f"target_keys/control_qpos length mismatch: {keys.shape[0]} vs {control.shape[0]}")

    if not bool(config.enable_contact_cem):
        return ContactCEMResult(
            control_qpos=control,
            dense_qpos=dense_from_control(control, substeps=substeps),
            metadata={"enabled": False, "changed_control_frames": 0, "used_masks": 0},
        )

    rng = np.random.default_rng(int(config.contact_cem_seed))
    masks: dict[tuple[int, ...], list[int]] = {}
    for frame, row in enumerate(keys):
        key = mask_key(row, float(config.threshold))
        if key:
            masks.setdefault(key, []).append(int(frame))

    selected_by_mask: dict[tuple[int, ...], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for key, frames in sorted(masks.items(), key=lambda item: (len(item[0]), item[0])):
        target = keys[frames[0]]
        seed_scores: list[tuple[tuple[float, int, int, int], np.ndarray, str]] = []
        baseline_scores = []
        for frame in frames:
            q = control[frame].copy()
            score = contact_score(target, kin.activation_for_qpos(q, settle_steps=1), float(config.threshold))
            baseline_scores.append(score)
            seed_scores.append((score, q, f"frame_{frame}"))
        seed_scores.sort(
            key=lambda item: rank_score(item[0], fp_weight=float(config.contact_cem_fp_rank_weight)),
            reverse=True,
        )
        base_mean_f1 = float(np.mean([score[0] for score in baseline_scores])) if baseline_scores else 0.0
        base_mean_fp = float(np.mean([score[2] for score in baseline_scores])) if baseline_scores else 0.0
        candidates = seed_scores[: max(int(config.contact_cem_top_seeds), 1)]
        centers = [q for _score, q, _name in candidates]
        finger_sigma = float(config.contact_cem_finger_sigma)
        forearm_sigma = float(config.contact_cem_forearm_sigma)
        for iteration in range(max(int(config.contact_cem_iterations), 1)):
            sampled: list[tuple[tuple[float, int, int, int], np.ndarray, str]] = []
            for sample in range(max(int(config.contact_cem_samples_per_mask), 0)):
                center = centers[int(sample % len(centers))]
                cand = kin.clip_qpos(
                    perturb_qpos(
                        center,
                        rng,
                        finger_sigma=finger_sigma,
                        forearm_sigma=forearm_sigma,
                        include_forearm=bool(config.contact_cem_include_forearm),
                        lift_forearm_ty=bool(config.contact_cem_lift_forearm_ty),
                    )
                )
                score = contact_score(target, kin.activation_for_qpos(cand, settle_steps=1), float(config.threshold))
                sampled.append((score, cand, f"iter{iteration}_sample{sample}"))
            candidates.extend(sampled)
            candidates.sort(
                key=lambda item: rank_score(item[0], fp_weight=float(config.contact_cem_fp_rank_weight)),
                reverse=True,
            )
            elite = candidates[: max(int(config.contact_cem_elite_count), 1)]
            centers = [q for _score, q, _name in elite]
            finger_sigma *= float(config.contact_cem_sigma_decay)
            forearm_sigma *= float(config.contact_cem_sigma_decay)

        candidates.sort(
            key=lambda item: rank_score(item[0], fp_weight=float(config.contact_cem_fp_rank_weight)),
            reverse=True,
        )
        best_score, best_qpos, best_source = candidates[0]
        use = (
            float(best_score[0]) >= base_mean_f1 + float(config.contact_cem_min_static_improvement)
            and float(best_score[2]) <= base_mean_fp + float(config.contact_cem_max_fp_increase)
        )
        if use:
            selected_by_mask[key] = best_qpos.astype(np.float32)
        rows.append(
            {
                "keys": list(key),
                "frames": int(len(frames)),
                "use": bool(use),
                "source": str(best_source),
                "baseline_mean_static_f1": base_mean_f1,
                "baseline_mean_fp": base_mean_fp,
                "best_static_f1": float(best_score[0]),
                "best_tp": int(best_score[1]),
                "best_fp": int(best_score[2]),
                "best_fn": int(best_score[3]),
            }
        )

    changed = 0
    for key, frames in masks.items():
        best = selected_by_mask.get(key)
        if best is None:
            continue
        for frame in frames:
            control[frame] = best
            changed += 1

    metadata = {
        "enabled": True,
        "masks": int(len(masks)),
        "used_masks": int(len(selected_by_mask)),
        "changed_control_frames": int(changed),
        "samples_per_mask": int(config.contact_cem_samples_per_mask),
        "elite_count": int(config.contact_cem_elite_count),
        "iterations": int(config.contact_cem_iterations),
        "top_seeds": int(config.contact_cem_top_seeds),
        "finger_sigma": float(config.contact_cem_finger_sigma),
        "forearm_sigma": float(config.contact_cem_forearm_sigma),
        "min_static_improvement": float(config.contact_cem_min_static_improvement),
        "max_fp_increase": float(config.contact_cem_max_fp_increase),
        "fp_rank_weight": float(config.contact_cem_fp_rank_weight),
        "sigma_decay": float(config.contact_cem_sigma_decay),
        "include_forearm": bool(config.contact_cem_include_forearm),
        "lift_forearm_ty": bool(config.contact_cem_lift_forearm_ty),
        "seed": int(config.contact_cem_seed),
        "mask_rows": rows[:300],
    }
    return ContactCEMResult(
        control_qpos=control.astype(np.float32),
        dense_qpos=dense_from_control(control, substeps=substeps),
        metadata=metadata,
    )
