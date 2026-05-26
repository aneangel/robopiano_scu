from __future__ import annotations

from dataclasses import dataclass, replace
import ctypes
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from bagatelle.assignment import FingerAssignmentResult, NUM_FINGERS
from bagatelle.config import BagatelleConfig
from bagatelle.paths import ensure_repo_paths


HAND_STATE_DIM = 46
FINGER_ORDER = "left_hand_sites_then_right_hand_sites"
JOINT_ORDER = "right_hand_joints_then_left_hand_joints"
JOINT_INDEX_RANGES_BY_HAND = {
    "right_hand": [0, 23],
    "left_hand": [23, 46],
}
FINGER_HAND_LABELS = ("left", "left", "left", "left", "left", "right", "right", "right", "right", "right")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little", "thumb", "index", "middle", "ring", "little")
UNASSIGNED_FINGERTIP_STRATEGIES = ("legacy", "avoid_mispresses")
RIGHT_FOREARM_TX_INDEX = 21
RIGHT_FOREARM_TY_INDEX = 22
LEFT_FOREARM_TX_INDEX = 44
LEFT_FOREARM_TY_INDEX = 45


def _model_element_name(element: Any) -> str:
    for attr in ("full_identifier", "identifier", "name"):
        value = getattr(element, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        if value:
            return str(value)
    return str(element)


@dataclass(frozen=True)
class IKResult:
    pose: np.ndarray
    fingertip_positions: np.ndarray
    assigned_distances: np.ndarray
    residual_norm: float
    max_residual: float
    success: bool
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_cost: float
    nfev: int
    active_keys: np.ndarray
    assigned_keys: np.ndarray
    assigned_finger_indices: np.ndarray
    unassigned_keys: np.ndarray
    contact_validation_used: bool = False
    contact_target_hit_count: int = 0
    contact_wrong_key_count: int = 0
    contact_missed_key_count: int = 0
    contact_played_key_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_keys": self.active_keys.astype(int).tolist(),
            "assigned_keys": self.assigned_keys.astype(int).tolist(),
            "assigned_finger_indices": self.assigned_finger_indices.astype(int).tolist(),
            "unassigned_keys": self.unassigned_keys.astype(int).tolist(),
            "assigned_distances": self.assigned_distances.astype(float).tolist(),
            "residual_norm": float(self.residual_norm),
            "max_residual": float(self.max_residual),
            "success": bool(self.success),
            "optimizer_success": bool(self.optimizer_success),
            "optimizer_status": int(self.optimizer_status),
            "optimizer_message": str(self.optimizer_message),
            "optimizer_cost": float(self.optimizer_cost),
            "nfev": int(self.nfev),
            "contact_validation_used": bool(self.contact_validation_used),
            "contact_target_hit_count": int(self.contact_target_hit_count),
            "contact_wrong_key_count": int(self.contact_wrong_key_count),
            "contact_missed_key_count": int(self.contact_missed_key_count),
            "contact_played_key_count": int(self.contact_played_key_count),
        }


def _preload_conda_libstdcxx() -> None:
    import sys

    lib = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not lib.is_file():
        return
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


class BagatelleKinematics:
    """RoboPianist-backed forward kinematics and bounded reduced-hand IK."""

    def __init__(
        self,
        config: BagatelleConfig | None = None,
        *,
        target_keys: np.ndarray | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or BagatelleConfig()
        ensure_repo_paths()
        os.environ.setdefault("MUJOCO_GL", "egl")
        _preload_conda_libstdcxx()

        from partita.evaluation.rollout import (
            _load_env,
            _locate_task_physics_piano,
            candidate_environment_names,
            write_goals_proto,
        )

        if output_dir is None:
            self._temp_context = tempfile.TemporaryDirectory(prefix="bagatelle_kinematics_")
            proto_dir = Path(self._temp_context.name)
        else:
            self._temp_context = None
            proto_dir = Path(output_dir)
            proto_dir.mkdir(parents=True, exist_ok=True)

        goals = np.asarray(target_keys, dtype=np.float32) if target_keys is not None else np.zeros((1, 88), dtype=np.float32)
        if goals.ndim != 2 or goals.shape[1] < 88:
            raise ValueError(f"target_keys must have shape [T, 88+], got {goals.shape}")
        self.midi_proto_path = write_goals_proto(
            goals[:, :88],
            proto_dir / "bagatelle_target_goals.proto",
            dt=float(self.config.control_timestep),
            title="Bagatelle target goals",
        )
        self.environment_name, self.env, self.load_info = _load_env(
            environment_names=candidate_environment_names(self.config.environment_name),
            midi_proto_path=self.midi_proto_path,
            control_timestep=float(self.config.control_timestep),
            seed=int(self.config.seed),
            reduced_action_space=bool(self.config.reduced_action_space),
            extra_task_kwargs={
                "disable_forearm_reward": True,
                "disable_fingering_reward": True,
                "disable_colorization": True,
                "disable_hand_collisions": False,
                "wrong_press_termination": False,
            },
            suite_load_kwargs=None,
            prefer_canonical_midi=False,
        )
        self.env.reset()
        self.task, self.physics, self.piano = _locate_task_physics_piano(self.env)
        self.joint_handles = self._hand_joint_handles()
        if len(self.joint_handles) != HAND_STATE_DIM:
            raise RuntimeError(f"Expected {HAND_STATE_DIM} reduced hand joints, got {len(self.joint_handles)}")
        self.fingertip_sites = tuple(self.task.left_hand.fingertip_sites) + tuple(self.task.right_hand.fingertip_sites)
        if len(self.fingertip_sites) != NUM_FINGERS:
            raise RuntimeError(f"Expected {NUM_FINGERS} fingertip sites, got {len(self.fingertip_sites)}")

        bounds = np.asarray(self.physics.bind(self.joint_handles).range, dtype=np.float64)
        if bounds.shape != (HAND_STATE_DIM, 2):
            raise RuntimeError(f"Expected joint bounds [{HAND_STATE_DIM}, 2], got {bounds.shape}")
        self.joint_lower = bounds[:, 0].astype(np.float32)
        self.joint_upper = bounds[:, 1].astype(np.float32)
        self._repair_bounds()
        self.neutral_qpos = self.current_qpos()
        self.neutral_fingertips = self.current_fingertips()
        self._rhapsody_solver = None
        self._rhapsody_checkpoint = ""
        self._rhapsody_device = "cpu"
        self._zero_action = None
        try:
            action_spec = self.env.action_spec()
            self._zero_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        except Exception:
            self._zero_action = None

    def close(self) -> None:
        close = getattr(getattr(self, "env", None), "close", None)
        if callable(close):
            close()
        if getattr(self, "_temp_context", None) is not None:
            self._temp_context.cleanup()
            self._temp_context = None

    def __enter__(self) -> "BagatelleKinematics":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def joint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.joint_lower.copy(), self.joint_upper.copy()

    @property
    def fingertip_site_names(self) -> tuple[str, ...]:
        return tuple(_model_element_name(site) for site in self.fingertip_sites)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(_model_element_name(joint) for joint in self.joint_handles)

    @property
    def joint_index_ranges_by_hand(self) -> dict[str, list[int]]:
        return {hand: list(bounds) for hand, bounds in JOINT_INDEX_RANGES_BY_HAND.items()}

    @property
    def order_metadata(self) -> dict[str, Any]:
        site_names = list(self.fingertip_site_names)
        joint_names = list(self.joint_names)
        return {
            "finger_order": FINGER_ORDER,
            "joint_order": JOINT_ORDER,
            "fingertip_site_names": site_names,
            "joint_names": joint_names,
            "finger_hands": list(FINGER_HAND_LABELS),
            "finger_names": list(FINGER_NAMES),
            "assignment_finger_to_site": [
                {"finger_index": index, "site_name": name}
                for index, name in enumerate(site_names)
            ],
            "joint_index_ranges_by_hand": self.joint_index_ranges_by_hand,
        }

    def _repair_bounds(self) -> None:
        for index in range(HAND_STATE_DIM):
            lo = float(self.joint_lower[index])
            hi = float(self.joint_upper[index])
            if not np.isfinite(lo):
                lo = -1.0
            if not np.isfinite(hi):
                hi = 1.0
            if hi <= lo:
                center = 0.5 * (lo + hi)
                lo = center - 1e-3
                hi = center + 1e-3
            self.joint_lower[index] = lo
            self.joint_upper[index] = hi

    def _hand_joint_handles(self) -> list[Any]:
        joints: list[Any] = []
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(self.task, hand_name)
            joints.extend(list(getattr(hand, "joints")))
        return joints

    def current_qpos(self) -> np.ndarray:
        values = []
        for joint in self.joint_handles:
            qpos = np.asarray(self.physics.bind(joint).qpos, dtype=np.float64).reshape(-1)
            values.append(float(qpos[0]) if qpos.size else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _set_qpos(self, qpos: np.ndarray) -> None:
        values = self.clip_qpos(qpos)
        for joint, value in zip(self.joint_handles, values):
            self.physics.bind(joint).qpos = float(value)
        if hasattr(self.physics, "forward"):
            self.physics.forward()

    def clip_qpos(self, qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if values.shape != (HAND_STATE_DIM,):
            raise ValueError(f"qpos must have shape [{HAND_STATE_DIM}], got {values.shape}")
        return np.clip(values, self.joint_lower, self.joint_upper).astype(np.float32)

    def current_fingertips(self) -> np.ndarray:
        positions = np.asarray(self.physics.bind(self.fingertip_sites).xpos, dtype=np.float32)
        if positions.shape != (NUM_FINGERS, 3):
            raise RuntimeError(f"Expected fingertip positions [10, 3], got {positions.shape}")
        return np.ascontiguousarray(positions, dtype=np.float32)

    def fingertip_positions_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        self._set_qpos(qpos)
        return self.current_fingertips()

    def _reset_piano_to_inactive(self) -> None:
        qpos_range = getattr(self.piano, "_qpos_range", None)
        joints = getattr(self.piano, "joints", None)
        if qpos_range is not None and joints is not None:
            ranges = np.asarray(qpos_range, dtype=np.float64)
            inactive = np.maximum(ranges[:, 0], 0.0)
            self.physics.bind(joints).qpos = inactive
        sustain_state = getattr(self.piano, "_sustain_state", None)
        if sustain_state is not None:
            try:
                sustain_state[0] = 0.0
            except Exception:
                pass
        if hasattr(self.physics, "forward"):
            self.physics.forward()
        update_key_state = getattr(self.piano, "_update_key_state", None)
        if callable(update_key_state):
            update_key_state(self.physics)
        update_key_color = getattr(self.piano, "_update_key_color", None)
        if callable(update_key_color):
            update_key_color(self.physics)

    def activation_for_qpos(self, qpos: np.ndarray, *, settle_steps: int = 1) -> np.ndarray:
        self._reset_piano_to_inactive()
        steps = max(int(settle_steps), 1)
        for _ in range(steps):
            self._set_qpos(qpos)
            if self._zero_action is not None:
                self.env.step(self._zero_action)
            elif hasattr(self.physics, "forward"):
                self.physics.forward()
        update_key_state = getattr(self.piano, "_update_key_state", None)
        if callable(update_key_state):
            update_key_state(self.physics)
        update_key_color = getattr(self.piano, "_update_key_color", None)
        if callable(update_key_color):
            update_key_color(self.physics)
        activation = getattr(self.piano, "activation", None)
        if activation is None:
            return np.zeros((int(self.piano.n_keys),), dtype=np.float32)
        return np.asarray(activation, dtype=np.float32).reshape(-1)[: int(self.piano.n_keys)].astype(np.float32)

    def static_contact_metrics(
        self,
        qpos: np.ndarray,
        assigned_keys: np.ndarray,
        *,
        threshold: float,
        settle_steps: int,
    ) -> dict[str, int]:
        activation = self.activation_for_qpos(qpos, settle_steps=settle_steps)
        played = activation[: int(self.piano.n_keys)] > float(threshold)
        assigned = np.zeros((int(self.piano.n_keys),), dtype=bool)
        keys = np.asarray(assigned_keys, dtype=np.int32).reshape(-1)
        valid = keys[(keys >= 0) & (keys < int(self.piano.n_keys))]
        if valid.size:
            assigned[valid.astype(np.int64)] = True
        hits = np.logical_and(played, assigned)
        wrong = np.logical_and(played, ~assigned)
        missed = np.logical_and(~played, assigned)
        return {
            "target_hit_count": int(np.count_nonzero(hits)),
            "wrong_key_count": int(np.count_nonzero(wrong)),
            "missed_key_count": int(np.count_nonzero(missed)),
            "played_key_count": int(np.count_nonzero(played)),
        }

    def _key_targets(self, keys: np.ndarray | None, *, press_depth: float) -> np.ndarray:
        if keys is None:
            key_indices = np.arange(88, dtype=np.int32)
        else:
            key_indices = np.asarray(keys, dtype=np.int32).reshape(-1)
        targets = []
        for key in key_indices:
            if int(key) < 0 or int(key) >= self.piano.n_keys:
                raise ValueError(f"Invalid piano key index: {int(key)}")
            key_geom = self.piano.keys[int(key)].geom[0]
            bind = self.physics.bind(key_geom)
            pos = np.asarray(bind.xpos, dtype=np.float32).copy()
            size = np.asarray(bind.size, dtype=np.float32)
            pos[-1] += float(self.config.key_target_top_offset) * float(size[2])
            pos[0] += float(self.config.key_target_front_offset) * float(size[0])
            pos[-1] -= float(press_depth)
            targets.append(pos)
        return np.stack(targets, axis=0).astype(np.float32) if targets else np.zeros((0, 3), dtype=np.float32)

    def key_contact_targets(self, keys: np.ndarray | None = None) -> np.ndarray:
        return self._key_targets(keys, press_depth=0.0)

    def key_press_targets(self, keys: np.ndarray | None = None, *, press_depth: float | None = None) -> np.ndarray:
        depth = self.config.key_press_depth if press_depth is None else float(press_depth)
        return self._key_targets(keys, press_depth=float(depth))

    @staticmethod
    def inactive_wrong_key_clearance_residual(
        inactive_fingertips: np.ndarray,
        wrong_key_targets: np.ndarray,
        *,
        clearance_z: float,
        radius: float,
    ) -> np.ndarray:
        fingertips = np.asarray(inactive_fingertips, dtype=np.float32)
        targets = np.asarray(wrong_key_targets, dtype=np.float32)
        if fingertips.size == 0 or targets.size == 0:
            return np.zeros((0,), dtype=np.float64)
        radius_value = max(float(radius), 1e-6)
        xy_distance = np.linalg.norm(fingertips[:, None, :2] - targets[None, :, :2], axis=2)
        proximity = np.maximum(radius_value - xy_distance, 0.0) / radius_value
        vertical_deficit = np.maximum(float(clearance_z) - fingertips[:, 2], 0.0)
        return (proximity * vertical_deficit[:, None]).reshape(-1).astype(np.float64)

    def _ranked_press_pose_initial_seeds(
        self,
        *,
        previous: np.ndarray,
        neutral: np.ndarray,
        finger_indices: np.ndarray,
        target_positions: np.ndarray,
        config: BagatelleConfig,
    ) -> list[np.ndarray]:
        seed_count = max(int(getattr(config, "ik_multistart_seed_count", 4)), 1)
        grid_count = max(int(getattr(config, "ik_multistart_forearm_tx_grid", 5)), 2)
        base_rows = [self.clip_qpos(previous), self.clip_qpos(neutral)]
        active_left = bool(np.any(np.asarray(finger_indices, dtype=np.int64) < 5))
        active_right = bool(np.any(np.asarray(finger_indices, dtype=np.int64) >= 5))
        tx_options: dict[str, np.ndarray] = {}
        if active_left:
            tx_options["left"] = np.linspace(
                float(self.joint_lower[LEFT_FOREARM_TX_INDEX]),
                float(self.joint_upper[LEFT_FOREARM_TX_INDEX]),
                grid_count,
                dtype=np.float32,
            )
        if active_right:
            tx_options["right"] = np.linspace(
                float(self.joint_lower[RIGHT_FOREARM_TX_INDEX]),
                float(self.joint_upper[RIGHT_FOREARM_TX_INDEX]),
                grid_count,
                dtype=np.float32,
            )

        candidates: list[np.ndarray] = []
        candidates.extend(base_rows)
        for base in base_rows:
            left_values = tx_options.get("left", np.asarray([base[LEFT_FOREARM_TX_INDEX]], dtype=np.float32))
            right_values = tx_options.get("right", np.asarray([base[RIGHT_FOREARM_TX_INDEX]], dtype=np.float32))
            for left_tx in left_values:
                for right_tx in right_values:
                    row = np.asarray(base, dtype=np.float32).copy()
                    if active_left:
                        row[LEFT_FOREARM_TX_INDEX] = np.float32(left_tx)
                        row[LEFT_FOREARM_TY_INDEX] = np.float32(
                            max(float(row[LEFT_FOREARM_TY_INDEX]), float(neutral[LEFT_FOREARM_TY_INDEX]))
                        )
                    if active_right:
                        row[RIGHT_FOREARM_TX_INDEX] = np.float32(right_tx)
                        row[RIGHT_FOREARM_TY_INDEX] = np.float32(
                            max(float(row[RIGHT_FOREARM_TY_INDEX]), float(neutral[RIGHT_FOREARM_TY_INDEX]))
                        )
                    candidates.append(self.clip_qpos(row))

        scored: list[tuple[float, tuple[float, ...], np.ndarray]] = []
        seen: set[bytes] = set()
        targets = np.asarray(target_positions, dtype=np.float32)
        fingers = np.asarray(finger_indices, dtype=np.int64)
        for row in candidates:
            key = np.round(row.astype(np.float32), 5).tobytes()
            if key in seen:
                continue
            seen.add(key)
            try:
                fingertips = self.fingertip_positions_for_qpos(row)
                if fingers.size and targets.size:
                    distances = np.linalg.norm(fingertips[fingers] - targets, axis=1)
                    max_distance = float(np.max(distances))
                    score = float(np.linalg.norm(distances))
                else:
                    max_distance = 0.0
                    score = 0.0
            except Exception:
                max_distance = float("inf")
                score = float("inf")
            scored.append((score, (max_distance, *row.astype(float).tolist()), row))
        scored.sort(key=lambda item: (float(item[0]), item[1]))
        return [row.astype(np.float32) for _score, _tie, row in scored[:seed_count]]

    def _load_rhapsody_solver(self, config: BagatelleConfig):
        checkpoint = str(getattr(config, "rhapsody_ik_checkpoint", "") or "")
        if not checkpoint:
            raise ValueError("rhapsody_ik_checkpoint must be set when rhapsody_ik_enabled is true")
        device = str(getattr(config, "rhapsody_ik_device", "cpu") or "cpu")
        if (
            self._rhapsody_solver is not None
            and self._rhapsody_checkpoint == checkpoint
            and self._rhapsody_device == device
        ):
            return self._rhapsody_solver
        from rhapsody.solver import RhapsodyIKSolver

        self._rhapsody_solver = RhapsodyIKSolver.from_checkpoint(checkpoint, device=device)
        self._rhapsody_checkpoint = checkpoint
        self._rhapsody_device = device
        return self._rhapsody_solver

    def rhapsody_seed_for_fingertips(
        self,
        target_fingertips: np.ndarray,
        active_mask: np.ndarray,
        previous_qpos: np.ndarray,
        *,
        config: BagatelleConfig | None = None,
    ) -> np.ndarray:
        cfg = config or self.config
        if not bool(getattr(cfg, "rhapsody_ik_enabled", False)):
            raise ValueError("Rhapsody IK is not enabled")
        solver = self._load_rhapsody_solver(cfg)
        previous = self.clip_qpos(previous_qpos)
        targets = np.asarray(target_fingertips, dtype=np.float32).reshape(NUM_FINGERS, 3).copy()
        mask = np.asarray(active_mask, dtype=np.float32).reshape(NUM_FINGERS).copy()
        if bool(getattr(cfg, "rhapsody_ik_fill_inactive_from_previous", True)):
            inactive = mask <= 0.0
            if bool(np.any(inactive)):
                previous_tips = np.asarray(self.fingertip_positions_for_qpos(previous), dtype=np.float32).reshape(NUM_FINGERS, 3)
                targets[inactive] = previous_tips[inactive]
            mask = np.ones((NUM_FINGERS,), dtype=np.float32)
        target_for_solver, mask_for_solver = self._rhapsody_target_transform(
            targets,
            mask,
            cfg,
        )
        solution = solver.solve(
            target_for_solver,
            active_mask=mask_for_solver,
            previous_qpos=previous,
            refinement_steps=max(int(getattr(cfg, "rhapsody_ik_refinement_steps", 16)), 0),
            refinement_lr=float(getattr(cfg, "rhapsody_ik_refinement_lr", 0.05)),
        )
        return self.clip_qpos(solution.qpos)

    @staticmethod
    def _rhapsody_target_transform(
        target_fingertips: np.ndarray,
        active_mask: np.ndarray,
        config: BagatelleConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(target_fingertips, dtype=np.float32).reshape(NUM_FINGERS, 3)
        mask = np.asarray(active_mask, dtype=np.float32).reshape(NUM_FINGERS)
        mode = str(getattr(config, "rhapsody_ik_coordinate_transform", "bagatelle_to_rp1m"))
        if mode in ("", "none"):
            return targets.copy(), mask.copy()
        if mode != "bagatelle_to_rp1m":
            raise ValueError(f"Unsupported Rhapsody target transform: {mode!r}")
        transformed = np.empty_like(targets, dtype=np.float32)
        transformed[:5] = targets[5:]
        transformed[5:] = targets[:5]
        transformed[:, 1] += np.float32(float(getattr(config, "rhapsody_ik_y_offset", 0.08289646)))
        transformed_mask = np.concatenate([mask[5:], mask[:5]], axis=0).astype(np.float32)
        return transformed.astype(np.float32), transformed_mask

    def _rhapsody_press_pose_seed(
        self,
        assignments: FingerAssignmentResult,
        previous_qpos: np.ndarray,
        config: BagatelleConfig,
    ) -> np.ndarray | None:
        if not bool(getattr(config, "rhapsody_ik_enabled", False)) or assignments.count == 0:
            return None
        targets = np.full((NUM_FINGERS, 3), np.nan, dtype=np.float32)
        active = np.zeros((NUM_FINGERS,), dtype=np.float32)
        fingers = assignments.assigned_finger_indices.astype(np.int64)
        targets[fingers] = assignments.target_positions.astype(np.float32)
        active[fingers] = 1.0
        try:
            seed = self.clip_qpos(self.rhapsody_seed_for_fingertips(targets, active, previous_qpos, config=config))
        except Exception:
            return None
        active_targets = assignments.target_positions.astype(np.float32)
        try:
            seed_tips = np.asarray(self.fingertip_positions_for_qpos(seed), dtype=np.float32)
            seed_errors = np.linalg.norm(seed_tips[fingers] - active_targets, axis=1)
        except Exception:
            return None
        if not np.all(np.isfinite(seed_errors)):
            return None
        max_allowed = float(getattr(config, "rhapsody_ik_seed_max_active_error", 0.08))
        if max_allowed > 0.0 and float(np.max(seed_errors)) > max_allowed:
            return None
        if bool(getattr(config, "rhapsody_ik_seed_require_previous_improvement", True)):
            try:
                previous = self.clip_qpos(previous_qpos)
                previous_tips = np.asarray(self.fingertip_positions_for_qpos(previous), dtype=np.float32)
                previous_errors = np.linalg.norm(previous_tips[fingers] - active_targets, axis=1)
                if float(np.mean(seed_errors)) >= float(np.mean(previous_errors)):
                    return None
            except Exception:
                return None
        return seed

    def solve_press_pose(
        self,
        assignments: FingerAssignmentResult,
        previous_qpos: np.ndarray,
        neutral_qpos: np.ndarray | None = None,
        config: BagatelleConfig | None = None,
        cache: "KeysetCache | None" = None,
    ) -> IKResult:
        cfg = config or self.config
        previous = self.clip_qpos(previous_qpos)
        neutral = self.clip_qpos(neutral_qpos if neutral_qpos is not None else self.neutral_qpos)
        if assignments.count == 0:
            fingertips = self.fingertip_positions_for_qpos(previous)
            return self._result_from_pose(
                previous,
                fingertips,
                assignments,
                optimizer_success=True,
                optimizer_status=0,
                optimizer_message="no active assignments",
                optimizer_cost=0.0,
                nfev=0,
                threshold=float(cfg.residual_success_threshold),
            )

        # Keyset cache exact-hit short-circuit. If a previous identical-assignment
        # solve produced an acceptable pose, reuse it and skip scipy LM entirely.
        cache_key = None
        if cache is not None and cache.mode != "off":
            cache_key = cache.make_key(
                active_keys=np.asarray(assignments.assigned_keys, dtype=np.int32),
                hand_assignment=tuple(
                    "L" if int(f) < 5 else "R"
                    for f in assignments.assigned_finger_indices.tolist()
                ),
                finger_assignment=tuple(int(f) for f in assignments.assigned_finger_indices.tolist()),
            )
            cached_qpos, hit_kind = cache.lookup(cache_key)
            if hit_kind == "exact" and cached_qpos is not None:
                fingertips = self.fingertip_positions_for_qpos(cached_qpos)
                return self._result_from_pose(
                    cached_qpos,
                    fingertips,
                    assignments,
                    optimizer_success=True,
                    optimizer_status=0,
                    optimizer_message="keyset_cache_exact_hit",
                    optimizer_cost=0.0,
                    nfev=0,
                    threshold=float(cfg.residual_success_threshold),
                )

        finger_indices = assignments.assigned_finger_indices.astype(np.int64)
        target_positions = assignments.target_positions.astype(np.float32)
        x0 = self.clip_qpos(previous)

        fingertip_axis_weights = np.asarray(
            [cfg.ik_key_front_weight, cfg.ik_key_width_weight, cfg.ik_key_height_weight],
            dtype=np.float64,
        )
        inactive_clearance_weight = float(getattr(cfg, "ik_inactive_fingertip_clearance_weight", 0.0))
        inactive_clearance = float(getattr(cfg, "ik_inactive_fingertip_clearance", 0.02))
        unassigned_strategy = str(getattr(cfg, "ik_unassigned_fingertip_strategy", "legacy"))
        if unassigned_strategy not in UNASSIGNED_FINGERTIP_STRATEGIES:
            raise ValueError(
                "ik_unassigned_fingertip_strategy must be one of "
                f"{UNASSIGNED_FINGERTIP_STRATEGIES}, got {unassigned_strategy!r}"
            )
        inactive_avoidance_weight = float(getattr(cfg, "ik_unassigned_fingertip_avoidance_weight", 0.5))
        inactive_avoidance_radius = float(getattr(cfg, "ik_unassigned_fingertip_avoidance_radius", 0.03))
        wrong_key_weight = float(getattr(cfg, "ik_wrong_key_xy_avoidance_weight", 0.0))
        wrong_key_radius = max(float(getattr(cfg, "ik_wrong_key_xy_avoidance_radius", 0.025)), 1e-6)
        inactive_indices = np.setdiff1d(np.arange(NUM_FINGERS, dtype=np.int64), finger_indices, assume_unique=False)
        clearance_z = float(np.max(target_positions[:, 2]) + inactive_clearance) if target_positions.size else 0.0
        wrong_key_xy = np.zeros((0, 2), dtype=np.float32)
        wrong_key_targets = np.zeros((0, 3), dtype=np.float32)
        if wrong_key_weight > 0.0 or unassigned_strategy == "avoid_mispresses":
            all_keys = np.arange(int(self.piano.n_keys), dtype=np.int32)
            assigned_key_set = set(int(key) for key in assignments.assigned_keys.tolist())
            wrong_keys = np.asarray([int(key) for key in all_keys if int(key) not in assigned_key_set], dtype=np.int32)
            if wrong_keys.size:
                wrong_key_targets = self.key_contact_targets(wrong_keys)
                wrong_key_xy = wrong_key_targets[:, :2]

        def residual(values: np.ndarray) -> np.ndarray:
            q = self.clip_qpos(values)
            fingertips = self.fingertip_positions_for_qpos(q)
            fingertip_error = (fingertips[finger_indices] - target_positions) * fingertip_axis_weights[None, :]
            parts = [
                fingertip_error.reshape(-1) * float(cfg.ik_fingertip_weight),
                (q - previous).reshape(-1) * float(cfg.ik_smoothness_weight),
                (q - neutral).reshape(-1) * float(cfg.ik_neutral_weight),
            ]
            if inactive_clearance_weight > 0.0 and inactive_indices.size:
                clearance_error = np.maximum(clearance_z - fingertips[inactive_indices, 2], 0.0)
                parts.append(clearance_error.reshape(-1) * inactive_clearance_weight)
            if (
                unassigned_strategy == "avoid_mispresses"
                and inactive_avoidance_weight > 0.0
                and inactive_indices.size
                and wrong_key_targets.size
            ):
                inactive_error = self.inactive_wrong_key_clearance_residual(
                    fingertips[inactive_indices],
                    wrong_key_targets,
                    clearance_z=clearance_z,
                    radius=inactive_avoidance_radius,
                )
                parts.append(inactive_error * inactive_avoidance_weight)
            if wrong_key_weight > 0.0 and wrong_key_xy.size:
                assigned_xy = fingertips[finger_indices, :2]
                xy_distance = np.linalg.norm(assigned_xy[:, None, :] - wrong_key_xy[None, :, :], axis=2)
                avoid_error = np.maximum(wrong_key_radius - xy_distance, 0.0) / wrong_key_radius
                parts.append(avoid_error.reshape(-1) * wrong_key_weight)
            return np.concatenate(parts, axis=0).astype(np.float64)

        def solve_from_seed(seed: np.ndarray, *, seed_index: int) -> IKResult:
            opt = least_squares(
                residual,
                self.clip_qpos(seed).astype(np.float64),
                bounds=(self.joint_lower.astype(np.float64), self.joint_upper.astype(np.float64)),
                max_nfev=max(int(cfg.ik_max_nfev), 1),
                ftol=float(cfg.ik_ftol),
                xtol=float(cfg.ik_xtol),
                gtol=float(cfg.ik_gtol),
            )
            pose = self.clip_qpos(opt.x)
            fingertips = self.fingertip_positions_for_qpos(pose)
            return self._result_from_pose(
                pose,
                fingertips,
                assignments,
                optimizer_success=bool(opt.success),
                optimizer_status=int(opt.status),
                optimizer_message=f"seed={seed_index}: {opt.message}",
                optimizer_cost=float(opt.cost),
                nfev=int(opt.nfev),
                threshold=float(cfg.residual_success_threshold),
            )

        validate_static_contacts = bool(getattr(cfg, "ik_static_contact_validation", False))
        contact_settle_steps = max(int(getattr(cfg, "ik_static_contact_settle_steps", 1)), 1)

        def with_static_contact_metrics(result: IKResult) -> IKResult:
            if not validate_static_contacts:
                return result
            metrics = self.static_contact_metrics(
                result.pose,
                result.assigned_keys,
                threshold=float(cfg.threshold),
                settle_steps=contact_settle_steps,
            )
            return replace(
                result,
                contact_validation_used=True,
                contact_target_hit_count=int(metrics["target_hit_count"]),
                contact_wrong_key_count=int(metrics["wrong_key_count"]),
                contact_missed_key_count=int(metrics["missed_key_count"]),
                contact_played_key_count=int(metrics["played_key_count"]),
            )

        def contact_rank(result: IKResult) -> tuple[float, int, int, int, float, float]:
            if not validate_static_contacts:
                return (
                    0.0,
                    0,
                    0,
                    0 if bool(result.success) else 1,
                    float(result.max_residual),
                    float(result.residual_norm),
                )
            missed = int(result.contact_missed_key_count)
            wrong = int(result.contact_wrong_key_count)
            failure = 0 if bool(result.success) else 1
            score = (
                float(getattr(cfg, "ik_static_contact_missed_key_weight", 2.0)) * float(missed)
                + float(getattr(cfg, "ik_static_contact_wrong_key_weight", 1.0)) * float(wrong)
                + float(getattr(cfg, "ik_static_contact_residual_weight", 10.0)) * float(result.max_residual)
                + float(getattr(cfg, "ik_static_contact_failure_weight", 25.0)) * float(failure)
                + (0.0 if bool(result.optimizer_success) else 100.0)
            )
            return (float(score), missed, wrong, failure, float(result.max_residual), float(result.residual_norm))

        def residual_rank(result: IKResult) -> tuple[float, float]:
            return (float(result.max_residual), float(result.residual_norm))

        try:
            result = with_static_contact_metrics(solve_from_seed(x0, seed_index=0))
            best = result
            rhapsody_seed = self._rhapsody_press_pose_seed(assignments, previous, cfg)
            try_extra_seeds = bool(validate_static_contacts) or (
                not bool(result.success) and bool(getattr(cfg, "ik_multistart_on_failure", True))
            ) or rhapsody_seed is not None
            if not try_extra_seeds:
                if cache is not None and cache_key is not None and bool(result.success):
                    cache.insert(cache_key, result.pose, float(result.residual_norm))
                return result
            extra_seeds: list[np.ndarray] = []
            if rhapsody_seed is not None:
                extra_seeds.append(self.clip_qpos(rhapsody_seed))
            extra_seeds.extend(
                self._ranked_press_pose_initial_seeds(
                    previous=previous,
                    neutral=neutral,
                    finger_indices=finger_indices,
                    target_positions=target_positions,
                    config=cfg,
                )
            )
            for seed_index, seed in enumerate(extra_seeds, start=1):
                if np.allclose(seed, x0, atol=1e-5, rtol=0.0):
                    continue
                candidate = with_static_contact_metrics(solve_from_seed(seed, seed_index=seed_index))
                if validate_static_contacts:
                    is_better = contact_rank(candidate) < contact_rank(best)
                else:
                    is_better = residual_rank(candidate) < residual_rank(best)
                if is_better:
                    best = candidate
                if not validate_static_contacts and bool(candidate.success):
                    if cache is not None and cache_key is not None:
                        cache.insert(cache_key, candidate.pose, float(candidate.residual_norm))
                    return candidate
            if cache is not None and cache_key is not None and bool(best.success):
                cache.insert(cache_key, best.pose, float(best.residual_norm))
            return best
        except Exception as exc:
            fingertips = self.fingertip_positions_for_qpos(previous)
            return self._result_from_pose(
                previous,
                fingertips,
                assignments,
                optimizer_success=False,
                optimizer_status=-1,
                optimizer_message=f"IK exception: {type(exc).__name__}: {exc}",
                optimizer_cost=float("nan"),
                nfev=0,
                threshold=float(cfg.residual_success_threshold),
            )

    def _result_from_pose(
        self,
        pose: np.ndarray,
        fingertips: np.ndarray,
        assignments: FingerAssignmentResult,
        *,
        optimizer_success: bool,
        optimizer_status: int,
        optimizer_message: str,
        optimizer_cost: float,
        nfev: int,
        threshold: float,
        contact_validation_used: bool = False,
        contact_target_hit_count: int = 0,
        contact_wrong_key_count: int = 0,
        contact_missed_key_count: int = 0,
        contact_played_key_count: int = 0,
    ) -> IKResult:
        if assignments.count:
            distances = np.linalg.norm(
                fingertips[assignments.assigned_finger_indices.astype(np.int64)] - assignments.target_positions,
                axis=1,
            ).astype(np.float32)
        else:
            distances = np.zeros((0,), dtype=np.float32)
        max_residual = float(np.max(distances)) if distances.size else 0.0
        residual_norm = float(np.linalg.norm(distances)) if distances.size else 0.0
        return IKResult(
            pose=np.asarray(pose, dtype=np.float32).copy(),
            fingertip_positions=np.asarray(fingertips, dtype=np.float32).copy(),
            assigned_distances=distances,
            residual_norm=residual_norm,
            max_residual=max_residual,
            success=bool(optimizer_success and max_residual <= float(threshold) and assignments.unassigned_keys.size == 0),
            optimizer_success=bool(optimizer_success),
            optimizer_status=int(optimizer_status),
            optimizer_message=str(optimizer_message),
            optimizer_cost=float(optimizer_cost),
            nfev=int(nfev),
            active_keys=assignments.active_keys.copy(),
            assigned_keys=assignments.assigned_keys.copy(),
            assigned_finger_indices=assignments.assigned_finger_indices.copy(),
            unassigned_keys=assignments.unassigned_keys.copy(),
            contact_validation_used=bool(contact_validation_used),
            contact_target_hit_count=int(contact_target_hit_count),
            contact_wrong_key_count=int(contact_wrong_key_count),
            contact_missed_key_count=int(contact_missed_key_count),
            contact_played_key_count=int(contact_played_key_count),
        )
