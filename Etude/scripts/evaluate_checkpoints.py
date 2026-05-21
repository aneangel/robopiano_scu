from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from etude.data.trajectory_io import finite_difference
from etude.evaluation.metrics import action_metrics, align_reference, error_profile_metrics, joint_metrics, note_metrics
from etude.evaluation.event_metrics import EventMetricsConfig, compute_event_metrics
from etude.evaluation.fingertip_metrics import compute_fingertip_assignment_metrics
from etude.evaluation.rollout import rollout_controller
from etude.evaluation.tracker_eval_utils import build_tracker_controller
from etude.experiments import load_experiment_config
from etude.robopianist.env_factory import make_robopianist_env
from etude.robopianist.state_mapping import resolve_mapping_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Etude checkpoints on a held-out split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-glob", default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--primary-metric", default="piano/event_f1")
    parser.add_argument("--selection-mode", choices=("max", "min"), default="max")
    parser.add_argument("--backend", default="dm_control")
    parser.add_argument("--task", default=None)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=0)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--save-rollout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend != "dm_control":
        raise ValueError(f"Only backend='dm_control' is currently supported, got {args.backend!r}")
    config = load_experiment_config(Path(args.config))
    checkpoints = _resolve_checkpoints(args.checkpoint, args.checkpoint_glob)
    if not checkpoints:
        raise ValueError("No checkpoints matched --checkpoint/--checkpoint-glob")
    episodes = _load_split_rows(Path(args.dataset_root), args.split, args.max_episodes)
    if not episodes:
        raise ValueError(f"No episodes found for split={args.split!r}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = make_robopianist_env(task=_resolve_task(config, args.task))
    action_dim = _checkpoint_action_dim(checkpoints[0])
    if action_dim is not None:
        env = _ActionDimAdapter(env, action_dim)
    mapping = resolve_mapping_from_env(env)
    for checkpoint in checkpoints:
        eval_name = _checkpoint_eval_name(checkpoint)
        eval_dir = output_root / eval_name
        episode_rows: list[dict[str, Any]] = []
        controller = build_tracker_controller(mapping, config, checkpoint_path=str(checkpoint))
        for row in episodes:
            episode_id = str(row["episode_id"])
            episode_dir = eval_dir / "episodes" / episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)
            traj = _load_dataset_episode(Path(args.dataset_root) / str(row["path"]))
            rollout = rollout_controller(
                env,
                controller,
                mapping,
                traj["q_ref"],
                traj["qdot_ref"],
                metadata=traj["metadata"],
                render_video=bool(args.render_video),
                render_height=int(args.render_height),
                render_width=int(args.render_width),
                render_every=int(args.render_every),
                camera_id=_coerce_camera_id(args.camera_id),
            )
            frames = rollout.pop("frames", None)
            metrics = compute_rollout_metrics(rollout, traj, mapping)
            metrics_payload = {
                "episode_id": episode_id,
                "path": row["path"],
                "source": row.get("source", ""),
                "checkpoint": str(checkpoint),
                "metrics": metrics,
            }
            (episode_dir / "metrics.json").write_text(
                json.dumps(metrics_payload, indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
            if args.save_rollout:
                np.savez_compressed(episode_dir / "rollout.npz", **rollout)
            if frames is not None:
                from etude.visualization.videos import save_video

                save_video(list(frames), episode_dir / "rollout.mp4", fps=_resolve_video_fps(args.video_fps, rollout, args.render_every))
            episode_rows.append({"episode_id": episode_id, "path": row["path"], **metrics})
        aggregate = aggregate_episode_metrics(episode_rows)
        aggregate_payload = {
            "checkpoint": str(checkpoint),
            "checkpoint_name": eval_name,
            "config": str(Path(args.config)),
            "split": args.split,
            "primary_metric": args.primary_metric,
            "selection_mode": args.selection_mode,
            "num_episodes": len(episode_rows),
            "metrics": aggregate,
        }
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "aggregate_metrics.json").write_text(
            json.dumps(aggregate_payload, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        _write_episode_metrics(eval_dir / "episode_metrics.csv", episode_rows)
        primary = aggregate.get(args.primary_metric, {}).get("mean")
        print(f"{checkpoint}: {args.primary_metric}_mean={primary}")


def compute_rollout_metrics(rollout: dict[str, np.ndarray], traj: dict[str, Any], mapping: Any) -> dict[str, float]:
    reference_indices = rollout.get("reference_indices")
    aligned_q_ref = align_reference(traj["q_ref"], reference_indices)
    aligned_qdot_ref = align_reference(traj["qdot_ref"], reference_indices)
    metrics = joint_metrics(
        rollout["q"],
        aligned_q_ref,
        rollout["qdot"],
        aligned_qdot_ref,
    )
    metrics.update(
        error_profile_metrics(
            rollout["q"],
            aligned_q_ref,
            prefix="tracking/joint",
            dt=float(np.asarray(rollout.get("env_dt", traj.get("dt", 0.005))).item()),
        )
    )
    metrics.update(action_metrics(rollout["actions"], getattr(mapping, "action_low", None), getattr(mapping, "action_high", None)))
    metrics.update(_fingertip_metrics(rollout, traj))
    metrics.update(_piano_metrics(rollout, traj))
    metrics.update(_timing_metrics(rollout))
    return metrics


def aggregate_episode_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.floating))})
    output: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row and _is_finite_number(row[key])], dtype=np.float64)
        if values.size == 0:
            continue
        output[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": int(values.size),
        }
    return output


def _load_split_rows(dataset_root: Path, split: str, max_episodes: int | None) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.csv"
    splits_path = dataset_root / "splits.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Split file is required for rollout evaluation: {splits_path}")
    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    splits = pd.read_csv(splits_path)
    if "episode_id" in manifest.columns and "episode_id" in splits.columns:
        merged = manifest.merge(splits[["episode_id", "split"]], on="episode_id", how="inner")
    else:
        merged = manifest.merge(splits[["path", "split"]], on="path", how="inner")
    filtered = merged[merged["split"].astype(str) == str(split)].copy()
    filtered = filtered.sort_values(["episode_id" if "episode_id" in filtered.columns else "path"]).reset_index(drop=True)
    if max_episodes is not None:
        filtered = filtered.head(int(max_episodes))
    return filtered.to_dict(orient="records")


def _load_dataset_episode(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        q_ref = np.asarray(data["q_ref"] if "q_ref" in data else data["q"], dtype=np.float32)
        dt = float(np.asarray(data["dt"]).item()) if "dt" in data else 0.005
        qdot_ref = (
            np.asarray(data["qdot_ref"], dtype=np.float32)
            if "qdot_ref" in data
            else finite_difference(q_ref, dt)
        )
        metadata: dict[str, Any] = {"dt": dt}
        for key in (
            "target_keys",
            "desired_fingertips",
            "fingertip_ref",
            "fingertip_weights",
            "active_finger_mask",
            "inactive_finger_mask",
            "contact_mask",
        ):
            if key in data:
                metadata[key] = np.asarray(data[key])
        if "desired_fingertips" not in metadata and "fingertips" in data:
            metadata["desired_fingertips"] = _reshape_fingertips(np.asarray(data["fingertips"]))
    return {"q_ref": q_ref, "qdot_ref": qdot_ref, "dt": dt, "metadata": metadata}


def _piano_metrics(rollout: dict[str, np.ndarray], traj: dict[str, Any]) -> dict[str, float]:
    predicted = rollout.get("key_state")
    target = traj.get("metadata", {}).get("target_keys")
    if predicted is None or target is None:
        return {}
    predicted_keys = np.asarray(predicted, dtype=np.float32)
    target_keys = align_reference(np.asarray(target, dtype=np.float32), rollout.get("reference_indices"))
    steps = min(predicted_keys.shape[0], target_keys.shape[0])
    if steps == 0:
        return {}
    predicted_keys = predicted_keys[:steps]
    target_keys = target_keys[:steps]
    metrics = note_metrics(predicted_keys, target_keys)
    metrics.update(compute_event_metrics(predicted_keys, target_keys, EventMetricsConfig(dt=float(traj.get("dt", 0.005)))))
    return metrics


def _fingertip_metrics(rollout: dict[str, np.ndarray], traj: dict[str, Any]) -> dict[str, float]:
    current = rollout.get("fingertips")
    desired = rollout.get("desired_fingertips")
    active_finger_mask = rollout.get("active_finger_mask")
    if current is None or desired is None:
        desired = align_reference(traj.get("metadata", {}).get("desired_fingertips"), rollout.get("reference_indices"))
    if active_finger_mask is None:
        active_finger_mask = align_reference(
            traj.get("metadata", {}).get("active_finger_mask"),
            rollout.get("reference_indices"),
        )
    if current is None or desired is None:
        return {}
    metrics = compute_fingertip_assignment_metrics(
        current,
        desired,
        active_finger_mask=active_finger_mask,
        contact_mask=traj.get("metadata", {}).get("contact_mask"),
    )
    metrics.update(
        error_profile_metrics(
            np.asarray(current, dtype=np.float32),
            np.asarray(desired, dtype=np.float32),
            prefix="tracking/fingertip",
            dt=float(np.asarray(rollout.get("env_dt", traj.get("dt", 0.005))).item()),
        )
    )
    return metrics


def _reshape_fingertips(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        return array
    if array.ndim == 2 and array.shape[1] % 3 == 0:
        return array.reshape(array.shape[0], array.shape[1] // 3, 3).astype(np.float32)
    return array


def _timing_metrics(rollout: dict[str, np.ndarray]) -> dict[str, float]:
    trajectory_dt = float(np.asarray(rollout.get("trajectory_dt", 0.0)).item())
    env_dt = float(np.asarray(rollout.get("env_dt", 0.0)).item())
    metrics = {
        "timing/reference_dt_s": trajectory_dt,
        "timing/env_dt_s": env_dt,
    }
    if env_dt > 0.0:
        metrics["timing/controller_rate_hz"] = 1.0 / env_dt
    if trajectory_dt > 0.0:
        metrics["timing/reference_rate_hz"] = 1.0 / trajectory_dt
    if env_dt > 0.0 and trajectory_dt > 0.0:
        metrics["timing/reference_repeat_factor"] = max(trajectory_dt / env_dt, 1.0)
    return metrics


def _resolve_video_fps(video_fps: int, rollout: dict[str, np.ndarray], render_every: int) -> int:
    if int(video_fps) > 0:
        return int(video_fps)
    env_dt = float(np.asarray(rollout.get("env_dt", 0.0)).item())
    stride = max(int(render_every), 1)
    if env_dt <= 0.0:
        return 30
    return max(int(round(1.0 / (env_dt * stride))), 1)


def _resolve_checkpoints(checkpoints: list[str], checkpoint_glob: str | None) -> list[Path]:
    paths = [Path(path) for path in checkpoints]
    if checkpoint_glob:
        paths.extend(Path(path) for path in glob.glob(checkpoint_glob))
    return sorted({path.resolve() for path in paths})


def _checkpoint_action_dim(path: Path) -> int | None:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if isinstance(payload, dict) and payload.get("action_dim") is not None:
        return int(payload["action_dim"])
    return None


class _ActionSpec:
    def __init__(self, minimum: np.ndarray, maximum: np.ndarray) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.shape = minimum.shape


class _ActionDimAdapter:
    def __init__(self, env: Any, action_dim: int) -> None:
        self._env = env
        self._action_dim = int(action_dim)
        base_spec = env.action_spec() if callable(getattr(env, "action_spec", None)) else env.action_spec
        self._base_minimum = np.asarray(base_spec.minimum, dtype=np.float32).reshape(-1)
        self._base_maximum = np.asarray(base_spec.maximum, dtype=np.float32).reshape(-1)
        if self._action_dim > self._base_minimum.size:
            raise ValueError(
                f"Checkpoint action_dim={self._action_dim} exceeds environment action_dim={self._base_minimum.size}"
            )
        self._spec = _ActionSpec(
            self._base_minimum[: self._action_dim],
            self._base_maximum[: self._action_dim],
        )

    def action_spec(self) -> _ActionSpec:
        return self._spec

    def reset(self) -> Any:
        return self._env.reset()

    def step(self, action: np.ndarray) -> Any:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self._action_dim,):
            raise ValueError(f"Expected action shape {(self._action_dim,)}, got {action.shape}")
        padded = np.zeros_like(self._base_minimum, dtype=np.float32)
        padded[: self._action_dim] = action
        padded = np.clip(padded, self._base_minimum, self._base_maximum)
        return self._env.step(padded)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _resolve_task(config: dict[str, Any], task: str | None) -> str:
    if task:
        return task
    rollout_cfg = config.get("rollout", {})
    if isinstance(rollout_cfg, dict) and rollout_cfg.get("task"):
        return str(rollout_cfg["task"])
    env_cfg = config.get("environment", {})
    if isinstance(env_cfg, dict) and env_cfg.get("task"):
        return str(env_cfg["task"])
    return "RoboPianist-repertoire-150-v0"


def _checkpoint_eval_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    parent = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    raw = f"{parent}_{path.stem}_{digest}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def _write_episode_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_keys = sorted({key for row in rows for key in row if key not in {"episode_id", "path"}})
    fieldnames = ["episode_id", "path", *metric_keys]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _coerce_camera_id(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
