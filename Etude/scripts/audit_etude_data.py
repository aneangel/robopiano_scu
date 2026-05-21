from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SEARCH_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders")
CHECKPOINT_NAMES = {"best.pt", "best_val.pt", "best_train.pt", "last.pt"}

DATASET_COLUMNS = (
    "dataset_root",
    "manifest_rows",
    "splits_exists",
    "train_count",
    "val_count",
    "test_count",
    "other_split_counts",
    "duplicate_episode_ids",
    "duplicate_paths",
    "split_overlap_ids",
    "split_overlap_paths",
    "missing_files",
    "q_shape_consistent",
    "qdot_shape_consistent",
    "action_dim_consistent",
    "q_dim_values",
    "action_dim_values",
    "target_keys_episodes",
    "fingertips_episodes",
    "desired_fingertips_episodes",
    "errors",
)

CHECKPOINT_COLUMNS = (
    "checkpoint_path",
    "exists",
    "controller_family",
    "controller_type",
    "feature_mode",
    "model_class",
    "input_dim",
    "action_dim",
    "epoch",
    "train_loss",
    "val_loss",
    "selection_metric",
    "dataset_root",
    "split_file",
    "config_hash",
    "refinement_used",
    "is_10_epoch_unrefined_baseline",
    "errors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Etude datasets, splits, and checkpoint lineage.")
    parser.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    parser.add_argument("--dataset-root", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    for child in ("analysis", "metrics", "manifests"):
        (output_root / child).mkdir(parents=True, exist_ok=True)

    search_root = Path(args.search_root)
    dataset_roots = [Path(path) for path in args.dataset_root]
    dataset_roots.extend(_find_dataset_roots(search_root, max_depth=args.max_depth))
    dataset_roots = sorted(set(path.resolve() for path in dataset_roots))
    checkpoint_paths = _find_checkpoint_paths(search_root, max_depth=args.max_depth)

    dataset_rows = [audit_dataset(root) for root in dataset_roots]
    checkpoint_rows = [audit_checkpoint(path) for path in checkpoint_paths]

    _write_csv(output_root / "metrics" / "dataset_split_audit.csv", dataset_rows, DATASET_COLUMNS)
    _write_csv(output_root / "metrics" / "checkpoint_lineage.csv", checkpoint_rows, CHECKPOINT_COLUMNS)
    _write_dataset_markdown(output_root / "analysis" / "dataset_split_audit.md", dataset_rows)
    _write_checkpoint_markdown(output_root / "analysis" / "checkpoint_lineage.md", checkpoint_rows)
    _write_file_manifest(output_root / "manifests" / "data_and_checkpoint_files.txt", dataset_roots, checkpoint_paths)


def _find_dataset_roots(search_root: Path, *, max_depth: int) -> list[Path]:
    return [path.parent for path in _bounded_rglob(search_root, "manifest.csv", max_depth=max_depth)]


def _find_checkpoint_paths(search_root: Path, *, max_depth: int) -> list[Path]:
    return sorted(
        set(path for path in _bounded_rglob(search_root, "*.pt", max_depth=max_depth) if path.name in CHECKPOINT_NAMES)
    )


def _bounded_rglob(root: Path, pattern: str, *, max_depth: int) -> list[Path]:
    if not root.exists():
        return []
    root = root.resolve()
    matches: list[Path] = []
    for path in root.rglob(pattern):
        try:
            depth = len(path.resolve().relative_to(root).parts)
        except ValueError:
            continue
        if depth <= max_depth:
            matches.append(path)
    return matches


def audit_dataset(root: Path) -> dict[str, Any]:
    row = {column: "" for column in DATASET_COLUMNS}
    row["dataset_root"] = str(root)
    errors: list[str] = []
    try:
        manifest = pd.read_csv(root / "manifest.csv")
    except Exception as exc:  # pragma: no cover
        row["errors"] = f"manifest read failed: {exc}"
        return row

    row["manifest_rows"] = len(manifest)
    row["duplicate_episode_ids"] = _duplicate_count(manifest, "episode_id")
    row["duplicate_paths"] = _duplicate_count(manifest, "path")

    split_path = root / "splits.csv"
    row["splits_exists"] = split_path.exists()
    if split_path.exists():
        try:
            splits = pd.read_csv(split_path)
            counts = splits["split"].astype(str).value_counts().to_dict() if "split" in splits.columns else {}
            row["train_count"] = int(counts.pop("train", 0))
            row["val_count"] = int(counts.pop("val", 0))
            row["test_count"] = int(counts.pop("test", 0))
            row["other_split_counts"] = json.dumps(counts, sort_keys=True)
            row["split_overlap_ids"] = _split_overlap_count(splits, "episode_id")
            row["split_overlap_paths"] = _split_overlap_count(splits, "path")
        except Exception as exc:  # pragma: no cover
            errors.append(f"splits read failed: {exc}")
    else:
        row["train_count"] = row["val_count"] = row["test_count"] = 0
        row["other_split_counts"] = "{}"
        row["split_overlap_ids"] = row["split_overlap_paths"] = 0

    stats = _scan_manifest_episodes(root, manifest)
    row.update(stats)
    if stats.get("errors"):
        errors.append(str(stats["errors"]))
    row["errors"] = "; ".join(error for error in errors if error)
    return row


def _scan_manifest_episodes(root: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    missing_files = 0
    q_shapes: set[tuple[int, ...]] = set()
    qdot_shapes: set[tuple[int, ...]] = set()
    action_dims: set[int] = set()
    q_dims: set[int] = set()
    target_keys = 0
    fingertips = 0
    desired_fingertips = 0
    errors: list[str] = []
    if "path" not in manifest.columns:
        return _episode_scan_error("manifest missing path column")
    for rel_path in manifest["path"].astype(str):
        path = root / rel_path
        if not path.exists():
            missing_files += 1
            continue
        try:
            with np.load(path, allow_pickle=False) as npz:
                if "q" in npz:
                    q_shape = tuple(np.asarray(npz["q"]).shape)
                    q_shapes.add(q_shape)
                    if len(q_shape) >= 2:
                        q_dims.add(int(q_shape[1]))
                if "qdot" in npz:
                    qdot_shapes.add(tuple(np.asarray(npz["qdot"]).shape))
                if "actions" in npz:
                    action_shape = tuple(np.asarray(npz["actions"]).shape)
                    if len(action_shape) >= 2:
                        action_dims.add(int(action_shape[1]))
                target_keys += int("target_keys" in npz)
                fingertips += int("fingertips" in npz)
                desired_fingertips += int("desired_fingertips" in npz)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{rel_path}: {exc}")
    return {
        "missing_files": missing_files,
        "q_shape_consistent": len(q_shapes) <= 1,
        "qdot_shape_consistent": len(qdot_shapes) <= 1 and (not q_shapes or not qdot_shapes or q_shapes == qdot_shapes),
        "action_dim_consistent": len(action_dims) <= 1,
        "q_dim_values": ";".join(str(value) for value in sorted(q_dims)),
        "action_dim_values": ";".join(str(value) for value in sorted(action_dims)),
        "target_keys_episodes": target_keys,
        "fingertips_episodes": fingertips,
        "desired_fingertips_episodes": desired_fingertips,
        "errors": "; ".join(errors[:10]),
    }


def _episode_scan_error(message: str) -> dict[str, Any]:
    return {
        "missing_files": "",
        "q_shape_consistent": "",
        "qdot_shape_consistent": "",
        "action_dim_consistent": "",
        "q_dim_values": "",
        "action_dim_values": "",
        "target_keys_episodes": "",
        "fingertips_episodes": "",
        "desired_fingertips_episodes": "",
        "errors": message,
    }


def audit_checkpoint(path: Path) -> dict[str, Any]:
    row = {column: "" for column in CHECKPOINT_COLUMNS}
    row["checkpoint_path"] = str(path)
    row["exists"] = path.exists()
    row["refinement_used"] = _looks_refined(path, {})
    row["is_10_epoch_unrefined_baseline"] = False
    if not path.exists():
        row["errors"] = "missing checkpoint file"
        return row
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover
        row["errors"] = f"torch.load failed: {exc}"
        return row
    if not isinstance(payload, dict):
        row["errors"] = f"unexpected checkpoint payload type: {type(payload).__name__}"
        return row

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    controller = config.get("controller") if isinstance(config.get("controller"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    row["controller_family"] = controller.get("family", "")
    row["controller_type"] = controller.get("type", "")
    row["feature_mode"] = _infer_feature_mode(row["controller_family"] or row["controller_type"])
    row["model_class"] = controller.get("model_module") or controller.get("residual_model") or controller.get("model_type") or ""
    row["input_dim"] = payload.get("input_dim", "")
    row["action_dim"] = payload.get("action_dim", "")
    row["epoch"] = payload.get("epoch", "")
    row["train_loss"] = payload.get("train_loss", "")
    row["val_loss"] = payload.get("val_loss", "")
    row["selection_metric"] = payload.get("selection_metric", "")
    row["dataset_root"] = data.get("dataset_root", "")
    row["split_file"] = _read_summary_value(path, "split_file")
    log_metadata = _read_training_log_metadata(path)
    for key in ("epoch", "train_loss", "val_loss"):
        if row.get(key) in {"", None} and log_metadata.get(key) not in {"", None}:
            row[key] = log_metadata[key]
    if not row["selection_metric"] and row["train_loss"] not in {"", None}:
        row["selection_metric"] = "train_loss"
    if not row["split_file"] and row["dataset_root"]:
        split_file = Path(str(row["dataset_root"])) / "splits.csv"
        row["split_file"] = str(split_file) if split_file.exists() else ""
    row["config_hash"] = _hash_config(config)
    row["refinement_used"] = _looks_refined(path, config)
    row["is_10_epoch_unrefined_baseline"] = _is_10_epoch_unrefined(row)
    return row


def _read_summary_value(checkpoint: Path, key: str) -> Any:
    summary_path = checkpoint.parent.parent / "training_summary.json"
    if not summary_path.exists():
        return ""
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return payload.get(key, "")


def _read_training_log_metadata(checkpoint: Path) -> dict[str, Any]:
    run_dir = checkpoint.parent.parent
    logs_dir = run_dir.parent / "logs"
    if not logs_dir.exists():
        return {}
    pattern = f"{run_dir.name}_*.out"
    best: dict[str, Any] = {}
    for log_path in sorted(logs_dir.glob(pattern)):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            match = re.search(
                r"epoch=(?P<epoch>\d+)\s+train_loss=(?P<train>[0-9.eE+-]+)(?:\s+val_loss=(?P<val>[0-9.eE+-]+))?",
                line,
            )
            if match is None:
                continue
            epoch = int(match.group("epoch"))
            if epoch >= int(best.get("epoch") or -1):
                best = {
                    "epoch": epoch,
                    "train_loss": float(match.group("train")),
                    "val_loss": float(match.group("val")) if match.group("val") is not None else "",
                }
    return best


def _infer_feature_mode(family: Any) -> str:
    value = str(family)
    if value == "key_aware_residual":
        return "key_aware"
    if value == "fingertip_residual":
        return "fingertip_phase"
    if value == "inverse_dynamics":
        return "inverse_dynamics"
    return "tracking" if value else ""


def _hash_config(config: dict[str, Any]) -> str:
    if not config:
        return ""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _looks_refined(path: Path, config: dict[str, Any]) -> bool:
    text = str(path).lower()
    if "refine" in text or "refinement" in text:
        return True
    return bool(config.get("refinement") or config.get("rollout_refinement"))


def _is_10_epoch_unrefined(row: dict[str, Any]) -> bool:
    try:
        epoch = int(row.get("epoch") or -1)
    except (TypeError, ValueError):
        return False
    return epoch == 10 and not bool(row.get("refinement_used"))


def _duplicate_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].astype(str).duplicated(keep=False).sum())


def _split_overlap_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns or "split" not in frame.columns:
        return 0
    counts = frame[[column, "split"]].astype(str).drop_duplicates().groupby(column)["split"].nunique()
    return int((counts > 1).sum())


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_file_manifest(path: Path, dataset_roots: list[Path], checkpoint_paths: list[Path]) -> None:
    files: list[Path] = []
    for root in dataset_roots:
        files.append(root / "manifest.csv")
        split_path = root / "splits.csv"
        if split_path.exists():
            files.append(split_path)
    files.extend(checkpoint_paths)
    path.write_text("\n".join(str(file_path) for file_path in sorted(files)) + "\n", encoding="utf-8")


def _write_dataset_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Etude Dataset and Split Audit",
        "",
        "This report audits manifest coverage, split integrity, file availability, and core tensor shapes.",
        "",
        "| Dataset | Rows | Splits | Train | Val | Test | Overlap IDs | Missing Files | q dims | action dims | Errors |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset_root} | {manifest_rows} | {splits_exists} | {train_count} | {val_count} | "
            "{test_count} | {split_overlap_ids} | {missing_files} | {q_dim_values} | "
            "{action_dim_values} | {errors} |".format(**{key: _md(row.get(key, "")) for key in DATASET_COLUMNS})
        )
    lines.extend(
        [
            "",
            "## Refinement Readiness Checklist",
            "",
            "- Clean train/val/test split with zero train-validation overlap.",
            "- Existing 10-epoch unrefined checkpoint baseline for each controller family.",
            "- Actuated rollout evaluator and short validation trajectory suite.",
            "- Output root, logging, and wandb configuration fixed before launch.",
            "- Success metrics and stopping criteria declared before any GPU refinement job.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checkpoint_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    baselines = [row for row in rows if row.get("is_10_epoch_unrefined_baseline")]
    lines = [
        "# Etude Checkpoint Lineage",
        "",
        "User-provided baseline fact: each controller already has a 10-epoch unrefined checkpoint. "
        "This audit marks files as verified baselines when checkpoint metadata or matching training logs show "
        "epoch 10 and no refinement signal is present.",
        "",
        f"Verified 10-epoch unrefined checkpoint files: {len(baselines)}",
        "",
        "| Checkpoint | Family | Feature Mode | Model | Epoch | Train Loss | Val Loss | Dataset | Split File | "
        "10-epoch unrefined | Errors |",
        "|---|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {checkpoint_path} | {controller_family} | {feature_mode} | {model_class} | {epoch} | "
            "{train_loss} | {val_loss} | {dataset_root} | {split_file} | "
            "{is_10_epoch_unrefined_baseline} | {errors} |".format(
                **{key: _md(row.get(key, "")) for key in CHECKPOINT_COLUMNS}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
