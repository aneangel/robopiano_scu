from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic Etude train/val/test splits.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-by-source", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    splits = create_split(
        manifest,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        group_by_source=args.group_by_source,
    )
    output_path = Path(args.output) if args.output else dataset_root / "splits.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_path, index=False)
    counts = splits["split"].value_counts().to_dict()
    print(f"Wrote {len(splits)} rows to {output_path} with counts={counts}")


def create_split(
    manifest: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 7,
    group_by_source: bool = False,
) -> pd.DataFrame:
    _validate_fractions(train_frac, val_frac, test_frac)
    if manifest.empty:
        raise ValueError("manifest.csv is empty")
    rows = manifest.copy()
    if "episode_id" not in rows.columns:
        rows["episode_id"] = np.arange(len(rows), dtype=np.int64)
    if "path" not in rows.columns:
        raise ValueError("manifest.csv must contain a 'path' column")
    if "source" not in rows.columns:
        rows["source"] = ""
    if "timesteps" not in rows.columns:
        rows["timesteps"] = ""

    if group_by_source and _has_usable_sources(rows["source"]):
        group_keys = rows["source"].astype(str).map(_source_group_key).to_numpy()
        split_by_key = _assign_group_splits(
            group_keys,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )
        rows["split"] = [split_by_key[str(source)] for source in group_keys]
    else:
        rows["split"] = _assign_splits(
            len(rows),
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )
    return rows[["episode_id", "path", "source", "timesteps", "split"]]


def _validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    fractions = (train_frac, val_frac, test_frac)
    if any(frac < 0.0 for frac in fractions):
        raise ValueError("split fractions must be non-negative")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1.0")
    if train_frac <= 0.0:
        raise ValueError("train fraction must be positive")


def _has_usable_sources(source: pd.Series) -> bool:
    values = [
        _source_group_key(value)
        for value in source.astype(str).tolist()
        if value and value.lower() != "nan"
    ]
    return len(set(values)) > 1


def _source_group_key(source: str) -> str:
    value = str(source)
    if "[" in value and value.endswith("]"):
        return value.rsplit("[", 1)[0]
    return value


def _assign_group_splits(
    keys: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, str]:
    unique_keys = np.array(sorted(set(str(key) for key in keys)), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_keys))
    labels = _assign_splits(
        len(unique_keys),
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        order=order,
    )
    return {str(unique_keys[index]): labels[index] for index in range(len(unique_keys))}


def _assign_splits(
    count: int,
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
    order: np.ndarray | None = None,
) -> list[str]:
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    permutation = np.asarray(order if order is not None else rng.permutation(count), dtype=np.int64)
    labels = np.empty(count, dtype=object)
    train_count = int(np.floor(count * train_frac))
    val_count = int(np.floor(count * val_frac))
    if count >= 3:
        train_count = max(1, min(train_count, count - 2))
        val_count = max(1, min(val_count, count - train_count - 1))
    elif count == 2:
        train_count = 1
        val_count = 0
    else:
        train_count = 1
        val_count = 0
    test_start = train_count + val_count
    labels[permutation[:train_count]] = "train"
    labels[permutation[train_count:test_start]] = "val"
    labels[permutation[test_start:]] = "test"
    return labels.tolist()


if __name__ == "__main__":
    main()
