from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(tmp, out)
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_npz(path: str | Path, **payload: Any) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    tmp = out.with_name(f"{out.stem}.tmp{out.suffix}")
    np.savez_compressed(tmp, **payload)
    saved = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    os.replace(saved, out)
    return out


def write_table(df: Any, path_without_suffix: str | Path) -> dict[str, str | None]:
    base = Path(path_without_suffix)
    ensure_dir(base.parent)
    csv_path = base.with_suffix(".csv")
    parquet_path = base.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    parquet_written = None
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_written = str(parquet_path)
    except Exception:
        parquet_written = None
    return {"csv": str(csv_path), "parquet": parquet_written}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
