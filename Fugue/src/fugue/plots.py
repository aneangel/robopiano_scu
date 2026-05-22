from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def plot_training_curves(history_csv: str | Path, output_path: str | Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    history = pd.read_csv(history_csv)
    if history.empty:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if "train_action_mse" in history:
        ax.plot(history["epoch"], history["train_action_mse"], label="train")
    if "val_action_mse" in history:
        ax.plot(history["epoch"], history["val_action_mse"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized action MSE")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_per_dim_mse(metrics: dict[str, Any], output_path: str | Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    values = metrics.get("per_dim_mse")
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(arr.size), arr)
    ax.set_xlabel("action dimension")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
