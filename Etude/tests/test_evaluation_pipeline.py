from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from create_dataset_split import create_split
from rank_checkpoints import rank_checkpoints


def test_split_generation_creates_all_splits() -> None:
    manifest = pd.DataFrame(
        {
            "episode_id": list(range(10)),
            "path": [f"episodes/episode_{index:06d}.npz" for index in range(10)],
            "source": [f"song_{index // 2}" for index in range(10)],
            "timesteps": [4] * 10,
        }
    )
    splits = create_split(manifest, seed=123)

    assert set(splits["split"]) == {"train", "val", "test"}
    assert splits["split"].value_counts().sum() == 10
    assert create_split(manifest, seed=123)["split"].tolist() == splits["split"].tolist()


def test_dataset_split_filtering(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from etude.data.rp1m_tracking_dataset import RP1MTrackingDataset

    episodes = tmp_path / "episodes"
    episodes.mkdir()
    rows = []
    split_rows = []
    for episode_id, split in enumerate(("train", "val", "test")):
        path = episodes / f"episode_{episode_id:06d}.npz"
        q = np.full((3, 2), episode_id, dtype=np.float32)
        np.savez_compressed(
            path,
            q=q,
            qdot=np.zeros_like(q),
            q_ref=q,
            qdot_ref=np.zeros_like(q),
            actions=np.zeros((3, 1), dtype=np.float32),
            target_keys=np.zeros((3, 88), dtype=np.float32),
            fingertips=np.zeros((3, 30), dtype=np.float32),
        )
        rel = f"episodes/{path.name}"
        rows.append({"episode_id": episode_id, "path": rel, "source": f"source_{episode_id}", "timesteps": 3})
        split_rows.append({**rows[-1], "split": split})
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    pd.DataFrame(split_rows).to_csv(tmp_path / "splits.csv", index=False)

    dataset = RP1MTrackingDataset(tmp_path, split="test")

    assert len(dataset) == 3
    assert len(dataset.manifest) == 1
    assert dataset.manifest.iloc[0]["episode_id"] == 2


def test_rank_checkpoints_uses_primary_then_tiebreakers(tmp_path: Path) -> None:
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_a.write_text("a", encoding="utf-8")
    ckpt_b.write_text("b", encoding="utf-8")
    _write_aggregate(tmp_path / "eval_a", ckpt_a, event_f1=0.8, missed=2.0)
    _write_aggregate(tmp_path / "eval_b", ckpt_b, event_f1=0.8, missed=1.0)

    rows = rank_checkpoints(tmp_path, primary_metric="piano/event_f1", selection_mode="max")

    assert rows[0]["checkpoint"] == str(ckpt_b)
    assert rows[0]["rank"] == 1


def test_eval_bc_epoch_runs_tiny_model() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from etude.training.bc_trainer import eval_bc_epoch

    model = torch.nn.Linear(2, 1)
    loader = DataLoader(
        [
            {
                "features": torch.zeros(2, 2),
                "actions": torch.zeros(2, 1),
            }
        ],
        batch_size=1,
    )

    result = eval_bc_epoch(model, loader, "cpu")

    assert result.eval_loss >= 0.0


def _write_aggregate(path: Path, checkpoint: Path, *, event_f1: float, missed: float) -> None:
    path.mkdir()
    payload = {
        "checkpoint": str(checkpoint),
        "config": "config.yaml",
        "num_episodes": 2,
        "metrics": {
            "piano/event_f1": {"mean": event_f1, "count": 2},
            "piano/missed_events": {"mean": missed, "count": 2},
            "piano/false_events": {"mean": 0.0, "count": 2},
            "piano/timing_abs_error_mean_s": {"mean": 0.0, "count": 2},
            "fingertip/active_l2_mean": {"mean": 0.0, "count": 2},
            "tracking/joint_pos_rmse": {"mean": 0.0, "count": 2},
            "control/action_clip_rate": {"mean": 0.0, "count": 2},
        },
    }
    (path / "aggregate_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
