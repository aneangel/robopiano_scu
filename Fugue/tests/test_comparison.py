from __future__ import annotations

import json
from pathlib import Path

from fugue.comparison import compare_runs, discover_run_dirs


def test_discover_and_rank_completed_runs(tmp_path: Path) -> None:
    model_b = _write_completed_run(tmp_path, "run_b", "model_b_history", delta=0, test_mse=0.02, oracle=False)
    model_c = _write_completed_run(tmp_path, "run_c", "model_c_oracle_inverse", delta=-1, test_mse=0.01, oracle=True)
    discovered = discover_run_dirs([tmp_path])
    assert model_b in discovered
    assert model_c in discovered

    out = compare_runs(run_dirs=[tmp_path], output_dir=tmp_path / "comparison")
    assert len(out["records"]) == 2
    assert out["recommendation"]["best_oracle"]["approach"] == "Model C oracle inverse dynamics"
    assert out["recommendation"]["best_deployable"]["approach"] == "Model B history + goals"
    assert "deployable next-step baseline" in out["recommendation"]["recommendation"]
    assert Path(out["csv_path"]).exists()
    assert Path(out["report_path"]).exists()


def test_incomplete_run_is_reported(tmp_path: Path) -> None:
    incomplete = tmp_path / "run_pending" / "model_b_history"
    incomplete.mkdir(parents=True)
    (incomplete / "alignment_summary.json").write_text(
        json.dumps({"best": {"delta": 0, "best_val_action_mse": 0.1}, "candidates": []}),
        encoding="utf-8",
    )
    out = compare_runs(run_dirs=[tmp_path], output_dir=tmp_path / "comparison")
    assert out["records"][0]["status"] == "incomplete"
    assert "test metrics" in out["records"][0]["notes"]


def _write_completed_run(tmp_path: Path, run_name: str, model_name: str, *, delta: int, test_mse: float, oracle: bool) -> Path:
    run_dir = tmp_path / run_name / model_name
    delta_dir = run_dir / ("delta_m1" if delta < 0 else "delta_0")
    test_dir = run_dir / f"test_{delta_dir.name}"
    ckpt_dir = delta_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (ckpt_dir / "best.pt").write_bytes(b"fake")
    feature_mode = "inverse" if oracle else "history"
    (delta_dir / "run_config.json").write_text(
        json.dumps(
            {
                "sample_config": {"feature_mode": feature_mode, "chunk_horizon": 1, "oracle_future_hand_state": oracle},
                "config": {"wandb": {"mode": "offline", "group": "Fugue-test"}},
            }
        ),
        encoding="utf-8",
    )
    (delta_dir / "training_summary.json").write_text(
        json.dumps({"best_val_action_mse": test_mse * 2, "epochs_ran": 10}),
        encoding="utf-8",
    )
    (run_dir / "alignment_summary.json").write_text(
        json.dumps({"best": {"delta": delta, "best_val_action_mse": test_mse * 2}, "candidates": [{"delta": delta, "best_val_action_mse": test_mse * 2}]}),
        encoding="utf-8",
    )
    (test_dir / "metrics.json").write_text(
        json.dumps(
            {
                "action_mse": test_mse,
                "action_l1": test_mse + 0.1,
                "press_action_mse": test_mse + 0.01,
                "press_action_l1": test_mse + 0.11,
                "num_samples": 123,
            }
        ),
        encoding="utf-8",
    )
    return run_dir
