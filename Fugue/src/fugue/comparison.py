from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fugue.evaluation import evaluate_checkpoint
from fugue.training import save_json


DELTA_DIR_NAMES = ("delta_m1", "delta_0", "delta_p1")


@dataclass(slots=True)
class RunComparisonRecord:
    run_name: str
    run_dir: str
    status: str
    approach: str
    feature_mode: str | None
    oracle: bool
    best_delta: int | None
    best_delta_dir: str | None
    val_action_mse: float | None
    val_press_action_mse: float | None
    test_action_mse: float | None
    test_action_l1: float | None
    test_press_action_mse: float | None
    test_press_action_l1: float | None
    test_smoothness_mse: float | None
    test_num_samples: int | None
    epochs_ran: int | None
    checkpoint: str | None
    test_metrics_path: str | None
    alignment_summary_path: str | None
    wandb_mode: str | None
    wandb_group: str | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_run_dirs(paths: list[str | Path]) -> list[Path]:
    """Return candidate model-level run dirs.

    A model-level run dir is normally `.../fugue_model_x_*/model_b_history`,
    not the top-level Slurm run directory. The function accepts either level.
    """
    discovered: list[Path] = []
    for path_like in paths:
        path = Path(path_like).expanduser().resolve()
        if not path.exists():
            continue
        if _is_model_run_dir(path):
            discovered.append(path)
            continue
        for candidate in sorted(path.rglob("alignment_summary.json")):
            discovered.append(candidate.parent)
        for candidate in sorted(path.rglob("training_summary.json")):
            parent = candidate.parent
            if parent.name in DELTA_DIR_NAMES and parent.parent not in discovered:
                discovered.append(parent.parent)
            elif _is_single_delta_dir(parent):
                discovered.append(parent)
    return _dedupe_paths(_collapse_delta_dirs(discovered))


def compare_runs(
    *,
    run_dirs: list[str | Path],
    output_dir: str | Path,
    dataset_root: str | Path | None = None,
    evaluate_missing_test: bool = False,
    batch_size: int = 2048,
    device: str = "cpu",
    include_incomplete: bool = True,
) -> dict[str, Any]:
    candidates = discover_run_dirs(run_dirs)
    records = []
    for run_dir in candidates:
        record = summarize_run(
            run_dir=run_dir,
            dataset_root=dataset_root,
            evaluate_missing_test=evaluate_missing_test,
            batch_size=batch_size,
            device=device,
        )
        if include_incomplete or record.status == "complete":
            records.append(record)
    records = sorted(records, key=_sort_key)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    df = pd.DataFrame(rows)
    csv_path = output_dir / "comparison_table.csv"
    json_path = output_dir / "comparison_table.json"
    report_path = output_dir / "comparison_report.md"
    df.to_csv(csv_path, index=False)
    save_json(json_path, {"records": rows, "recommendation": build_recommendation(rows)})
    report_path.write_text(render_markdown_report(rows, csv_path=csv_path, json_path=json_path), encoding="utf-8")
    return {
        "records": rows,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "report_path": str(report_path),
        "recommendation": build_recommendation(rows),
    }


def summarize_run(
    *,
    run_dir: str | Path,
    dataset_root: str | Path | None = None,
    evaluate_missing_test: bool = False,
    batch_size: int = 2048,
    device: str = "cpu",
) -> RunComparisonRecord:
    run_dir = Path(run_dir).expanduser().resolve()
    alignment_path = run_dir / "alignment_summary.json"
    alignment = _read_json(alignment_path)
    best_delta = _best_delta(alignment)
    best_delta_dir = _best_delta_dir(run_dir, best_delta)
    if best_delta_dir is None and _is_single_delta_dir(run_dir):
        best_delta_dir = run_dir
    run_config = _read_json(best_delta_dir / "run_config.json") if best_delta_dir is not None else {}
    sample_config = dict(run_config.get("sample_config", {}))
    model_config = dict(run_config.get("model_config", {}))
    feature_mode = sample_config.get("feature_mode")
    oracle = bool(sample_config.get("oracle_future_hand_state", False)) or str(feature_mode) == "inverse"
    approach = _approach_name(run_dir, sample_config, model_config)
    training_summary = _read_json(best_delta_dir / "training_summary.json") if best_delta_dir is not None else {}
    checkpoint = _checkpoint_path(best_delta_dir)
    test_metrics_path = _find_test_metrics_path(run_dir=run_dir, best_delta_dir=best_delta_dir)
    if test_metrics_path is None and evaluate_missing_test and checkpoint is not None:
        test_metrics_path = _evaluate_missing_test(
            run_dir=run_dir,
            best_delta_dir=best_delta_dir,
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            run_config=run_config,
            batch_size=batch_size,
            device=device,
        )
    test_metrics = _read_json(test_metrics_path) if test_metrics_path is not None else {}
    val_metrics = _best_val_metrics(alignment=alignment, training_summary=training_summary, best_delta=best_delta)
    status, notes = _status_and_notes(
        alignment_path=alignment_path,
        best_delta_dir=best_delta_dir,
        checkpoint=checkpoint,
        test_metrics=test_metrics,
    )
    wandb_config = dict(run_config.get("config", {}).get("wandb", {}))
    return RunComparisonRecord(
        run_name=run_dir.parent.name if run_dir.name.startswith("model_") else run_dir.name,
        run_dir=str(run_dir),
        status=status,
        approach=approach,
        feature_mode=None if feature_mode is None else str(feature_mode),
        oracle=oracle,
        best_delta=best_delta,
        best_delta_dir=None if best_delta_dir is None else best_delta_dir.name,
        val_action_mse=_float_or_none(val_metrics.get("action_mse")),
        val_press_action_mse=_float_or_none(val_metrics.get("press_action_mse")),
        test_action_mse=_float_or_none(test_metrics.get("action_mse")),
        test_action_l1=_float_or_none(test_metrics.get("action_l1")),
        test_press_action_mse=_float_or_none(test_metrics.get("press_action_mse")),
        test_press_action_l1=_float_or_none(test_metrics.get("press_action_l1")),
        test_smoothness_mse=_float_or_none(test_metrics.get("smoothness_mse")),
        test_num_samples=_int_or_none(test_metrics.get("num_samples")),
        epochs_ran=_int_or_none(training_summary.get("epochs_ran")),
        checkpoint=None if checkpoint is None else str(checkpoint),
        test_metrics_path=None if test_metrics_path is None else str(test_metrics_path),
        alignment_summary_path=str(alignment_path) if alignment_path.exists() else None,
        wandb_mode=None if wandb_config.get("mode") is None else str(wandb_config.get("mode")),
        wandb_group=None if wandb_config.get("group") is None else str(wandb_config.get("group")),
        notes=notes,
    )


def build_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete" and row.get("test_action_mse") is not None]
    deployable = [row for row in complete if not bool(row.get("oracle"))]
    oracle = [row for row in complete if bool(row.get("oracle"))]
    best_deployable = min(deployable, key=_row_metric_key) if deployable else None
    best_oracle = min(oracle, key=_row_metric_key) if oracle else None
    overall = min(complete, key=_row_metric_key) if complete else None
    oracle_gap = None
    if best_deployable is not None and best_oracle is not None:
        oracle_gap = float(best_deployable["test_action_mse"]) - float(best_oracle["test_action_mse"])
    if best_deployable is not None:
        recommendation = (
            f"Use {best_deployable['approach']} as the deployable next-step baseline "
            f"(test action MSE {best_deployable['test_action_mse']:.6g})."
        )
    elif best_oracle is not None:
        recommendation = (
            f"Only oracle-complete runs are available; {best_oracle['approach']} is an upper bound, "
            "not a deployable controller choice."
        )
    else:
        recommendation = "No completed test-evaluated runs are available yet."
    return {
        "recommendation": recommendation,
        "best_overall": overall,
        "best_deployable": best_deployable,
        "best_oracle": best_oracle,
        "oracle_gap_action_mse": oracle_gap,
    }


def render_markdown_report(rows: list[dict[str, Any]], *, csv_path: Path, json_path: Path) -> str:
    recommendation = build_recommendation(rows)
    lines = [
        "# Fugue Run Comparison",
        "",
        recommendation["recommendation"],
        "",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
        "",
        "## Ranked Runs",
        "",
        "| Rank | Status | Approach | Oracle | Delta | Test MSE | Test L1 | Press MSE | Val MSE | Epochs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(rows, key=_row_metric_key)
    for index, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{index} | {row.get('status')} | {row.get('approach')} | {bool(row.get('oracle'))} | "
            f"{_fmt(row.get('best_delta'))} | {_fmt(row.get('test_action_mse'))} | "
            f"{_fmt(row.get('test_action_l1'))} | {_fmt(row.get('test_press_action_mse'))} | "
            f"{_fmt(row.get('val_action_mse'))} | {_fmt(row.get('epochs_ran'))} |"
        )
    incomplete = [row for row in rows if row.get("status") != "complete"]
    if incomplete:
        lines.extend(["", "## Incomplete Or Not Yet Comparable", ""])
        for row in incomplete:
            lines.append(f"- `{row.get('run_dir')}`: {row.get('notes')}")
    if recommendation.get("oracle_gap_action_mse") is not None:
        lines.extend(
            [
                "",
                "## Oracle Gap",
                "",
                "Oracle gap is deployable test MSE minus oracle test MSE. "
                f"Current gap: `{recommendation['oracle_gap_action_mse']:.6g}`.",
            ]
        )
    return "\n".join(lines) + "\n"


def _is_model_run_dir(path: Path) -> bool:
    return (path / "alignment_summary.json").exists() or any((path / name / "training_summary.json").exists() for name in DELTA_DIR_NAMES)


def _is_single_delta_dir(path: Path) -> bool:
    return (path / "training_summary.json").exists() and (path / "checkpoints" / "best.pt").exists()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    output = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output


def _collapse_delta_dirs(paths: list[Path]) -> list[Path]:
    collapsed = []
    for path in paths:
        if path.name in DELTA_DIR_NAMES and _is_model_run_dir(path.parent):
            collapsed.append(path.parent)
        else:
            collapsed.append(path)
    parent_keys = {str(path.resolve()) for path in collapsed}
    output = []
    for path in collapsed:
        if path.name in DELTA_DIR_NAMES and str(path.parent.resolve()) in parent_keys:
            continue
        output.append(path)
    return output


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_delta(alignment: dict[str, Any]) -> int | None:
    best = alignment.get("best")
    if isinstance(best, dict) and best.get("delta") is not None:
        return int(best["delta"])
    return None


def _delta_dir_name(delta: int) -> str:
    if delta < 0:
        return f"delta_m{abs(delta)}"
    if delta > 0:
        return f"delta_p{delta}"
    return "delta_0"


def _best_delta_dir(run_dir: Path, best_delta: int | None) -> Path | None:
    if best_delta is not None:
        candidate = run_dir / _delta_dir_name(int(best_delta))
        if candidate.exists():
            return candidate
    for name in DELTA_DIR_NAMES:
        candidate = run_dir / name
        if (candidate / "training_summary.json").exists():
            return candidate
    return None


def _approach_name(run_dir: Path, sample_config: dict[str, Any], model_config: dict[str, Any]) -> str:
    mode = str(sample_config.get("feature_mode") or "")
    model_type = str(model_config.get("type") or "")
    if bool(sample_config.get("oracle_future_hand_state", False)) or mode == "inverse":
        return "Model C oracle inverse dynamics"
    if mode == "sequence":
        if int(sample_config.get("chunk_horizon", 1)) > 1:
            return "Model E action-chunk transformer"
        if model_type in {"tcn", "temporal_conv"}:
            return "Model D temporal convolution"
        if model_type in {"transformer", "action_chunk_transformer"}:
            return "Model D causal transformer"
        return "Model D sequence"
    if mode == "history":
        if int(sample_config.get("chunk_horizon", 1)) > 1:
            return "Model E history action chunk"
        return "Model B history + goals"
    if mode == "stateless":
        return "Model A stateless MLP"
    return run_dir.name


def _checkpoint_path(best_delta_dir: Path | None) -> Path | None:
    if best_delta_dir is None:
        return None
    checkpoint = best_delta_dir / "checkpoints" / "best.pt"
    return checkpoint if checkpoint.exists() else None


def _find_test_metrics_path(*, run_dir: Path, best_delta_dir: Path | None) -> Path | None:
    if best_delta_dir is not None:
        preferred = run_dir / f"test_{best_delta_dir.name}" / "metrics.json"
        if preferred.exists():
            return preferred
        alt = best_delta_dir / "eval_test" / "metrics.json"
        if alt.exists():
            return alt
    matches = sorted(run_dir.glob("test_*/metrics.json"))
    if matches:
        return matches[0]
    return None


def _evaluate_missing_test(
    *,
    run_dir: Path,
    best_delta_dir: Path | None,
    checkpoint: Path,
    dataset_root: str | Path | None,
    run_config: dict[str, Any],
    batch_size: int,
    device: str,
) -> Path | None:
    if dataset_root is None:
        dataset_root = run_config.get("dataset_root")
    artifact_root = run_config.get("dataset_artifact_root") or str(run_dir / "dataset")
    if dataset_root is None or not Path(str(artifact_root)).exists():
        return None
    label = best_delta_dir.name if best_delta_dir is not None else "single"
    output_dir = run_dir / f"test_{label}"
    metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        dataset_root=dataset_root,
        dataset_artifact_root=artifact_root,
        split="test",
        batch_size=int(batch_size),
        device=device,
    )
    save_json(output_dir / "metrics.json", metrics)
    return output_dir / "metrics.json"


def _best_val_metrics(
    *,
    alignment: dict[str, Any],
    training_summary: dict[str, Any],
    best_delta: int | None,
) -> dict[str, Any]:
    candidates = alignment.get("candidates", [])
    if isinstance(candidates, list) and best_delta is not None:
        for row in candidates:
            if isinstance(row, dict) and int(row.get("delta", 999)) == int(best_delta):
                return {
                    "action_mse": row.get("best_val_action_mse"),
                    "press_action_mse": row.get("best_val_press_action_mse"),
                }
    return {"action_mse": training_summary.get("best_val_action_mse")}


def _status_and_notes(
    *,
    alignment_path: Path,
    best_delta_dir: Path | None,
    checkpoint: Path | None,
    test_metrics: dict[str, Any],
) -> tuple[str, str]:
    missing = []
    if not alignment_path.exists() and best_delta_dir is None:
        missing.append("alignment summary or single-delta training summary")
    if checkpoint is None:
        missing.append("best checkpoint")
    if not test_metrics:
        missing.append("test metrics")
    if missing:
        return "incomplete", "missing " + ", ".join(missing)
    return "complete", ""


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sort_key(record: RunComparisonRecord) -> tuple[int, float, float, str]:
    if record.test_action_mse is not None:
        primary = record.test_action_mse
        tier = 0
    elif record.val_action_mse is not None:
        primary = record.val_action_mse
        tier = 1
    else:
        primary = float("inf")
        tier = 2
    press = record.test_press_action_mse if record.test_press_action_mse is not None else float("inf")
    return (tier, float(primary), float(press), record.run_name)


def _row_metric_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    test = row.get("test_action_mse")
    val = row.get("val_action_mse")
    if test is not None:
        tier = 0
        primary = float(test)
    elif val is not None:
        tier = 1
        primary = float(val)
    else:
        tier = 2
        primary = float("inf")
    press = row.get("test_press_action_mse")
    return (tier, primary, float(press) if press is not None else float("inf"), str(row.get("run_name", "")))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
