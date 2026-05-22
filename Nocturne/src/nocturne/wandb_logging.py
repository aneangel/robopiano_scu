from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import json

DEFAULT_WANDB_ENTITY = "tnguyen31-santa-clara-university"
DEFAULT_WANDB_PROJECT = "robopianist"


def init_training_run(
    *,
    output_root: str | Path,
    config: dict[str, Any],
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
):
    """Start a required W&B run for Nocturne training."""
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "Nocturne training requires wandb logging. Install wandb or run in an "
            "environment where it is available before starting training."
        ) from exc

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    resolved_entity = entity if entity else os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)
    resolved_mode = mode if mode else os.environ.get("WANDB_MODE", "online")
    run = wandb.init(
        project=project,
        entity=resolved_entity,
        name=name or output.name,
        group=group,
        notes=notes,
        tags=tags or ["nocturne", "stitched-controller"],
        config=_json_ready(config),
        dir=str(output),
        mode=resolved_mode,
        job_type="nocturne-train-controller",
        reinit=True,
    )
    if run is None:
        raise RuntimeError("wandb.init returned None; Nocturne training requires an active W&B run.")
    return run


def log_metrics(run: Any, metrics: dict[str, Any], *, step: int) -> None:
    run.log(metrics, step=int(step))


def finish_run(run: Any, *, summary: dict[str, Any], files: list[str | Path] | None = None) -> None:
    for key, value in summary.items():
        try:
            run.summary[key] = value
        except Exception:
            pass
    for path in files or []:
        file_path = Path(path)
        if file_path.exists():
            run.save(str(file_path))
    _log_artifact_bundle(run, summary=summary, files=files or [])
    run.finish()


def _json_ready(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=str))


def _artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "artifact"


def _log_artifact_bundle(run: Any, *, summary: dict[str, Any], files: list[str | Path]) -> None:
    try:
        import wandb
    except Exception:
        return
    artifact = wandb.Artifact(
        name=_artifact_name(f"{getattr(run, 'name', 'nocturne')}-controller-training"),
        type="nocturne-controller-training",
        metadata=_json_ready(summary),
    )
    added = False
    for file_path_like in files:
        file_path = Path(file_path_like)
        if not file_path.exists() or file_path.is_dir():
            continue
        artifact.add_file(str(file_path.resolve()), name=file_path.name)
        added = True
    if added:
        run.log_artifact(artifact, aliases=["latest"])
