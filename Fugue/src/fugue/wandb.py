from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_WANDB_ENTITY = "tnguyen31-santa-clara-university"
DEFAULT_WANDB_PROJECT = "robopianist"


def add_wandb_arguments(parser: Any) -> None:
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--wandb-notes", default=None)
    parser.add_argument("--wandb-run-name", default=None)


def apply_wandb_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    wandb_config = dict(config.get("wandb", {}))
    if getattr(args, "wandb_enabled", None) is not None:
        wandb_config["enabled"] = bool(args.wandb_enabled)
    if getattr(args, "wandb_project", None):
        wandb_config["project"] = args.wandb_project
    if getattr(args, "wandb_entity", None):
        wandb_config["entity"] = args.wandb_entity
    if getattr(args, "wandb_mode", None):
        wandb_config["mode"] = args.wandb_mode
    if getattr(args, "wandb_group", None):
        wandb_config["group"] = args.wandb_group
    if getattr(args, "wandb_tags", None) is not None:
        wandb_config["tags"] = [tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()]
    if getattr(args, "wandb_notes", None):
        wandb_config["notes"] = args.wandb_notes
    config["wandb"] = wandb_config
    return config


def _json_ready(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=str))


def _artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "artifact"


class WandbRun:
    """Small W&B wrapper mirroring Sonata's train-time logging pattern."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        run_name: str,
        config_payload: dict[str, Any],
        job_type: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        raw = dict(config or {})
        raw_tags = raw.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
        mode = str(raw.get("mode", "online") or "online").strip() or "online"
        self.config = {
            "enabled": bool(raw.get("enabled", False)),
            "project": str(raw.get("project", DEFAULT_WANDB_PROJECT)),
            "entity": str(raw.get("entity", DEFAULT_WANDB_ENTITY)).strip(),
            "mode": mode,
            "group": _optional_text(raw.get("group")),
            "notes": _optional_text(raw.get("notes")),
            "tags": [str(tag).strip() for tag in raw_tags if str(tag).strip()],
            "dir": raw.get("dir"),
        }
        if tags:
            merged = list(self.config["tags"])
            merged.extend(tag for tag in tags if tag and tag not in merged)
            self.config["tags"] = merged
        self.run = None
        self._wandb = None
        if not self.config["enabled"] or self.config["mode"] == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "W&B logging is enabled for Fugue, but `wandb` is not installed. "
                "Install wandb or rerun with --no-wandb."
            ) from exc
        init_kwargs = {
            "project": self.config["project"],
            "entity": self.config["entity"],
            "group": self.config["group"],
            "tags": self.config["tags"],
            "notes": self.config["notes"],
            "mode": self.config["mode"],
            "name": run_name,
            "config": _json_ready(config_payload),
            "job_type": job_type,
            "reinit": True,
        }
        if self.config["dir"]:
            init_kwargs["dir"] = str(Path(self.config["dir"]).resolve())
        try:
            self.run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to initialize W&B for Fugue. Run `wandb login`, set `WANDB_API_KEY`, "
                "or rerun with --no-wandb."
            ) from exc
        self._wandb = wandb

    @property
    def active(self) -> bool:
        return self.run is not None and self._wandb is not None

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(_json_ready(payload), step=step)

    def summary(self, payload: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in _json_ready(payload).items():
            self.run.summary[key] = value

    def log_artifact_bundle(
        self,
        *,
        artifact_name: str,
        artifact_type: str,
        entries: dict[str, str | Path],
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.run is None or self._wandb is None:
            return
        artifact = self._wandb.Artifact(
            name=_artifact_name(artifact_name),
            type=artifact_type,
            metadata=_json_ready(metadata or {}),
        )
        added = False
        for name, path_like in entries.items():
            path = Path(path_like).resolve()
            if not path.exists():
                continue
            if path.is_dir():
                artifact.add_dir(local_path=str(path), name=name)
            else:
                artifact.add_file(local_path=str(path), name=name)
            added = True
        if added:
            self.run.log_artifact(artifact, aliases=list(aliases or []))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
