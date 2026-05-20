from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from etude.experiments import load_experiment_config


def test_bagatelle_refinement_configs_load() -> None:
    for name in ("bagatelle_fingertip_refine.yaml", "bagatelle_keypress_temporal_refine.yaml"):
        config = load_experiment_config(Path("configs/experiments") / name)
        assert "controller" in config
        assert "training" in config


def test_bagatelle_refinement_scripts_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for script_name in ("refine_bagatelle_fingertips.py", "refine_keypress_temporal.py"):
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / script_name), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
