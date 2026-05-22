from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SONG_KEY = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0_0"
DEFAULT_ENVIRONMENT_NAME = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
DEFAULT_RP1M_ROOT = Path(
    os.environ.get("RP1M_300_ROOT", "/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr")
)
DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Fugue")
DEFAULT_DT = 0.05

HAND_QPOS_DIM = 46
ACTION_DIM = 39
GOAL_DIM = 88
FINGERTIP_DIM = 30
