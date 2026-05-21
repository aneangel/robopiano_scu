from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from impromptu.anchor_selection import select_ik_anchor_frames  # noqa: E402
from impromptu.config import ImpromptuConfig  # noqa: E402


def test_anchor_policy_includes_required_frames_and_sorted_unique_int64() -> None:
    targets = np.full((12, 10, 3), np.nan, dtype=np.float32)
    weights = np.zeros((12, 10), dtype=np.float32)
    targets[3:7, 0] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    weights[3:7, 0] = 1.0
    targets[6, 0] = np.asarray([1.0, 2.0, 2.9], dtype=np.float32)

    anchors = select_ik_anchor_frames(
        fingertip_targets=targets,
        fingertip_weights=weights,
        waypoint_frames_dense=np.asarray([3, 9], dtype=np.int64),
        config=ImpromptuConfig(anchor_stride=2),
    )

    assert anchors.dtype == np.int64
    assert np.array_equal(anchors, np.unique(anchors))
    assert 0 in anchors
    assert 11 in anchors
    assert 3 in anchors
    assert 9 in anchors
    assert 2 in anchors and 3 in anchors
    assert 6 in anchors


def test_smooth_target_motion_does_not_anchor_every_frame() -> None:
    total = 100
    targets = np.full((total, 10, 3), np.nan, dtype=np.float32)
    weights = np.zeros((total, 10), dtype=np.float32)
    targets[:, 0, 0] = np.linspace(0.0, 0.05, total, dtype=np.float32)
    targets[:, 0, 1] = 0.0
    targets[:, 0, 2] = 0.0
    weights[:, 0] = 1.0

    anchors = select_ik_anchor_frames(
        fingertip_targets=targets,
        fingertip_weights=weights,
        waypoint_frames_dense=np.asarray([0, total - 1], dtype=np.int64),
        config=ImpromptuConfig(anchor_stride=25, include_midpoint_anchors=False),
    )

    assert anchors.size < 15
    assert anchors.size < total // 2
