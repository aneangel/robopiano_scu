from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from impromptu.active_window import crop_active_window  # noqa: E402


def test_active_window_keeps_final_active_seconds_with_padding() -> None:
    keys = np.zeros((100, 88), dtype=np.float32)
    keys[70:95, 40] = 1.0

    crop = crop_active_window(
        keys,
        dt=0.05,
        threshold=0.5,
        active_window_last_s=0.5,
        active_window_preroll_s=0.5,
        active_window_postroll_s=0.25,
    )

    assert crop.start_frame == 85
    assert crop.end_frame == 100
    assert crop.target_keys.shape == (15, 88)
    assert crop.metadata["original_steps"] == 100
    assert crop.metadata["cropped_steps"] == 15


def test_active_window_no_active_frames_is_identity() -> None:
    keys = np.zeros((12, 88), dtype=np.float32)
    crop = crop_active_window(
        keys,
        dt=0.05,
        threshold=0.5,
        active_window_last_s=10.0,
        active_window_preroll_s=0.5,
        active_window_postroll_s=0.25,
    )

    assert crop.start_frame == 0
    assert crop.end_frame == 12
    np.testing.assert_allclose(crop.target_keys, keys)
