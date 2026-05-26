from __future__ import annotations

import numpy as np
import pytest

from ballade.interpolation import build_micro_q_targets, hermite_interpolate, interpolation_phases, linear_interpolate


def test_linear_interpolation_shape_and_endpoint() -> None:
    q = np.asarray([[0.0, 10.0], [10.0, 20.0], [20.0, 40.0]], dtype=np.float32)
    micro = build_micro_q_targets(q, source_dt=0.05, control_dt=0.005)
    assert micro.shape == (2, 10, 2)
    np.testing.assert_allclose(micro[0, 0], [1.0, 11.0])
    np.testing.assert_allclose(micro[0, -1], q[1])
    np.testing.assert_allclose(micro[1, -1], q[2])


def test_phases_are_endpoint_inclusive() -> None:
    phases = interpolation_phases(0.05, 0.005)
    np.testing.assert_allclose(phases[0], 0.1)
    np.testing.assert_allclose(phases[-1], 1.0)


def test_hermite_endpoint() -> None:
    out = hermite_interpolate(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        1.0,
        0.05,
    )
    np.testing.assert_allclose(out, [1.0])


def test_rejects_non_integer_ratio() -> None:
    with pytest.raises(ValueError):
        build_micro_q_targets(np.zeros((2, 1), dtype=np.float32), source_dt=0.05, control_dt=0.007)


def test_linear_interpolate_scalar_phase() -> None:
    out = linear_interpolate(np.asarray([0.0, 2.0]), np.asarray([2.0, 4.0]), 0.5)
    np.testing.assert_allclose(out, [1.0, 3.0])
