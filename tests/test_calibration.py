"""Tests for calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.optimization.calibration import fit_linear_demand


def test_recovers_known_parameters() -> None:
    a_true, b_true = 100.0, 2.0
    q = np.linspace(5.0, 40.0, 20)
    p = a_true - b_true * q
    fit = fit_linear_demand(q, p)
    assert fit.demand.a == pytest.approx(a_true, rel=1e-6)
    assert fit.demand.b == pytest.approx(b_true, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0)


def test_rejects_upward_sloping_data() -> None:
    q = np.linspace(1.0, 10.0, 20)
    p = 10.0 + 2.0 * q
    with pytest.raises(ValueError):
        fit_linear_demand(q, p)


def test_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError):
        fit_linear_demand([1.0, 2.0], [5.0, 4.0])


def test_noisy_fit_has_high_r_squared_when_signal_strong() -> None:
    rng = np.random.default_rng(1)
    q = np.linspace(5.0, 40.0, 50)
    p = 100.0 - 2.0 * q + rng.normal(scale=0.5, size=q.size)
    fit = fit_linear_demand(q, p)
    assert fit.r_squared > 0.99
