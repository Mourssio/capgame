"""Parameter calibration helpers.

Planned scope (Phase 4):
- Fit inverse-demand (a, b) from observed price-quantity pairs.
- Fit a Markov demand chain from a residual-load time series.
- Fit forced-outage rates from NERC GADS-style reliability data.

Each routine returns the calibrated object consumed by the stochastic /
game layers, and reports a goodness-of-fit summary. For now we ship a
minimal OLS-based demand fit; the rest will follow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from capgame.game.cournot import LinearDemand


@dataclass(frozen=True)
class DemandFit:
    demand: LinearDemand
    r_squared: float
    residual_std: float


def fit_linear_demand(quantities: Sequence[float], prices: Sequence[float]) -> DemandFit:
    """Ordinary-least-squares fit of ``P = a - b * Q``.

    Enforces ``a > 0`` and ``b > 0``; if the raw fit violates either, the
    opposite sign is inverted (a truly upward-sloping demand curve is
    rejected by raising ``ValueError``).
    """
    q = np.asarray(quantities, dtype=float)
    p = np.asarray(prices, dtype=float)
    if q.shape != p.shape or q.ndim != 1:
        raise ValueError("quantities and prices must be 1-D arrays of the same length.")
    if q.size < 3:
        raise ValueError("At least three observations are required for a meaningful fit.")

    X = np.column_stack([np.ones_like(q), q])
    coefs, _residuals, _rank, _sv = np.linalg.lstsq(X, p, rcond=None)
    a_hat, slope = float(coefs[0]), float(coefs[1])
    b_hat = -slope

    if a_hat <= 0 or b_hat <= 0:
        raise ValueError(
            f"Fit produced implausible parameters (a={a_hat:.3f}, b={b_hat:.3f}); "
            "data is inconsistent with a downward-sloping linear demand."
        )

    fitted = a_hat - b_hat * q
    ss_res = float(((p - fitted) ** 2).sum())
    ss_tot = float(((p - p.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    resid_std = float(np.sqrt(ss_res / max(1, p.size - 2)))

    return DemandFit(
        demand=LinearDemand(a=a_hat, b=b_hat),
        r_squared=r2,
        residual_std=resid_std,
    )
