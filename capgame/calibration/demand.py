"""Calibrate a linear inverse demand curve from hourly data.

We target the affine model the Cournot solver consumes:

.. math::

    P(Q) = a - b Q,  \\qquad a > 0, \\; b > 0.

The **economics** of Ontario's public data force a choice here. Raw OLS
on hourly ``(Q_t, P_t)`` equilibrium pairs does **not** identify the
demand curve: both schedules shift each hour, and in practice the
*supply* curve shifts more, so OLS traces out a supply-side slope
(often with the wrong sign for demand). This is a textbook simultaneity
problem; see e.g. Wolak (2003) or Wang & Adams (2019).

We therefore prefer an **elasticity-based point calibration**:

1.  Pick a reference ``(P_ref, Q_ref)`` — the sample mean is a sensible
    default.
2.  Pick a short-run price elasticity of demand ``epsilon < 0`` from the
    literature. Ontario estimates cluster around ``-0.05`` to ``-0.2``
    for the hourly horizon; ``-0.1`` is a standard baseline.
3.  Solve

    .. math::
        b = -\\frac{1}{\\epsilon} \\cdot \\frac{P_{\\mathrm{ref}}}{Q_{\\mathrm{ref}}},
        \\qquad a = P_{\\mathrm{ref}} + b\\, Q_{\\mathrm{ref}}.

This is exact for the linear curve that passes through the reference
point with the specified elasticity.

The naive OLS estimate is still computed and returned as a
**diagnostic** (along with its sign), so the caller can see when the
data are too endogenous for an observational fit and explain the
discrepancy in the paper's methodology section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from capgame.game.cournot import LinearDemand

__all__ = ["DemandFit", "calibrate_demand_from_elasticity", "fit_linear_demand", "ols_slope"]


@dataclass(frozen=True)
class DemandFit:
    """Calibrated linear demand plus provenance.

    Attributes
    ----------
    demand
        The calibrated :class:`LinearDemand`.
    a, b
        Intercept and slope (redundant with ``demand``, exposed for
        convenience in reports).
    method
        Which estimator produced the coefficients: ``"elasticity"`` (the
        recommended path) or ``"ols"`` (raw OLS, only if the sign came
        out right and the caller explicitly opted in).
    elasticity
        Point elasticity used (``None`` for OLS).
    reference_price, reference_quantity
        Means of the price and quantity input.
    ols_slope
        Raw OLS slope ``dP/dQ`` from the data, kept as a diagnostic. A
        positive value here (i.e. ``b_ols = -ols_slope < 0``) means the
        data are supply-dominated and the elasticity path is correct.
    n_observations
        Observations used after masking.
    """

    demand: LinearDemand
    a: float
    b: float
    method: Literal["elasticity", "ols"]
    elasticity: float | None
    reference_price: float
    reference_quantity: float
    ols_slope: float
    n_observations: int


def _clean(
    q: np.ndarray,
    p: np.ndarray,
    pmin: float,
    pmax: float,
    min_obs: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    if q.shape != p.shape or q.ndim != 1:
        raise ValueError("quantity_mw and price_per_mwh must be 1-D and the same length.")
    mask = np.isfinite(q) & np.isfinite(p) & (p >= pmin) & (p <= pmax) & (q > 0)
    if mask.sum() < min_obs:
        raise ValueError(f"Too few valid observations after masking: {mask.sum()} < {min_obs}.")
    return q[mask], p[mask]


def ols_slope(quantity_mw: np.ndarray, price_per_mwh: np.ndarray) -> float:
    """Return the raw OLS slope ``dP/dQ`` from a hourly ``(Q, P)`` regression.

    Exposed separately for diagnostics: a positive value indicates that
    observational data are identifying the (upward-sloping) supply
    curve, not the demand curve.
    """
    q, p = _clean(quantity_mw, price_per_mwh, pmin=-np.inf, pmax=np.inf)
    x = np.column_stack([np.ones_like(q), q])
    coef, *_ = np.linalg.lstsq(x, p, rcond=None)
    return float(coef[1])


def calibrate_demand_from_elasticity(
    reference_price: float,
    reference_quantity: float,
    elasticity: float = -0.1,
) -> LinearDemand:
    """Construct ``P(Q) = a - bQ`` that passes through a reference point
    with a specified point elasticity.

    Parameters
    ----------
    reference_price
        $/MWh. Must be > 0.
    reference_quantity
        MW. Must be > 0.
    elasticity
        Short-run price elasticity of demand. Must be strictly negative.

    Returns
    -------
    LinearDemand
        Such that ``P(Q_ref) = P_ref`` and ``dQ/dP x P/Q|_{ref} =
        elasticity``.
    """
    if reference_price <= 0 or reference_quantity <= 0:
        raise ValueError("reference_price and reference_quantity must be positive.")
    if elasticity >= 0:
        raise ValueError(f"Price elasticity of demand must be < 0, got {elasticity}.")
    b = -1.0 / elasticity * reference_price / reference_quantity
    a = reference_price + b * reference_quantity
    return LinearDemand(a=a, b=b)


def fit_linear_demand(
    quantity_mw: np.ndarray,
    price_per_mwh: np.ndarray,
    *,
    elasticity: float = -0.1,
    price_floor: float = 1.0,
    price_cap: float = 2000.0,
    prefer: Literal["elasticity", "ols"] = "elasticity",
) -> DemandFit:
    """Calibrate a linear inverse demand curve.

    The default (``prefer="elasticity"``) anchors the curve at the
    sample means and uses the supplied elasticity — the approach
    recommended in the module docstring.

    Setting ``prefer="ols"`` falls back to elasticity if the OLS slope
    comes out with the wrong sign (``dP/dQ >= 0``), which is the
    typical case for Ontario hourly data. Genuine OLS output is only
    returned when the data actually identify a downward-sloping curve
    (unusual, but possible on aggregated or instrumented series).
    """
    q, p = _clean(quantity_mw, price_per_mwh, pmin=price_floor, pmax=price_cap)
    p_ref = float(p.mean())
    q_ref = float(q.mean())
    slope = ols_slope(q, p)
    method: Literal["elasticity", "ols"] = "elasticity"
    if prefer == "ols" and slope < 0.0:
        a = float(p.mean() - slope * q.mean())
        b = float(-slope)
        method = "ols"
        demand = LinearDemand(a=a, b=b)
        used_elasticity: float | None = None
    else:
        demand = calibrate_demand_from_elasticity(p_ref, q_ref, elasticity=elasticity)
        a = demand.a
        b = demand.b
        used_elasticity = elasticity

    return DemandFit(
        demand=demand,
        a=a,
        b=b,
        method=method,
        elasticity=used_elasticity,
        reference_price=p_ref,
        reference_quantity=q_ref,
        ols_slope=slope,
        n_observations=int(q.size),
    )
