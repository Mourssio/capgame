"""Estimate forced outage rates from the monthly capability reports.

The IESO ``GenOutputCapabilityMonth`` file reports, per hour and per
generator, both the ``Capability`` the plant declared available and the
``Output`` it actually produced. Hours where a thermal plant is
**offline while scheduled to be available** are (approximately) forced
outages. We estimate the fleet forced-outage rate (FOR) by fuel as:

.. math::

    \\hat{\\mathrm{FOR}}_f
    = \\frac{\\#\\{(g, t) : \\mathrm{capability}_{g,t} > 0,
      \\; \\mathrm{output}_{g,t} = 0,\\; \\mathrm{fuel}(g) = f\\}}
      {\\#\\{(g, t) : \\mathrm{capability}_{g,t} > 0,\\; \\mathrm{fuel}(g) = f\\}}

with a Jeffreys prior (``Beta(1/2, 1/2)``) so small samples do not
collapse to 0 or 1.

This is a **proxy**, not the NERC GADS-style FOR: a dispatch-driven
zero (plant is available but the operator didn't call it) is counted
the same as a true forced outage. For relative comparison between
technologies and as an input to adequacy metrics this proxy is
adequate; a true GADS comparison would need outage-reason data that
IESO does not publish in the public stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["OutageEstimate", "estimate_outage_rates"]


@dataclass(frozen=True)
class OutageEstimate:
    """Per-fuel FOR estimate with the raw counts for traceability."""

    fuel: str
    outage_rate: float
    n_hours_available: int
    n_hours_zero_output: int


def estimate_outage_rates(
    capability_df: pd.DataFrame,
    *,
    fuels: tuple[str, ...] = ("NUCLEAR", "GAS", "HYDRO", "WIND", "SOLAR", "BIOFUEL"),
    output_tolerance_mw: float = 1.0,
    prior_successes: float = 0.5,
    prior_failures: float = 0.5,
) -> list[OutageEstimate]:
    """Estimate a per-fuel FOR from a fleet capability DataFrame.

    Parameters
    ----------
    capability_df
        Output of :func:`load_fleet_capability_month` (or a
        concatenation of several months).
    fuels
        Fuel types to estimate. Case-insensitive match against the
        ``fuel_type`` column.
    output_tolerance_mw
        ``output_mw <= output_tolerance_mw`` counts as a zero for the
        proxy. Guards against trivial metering noise around zero.
    prior_successes, prior_failures
        Beta prior. Default ``(1/2, 1/2)`` is Jeffreys (uniform under
        the arcsine transformation), a neutral choice for a bounded
        proportion.

    Notes
    -----
    For solar/wind this proxy is mostly a capacity-factor
    complement and should not be interpreted as a forced outage;
    downstream code in :mod:`capgame.calibration.ontario` uses the
    thermal fuels only.
    """
    required = {"fuel_type", "capability_mw", "output_mw"}
    if not required.issubset(capability_df.columns):
        raise ValueError(f"capability_df must have columns {required}")
    estimates: list[OutageEstimate] = []
    df = capability_df.copy()
    df["fuel_norm"] = df["fuel_type"].astype(str).str.upper()
    for fuel in fuels:
        sub = df[df["fuel_norm"] == fuel.upper()]
        # "Available" = declared positive capability in that hour.
        available = sub[pd.to_numeric(sub["capability_mw"], errors="coerce") > 0]
        n_avail = len(available)
        if n_avail == 0:
            estimates.append(
                OutageEstimate(
                    fuel=fuel,
                    outage_rate=0.0,
                    n_hours_available=0,
                    n_hours_zero_output=0,
                )
            )
            continue
        output = pd.to_numeric(available["output_mw"], errors="coerce").fillna(0.0)
        n_zero = int((output <= output_tolerance_mw).sum())
        rate = (n_zero + prior_successes) / (n_avail + prior_successes + prior_failures)
        estimates.append(
            OutageEstimate(
                fuel=fuel,
                outage_rate=float(min(max(rate, 0.0), 0.999)),
                n_hours_available=n_avail,
                n_hours_zero_output=n_zero,
            )
        )
    return estimates
