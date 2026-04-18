"""Loss of load expectation (LOLE).

Definition used here: expected number of periods in which available capacity
falls short of demand, over a specified horizon. When demand is itself a
Markov random variable and generators have forced outages, we integrate over
both sources of uncertainty.

    LOLE = sum over demand states s of  P(s) * P(available capacity < d_s)

The caller chooses the time unit (hours per year, days per year, etc.) by
scaling the resulting probability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from capgame.stochastic.outages import effective_capacity_distribution


def loss_of_load_probability(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    peak_load: float,
) -> float:
    """Probability that available capacity falls short of ``peak_load``."""
    if peak_load <= 0:
        raise ValueError(f"peak_load must be > 0, got {peak_load}")
    support, probs = effective_capacity_distribution(capacities, outage_rates)
    short = support < peak_load
    return float(probs[short].sum())


def loss_of_load_expectation(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    demand_distribution: Sequence[tuple[float, float]],
    periods_per_unit: float = 1.0,
) -> float:
    """Expected loss-of-load count over one assessment period.

    Parameters
    ----------
    capacities
        Per-unit nameplate capacities in MW.
    outage_rates
        Per-unit forced-outage rates in [0, 1).
    demand_distribution
        Iterable of ``(peak_load_MW, probability)`` pairs. Probabilities must
        sum to 1.
    periods_per_unit
        Scaling from a probability of short-fall per period to a count (e.g.
        ``8760`` to report LOLE in hours per year from an hourly study).
    """
    demand = list(demand_distribution)
    probs = np.array([p for _, p in demand], dtype=float)
    if probs.size == 0:
        raise ValueError("demand_distribution must be non-empty.")
    if not np.isclose(probs.sum(), 1.0, atol=1e-8) or np.any(probs < 0):
        raise ValueError("demand_distribution probabilities must form a pmf.")
    if periods_per_unit <= 0:
        raise ValueError(f"periods_per_unit must be > 0, got {periods_per_unit}")

    support, cap_probs = effective_capacity_distribution(capacities, outage_rates)

    total = 0.0
    for (load, prob), p_demand in zip(demand, probs, strict=True):
        del p_demand
        short = support < load
        p_lolp = float(cap_probs[short].sum())
        total += prob * p_lolp
    return total * periods_per_unit


def lole_from_capacity_distribution(
    support: npt.NDArray[np.float64],
    probs: npt.NDArray[np.float64],
    demand_distribution: Sequence[tuple[float, float]],
    periods_per_unit: float = 1.0,
) -> float:
    """Lower-level LOLE accepting a pre-computed capacity distribution.

    Useful when the same fleet is assessed against many demand scenarios
    and computing the capacity distribution once amortizes convolution cost.
    """
    demand = list(demand_distribution)
    d_probs = np.array([p for _, p in demand], dtype=float)
    if not np.isclose(d_probs.sum(), 1.0, atol=1e-8):
        raise ValueError("demand_distribution probabilities must form a pmf.")
    total = 0.0
    for load, prob in demand:
        short = support < load
        total += prob * float(probs[short].sum())
    return total * periods_per_unit
