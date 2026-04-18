"""Loss of load expectation (LOLE).

Definition used here: expected number of periods in which available capacity
falls short of demand, over a specified horizon. When demand is itself a
Markov random variable and generators have forced outages, we integrate over
both sources of uncertainty.

    LOLE = sum over demand states s of  P(s) * P(available capacity < d_s)

The caller picks the reporting unit (hours/year, days/year, ...) via the
``periods_per_year`` scaling factor. Set it to ``8760`` to report LOLE in
hours per year from an hourly snapshot, ``365`` for days per year, etc.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from capgame.adequacy._validation import (
    validate_demand_pmf,
    validate_fleet,
    validate_scaling,
)
from capgame.stochastic.outages import effective_capacity_distribution


def loss_of_load_probability(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    peak_load: float,
) -> float:
    """Probability that available capacity falls short of ``peak_load``."""
    if peak_load <= 0:
        raise ValueError(f"peak_load must be > 0, got {peak_load}")
    caps, fors = validate_fleet(capacities, outage_rates)
    support, probs = effective_capacity_distribution(caps, fors)
    short = support < peak_load
    return float(probs[short].sum())


def loss_of_load_expectation(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    demand_distribution: Sequence[tuple[float, float]],
    periods_per_year: float = 1.0,
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
        sum to one.
    periods_per_year
        Scaling from a per-period shortfall probability to an annual count.
        Use ``1.0`` to report the raw probability, ``8760.0`` for hours per
        year from an hourly snapshot, ``365.0`` for days per year.
    """
    validate_demand_pmf(demand_distribution)
    caps, fors = validate_fleet(capacities, outage_rates)
    scaling = validate_scaling(periods_per_year)

    support, cap_probs = effective_capacity_distribution(caps, fors)

    total = 0.0
    for load, prob in demand_distribution:
        short = support < load
        total += prob * float(cap_probs[short].sum())
    return total * scaling


def lole_from_capacity_distribution(
    support: npt.NDArray[np.float64],
    probs: npt.NDArray[np.float64],
    demand_distribution: Sequence[tuple[float, float]],
    periods_per_year: float = 1.0,
) -> float:
    """Lower-level LOLE accepting a pre-computed capacity distribution.

    Useful when the same fleet is assessed against many demand scenarios
    (e.g. inside SDP backward induction): convolving the COPT once and
    reusing it amortizes the ``O(2^N)`` convolution cost.
    """
    validate_demand_pmf(demand_distribution)
    scaling = validate_scaling(periods_per_year)
    total = 0.0
    for load, prob in demand_distribution:
        short = support < load
        total += prob * float(probs[short].sum())
    return total * scaling
