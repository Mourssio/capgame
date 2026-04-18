"""Expected unserved energy (EUE).

    EUE = sum over demand states s of  P(s) * E[max(d_s - C, 0)]

where ``C`` is available capacity. We compute the inner expectation exactly
against the convolved outage distribution when the unit count permits, and
expose a Monte-Carlo fallback for larger fleets.

See :mod:`capgame.adequacy.lole` for the companion probability metric and
for the definition of ``periods_per_year``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from capgame.adequacy._validation import (
    validate_demand_pmf,
    validate_fleet,
    validate_scaling,
)
from capgame.stochastic.outages import (
    effective_capacity_distribution,
    sample_effective_capacity,
)


def expected_unserved_energy(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    demand_distribution: Sequence[tuple[float, float]],
    periods_per_year: float = 1.0,
) -> float:
    """Exact EUE by convolution over unit availability."""
    validate_demand_pmf(demand_distribution)
    caps, fors = validate_fleet(capacities, outage_rates)
    scaling = validate_scaling(periods_per_year)

    support, cap_probs = effective_capacity_distribution(caps, fors)

    total = 0.0
    for load, prob in demand_distribution:
        shortfall = np.maximum(load - support, 0.0)
        total += prob * float((shortfall * cap_probs).sum())
    return total * scaling


def expected_unserved_energy_monte_carlo(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    demand_distribution: Sequence[tuple[float, float]],
    n_samples: int = 100_000,
    rng: np.random.Generator | None = None,
    periods_per_year: float = 1.0,
) -> float:
    """Monte-Carlo EUE for fleets too large for exact convolution."""
    validate_demand_pmf(demand_distribution)
    caps, fors = validate_fleet(capacities, outage_rates)
    scaling = validate_scaling(periods_per_year)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    rng = rng if rng is not None else np.random.default_rng()
    draws = sample_effective_capacity(caps, fors, n_samples, rng=rng)

    total = 0.0
    for load, prob in demand_distribution:
        shortfall = np.maximum(load - draws, 0.0)
        total += prob * float(shortfall.mean())
    return total * scaling
