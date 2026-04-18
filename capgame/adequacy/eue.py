"""Expected unserved energy (EUE).

    EUE = sum over demand states s of  P(s) * E[max(d_s - C, 0)]

where ``C`` is available capacity. We compute the inner expectation exactly
against the convolved outage distribution when the unit count permits, and
expose a Monte-Carlo fallback for larger fleets.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from capgame.stochastic.outages import (
    effective_capacity_distribution,
    sample_effective_capacity,
)


def expected_unserved_energy(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    demand_distribution: Sequence[tuple[float, float]],
    periods_per_unit: float = 1.0,
) -> float:
    """Exact EUE by convolution over unit availability."""
    probs = np.array([p for _, p in demand_distribution], dtype=float)
    if probs.size == 0:
        raise ValueError("demand_distribution must be non-empty.")
    if not np.isclose(probs.sum(), 1.0, atol=1e-8) or np.any(probs < 0):
        raise ValueError("demand_distribution probabilities must form a pmf.")
    if periods_per_unit <= 0:
        raise ValueError(f"periods_per_unit must be > 0, got {periods_per_unit}")

    support, cap_probs = effective_capacity_distribution(capacities, outage_rates)

    total = 0.0
    for load, prob in demand_distribution:
        shortfall = np.maximum(load - support, 0.0)
        total += prob * float((shortfall * cap_probs).sum())
    return total * periods_per_unit


def expected_unserved_energy_monte_carlo(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    demand_distribution: Sequence[tuple[float, float]],
    n_samples: int = 100_000,
    rng: np.random.Generator | None = None,
    periods_per_unit: float = 1.0,
) -> float:
    """Monte-Carlo EUE for fleets too large for exact convolution."""
    rng = rng if rng is not None else np.random.default_rng()
    draws = sample_effective_capacity(capacities, outage_rates, n_samples, rng=rng)

    total = 0.0
    for load, prob in demand_distribution:
        shortfall = np.maximum(load - draws, 0.0)
        total += prob * float(shortfall.mean())
    return total * periods_per_unit
