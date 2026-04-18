"""Forced outages of generators.

Each generator has a forced-outage rate ``f_i`` in [0, 1). Availability is
Bernoulli(1 - f_i), independent across units. Effective capacity is the sum
of available capacities, a compound random variable used by the adequacy
metrics (LOLE, EUE).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ForcedOutage:
    """Per-unit availability distribution."""

    outage_rate: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.outage_rate < 1.0):
            raise ValueError(f"outage_rate must be in [0, 1), got {self.outage_rate}")

    @property
    def availability(self) -> float:
        return 1.0 - self.outage_rate


def effective_capacity_distribution(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Exact distribution of total available capacity.

    Each of the N units has capacity ``c_i`` with independent availability
    Bernoulli(1 - f_i). This returns the support and probabilities of the
    sum, computed by convolution. Cost is O(2^N) on the support; for large
    fleets use Monte Carlo instead.

    Returns
    -------
    support : (K,) array of achievable capacity totals, sorted ascending
    probs   : (K,) array of probabilities that sum to 1

    Raises
    ------
    ValueError if the unit count exceeds 20 (to guard against explosion).
    """
    caps = np.asarray(capacities, dtype=float)
    fors = np.asarray(outage_rates, dtype=float)
    if caps.shape != fors.shape:
        raise ValueError("capacities and outage_rates must have the same shape.")
    n = caps.size
    if n == 0:
        return np.array([0.0]), np.array([1.0])
    if n > 20:
        raise ValueError(
            f"Exact convolution supports at most 20 units, got {n}. "
            "Use Monte Carlo sampling instead."
        )
    if np.any((fors < 0.0) | (fors >= 1.0)):
        raise ValueError("outage rates must be in [0, 1).")

    support: dict[float, float] = {0.0: 1.0}
    for c, f in zip(caps, fors, strict=True):
        new_support: dict[float, float] = {}
        p_up = 1.0 - f
        p_down = f
        for s, p in support.items():
            if p_up > 0.0:
                key_up = s + c
                new_support[key_up] = new_support.get(key_up, 0.0) + p * p_up
            if p_down > 0.0:
                new_support[s] = new_support.get(s, 0.0) + p * p_down
        support = new_support

    xs = np.array(sorted(support.keys()))
    ps = np.array([support[x] for x in xs])
    return xs, ps


def sample_effective_capacity(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Monte Carlo draws of total available capacity."""
    rng = rng if rng is not None else np.random.default_rng()
    caps = np.asarray(capacities, dtype=float)
    fors = np.asarray(outage_rates, dtype=float)
    up_probs = 1.0 - fors
    draws = rng.binomial(n=1, p=up_probs, size=(n_samples, caps.size))
    return draws @ caps
