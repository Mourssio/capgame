"""Forced outages of generators.

Each generator has a forced-outage rate ``f_i`` in [0, 1). Availability is
Bernoulli(1 - f_i), independent across units. Effective capacity is the sum
of available capacities, a compound random variable whose distribution is
the capacity-outage probability table (COPT) used by the adequacy metrics
(LOLE, EUE).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

MAX_UNITS_EXACT = 20
"""Hard cap on the unit count for exact convolution.

The support of the COPT grows as ``O(2^N)`` in the worst case (all units
distinct), so exact enumeration becomes both memory-bound and slow well
before it becomes incorrect. Fleets larger than this must use
:func:`sample_effective_capacity` for a Monte-Carlo estimate.
"""


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
    """Exact distribution of total available capacity (the COPT).

    Builds the compound Bernoulli distribution by online convolution. The
    support is merged across duplicate totals, so identical-unit fleets
    stay compact rather than blowing up to ``2^N``.

    Returns
    -------
    support
        Achievable capacity totals, sorted ascending.
    probs
        Probability mass at each point on ``support``; sums to 1.

    Raises
    ------
    ValueError
        If inputs are malformed or the unit count exceeds
        :data:`MAX_UNITS_EXACT`. For larger fleets use
        :func:`sample_effective_capacity`.
    """
    caps = np.asarray(capacities, dtype=float)
    fors = np.asarray(outage_rates, dtype=float)
    if caps.shape != fors.shape:
        raise ValueError("capacities and outage_rates must have the same shape.")
    if np.any(caps < 0):
        raise ValueError("capacities must be non-negative.")
    if np.any((fors < 0.0) | (fors >= 1.0)):
        raise ValueError("outage rates must lie in [0, 1).")

    n = caps.size
    if n == 0:
        return np.array([0.0]), np.array([1.0])
    if n > MAX_UNITS_EXACT:
        raise ValueError(
            f"Exact convolution supports at most {MAX_UNITS_EXACT} units, got {n}. "
            "Use sample_effective_capacity() for a Monte-Carlo estimate."
        )

    support: dict[float, float] = {0.0: 1.0}
    for c, f in zip(caps, fors, strict=True):
        p_up = 1.0 - f
        next_support: dict[float, float] = {}
        for s, p in support.items():
            next_support[s + c] = next_support.get(s + c, 0.0) + p * p_up
            next_support[s] = next_support.get(s, 0.0) + p * f
        support = next_support

    xs = np.fromiter(sorted(support.keys()), dtype=float, count=len(support))
    ps = np.array([support[x] for x in xs])
    return xs, ps


def sample_effective_capacity(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Monte Carlo draws of total available capacity.

    ``(1 - outage_rate)`` is treated as the per-unit independent availability
    probability; the sum of available nameplate capacities is returned for
    each of the ``n_samples`` draws.
    """
    rng = rng if rng is not None else np.random.default_rng()
    caps = np.asarray(capacities, dtype=float)
    fors = np.asarray(outage_rates, dtype=float)
    up_probs = 1.0 - fors
    draws = rng.binomial(n=1, p=up_probs, size=(n_samples, caps.size))
    return draws @ caps
