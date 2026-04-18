"""Shared input validators for the adequacy layer.

Keeping these in one place ensures that every LOLE / EUE entry point
applies identical rules -- and that changes to the rules (e.g. tighter
probability tolerance, richer demand distributions) propagate uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

PMF_ATOL = 1e-8


def validate_demand_pmf(
    demand: Sequence[tuple[float, float]],
) -> npt.NDArray[np.float64]:
    """Validate an iterable of ``(peak_load, probability)`` pairs.

    Returns the probabilities as a numpy array for convenience. Raises
    ``ValueError`` if the iterable is empty, contains negative probabilities,
    or does not sum to one within ``PMF_ATOL``.
    """
    probs = np.array([p for _, p in demand], dtype=float)
    if probs.size == 0:
        raise ValueError("demand_distribution must be non-empty.")
    if np.any(probs < 0):
        raise ValueError("demand_distribution probabilities must be non-negative.")
    if not np.isclose(probs.sum(), 1.0, atol=PMF_ATOL):
        raise ValueError(
            "demand_distribution probabilities must sum to 1 " f"(got {float(probs.sum())})."
        )
    return probs


def validate_fleet(
    capacities: Sequence[float],
    outage_rates: Sequence[float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate a fleet description and return it as numpy arrays.

    Raises ``ValueError`` if the two sequences differ in length, contain
    negative capacities, or contain outage rates outside ``[0, 1)``.
    """
    caps = np.asarray(list(capacities), dtype=float)
    fors = np.asarray(list(outage_rates), dtype=float)
    if caps.shape != fors.shape:
        raise ValueError(
            f"capacities and outage_rates must have the same length; "
            f"got {caps.shape} and {fors.shape}."
        )
    if np.any(caps < 0):
        raise ValueError("capacities must be non-negative.")
    if np.any((fors < 0.0) | (fors >= 1.0)):
        raise ValueError("outage_rates must lie in [0, 1).")
    return caps, fors


def validate_scaling(periods_per_year: float) -> float:
    """Validate the ``periods_per_year`` scaling factor (must be > 0)."""
    if periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year must be > 0, got {periods_per_year}. "
            "Use 8760.0 for hours-per-year scaling from an hourly snapshot, "
            "365.0 for days-per-year, etc."
        )
    return float(periods_per_year)
