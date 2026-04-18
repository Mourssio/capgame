"""Deterministic reserve margin.

The standard one-number adequacy metric used in Khalfallah's baseline:

    reserve_margin = (total_capacity - peak_load) / peak_load
"""

from __future__ import annotations

from collections.abc import Sequence


def reserve_margin(total_capacity: float, peak_load: float) -> float:
    """Compute the planning reserve margin as a fraction of peak load.

    Raises
    ------
    ValueError if ``peak_load`` is non-positive.
    """
    if peak_load <= 0:
        raise ValueError(f"peak_load must be > 0, got {peak_load}")
    return (float(total_capacity) - float(peak_load)) / float(peak_load)


def system_capacity(capacities: Sequence[float]) -> float:
    """Sum of installed capacities; convenience wrapper."""
    return float(sum(capacities))
