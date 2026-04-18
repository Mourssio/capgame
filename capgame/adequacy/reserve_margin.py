"""Deterministic reserve margin.

The standard one-number adequacy metric used in Khalfallah's baseline:

    reserve_margin = (total_capacity - peak_load) / peak_load

Equivalently, for a target reserve margin ``m`` (a fraction, e.g. ``0.15``
for 15%), the required installed capacity is

    capacity_required = (1 + m) * peak_load
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


def capacity_required(peak_load: float, target_margin: float) -> float:
    """Installed capacity needed to meet ``target_margin`` above ``peak_load``.

    Parameters
    ----------
    peak_load
        Peak demand in MW.
    target_margin
        Reliability target expressed as a fraction (``0.15`` for 15%). May
        be zero; negative values are rejected since they correspond to
        chronic shortages rather than "reserve".
    """
    if peak_load <= 0:
        raise ValueError(f"peak_load must be > 0, got {peak_load}")
    if target_margin < 0:
        raise ValueError(f"target_margin must be >= 0, got {target_margin}")
    return (1.0 + float(target_margin)) * float(peak_load)


def system_capacity(capacities: Sequence[float]) -> float:
    """Sum of installed capacities; convenience wrapper."""
    return float(sum(capacities))
