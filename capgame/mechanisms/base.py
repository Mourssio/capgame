"""Common types for capacity mechanisms.

All mechanism ``apply`` functions return a :class:`MechanismOutcome` that
carries (i) per-firm net profits, (ii) the per-firm capacity payment (which
may be zero), and (iii) the per-MWh consumer cost impact. These are the
quantities consumed downstream by the dynamic game and by welfare analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MechanismOutcome:
    """Result of applying a capacity mechanism to a Cournot subgame outcome.

    Attributes
    ----------
    energy_profits
        Per-firm profits from energy sales only, ``(P(Q) - c_i) * q_i``.
    capacity_payments
        Per-firm payments from the capacity mechanism. Zero for energy-only.
    refunds
        Per-firm refunds to consumers (e.g. reliability-option claw-back in
        scarcity). Stored as positive outflows from the firm.
    net_profits
        ``energy_profits + capacity_payments - refunds``.
    consumer_cost
        Aggregate payment from consumers above the energy cost of generation,
        i.e. capacity payments minus refunds.
    """

    energy_profits: npt.NDArray[np.float64]
    capacity_payments: npt.NDArray[np.float64]
    refunds: npt.NDArray[np.float64]
    net_profits: npt.NDArray[np.float64]
    consumer_cost: float
