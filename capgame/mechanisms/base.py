"""Common types for capacity mechanisms.

All mechanism ``apply`` functions return a :class:`MechanismOutcome` that
carries (i) per-firm net profits, (ii) the per-firm capacity payment (which
may be zero), and (iii) the per-MWh consumer cost impact. These are the
quantities consumed downstream by the dynamic game and by welfare analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from capgame.game.cournot import CournotEquilibrium


@runtime_checkable
class Mechanism(Protocol):
    """Structural type shared by every capacity-mechanism implementation.

    All four concrete mechanisms (:class:`~capgame.mechanisms.energy_only.EnergyOnly`,
    :class:`~capgame.mechanisms.capacity_payment.CapacityPayment`,
    :class:`~capgame.mechanisms.forward_capacity.ForwardCapacityMarket`,
    :class:`~capgame.mechanisms.reliability_options.ReliabilityOption`) expose
    an ``apply(equilibrium, capacities) -> MechanismOutcome`` method. Code in
    the experiments and UI layers is typed against this Protocol rather than
    the concrete classes so that new mechanisms plug in without edits
    downstream.
    """

    def apply(
        self,
        equilibrium: CournotEquilibrium,
        capacities: Sequence[float],
    ) -> MechanismOutcome: ...


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
