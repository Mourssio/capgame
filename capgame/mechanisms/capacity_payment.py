"""Administratively-set capacity payment.

Each MW of installed capacity receives an exogenous payment ``rho`` per
period, regardless of market conditions. This is the simplest non-trivial
capacity mechanism and the one most prone to the well-known over-investment
bias when ``rho`` is set above the long-run marginal scarcity value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from capgame.game.cournot import CournotEquilibrium
from capgame.mechanisms.base import MechanismOutcome


@dataclass(frozen=True)
class CapacityPayment:
    """A flat payment per unit of installed capacity.

    Parameters
    ----------
    rho
        Capacity payment in $/MW per period. Must be non-negative.
    """

    rho: float

    def __post_init__(self) -> None:
        if self.rho < 0:
            raise ValueError(f"rho must be >= 0, got {self.rho}")

    def apply(
        self,
        equilibrium: CournotEquilibrium,
        capacities: Sequence[float],
    ) -> MechanismOutcome:
        caps = np.asarray(capacities, dtype=float)
        if caps.shape != equilibrium.quantities.shape:
            raise ValueError(
                f"capacities shape {caps.shape} does not match equilibrium shape "
                f"{equilibrium.quantities.shape}"
            )
        energy_profits = np.asarray(equilibrium.profits, dtype=float).copy()
        payments = self.rho * caps
        refunds = np.zeros_like(energy_profits)
        net = energy_profits + payments - refunds
        return MechanismOutcome(
            energy_profits=energy_profits,
            capacity_payments=payments,
            refunds=refunds,
            net_profits=net,
            consumer_cost=float(payments.sum()),
        )


def apply(
    equilibrium: CournotEquilibrium,
    capacities: Sequence[float],
    rho: float,
) -> MechanismOutcome:
    """Function-form wrapper."""
    return CapacityPayment(rho=rho).apply(equilibrium, capacities)
