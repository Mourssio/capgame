"""Reliability options (Vazquez-Rivier-Perez-Arriaga, 2002).

Each MW of installed capacity sold under a reliability option receives an
upfront premium ``pi_0`` per period. In exchange, whenever the spot price
exceeds a strike ``K``, the firm refunds ``(P(Q) - K) * x_i`` to consumers.
At ``K -> infinity`` the mechanism collapses to energy-only; at ``K = 0`` it
collapses to a flat capacity payment equal to the premium.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from capgame.game.cournot import CournotEquilibrium
from capgame.mechanisms.base import MechanismOutcome


@dataclass(frozen=True)
class ReliabilityOption:
    """Financial-call reliability option.

    Parameters
    ----------
    premium
        Option premium ``pi_0`` paid to each MW of committed capacity per period
        ($/MW).
    strike_price
        Strike price ``K`` above which the firm refunds the difference ($/MWh).
    coverage
        Fraction of each firm's capacity that is covered by options, in [0, 1].
        A coverage below one allows partial participation.
    """

    premium: float
    strike_price: float
    coverage: float = 1.0

    def __post_init__(self) -> None:
        if self.premium < 0:
            raise ValueError(f"premium must be >= 0, got {self.premium}")
        if self.strike_price < 0:
            raise ValueError(f"strike_price must be >= 0, got {self.strike_price}")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError(f"coverage must be in [0, 1], got {self.coverage}")

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
        covered = self.coverage * caps
        energy_profits = np.asarray(equilibrium.profits, dtype=float).copy()
        payments = self.premium * covered

        spread = max(0.0, equilibrium.price - self.strike_price)
        refunds = spread * covered

        net = energy_profits + payments - refunds
        return MechanismOutcome(
            energy_profits=energy_profits,
            capacity_payments=payments,
            refunds=refunds,
            net_profits=net,
            consumer_cost=float(payments.sum() - refunds.sum()),
        )


def apply(
    equilibrium: CournotEquilibrium,
    capacities: Sequence[float],
    option: ReliabilityOption | None = None,
    *,
    premium: float | None = None,
    strike_price: float | None = None,
    coverage: float = 1.0,
) -> MechanismOutcome:
    """Function-form wrapper accepting either an option or scalar parameters."""
    if option is None:
        if premium is None or strike_price is None:
            raise ValueError("Supply `option=` or both `premium=` and `strike_price=`.")
        option = ReliabilityOption(
            premium=premium, strike_price=strike_price, coverage=coverage
        )
    return option.apply(equilibrium, capacities)
