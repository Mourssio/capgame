"""Reliability options (Vazquez-Rivier-Perez-Arriaga, 2002).

Each MW of committed capacity is paid an upfront premium ``pi_0`` per period
(units: $ per MW per period). In exchange, whenever the spot price exceeds
a strike ``K`` (units: $/MWh), the firm refunds ``(P - K) * x_i * H`` to
consumers, where ``H`` is the number of hours the price obtains within the
period.

Limits of interest:

- ``K -> infinity``: refund is identically zero, so the mechanism collapses
  to energy-only.
- ``K = 0``: the firm forfeits *all* spot revenue on covered capacity in
  exchange for the premium. This is strictly stronger than a flat capacity
  payment of ``pi_0`` per MW, because the firm also gives up the energy
  margin above zero.

Units convention (important).
    premium           -- $/MW per period
    strike_price      -- $/MWh
    equilibrium.price -- $/MWh
    hours_per_period  -- hours per "period" in the surrounding model

Payments: ``premium * covered`` is already ``$/period``.
Refunds:  ``(P - K) * covered * hours_per_period`` is also ``$/period``.

The default ``hours_per_period = 1.0`` matches the common stylized setup in
which each Cournot snapshot represents a single-hour peak; change it when
integrating the mechanism into a multi-hour or seasonal block model.
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
        Option premium ``pi_0`` paid per MW of committed capacity per period
        ($ / MW / period).
    strike_price
        Strike price ``K`` above which the firm refunds the spread ($/MWh).
    coverage
        Fraction of each firm's capacity sold under options, in [0, 1].
    hours_per_period
        Length of the period in hours, used to convert the $/MWh spread into
        a $ refund. Defaults to 1.0 (one-hour snapshot).
    """

    premium: float
    strike_price: float
    coverage: float = 1.0
    hours_per_period: float = 1.0

    def __post_init__(self) -> None:
        if self.premium < 0:
            raise ValueError(f"premium must be >= 0, got {self.premium}")
        if self.strike_price < 0:
            raise ValueError(f"strike_price must be >= 0, got {self.strike_price}")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError(f"coverage must be in [0, 1], got {self.coverage}")
        if self.hours_per_period <= 0:
            raise ValueError(f"hours_per_period must be > 0, got {self.hours_per_period}")

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

        # TODO(phase-5a): when forced outages or nodal pricing land, the
        # spread becomes firm-specific (each firm sees its own realized
        # price and availability). Replace the scalar `spread` with a
        # per-firm vector at that point.
        spread = max(0.0, equilibrium.price - self.strike_price)
        refunds = spread * covered * self.hours_per_period

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
    option: ReliabilityOption,
) -> MechanismOutcome:
    """Thin forwarder to :meth:`ReliabilityOption.apply`.

    Kept for symmetry with the other mechanisms' ``apply`` free functions;
    construct a :class:`ReliabilityOption` directly for any non-trivial use.
    """
    return option.apply(equilibrium, capacities)
