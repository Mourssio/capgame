"""Forward capacity market (ISO-NE / PJM / IESO style).

The regulator specifies a downward-sloping procurement curve ``D(rho)``
giving the quantity of capacity the system is willing to buy at price
``rho``. Firms offer capacity at a reservation price and the auction clears
at the intersection of supply and demand.

In the simplest implementation (used here), offers are accepted in ascending
order of reservation price until the procurement curve is exhausted, and the
clearing price is the marginal offer. All accepted offers are paid the
clearing price (pay-as-cleared).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from capgame.game.cournot import CournotEquilibrium
from capgame.mechanisms.base import MechanismOutcome


@dataclass(frozen=True)
class ProcurementCurve:
    """Downward-sloping linear procurement curve D(rho) = max(0, cap_target - k*rho).

    Parameters
    ----------
    cap_target
        Target procurement in MW at zero price (intercept of the demand curve).
    slope
        Slope in MW per $/MW of price. A higher slope makes the curve more
        elastic (the regulator buys less as price rises).
    """

    cap_target: float
    slope: float

    def __post_init__(self) -> None:
        if self.cap_target <= 0:
            raise ValueError(f"cap_target must be > 0, got {self.cap_target}")
        if self.slope <= 0:
            raise ValueError(f"slope must be > 0, got {self.slope}")

    def quantity(self, rho: float) -> float:
        return max(0.0, self.cap_target - self.slope * rho)

    def price(self, quantity: float) -> float:
        return max(0.0, (self.cap_target - quantity) / self.slope)


@dataclass(frozen=True)
class CapacityOffer:
    firm_index: int
    quantity: float
    reservation_price: float

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(f"quantity must be >= 0, got {self.quantity}")
        if self.reservation_price < 0:
            raise ValueError(f"reservation_price must be >= 0, got {self.reservation_price}")


@dataclass(frozen=True)
class AuctionResult:
    clearing_price: float
    accepted_quantities: np.ndarray
    total_payment: float


def clear_auction(
    offers: Sequence[CapacityOffer],
    curve: ProcurementCurve,
    n_firms: int,
) -> AuctionResult:
    """Clear a pay-as-cleared capacity auction against a linear procurement curve.

    Offers are sorted by ascending reservation price; the auctioneer keeps
    accepting quantity until marginal supply crosses demand. The clearing
    price is the last accepted reservation price (or the demand-curve
    intersection, whichever is lower).
    """
    accepted = np.zeros(n_firms, dtype=float)
    sorted_offers = sorted(offers, key=lambda o: o.reservation_price)
    total_cleared = 0.0
    clearing_price = 0.0

    for offer in sorted_offers:
        demand_at_price = curve.quantity(offer.reservation_price)
        slack = demand_at_price - total_cleared
        if slack <= 0:
            break
        take = min(offer.quantity, slack)
        if take > 0:
            accepted[offer.firm_index] += take
            total_cleared += take
            clearing_price = offer.reservation_price

    inferred = curve.price(total_cleared)
    clearing_price = min(clearing_price, inferred) if total_cleared > 0 else 0.0

    total_payment = clearing_price * total_cleared
    return AuctionResult(
        clearing_price=float(clearing_price),
        accepted_quantities=accepted,
        total_payment=float(total_payment),
    )


@dataclass(frozen=True)
class ForwardCapacityMarket:
    """A forward capacity market parameterized by a procurement curve."""

    curve: ProcurementCurve

    def apply(
        self,
        equilibrium: CournotEquilibrium,
        capacities: Sequence[float],
        offers: Sequence[CapacityOffer] | None = None,
    ) -> MechanismOutcome:
        caps = np.asarray(capacities, dtype=float)
        n = caps.size

        if offers is None:
            offers = [
                CapacityOffer(firm_index=i, quantity=float(caps[i]), reservation_price=0.0)
                for i in range(n)
            ]

        auction = clear_auction(offers, self.curve, n_firms=n)

        energy_profits = np.asarray(equilibrium.profits, dtype=float).copy()
        payments = auction.clearing_price * auction.accepted_quantities
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
    curve: ProcurementCurve,
    offers: Sequence[CapacityOffer] | None = None,
) -> MechanismOutcome:
    """Function-form wrapper."""
    return ForwardCapacityMarket(curve=curve).apply(equilibrium, capacities, offers)
