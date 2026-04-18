"""Tests for capacity mechanisms (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.game.cournot import Firm, LinearDemand, solve_constrained
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from capgame.mechanisms.forward_capacity import (
    CapacityOffer,
    ForwardCapacityMarket,
    ProcurementCurve,
    clear_auction,
)


@pytest.fixture
def demand() -> LinearDemand:
    return LinearDemand(a=100.0, b=1.0)


@pytest.fixture
def firms() -> list[Firm]:
    return [
        Firm(marginal_cost=10.0, capacity=30.0),
        Firm(marginal_cost=20.0, capacity=20.0),
        Firm(marginal_cost=40.0, capacity=15.0),
    ]


class TestEnergyOnly:
    def test_returns_cournot_profits_unchanged(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        out = EnergyOnly().apply(eq, [f.capacity for f in firms])
        np.testing.assert_allclose(out.net_profits, eq.profits)
        assert out.consumer_cost == 0.0
        assert np.all(out.capacity_payments == 0.0)


class TestCapacityPayment:
    def test_adds_rho_times_capacity(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        mech = CapacityPayment(rho=5.0)
        out = mech.apply(eq, caps)
        expected = np.asarray(eq.profits) + 5.0 * np.asarray(caps)
        np.testing.assert_allclose(out.net_profits, expected)
        assert out.consumer_cost == pytest.approx(5.0 * sum(caps))

    def test_rho_zero_reduces_to_energy_only(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        out = CapacityPayment(rho=0.0).apply(eq, caps)
        np.testing.assert_allclose(out.net_profits, eq.profits)

    def test_negative_rho_rejected(self) -> None:
        with pytest.raises(ValueError):
            CapacityPayment(rho=-1.0)

    def test_shape_mismatch_rejected(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        with pytest.raises(ValueError):
            CapacityPayment(rho=5.0).apply(eq, [10.0, 10.0])


class TestForwardCapacityMarket:
    def test_procurement_curve_properties(self) -> None:
        curve = ProcurementCurve(cap_target=100.0, slope=2.0)
        assert curve.quantity(0.0) == 100.0
        assert curve.quantity(50.0) == 0.0
        assert curve.quantity(100.0) == 0.0

    def test_clear_auction_single_firm(self) -> None:
        curve = ProcurementCurve(cap_target=100.0, slope=1.0)
        offers = [CapacityOffer(firm_index=0, quantity=50.0, reservation_price=20.0)]
        result = clear_auction(offers, curve, n_firms=1)
        assert result.accepted_quantities[0] == pytest.approx(50.0)
        assert result.clearing_price <= 20.0

    def test_clear_auction_merit_order(self) -> None:
        curve = ProcurementCurve(cap_target=100.0, slope=1.0)
        offers = [
            CapacityOffer(firm_index=0, quantity=40.0, reservation_price=5.0),
            CapacityOffer(firm_index=1, quantity=40.0, reservation_price=15.0),
            CapacityOffer(firm_index=2, quantity=40.0, reservation_price=25.0),
        ]
        result = clear_auction(offers, curve, n_firms=3)
        assert result.accepted_quantities[0] == pytest.approx(40.0)
        assert result.accepted_quantities[1] == pytest.approx(40.0)
        assert result.accepted_quantities[2] >= 0.0

    def test_no_offers_yields_zero(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        curve = ProcurementCurve(cap_target=100.0, slope=1.0)
        mech = ForwardCapacityMarket(curve=curve)
        out = mech.apply(eq, [f.capacity for f in firms], offers=[])
        assert out.consumer_cost == 0.0
        assert np.all(out.capacity_payments == 0.0)


# Reliability-option tests live in tests/test_reliability_options.py.
