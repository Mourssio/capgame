"""Tests for the market-structure solvers used by RQ3."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.game.cournot import Firm, LinearDemand, solve
from capgame.game.market_structure import (
    solve_cartel,
    solve_market,
    solve_monopoly,
)


class TestMonopoly:
    def test_single_firm_uncapped(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=1000.0)]
        eq = solve_monopoly(demand, firms)
        assert eq.total_quantity == pytest.approx((100.0 - 10.0) / (2.0 * 1.0), abs=1e-6)
        assert eq.price == pytest.approx(55.0, abs=1e-6)

    def test_single_firm_binding_cap(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=30.0)]
        eq = solve_monopoly(demand, firms)
        assert eq.total_quantity == pytest.approx(30.0, abs=1e-6)

    def test_monopoly_price_exceeds_cournot(self) -> None:
        demand = LinearDemand(a=120.0, b=1.0)
        firms = [
            Firm(marginal_cost=10.0, capacity=100.0),
            Firm(marginal_cost=15.0, capacity=100.0),
            Firm(marginal_cost=20.0, capacity=100.0),
        ]
        mono = solve_monopoly(demand, firms)
        cournot = solve(demand, firms)
        assert mono.price > cournot.price
        assert mono.total_quantity < cournot.total_quantity

    def test_rejects_empty_fleet(self) -> None:
        with pytest.raises(ValueError):
            solve_monopoly(LinearDemand(a=100.0, b=1.0), [])


class TestCartel:
    def test_merit_order_dispatch(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [
            Firm(marginal_cost=30.0, capacity=10.0),
            Firm(marginal_cost=10.0, capacity=10.0),
            Firm(marginal_cost=20.0, capacity=10.0),
        ]
        eq = solve_cartel(demand, firms)
        # cheapest firm (idx=1) should be dispatched first
        assert eq.quantities[1] >= eq.quantities[2]
        assert eq.quantities[2] >= eq.quantities[0]

    def test_cartel_matches_monopoly_aggregate(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [
            Firm(marginal_cost=15.0, capacity=40.0),
            Firm(marginal_cost=15.0, capacity=40.0),
        ]
        cartel = solve_cartel(demand, firms)
        mono = solve_monopoly(demand, firms)
        assert cartel.total_quantity == pytest.approx(mono.total_quantity, abs=1e-6)
        assert cartel.price == pytest.approx(mono.price, abs=1e-6)


class TestSolveMarket:
    def test_dispatch_by_name(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [
            Firm(marginal_cost=10.0, capacity=30.0),
            Firm(marginal_cost=15.0, capacity=30.0),
        ]
        oligo = solve_market(demand, firms, "oligopoly")
        cartel = solve_market(demand, firms, "cartel")
        mono = solve_market(demand, firms, "monopoly")
        assert oligo.total_quantity > cartel.total_quantity
        assert cartel.total_quantity == pytest.approx(mono.total_quantity, abs=1e-6)

    def test_rejects_unknown_structure(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=20.0)]
        with pytest.raises(ValueError):
            solve_market(demand, firms, "auction")  # type: ignore[arg-type]


class TestEquilibriumShape:
    def test_quantities_respect_capacities(self) -> None:
        demand = LinearDemand(a=200.0, b=0.5)
        firms = [
            Firm(marginal_cost=10.0, capacity=20.0),
            Firm(marginal_cost=20.0, capacity=20.0),
            Firm(marginal_cost=30.0, capacity=20.0),
        ]
        for structure in ("oligopoly", "cartel", "monopoly"):
            eq = solve_market(demand, firms, structure)  # type: ignore[arg-type]
            caps = np.array([f.capacity for f in firms])
            assert np.all(eq.quantities >= -1e-9)
            assert np.all(eq.quantities <= caps + 1e-6)
