"""Tests for the static Cournot solver."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.game.cournot import (
    CournotEquilibrium,
    Firm,
    LinearDemand,
    consumer_surplus,
    solve,
    solve_constrained,
    solve_unconstrained,
)


class TestLinearDemand:
    def test_price_is_clipped_at_zero(self) -> None:
        d = LinearDemand(a=10.0, b=1.0)
        assert d.price(100.0) == 0.0
        assert d.price(5.0) == pytest.approx(5.0)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            LinearDemand(a=-1.0, b=1.0)
        with pytest.raises(ValueError):
            LinearDemand(a=1.0, b=0.0)

    def test_inverse_is_consistent(self) -> None:
        d = LinearDemand(a=100.0, b=2.0)
        Q = 30.0
        assert d.inverse(d.price(Q)) == pytest.approx(Q)


class TestFirm:
    def test_validates_inputs(self) -> None:
        with pytest.raises(ValueError):
            Firm(marginal_cost=-1.0, capacity=10.0)
        with pytest.raises(ValueError):
            Firm(marginal_cost=10.0, capacity=-1.0)
        with pytest.raises(ValueError):
            Firm(marginal_cost=10.0, capacity=10.0, outage_rate=1.0)


class TestUnconstrainedCournot:
    def test_monopoly_matches_textbook(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=1e9)]
        eq = solve_unconstrained(demand, firms)

        expected_q = (100.0 - 10.0) / 2.0
        expected_p = 100.0 - expected_q
        expected_profit = (expected_p - 10.0) * expected_q
        assert eq.quantities[0] == pytest.approx(expected_q)
        assert eq.price == pytest.approx(expected_p)
        assert eq.profits[0] == pytest.approx(expected_profit)
        assert eq.hhi == pytest.approx(10_000.0)

    def test_symmetric_duopoly(self, symmetric_duopoly) -> None:
        demand, firms = symmetric_duopoly
        eq = solve_unconstrained(demand, firms)

        expected_q = (demand.a - firms[0].marginal_cost) / (demand.b * 3.0)
        assert eq.quantities[0] == pytest.approx(expected_q)
        assert eq.quantities[1] == pytest.approx(expected_q)
        assert eq.hhi == pytest.approx(5_000.0)

    def test_asymmetric_triopoly_closed_form(self, asymmetric_triopoly) -> None:
        demand, firms = asymmetric_triopoly
        eq = solve_unconstrained(demand, firms)

        c = np.array([f.marginal_cost for f in firms])
        a, b, N = demand.a, demand.b, len(firms)
        sum_c = c.sum()
        expected = (a - N * c + (sum_c - c)) / (b * (N + 1))
        np.testing.assert_allclose(eq.quantities, expected, rtol=1e-10)

    def test_dropout_of_uncompetitive_firm(self) -> None:
        demand = LinearDemand(a=50.0, b=1.0)
        firms = [
            Firm(marginal_cost=10.0, capacity=1e9),
            Firm(marginal_cost=200.0, capacity=1e9),
        ]
        eq = solve_unconstrained(demand, firms)
        assert eq.quantities[1] == pytest.approx(0.0, abs=1e-9)
        assert eq.quantities[0] > 0


class TestConstrainedCournot:
    def test_matches_unconstrained_when_slack(self, asymmetric_triopoly) -> None:
        demand, firms = asymmetric_triopoly
        unc = solve_unconstrained(demand, firms)
        con = solve_constrained(demand, firms)
        np.testing.assert_allclose(unc.quantities, con.quantities, atol=1e-6)
        assert not con.binding.any()

    def test_binding_capacity_shifts_price(
        self,
        symmetric_duopoly: tuple[LinearDemand, list[Firm]],
        tight_capacity_duopoly: tuple[LinearDemand, list[Firm]],
    ) -> None:
        _, slack_firms = symmetric_duopoly
        demand, tight_firms = tight_capacity_duopoly
        slack = solve_constrained(demand, slack_firms)
        tight = solve_constrained(demand, tight_firms)
        assert tight.price > slack.price
        assert tight.binding.all()

    def test_zero_capacity_firm_produces_nothing(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=50.0), Firm(marginal_cost=20.0, capacity=0.0)]
        eq = solve_constrained(demand, firms)
        assert eq.quantities[1] == pytest.approx(0.0, abs=1e-9)

    def test_converges(self, asymmetric_triopoly) -> None:
        demand, firms = asymmetric_triopoly
        eq = solve_constrained(demand, firms)
        assert eq.converged
        assert eq.iterations < 500


class TestHHIBounds:
    def test_hhi_in_zero_to_ten_thousand(self, asymmetric_triopoly) -> None:
        demand, firms = asymmetric_triopoly
        eq = solve_constrained(demand, firms)
        assert 0 <= eq.hhi <= 10_000

    def test_zero_output_yields_zero_hhi(self) -> None:
        eq = CournotEquilibrium(
            quantities=np.zeros(3),
            price=0.0,
            profits=np.zeros(3),
            binding=np.zeros(3, dtype=bool),
        )
        assert eq.hhi == 0.0


class TestConsumerSurplus:
    def test_positive_under_normal_conditions(self, asymmetric_triopoly) -> None:
        demand, firms = asymmetric_triopoly
        eq = solve_constrained(demand, firms)
        assert consumer_surplus(demand, eq) > 0

    def test_matches_triangle_formula(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=1e9)] * 2
        eq = solve_unconstrained(demand, firms)
        cs = consumer_surplus(demand, eq)
        assert cs == pytest.approx(0.5 * demand.b * eq.total_quantity**2)


class TestSolveDispatch:
    def test_solve_picks_unconstrained_when_safe(self, asymmetric_triopoly) -> None:
        demand, firms = asymmetric_triopoly
        eq = solve(demand, firms)
        assert not eq.binding.any()

    def test_solve_falls_back_to_constrained(self, tight_capacity_duopoly) -> None:
        demand, firms = tight_capacity_duopoly
        eq = solve(demand, firms)
        assert eq.binding.any()
