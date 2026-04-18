"""Tests for the reliability-option mechanism.

These exercise the two limiting cases documented in the module docstring,
a coverage-scaling sanity check, unit validation of ``hours_per_period``,
and a non-arbitrage (risk-neutral) fair-premium test.
"""

from __future__ import annotations

import numpy as np
import pytest

from capgame.game.cournot import Firm, LinearDemand, solve_constrained
from capgame.mechanisms import reliability_options as ro_mod
from capgame.mechanisms.reliability_options import ReliabilityOption, apply


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


class TestValidation:
    def test_negative_premium_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReliabilityOption(premium=-1.0, strike_price=50.0)

    def test_negative_strike_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReliabilityOption(premium=1.0, strike_price=-0.01)

    def test_out_of_range_coverage_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReliabilityOption(premium=1.0, strike_price=1.0, coverage=1.5)
        with pytest.raises(ValueError):
            ReliabilityOption(premium=1.0, strike_price=1.0, coverage=-0.1)

    def test_non_positive_hours_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReliabilityOption(premium=1.0, strike_price=1.0, hours_per_period=0.0)
        with pytest.raises(ValueError):
            ReliabilityOption(premium=1.0, strike_price=1.0, hours_per_period=-1.0)

    def test_shape_mismatch_rejected(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        with pytest.raises(ValueError):
            ReliabilityOption(premium=1.0, strike_price=50.0).apply(eq, [1.0, 2.0])


class TestLimitingCases:
    def test_infinite_strike_collapses_to_energy_only(self, demand, firms) -> None:
        """K -> infinity: no refund path, so only the (zero) premium matters."""
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        out = ReliabilityOption(premium=0.0, strike_price=1e12).apply(eq, caps)
        np.testing.assert_allclose(out.net_profits, eq.profits)
        np.testing.assert_allclose(out.refunds, 0.0)

    def test_zero_strike_is_stronger_than_capacity_payment(self, demand, firms) -> None:
        """At K=0 the firm forfeits all spot revenue on covered capacity.

        Extra profit per covered MW equals ``premium - price * H``, not simply
        ``premium`` (which would be a pure capacity payment). This is the
        precise form of the "stronger than a capacity payment" claim in the
        module docstring.
        """
        eq = solve_constrained(demand, firms)
        caps = np.array([f.capacity for f in firms])
        premium = 15.0
        H = 1.0
        out = ReliabilityOption(premium=premium, strike_price=0.0, hours_per_period=H).apply(
            eq, caps
        )
        extra = out.net_profits - eq.profits
        expected = (premium - eq.price * H) * caps
        np.testing.assert_allclose(extra, expected, atol=1e-9)

    def test_price_below_strike_yields_zero_refunds(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        option = ReliabilityOption(premium=10.0, strike_price=eq.price + 5.0)
        out = option.apply(eq, caps)
        np.testing.assert_allclose(out.refunds, 0.0)
        np.testing.assert_allclose(
            out.net_profits,
            np.asarray(eq.profits) + 10.0 * np.asarray(caps),
        )


class TestCoverage:
    def test_coverage_zero_zeros_everything(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        out = ReliabilityOption(premium=10.0, strike_price=20.0, coverage=0.0).apply(eq, caps)
        np.testing.assert_allclose(out.capacity_payments, 0.0)
        np.testing.assert_allclose(out.refunds, 0.0)
        np.testing.assert_allclose(out.net_profits, eq.profits)

    def test_coverage_one_applies_fully(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = np.array([f.capacity for f in firms])
        out = ReliabilityOption(premium=10.0, strike_price=20.0, coverage=1.0).apply(eq, caps)
        np.testing.assert_allclose(out.capacity_payments, 10.0 * caps)

    def test_coverage_scales_linearly(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        full = ReliabilityOption(premium=10.0, strike_price=20.0, coverage=1.0).apply(eq, caps)
        half = ReliabilityOption(premium=10.0, strike_price=20.0, coverage=0.5).apply(eq, caps)
        np.testing.assert_allclose(full.capacity_payments, 2.0 * half.capacity_payments)
        np.testing.assert_allclose(full.refunds, 2.0 * half.refunds)


class TestUnits:
    def test_hours_per_period_scales_refunds_not_payments(self, demand, firms) -> None:
        """Refund is a $/MWh spread integrated over H hours; payment is already $/period."""
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        strike = max(0.0, eq.price - 1.0)
        short = ReliabilityOption(premium=5.0, strike_price=strike, hours_per_period=1.0).apply(
            eq, caps
        )
        long = ReliabilityOption(premium=5.0, strike_price=strike, hours_per_period=4.0).apply(
            eq, caps
        )
        np.testing.assert_allclose(long.capacity_payments, short.capacity_payments)
        np.testing.assert_allclose(long.refunds, 4.0 * short.refunds)


class TestNoArbitrage:
    """Under risk-neutrality, the fair premium equals E[max(P-K,0) * H].

    If the regulator sets ``premium`` to that expectation, then across the
    demand states the firm's *capacity-side* cashflow (payment minus refund)
    nets to zero in expectation for every firm.
    """

    def test_fair_premium_gives_zero_expected_capacity_cashflow(self) -> None:
        demand_states = [
            (LinearDemand(a=80.0, b=1.0), 0.6),
            (LinearDemand(a=130.0, b=1.0), 0.4),
        ]
        firms = [
            Firm(marginal_cost=10.0, capacity=25.0),
            Firm(marginal_cost=30.0, capacity=25.0),
        ]
        caps = [f.capacity for f in firms]
        strike = 40.0
        H = 2.5

        equilibria = [solve_constrained(d, firms) for d, _ in demand_states]
        probs = np.array([p for _, p in demand_states])
        prices = np.array([eq.price for eq in equilibria])

        fair_premium = float((probs * np.maximum(prices - strike, 0.0)).sum() * H)

        option = ReliabilityOption(premium=fair_premium, strike_price=strike, hours_per_period=H)
        cashflows = np.zeros(len(firms), dtype=float)
        for eq, (_, prob) in zip(equilibria, demand_states, strict=True):
            out = option.apply(eq, caps)
            cashflows += prob * (out.capacity_payments - out.refunds)
        np.testing.assert_allclose(cashflows, 0.0, atol=1e-10)


class TestFreeFunction:
    def test_forwards_to_method(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        option = ReliabilityOption(premium=5.0, strike_price=20.0)
        a = option.apply(eq, caps)
        b = apply(eq, caps, option)
        np.testing.assert_allclose(a.net_profits, b.net_profits)
        np.testing.assert_allclose(a.capacity_payments, b.capacity_payments)
        np.testing.assert_allclose(a.refunds, b.refunds)

    def test_missing_option_is_a_type_error(self, demand, firms) -> None:
        eq = solve_constrained(demand, firms)
        caps = [f.capacity for f in firms]
        with pytest.raises(TypeError):
            ro_mod.apply(eq, caps)  # type: ignore[call-arg]
