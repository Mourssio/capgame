"""Unit tests for the missing-money diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.experiments.scenarios import (
    MissingMoneyReport,
    ScenarioConfig,
    missing_money,
    run_scenario,
)
from capgame.game.cournot import Firm, LinearDemand
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly


@pytest.fixture
def small_scenario() -> ScenarioConfig:
    """3-firm static scenario with known closed-form equilibrium."""
    return ScenarioConfig(
        demand=LinearDemand(a=100.0, b=1.0),
        firms=(
            Firm(marginal_cost=10.0, capacity=30.0, fixed_cost=5.0, name="Baseload"),
            Firm(marginal_cost=25.0, capacity=25.0, fixed_cost=3.0, name="Midmerit"),
            Firm(marginal_cost=50.0, capacity=20.0, fixed_cost=10.0, name="Peaker"),
        ),
        mechanism=EnergyOnly(),
        target_reserve_margin=0.15,
        hours_per_period=1.0,
    )


def test_missing_money_returns_correct_shapes(small_scenario: ScenarioConfig) -> None:
    res = run_scenario(small_scenario)
    mm = missing_money(res)
    n = len(small_scenario.firms)
    assert isinstance(mm, MissingMoneyReport)
    assert mm.per_firm_gap_per_mw_year.shape == (n,)
    assert mm.per_firm_annual_net_revenue.shape == (n,)
    assert mm.per_firm_annual_fixed_requirement.shape == (n,)
    assert 0.0 <= mm.fraction_firms_short <= 1.0


def test_missing_money_matches_hand_calculation(small_scenario: ScenarioConfig) -> None:
    """gap_per_year = 8760 * per_period_net_profit - fixed_cost * capacity."""
    res = run_scenario(small_scenario)
    mm = missing_money(res, periods_per_year=8760.0)
    expected_net = np.asarray(res.per_firm_expected_net_profit) * 8760.0
    expected_fixed = np.array([f.fixed_cost * f.capacity for f in small_scenario.firms])
    np.testing.assert_allclose(mm.per_firm_annual_net_revenue, expected_net)
    np.testing.assert_allclose(mm.per_firm_annual_fixed_requirement, expected_fixed)
    expected_gap = expected_net - expected_fixed
    np.testing.assert_allclose(
        mm.per_firm_gap_per_mw_year,
        expected_gap / np.array([f.capacity for f in small_scenario.firms]),
    )


def test_fixed_cost_zero_implies_no_firm_short() -> None:
    """With zero fixed cost, any positive dispatch satisfies break-even."""
    cfg = ScenarioConfig(
        demand=LinearDemand(a=100.0, b=1.0),
        firms=(
            Firm(marginal_cost=10.0, capacity=30.0, fixed_cost=0.0, name="A"),
            Firm(marginal_cost=20.0, capacity=25.0, fixed_cost=0.0, name="B"),
        ),
        mechanism=EnergyOnly(),
    )
    res = run_scenario(cfg)
    mm = missing_money(res)
    assert mm.fraction_firms_short == 0.0
    assert mm.largest_deficit_firm is None
    assert (mm.per_firm_gap_per_mw_year >= 0).all()


def test_enormous_fixed_cost_makes_all_firms_short() -> None:
    cfg = ScenarioConfig(
        demand=LinearDemand(a=100.0, b=1.0),
        firms=(
            Firm(marginal_cost=10.0, capacity=30.0, fixed_cost=1e9, name="A"),
            Firm(marginal_cost=20.0, capacity=25.0, fixed_cost=1e9, name="B"),
        ),
        mechanism=EnergyOnly(),
    )
    res = run_scenario(cfg)
    mm = missing_money(res)
    assert mm.fraction_firms_short == 1.0
    assert mm.largest_deficit_firm in {"A", "B"}
    assert (mm.per_firm_gap_per_mw_year < 0).all()


def test_capacity_payment_can_close_missing_money(small_scenario: ScenarioConfig) -> None:
    """A large-enough capacity payment pushes every firm into the black."""
    res_zero = run_scenario(small_scenario)
    mm_zero = missing_money(res_zero)
    baseline_gap = mm_zero.fleet_gap_per_year

    # rho is $/MW/period; pick a rho big enough to reverse every gap.
    largest_deficit = -mm_zero.per_firm_gap_per_mw_year.min() / 8760 + 1.0
    boosted = ScenarioConfig(
        demand=small_scenario.demand,
        firms=small_scenario.firms,
        mechanism=CapacityPayment(rho=largest_deficit),
    )
    res_boost = run_scenario(boosted)
    mm_boost = missing_money(res_boost)
    assert mm_boost.fleet_gap_per_year > baseline_gap
    assert (mm_boost.per_firm_gap_per_mw_year >= 0.0).all()


def test_periods_per_year_scales_net_revenue_linearly(small_scenario: ScenarioConfig) -> None:
    res = run_scenario(small_scenario)
    mm_a = missing_money(res, periods_per_year=1000.0)
    mm_b = missing_money(res, periods_per_year=2000.0)
    np.testing.assert_allclose(
        mm_b.per_firm_annual_net_revenue, 2 * mm_a.per_firm_annual_net_revenue
    )


def test_periods_per_year_must_be_positive(small_scenario: ScenarioConfig) -> None:
    res = run_scenario(small_scenario)
    with pytest.raises(ValueError, match="periods_per_year"):
        missing_money(res, periods_per_year=0.0)


def test_zero_capacity_firm_does_not_break_gap(small_scenario: ScenarioConfig) -> None:
    cfg = ScenarioConfig(
        demand=small_scenario.demand,
        firms=(*small_scenario.firms, Firm(marginal_cost=99.0, capacity=0.0, name="Idle")),
        mechanism=EnergyOnly(),
    )
    res = run_scenario(cfg)
    mm = missing_money(res)
    assert np.isfinite(mm.per_firm_gap_per_mw_year).all()
    assert mm.per_firm_gap_per_mw_year[-1] == 0.0
