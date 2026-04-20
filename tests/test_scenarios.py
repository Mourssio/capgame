"""Tests for the single-entry-point scenario runner."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.experiments.scenarios import (
    ScenarioConfig,
    ScenarioResult,
    run_scenario,
)
from capgame.game.cournot import Firm, LinearDemand
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from capgame.mechanisms.reliability_options import ReliabilityOption
from capgame.stochastic.renewables import simple_two_state_renewables


def _base_firms() -> tuple[Firm, ...]:
    return (
        Firm(marginal_cost=10.0, capacity=30.0, name="A"),
        Firm(marginal_cost=20.0, capacity=30.0, name="B"),
        Firm(marginal_cost=30.0, capacity=30.0, name="C"),
    )


class TestValidation:
    def test_empty_firms_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScenarioConfig(
                demand=LinearDemand(a=100.0, b=1.0),
                firms=(),
                mechanism=EnergyOnly(),
            )

    def test_negative_renewable_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScenarioConfig(
                demand=LinearDemand(a=100.0, b=1.0),
                firms=_base_firms(),
                mechanism=EnergyOnly(),
                wind_capacity_mw=-1.0,
            )

    def test_outage_rates_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            ScenarioConfig(
                demand=LinearDemand(a=100.0, b=1.0),
                firms=_base_firms(),
                mechanism=EnergyOnly(),
                outage_rates=(0.05, 0.05),
            )


class TestDeterministic:
    def test_returns_scenario_result(self) -> None:
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=EnergyOnly(),
        )
        r = run_scenario(cfg)
        assert isinstance(r, ScenarioResult)
        assert len(r.states) == 1
        assert r.states[0].label == "deterministic"
        assert r.states[0].probability == pytest.approx(1.0)

    def test_aggregates_match_single_state(self) -> None:
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=EnergyOnly(),
        )
        r = run_scenario(cfg)
        s = r.states[0]
        assert r.expected_price == pytest.approx(s.equilibrium.price)
        assert r.expected_quantity == pytest.approx(s.equilibrium.total_quantity)

    def test_capacity_payment_raises_producer_surplus(self) -> None:
        firms = _base_firms()
        cfg_ref = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=firms,
            mechanism=EnergyOnly(),
        )
        cfg_pay = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=firms,
            mechanism=CapacityPayment(rho=10.0),
        )
        ps_ref = run_scenario(cfg_ref).expected_producer_surplus
        ps_pay = run_scenario(cfg_pay).expected_producer_surplus
        assert ps_pay > ps_ref


class TestAdequacy:
    def test_no_outage_rates_leaves_metrics_none(self) -> None:
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=EnergyOnly(),
        )
        r = run_scenario(cfg)
        assert r.adequacy.lole_hours_per_year is None
        assert r.adequacy.eue_mwh_per_year is None
        assert r.adequacy.reserve_margin is not None

    def test_with_outage_rates_populates_metrics(self) -> None:
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=EnergyOnly(),
            outage_rates=(0.05, 0.05, 0.05),
        )
        r = run_scenario(cfg)
        assert r.adequacy.lole_hours_per_year is not None
        assert r.adequacy.eue_mwh_per_year is not None
        assert r.adequacy.lole_hours_per_year >= 0.0
        assert r.adequacy.eue_mwh_per_year >= 0.0

    def test_capacity_required_matches_target(self) -> None:
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=EnergyOnly(),
            target_reserve_margin=0.15,
        )
        r = run_scenario(cfg)
        assert r.adequacy.capacity_required_mw == pytest.approx(1.15 * 100.0)


class TestRenewableChain:
    def test_four_states_expected_from_chain(self) -> None:
        chain = simple_two_state_renewables(correlation=0.2)
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=EnergyOnly(),
            renewable_chain=chain,
            wind_capacity_mw=20.0,
            solar_capacity_mw=10.0,
        )
        r = run_scenario(cfg)
        assert len(r.states) == 4
        total_prob = sum(s.probability for s in r.states)
        assert total_prob == pytest.approx(1.0, abs=1e-8)

    def test_more_renewables_reduces_thermal_dispatch(self) -> None:
        firms = _base_firms()
        demand = LinearDemand(a=120.0, b=1.0)
        low = ScenarioConfig(
            demand=demand,
            firms=firms,
            mechanism=EnergyOnly(),
            renewable_chain=simple_two_state_renewables(),
            wind_capacity_mw=0.0,
            solar_capacity_mw=0.0,
        )
        high = ScenarioConfig(
            demand=demand,
            firms=firms,
            mechanism=EnergyOnly(),
            renewable_chain=simple_two_state_renewables(),
            wind_capacity_mw=40.0,
            solar_capacity_mw=20.0,
        )
        r_low = run_scenario(low)
        r_high = run_scenario(high)
        assert r_high.expected_quantity < r_low.expected_quantity


class TestPerFirmAggregates:
    def test_per_firm_shape(self) -> None:
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=EnergyOnly(),
        )
        r = run_scenario(cfg)
        assert isinstance(r.per_firm_expected_quantity, np.ndarray)
        assert r.per_firm_expected_quantity.shape == (3,)
        assert r.per_firm_expected_net_profit.shape == (3,)


class TestReliabilityOptionsPassThrough:
    def test_runs_cleanly(self) -> None:
        cfg = ScenarioConfig(
            demand=LinearDemand(a=100.0, b=1.0),
            firms=_base_firms(),
            mechanism=ReliabilityOption(premium=10.0, strike_price=45.0),
        )
        r = run_scenario(cfg)
        assert np.all(r.per_firm_expected_net_profit > 0.0) or np.all(
            r.per_firm_expected_net_profit >= 0.0
        )
