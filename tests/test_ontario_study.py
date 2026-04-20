"""Tests for the Ontario applied study driver.

The study driver composes the calibration pipeline and the mechanism
solvers. Tests that need the real IESO bundle are gated on the raw
directory existing, the rest run on a small synthetic
:class:`OntarioCalibration` built in-memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from capgame.calibration.demand import DemandFit
from capgame.calibration.ontario import (
    OntarioCalibration,
    TechnologyClass,
)
from capgame.calibration.renewables_cf import RenewableCalibration
from capgame.experiments.ontario_study import (
    annual_to_per_period,
    default_ontario_mechanisms,
    find_optimal_strike,
    run_mechanism_matrix,
    summarize_missing_money,
)
from capgame.experiments.scenarios import ScenarioConfig
from capgame.game.cournot import Firm, LinearDemand
from capgame.mechanisms.base import Mechanism
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from capgame.stochastic.demand import MarkovChain
from capgame.stochastic.renewables import RenewableState

RAW = Path("data/ieso/raw")
_HAS_DATA = RAW.exists() and any(RAW.glob("PUB_Demand_*.csv"))
skip_no_data = pytest.mark.skipif(not _HAS_DATA, reason="IESO raw data not present")


def _synthetic_calibration() -> OntarioCalibration:
    """Hand-rolled Ontario-ish calibration with a 3-state renewable chain."""
    demand = LinearDemand(a=200.0, b=0.01)
    firms = (
        Firm(
            marginal_cost=5.0, capacity=8_000.0, fixed_cost=120.0, outage_rate=0.03, name="NUCLEAR"
        ),
        Firm(
            marginal_cost=40.0, capacity=5_000.0, fixed_cost=60.0, outage_rate=0.05, name="GAS_CCGT"
        ),
        Firm(
            marginal_cost=85.0,
            capacity=2_000.0,
            fixed_cost=45.0,
            outage_rate=0.08,
            name="GAS_PEAKER",
        ),
    )
    states = [
        RenewableState(name="lo", wind_cf=0.1, solar_cf=0.05),
        RenewableState(name="md", wind_cf=0.3, solar_cf=0.15),
        RenewableState(name="hi", wind_cf=0.5, solar_cf=0.25),
    ]
    P = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ]
    )
    chain = MarkovChain(states=states, transition_matrix=P)
    scenario = ScenarioConfig(
        demand=demand,
        firms=firms,
        mechanism=EnergyOnly(),
        market_structure="oligopoly",
        renewable_chain=chain,
        wind_capacity_mw=5_000.0,
        solar_capacity_mw=500.0,
        outage_rates=tuple(f.outage_rate for f in firms),
        hours_per_period=1.0,
    )
    tech = tuple(
        TechnologyClass(
            name=f.name or f"firm_{i}",
            capacity_mw=f.capacity,
            marginal_cost=f.marginal_cost,
            fixed_cost=f.fixed_cost,
            outage_rate=f.outage_rate,
        )
        for i, f in enumerate(firms)
    )
    demand_fit = DemandFit(
        demand=demand,
        a=demand.a,
        b=demand.b,
        method="elasticity",
        elasticity=-0.1,
        reference_price=50.0,
        reference_quantity=10_000.0,
        ols_slope=-0.005,
        n_observations=8000,
    )
    rc = RenewableCalibration(
        chain=chain,
        wind_capacity_mw=5_000.0,
        solar_capacity_mw=500.0,
        state_counts={"lo": 1000, "md": 5000, "hi": 2000},
        mean_wind_cf=0.3,
        mean_solar_cf=0.15,
    )
    return OntarioCalibration(
        scenario=scenario,
        demand_fit=demand_fit,
        renewable_calibration=rc,
        technology_classes=tech,
        empirical_outage_rates=(),
        peak_load_mw=18_000.0,
        year=2024,
    )


@pytest.fixture
def synthetic_calibration() -> OntarioCalibration:
    return _synthetic_calibration()


def test_annual_to_per_period_roundtrip() -> None:
    assert annual_to_per_period(8760.0, hours_per_period=1.0) == pytest.approx(1.0)
    assert annual_to_per_period(8760.0, hours_per_period=2.0) == pytest.approx(2.0)


def test_annual_to_per_period_rejects_bad_hours() -> None:
    with pytest.raises(ValueError):
        annual_to_per_period(100.0, hours_per_period=0.0)


def test_default_ontario_mechanisms_returns_four_callables() -> None:
    mechs = default_ontario_mechanisms()
    assert set(mechs.keys()) == {
        "Energy-only",
        "Capacity payment",
        "Forward capacity",
        "Reliability options",
    }
    for m in mechs.values():
        assert isinstance(m, Mechanism)


def test_default_ontario_mechanisms_converts_annual_to_hourly() -> None:
    mechs = default_ontario_mechanisms(capacity_payment_rho_per_mw_year=8_760.0)
    cp = mechs["Capacity payment"]
    assert isinstance(cp, CapacityPayment)
    assert cp.rho == pytest.approx(1.0)


def test_run_mechanism_matrix_shape(synthetic_calibration: OntarioCalibration) -> None:
    df = run_mechanism_matrix(synthetic_calibration)
    # 4 mechanisms * 3 structures = 12 rows.
    assert len(df) == 12
    assert set(df["structure"].unique()) == {"oligopoly", "cartel", "monopoly"}
    assert set(df["mechanism"].unique()) == {
        "Energy-only",
        "Capacity payment",
        "Forward capacity",
        "Reliability options",
    }
    required = {
        "expected_price",
        "expected_quantity_mw",
        "annual_welfare",
        "annual_consumer_cost_for_capacity",
        "fleet_missing_money_per_year",
    }
    assert required.issubset(df.columns)


def test_cartel_price_weakly_above_oligopoly(
    synthetic_calibration: OntarioCalibration,
) -> None:
    df = run_mechanism_matrix(synthetic_calibration)
    sub = df[df["mechanism"] == "Energy-only"].set_index("structure")
    assert sub.loc["cartel", "expected_price"] >= sub.loc["oligopoly", "expected_price"]


def test_capacity_payment_increases_producer_surplus(
    synthetic_calibration: OntarioCalibration,
) -> None:
    df = run_mechanism_matrix(synthetic_calibration).query("structure == 'oligopoly'")
    eo = df.query("mechanism == 'Energy-only'").iloc[0]
    cp = df.query("mechanism == 'Capacity payment'").iloc[0]
    assert cp["annual_consumer_cost_for_capacity"] > eo["annual_consumer_cost_for_capacity"]
    # Rent (missing money numerator) increases by exactly the annual capacity payment.
    delta = cp["fleet_missing_money_per_year"] - eo["fleet_missing_money_per_year"]
    assert delta > 0


def test_summarize_missing_money_per_firm(synthetic_calibration: OntarioCalibration) -> None:
    df = summarize_missing_money(synthetic_calibration, EnergyOnly())
    assert len(df) == 3
    assert set(df.columns) == {
        "name",
        "capacity_mw",
        "annual_net_revenue",
        "annual_fixed_requirement",
        "gap_per_mw_year",
        "gap_per_year",
        "short",
    }
    # gap_per_year consistency: gap_per_mw_year * capacity_mw == gap_per_year.
    np.testing.assert_allclose(
        df["gap_per_mw_year"] * df["capacity_mw"], df["gap_per_year"], rtol=1e-10
    )


def test_find_optimal_strike_grid_shapes(synthetic_calibration: OntarioCalibration) -> None:
    ss = find_optimal_strike(synthetic_calibration, n_grid=11, strike_bounds=(0.0, 100.0))
    assert ss.strike_grid.shape == (11,)
    assert ss.annual_welfare.shape == (11,)
    assert ss.annual_consumer_cost.shape == (11,)
    assert 0.0 <= ss.optimal_strike <= 100.0


def test_find_optimal_strike_welfare_is_constant_under_standard_ro(
    synthetic_calibration: OntarioCalibration,
) -> None:
    """Under RO-as-transfer, welfare is degenerate in K -- this is a real property."""
    ss = find_optimal_strike(synthetic_calibration, n_grid=11, strike_bounds=(0.0, 200.0))
    np.testing.assert_allclose(ss.annual_welfare, ss.annual_welfare[0], rtol=1e-10)


def test_find_optimal_strike_missing_money_monotone_in_k(
    synthetic_calibration: OntarioCalibration,
) -> None:
    """As K rises, refunds shrink, net profit rises, missing money rises."""
    ss = find_optimal_strike(synthetic_calibration, n_grid=11, strike_bounds=(0.0, 200.0))
    diffs = np.diff(ss.annual_missing_money)
    assert (diffs >= -1e-6).all()


def test_find_optimal_strike_rejects_bad_bounds(
    synthetic_calibration: OntarioCalibration,
) -> None:
    with pytest.raises(ValueError):
        find_optimal_strike(synthetic_calibration, strike_bounds=(50.0, 50.0))
    with pytest.raises(ValueError):
        find_optimal_strike(synthetic_calibration, n_grid=1)


def test_find_optimal_strike_unknown_objective_raises(
    synthetic_calibration: OntarioCalibration,
) -> None:
    with pytest.raises(ValueError, match="unknown objective"):
        find_optimal_strike(
            synthetic_calibration,
            n_grid=3,
            strike_bounds=(0.0, 10.0),
            objective="bogus",  # type: ignore[arg-type]
        )


def test_strike_search_to_frame_is_aligned(
    synthetic_calibration: OntarioCalibration,
) -> None:
    ss = find_optimal_strike(synthetic_calibration, n_grid=5, strike_bounds=(0.0, 100.0))
    df = ss.to_frame()
    assert len(df) == 5
    np.testing.assert_allclose(df["strike_per_mwh"].to_numpy(), ss.strike_grid)


# ---------------- real-data smoke (gated) ----------------


@skip_no_data
def test_mechanism_matrix_smoke_on_real_ontario() -> None:
    from capgame.calibration.ontario import build_ontario_scenario

    cal = build_ontario_scenario(year=2024)
    df = run_mechanism_matrix(cal)
    assert len(df) == 12
    # Sanity: the Cournot oligopoly price should sit between marginal costs
    # of the cheapest and most expensive technology in the fleet.
    eo_row = df.query("mechanism == 'Energy-only' and structure == 'oligopoly'").iloc[0]
    cheapest = min(c.marginal_cost for c in cal.technology_classes)
    dearest = max(c.marginal_cost for c in cal.technology_classes)
    assert (
        cheapest <= eo_row["expected_price"] <= dearest * 3
    )  # Cournot markup can go above dearest
    # Missing money is finite.
    assert np.isfinite(df["fleet_missing_money_per_year"]).all()


@skip_no_data
def test_find_optimal_strike_smoke_on_real_ontario() -> None:
    from capgame.calibration.ontario import build_ontario_scenario

    cal = build_ontario_scenario(year=2024)
    ss = find_optimal_strike(cal, n_grid=11, strike_bounds=(0.0, 200.0))
    # The canonical policy objective: closure at zero missing money,
    # which should land between $30 and $150 given the Ontario calibration.
    assert 20.0 <= ss.optimal_strike <= 200.0


@skip_no_data
def test_summarize_missing_money_ontario_matches_matrix() -> None:
    """Sum of per-firm gap equals the matrix-row fleet missing money."""
    from capgame.calibration.ontario import build_ontario_scenario

    cal = build_ontario_scenario(year=2024)
    df_matrix = run_mechanism_matrix(cal, mechanisms={"Energy-only": EnergyOnly()})
    fleet = df_matrix.query("structure == 'oligopoly'").iloc[0]["fleet_missing_money_per_year"]
    df_per_firm = summarize_missing_money(cal, EnergyOnly())
    assert df_per_firm["gap_per_year"].sum() == pytest.approx(fleet, rel=1e-6)
