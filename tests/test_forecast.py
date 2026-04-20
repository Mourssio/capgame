"""Tests for the forecast package (pathways, trajectory, monte_carlo).

Most tests use a synthetic :class:`OntarioCalibration` (re-used from
``test_ontario_study``) so they don't require the IESO data bundle.
A couple of smoke tests on the real 2024 bundle are gated on the raw
directory existing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from capgame.forecast import (
    CapacityTrajectory,
    FixedCostTrajectory,
    FuelPriceTrajectory,
    MonteCarloConfig,
    Pathway,
    build_trajectory,
    default_ontario_pathway,
    run_mechanism_matrix_trajectory,
    run_monte_carlo,
    run_trajectory,
    summarize_paths,
)
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from tests.test_ontario_study import _synthetic_calibration

RAW = Path("data/ieso/raw")
_HAS_DATA = RAW.exists() and any(RAW.glob("PUB_Demand_*.csv"))
skip_no_data = pytest.mark.skipif(not _HAS_DATA, reason="IESO raw data not present")


# ----------------- CapacityTrajectory -----------------


def test_capacity_trajectory_single_anchor_is_constant() -> None:
    t = CapacityTrajectory(name="X", anchors={2024: 1000.0})
    for y in (2000, 2024, 2050):
        assert t.mw_at(y) == 1000.0


def test_capacity_trajectory_linear_between_anchors() -> None:
    t = CapacityTrajectory(name="X", anchors={2024: 0.0, 2034: 1000.0})
    assert t.mw_at(2024) == pytest.approx(0.0)
    assert t.mw_at(2034) == pytest.approx(1000.0)
    assert t.mw_at(2029) == pytest.approx(500.0)


def test_capacity_trajectory_clamps_outside_range() -> None:
    t = CapacityTrajectory(name="X", anchors={2024: 100.0, 2030: 200.0})
    assert t.mw_at(2020) == 100.0  # below first anchor -> clamp
    assert t.mw_at(2040) == 200.0  # above last anchor -> clamp


def test_capacity_trajectory_rejects_negative_mw() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        CapacityTrajectory(name="X", anchors={2024: -1.0})


def test_capacity_trajectory_rejects_empty_anchors() -> None:
    with pytest.raises(ValueError, match="at least one anchor"):
        CapacityTrajectory(name="X", anchors={})


# ----------------- FuelPriceTrajectory -----------------


def test_fuel_price_interpolation() -> None:
    fp = FuelPriceTrajectory(anchors={2024: 3.0, 2030: 5.0})
    assert fp.price_at(2024) == 3.0
    assert fp.price_at(2027) == pytest.approx(4.0)
    assert fp.price_at(2030) == 5.0


def test_fuel_price_rejects_negative() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        FuelPriceTrajectory(anchors={2024: -1.0})


# ----------------- FixedCostTrajectory -----------------


def test_fixed_cost_decline_reduces_over_time() -> None:
    fc = FixedCostTrajectory(
        name="WIND", mode="decline", base_year=2024, base_value=100_000.0, annual_decline=0.02
    )
    assert fc.cost_at(2024) == pytest.approx(100_000.0)
    assert fc.cost_at(2034) == pytest.approx(100_000.0 * (0.98**10))
    # Before base-year, cost is held at the base.
    assert fc.cost_at(2020) == pytest.approx(100_000.0)


def test_fixed_cost_anchors_interpolate() -> None:
    fc = FixedCostTrajectory(name="X", mode="anchors", anchors={2024: 100.0, 2030: 40.0})
    assert fc.cost_at(2024) == 100.0
    assert fc.cost_at(2027) == pytest.approx(70.0)
    assert fc.cost_at(2030) == 40.0


def test_fixed_cost_anchors_require_anchors() -> None:
    fc = FixedCostTrajectory(name="X", mode="anchors", anchors={})
    with pytest.raises(ValueError, match="anchors mode requires anchors"):
        fc.cost_at(2030)


# ----------------- Pathway + marginal cost -----------------


def test_pathway_marginal_cost_uses_heat_rate_and_gas_price() -> None:
    pw = default_ontario_pathway()
    # At 2024 gas=$3.5, CCGT heat rate 7 + VOM 3.5 -> $28
    mc = pw.marginal_cost_at("GAS_CCGT", 2024)
    assert mc == pytest.approx(7.0 * 3.5 + 3.5, rel=1e-10)
    # At 2050 gas=$6.0
    mc2050 = pw.marginal_cost_at("GAS_CCGT", 2050)
    assert mc2050 == pytest.approx(7.0 * 6.0 + 3.5, rel=1e-10)
    assert mc2050 > mc


def test_pathway_fleet_mw_has_all_categories() -> None:
    pw = default_ontario_pathway()
    got = pw.fleet_mw_at(2030)
    expected = {
        "NUCLEAR",
        "HYDRO",
        "GAS_CCGT",
        "GAS_PEAKER",
        "BIOFUEL",
        "WIND",
        "SOLAR",
        "STORAGE",
    }
    assert set(got.keys()) == expected
    assert all(v >= 0 for v in got.values())


def test_default_ontario_pathway_monotone_renewables() -> None:
    pw = default_ontario_pathway()
    prev_wind = 0.0
    for year in (2024, 2030, 2040, 2050):
        w = pw.fleet_mw_at(year)["WIND"]
        assert w >= prev_wind
        prev_wind = w


# ----------------- build_trajectory / run_trajectory -----------------


def test_build_trajectory_returns_one_per_year() -> None:
    cal = _synthetic_calibration()
    pw = default_ontario_pathway()
    years = range(2024, 2031)
    traj = build_trajectory(cal, pw, years)
    assert len(traj) == len(years)
    assert [ys.year for ys in traj] == list(years)


def test_build_trajectory_scenario_has_consistent_firms() -> None:
    cal = _synthetic_calibration()
    pw = default_ontario_pathway()
    traj = build_trajectory(cal, pw, [2024, 2035, 2050])
    for ys in traj:
        # Every firm has positive capacity.
        assert all(f.capacity > 0 for f in ys.scenario.firms)
        # Wind/solar nameplate comes through.
        assert ys.scenario.wind_capacity_mw >= 0
        assert ys.scenario.solar_capacity_mw >= 0
        # marginal costs map covers every firm.
        assert set(ys.marginal_costs.keys()).issubset({f.name for f in ys.scenario.firms})


def test_run_trajectory_produces_tidy_frame() -> None:
    cal = _synthetic_calibration()
    pw = default_ontario_pathway()
    traj = build_trajectory(cal, pw, range(2024, 2030))
    df = run_trajectory(traj)
    assert len(df) == 6
    required = {
        "year",
        "expected_price",
        "expected_quantity_mw",
        "annual_welfare",
        "fleet_missing_money_per_year",
        "reserve_margin",
        "gas_price_per_mmbtu",
        "wind_mw",
        "solar_mw",
    }
    assert required.issubset(df.columns)
    assert df["year"].tolist() == list(range(2024, 2030))


def test_run_trajectory_with_per_firm_attaches_detail() -> None:
    cal = _synthetic_calibration()
    pw = default_ontario_pathway()
    traj = build_trajectory(cal, pw, [2024, 2030])
    df = run_trajectory(traj, include_per_firm=True)
    assert "per_firm" in df.attrs
    pf = df.attrs["per_firm"]
    assert {"year", "firm", "gap_per_year", "short"}.issubset(pf.columns)
    assert set(pf["year"].unique()) == {2024, 2030}


def test_trajectory_rising_gas_prices_raise_ccgt_mc() -> None:
    cal = _synthetic_calibration()
    pw = default_ontario_pathway()
    traj = build_trajectory(cal, pw, [2024, 2050])
    mc24 = traj[0].marginal_costs["GAS_CCGT"]
    mc50 = traj[1].marginal_costs["GAS_CCGT"]
    assert mc50 > mc24


def test_build_trajectory_rejects_empty_fleet_year() -> None:
    cal = _synthetic_calibration()
    # All-zero fleet for a single year.
    empty_pw = Pathway(
        name="empty",
        fleet={"NUCLEAR": CapacityTrajectory(name="NUCLEAR", anchors={2024: 0.0})},
        peak_demand=CapacityTrajectory(name="PEAK", anchors={2024: 20_000.0}),
        mean_demand=CapacityTrajectory(name="MEAN", anchors={2024: 10_000.0}),
        gas_price=FuelPriceTrajectory(anchors={2024: 3.0}),
    )
    with pytest.raises(ValueError, match="no firms with positive capacity"):
        build_trajectory(cal, empty_pw, [2024])


def test_run_trajectory_applies_mechanism() -> None:
    cal = _synthetic_calibration()
    pw = default_ontario_pathway()
    traj_eo = build_trajectory(cal, pw, [2030], mechanism=EnergyOnly())
    traj_cp = build_trajectory(cal, pw, [2030], mechanism=CapacityPayment(rho=1.0))
    df_eo = run_trajectory(traj_eo)
    df_cp = run_trajectory(traj_cp)
    # Capacity payment raises net revenue (smaller negative/larger positive gap).
    assert (
        df_cp["fleet_missing_money_per_year"].iloc[0]
        > df_eo["fleet_missing_money_per_year"].iloc[0]
    )


def test_run_mechanism_matrix_trajectory_long_format() -> None:
    cal = _synthetic_calibration()
    pw = default_ontario_pathway()
    traj = build_trajectory(cal, pw, [2024, 2035])
    df = run_mechanism_matrix_trajectory(traj, structures=("oligopoly",))
    # 2 years x 4 mechanisms x 1 structure = 8 rows.
    assert len(df) == 8
    assert set(df["year"].unique()) == {2024, 2035}


# ----------------- Monte Carlo -----------------


def test_monte_carlo_returns_per_path_rows() -> None:
    cal = _synthetic_calibration()
    cfg = MonteCarloConfig(n_paths=4, seed=1)
    mc = run_monte_carlo(cal, years=range(2024, 2028), config=cfg)
    assert len(mc) == 4 * 4
    assert set(mc["path"].unique()) == {0, 1, 2, 3}


def test_monte_carlo_bands_quantile_ordering() -> None:
    cal = _synthetic_calibration()
    cfg = MonteCarloConfig(n_paths=20, seed=2)
    mc = run_monte_carlo(cal, years=[2024, 2030, 2040, 2050], config=cfg)
    bands = summarize_paths(mc)
    for col in ("expected_price", "fleet_missing_money_per_year"):
        q10 = f"{col}_q10"
        q50 = f"{col}_q50"
        q90 = f"{col}_q90"
        assert (bands[q10] <= bands[q50] + 1e-6).all(), f"{col} q10 > q50"
        assert (bands[q50] <= bands[q90] + 1e-6).all(), f"{col} q50 > q90"


def test_monte_carlo_base_year_has_no_shock() -> None:
    """Shocks start at shock_year; base year values are deterministic."""
    cal = _synthetic_calibration()
    cfg = MonteCarloConfig(n_paths=6, seed=3, shock_year=2030)
    mc = run_monte_carlo(cal, years=[2024, 2030], config=cfg)
    first_year = mc[mc["year"] == 2024]
    assert first_year["expected_price"].nunique() == 1


def test_monte_carlo_zero_sigma_is_deterministic() -> None:
    cal = _synthetic_calibration()
    cfg = MonteCarloConfig(
        n_paths=5,
        seed=4,
        demand_sigma=0.0,
        gas_sigma=0.0,
        renewable_sigma=0.0,
        nuclear_sigma=0.0,
    )
    mc = run_monte_carlo(cal, years=[2024, 2040], config=cfg)
    for y in (2024, 2040):
        sub = mc[mc["year"] == y]
        assert sub["expected_price"].nunique() == 1
        assert sub["fleet_missing_money_per_year"].nunique() == 1


def test_summarize_paths_requires_year_and_path_columns() -> None:
    bad = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match=r"path.*year"):
        summarize_paths(bad)


# ----------------- Real-data smoke -----------------


@skip_no_data
def test_real_trajectory_ontario_2024_2035() -> None:
    from capgame.calibration.ontario import build_ontario_scenario

    cal = build_ontario_scenario(year=2024)
    pw = default_ontario_pathway()
    traj = build_trajectory(cal, pw, range(2024, 2036))
    df = run_trajectory(traj)
    # Prices should be positive and in a plausible range.
    assert (df["expected_price"] > 0).all()
    assert (df["expected_price"] < 300).all()
    # Fleet size monotone-ish across the trajectory; just sanity-check range.
    assert df["total_fleet_mw"].min() > 20_000
    assert df["total_fleet_mw"].max() < 70_000


@skip_no_data
def test_real_monte_carlo_bands_order() -> None:
    from capgame.calibration.ontario import build_ontario_scenario

    cal = build_ontario_scenario(year=2024)
    mc = run_monte_carlo(
        cal,
        years=[2024, 2030, 2040, 2050],
        config=MonteCarloConfig(n_paths=8, seed=0),
    )
    bands = summarize_paths(mc)
    assert (bands["expected_price_q10"] <= bands["expected_price_q90"]).all()
