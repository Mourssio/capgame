"""Tests for the IESO loaders, estimators, and Ontario orchestrator.

Strategy: the test suite does not hit the network. A small synthetic
fixture reproduces the IESO file formats exactly (backslash header
lines, ``Hour 1..Hour 24`` wide schema, XML ``DailyData/HourlyData``
nesting, trailing-comma quirk), and the tests run the real loaders on
that fixture. If IESO's schema changes, these tests will surface the
drift without depending on an internet round-trip.

A second tier of tests (guarded by ``pytest.mark.skipif(not RAW.exists())``)
runs against the real 2024 data when it is on disk, so we get both
schema and domain-level coverage when the data bundle is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capgame.calibration.demand import (
    calibrate_demand_from_elasticity,
    fit_linear_demand,
    ols_slope,
)
from capgame.calibration.ieso_loaders import (
    load_demand,
    load_fleet_capability_month,
    load_fuel_hourly_xml,
    load_hoep,
)
from capgame.calibration.outages import estimate_outage_rates
from capgame.calibration.renewables_cf import build_renewable_chain

RAW = Path("data/ieso/raw")
_HAS_DATA = RAW.exists() and any(RAW.glob("PUB_Demand_*.csv"))
skip_no_data = pytest.mark.skipif(not _HAS_DATA, reason="IESO raw data not present")


# ---------------- synthetic fixtures ----------------


def _write_demand_csv(path: Path, n_hours: int = 48) -> None:
    lines = [
        "\\Hourly Demand Report,,,",
        "\\Created at 2024-01-01 00:00:00,,,",
        "\\For 2024,,,",
        "Date,Hour,Market Demand,Ontario Demand",
    ]
    for t in range(n_hours):
        date = "2024-01-01" if t < 24 else "2024-01-02"
        hour = (t % 24) + 1
        lines.append(f"{date},{hour},{15000 + t * 10},{13000 + t * 8}")
    path.write_text("\n".join(lines))


def _write_hoep_csv(path: Path, n_hours: int = 48) -> None:
    lines = [
        "\\Yearly HOEP OR Predispatch Report,,,,,,,,",
        "\\Created at 2024-01-01 00:00:00,,,,,,,,",
        "\\For 2024,,,,,,,,",
        (
            "Date,Hour,HOEP,Hour 1 Predispatch,Hour 2 Predispatch,"
            "Hour 3 Predispatch,OR 10 Min Sync,OR 10 Min non-sync,OR 30 Min"
        ),
    ]
    for t in range(n_hours):
        date = "2024-01-01" if t < 24 else "2024-01-02"
        hour = (t % 24) + 1
        lines.append(f"{date},{hour},{30 + (t % 10)},32,31,30,1,0.2,0.2")
    path.write_text("\n".join(lines))


def _write_capability_csv(path: Path) -> None:
    header = (
        "\\Generator Output Capability Month Report,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"
        "\\Created at 2024-06-01 00:00:00,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"
        "\\For June 2024,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"
        "Delivery Date,Generator,Fuel Type,Measurement,"
        + ",".join(f"Hour {h}" for h in range(1, 25))
        + "\n"
    )
    rows = []
    hours24 = ",".join(["100"] * 24) + ","  # trailing comma per IESO quirk
    rows.append(f"2024-06-01,NPLANT,NUCLEAR,Capability,{hours24}")
    rows.append(f"2024-06-01,NPLANT,NUCLEAR,Output,{hours24}")
    zeros24 = ",".join(["0"] * 24) + ","
    rows.append(f"2024-06-01,GPLANT1,GAS,Capability,{hours24}")
    rows.append(f"2024-06-01,GPLANT1,GAS,Output,{zeros24}")  # idle -> counts as 'outage'
    rows.append(f"2024-06-01,HPLANT,HYDRO,Capability,{hours24}")
    rows.append(f"2024-06-01,HPLANT,HYDRO,Output,{hours24}")
    wind24 = ",".join(["50"] * 24) + ","
    rows.append(f"2024-06-01,WFARM,WIND,Available Capacity,{wind24}")
    rows.append(f"2024-06-01,WFARM,WIND,Forecast,{wind24}")
    rows.append(f"2024-06-01,WFARM,WIND,Output,{wind24}")
    solar24 = ",".join(["20"] * 24) + ","
    rows.append(f"2024-06-01,SFARM,SOLAR,Available Capacity,{solar24}")
    rows.append(f"2024-06-01,SFARM,SOLAR,Output,{solar24}")
    path.write_text(header + "\n".join(rows))


def _write_fuel_xml(path: Path) -> None:
    blocks = []
    for day in (1, 2):
        hours = []
        for hour in range(1, 25):
            hours.append(
                f"<HourlyData><Hour>{hour}</Hour>"
                "<FuelTotal><Fuel>WIND</Fuel>"
                "<EnergyValue><OutputQuality>0</OutputQuality>"
                f"<Output>{1000 + hour * 10}</Output></EnergyValue></FuelTotal>"
                "<FuelTotal><Fuel>SOLAR</Fuel>"
                "<EnergyValue><OutputQuality>0</OutputQuality>"
                f"<Output>{50 + hour * 2}</Output></EnergyValue></FuelTotal>"
                "</HourlyData>"
            )
        blocks.append(f"<DailyData><Day>2024-01-0{day}</Day>" + "".join(hours) + "</DailyData>")
    body = "<DocBody>" + "".join(blocks) + "</DocBody>"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Document xmlns="http://www.ieso.ca/schema">\n'
        "<DocHeader></DocHeader>\n" + body + "\n</Document>"
    )
    path.write_text(xml)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("ieso_fixture")
    _write_demand_csv(d / "PUB_Demand_2024.csv")
    _write_hoep_csv(d / "PUB_PriceHOEPPredispOR_2024.csv")
    _write_capability_csv(d / "PUB_GenOutputCapabilityMonth_202406.csv")
    _write_fuel_xml(d / "PUB_GenOutputbyFuelHourly_2024.xml")
    return d


# ---------------- loader tests against synthetic files ----------------


def test_load_demand_synthetic(synth_dir: Path) -> None:
    d = load_demand(synth_dir / "PUB_Demand_2024.csv")
    assert d.n == 48
    assert d.timestamps[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert d.timestamps[23] == pd.Timestamp("2024-01-01 23:00:00")
    assert d.timestamps[24] == pd.Timestamp("2024-01-02 00:00:00")
    np.testing.assert_allclose(d.market_demand_mw[:3], [15000, 15010, 15020])
    np.testing.assert_allclose(d.ontario_demand_mw[:3], [13000, 13008, 13016])


def test_load_hoep_synthetic(synth_dir: Path) -> None:
    p = load_hoep(synth_dir / "PUB_PriceHOEPPredispOR_2024.csv")
    assert p.n == 48
    assert p.hoep.min() >= 30 and p.hoep.max() <= 39


def test_load_fleet_capability_synthetic(synth_dir: Path) -> None:
    fc = load_fleet_capability_month(synth_dir / "PUB_GenOutputCapabilityMonth_202406.csv")
    df = fc.data
    # Delivery date parsed as datetime (regression: trailing-comma quirk).
    assert pd.api.types.is_datetime64_any_dtype(df["delivery_date"])
    assert set(fc.fuel_types) == {"NUCLEAR", "GAS", "HYDRO", "WIND", "SOLAR"}
    assert set(fc.generators) == {"NPLANT", "GPLANT1", "HPLANT", "WFARM", "SFARM"}
    wind_rows = df[(df.fuel_type == "WIND")]
    # 24 hours per measurement column; WIND has 3 measurements (AC, Forecast, Output).
    assert len(wind_rows) == 24
    assert "available_capacity_mw" in df.columns
    assert "capability_mw" in df.columns
    # Gas idle: output == 0 while capability == 100.
    gas = df[df.fuel_type == "GAS"]
    assert (gas["capability_mw"] > 0).all()
    assert (gas["output_mw"] == 0).all()


def test_load_fuel_hourly_xml_synthetic(synth_dir: Path) -> None:
    fx = load_fuel_hourly_xml(synth_dir / "PUB_GenOutputbyFuelHourly_2024.xml")
    assert set(fx.columns) == {"timestamp", "fuel", "output_mwh"}
    assert set(fx["fuel"].unique()) == {"WIND", "SOLAR"}
    # 2 days x 24 hours x 2 fuels.
    assert len(fx) == 2 * 24 * 2
    assert fx["timestamp"].min() == pd.Timestamp("2024-01-01 00:00:00")
    assert fx["timestamp"].max() == pd.Timestamp("2024-01-02 23:00:00")


# ---------------- estimator tests ----------------


def test_elasticity_calibration_passes_through_reference_point() -> None:
    d = calibrate_demand_from_elasticity(
        reference_price=50.0, reference_quantity=10_000.0, elasticity=-0.1
    )
    assert d.price(10_000.0) == pytest.approx(50.0)
    # dQ/dP at ref = -1/b; elasticity = dQ/dP * P/Q.
    implied_eps = -(1.0 / d.b) * 50.0 / 10_000.0
    assert implied_eps == pytest.approx(-0.1, rel=1e-12)


def test_elasticity_calibration_rejects_nonnegative_elasticity() -> None:
    with pytest.raises(ValueError, match="elasticity of demand must be"):
        calibrate_demand_from_elasticity(50.0, 10_000.0, elasticity=0.0)


def test_fit_linear_demand_uses_elasticity_path_when_ols_wrong_sign() -> None:
    # Synthetic supply-dominated data: P positively correlated with Q.
    rng = np.random.default_rng(0)
    q = 10_000 + rng.normal(0, 500, size=500)
    p = 30 + 0.002 * q + rng.normal(0, 2, size=500)
    fit = fit_linear_demand(q, p, elasticity=-0.1, prefer="ols")
    # OLS would produce positive dP/dQ -> should fall back to elasticity.
    assert fit.method == "elasticity"
    assert fit.b > 0 and fit.a > 0
    assert fit.ols_slope > 0  # diagnostic preserved


def test_fit_linear_demand_can_use_ols_when_downward_sloping() -> None:
    rng = np.random.default_rng(1)
    q = 1_000 + rng.normal(0, 50, size=500)
    p = 100 - 0.05 * q + rng.normal(0, 1, size=500)
    fit = fit_linear_demand(q, p, elasticity=-0.1, prefer="ols", price_floor=-1000)
    assert fit.method == "ols"
    assert fit.b == pytest.approx(0.05, rel=0.2)


def test_ols_slope_identity() -> None:
    rng = np.random.default_rng(2)
    q = rng.normal(100, 10, 200)
    p = 50 - 0.3 * q + rng.normal(0, 0.1, 200)
    assert ols_slope(q, p) == pytest.approx(-0.3, rel=0.05)


def test_renewable_chain_from_synthetic(synth_dir: Path) -> None:
    fx = load_fuel_hourly_xml(synth_dir / "PUB_GenOutputbyFuelHourly_2024.xml")
    ref = load_fleet_capability_month(synth_dir / "PUB_GenOutputCapabilityMonth_202406.csv").data
    rc = build_renewable_chain(fx, ref)
    assert rc.wind_capacity_mw == 50.0
    assert rc.solar_capacity_mw == 20.0
    pi = rc.chain.stationary_distribution()
    assert pi.shape == (4,)
    assert pi.sum() == pytest.approx(1.0)
    assert len(rc.chain.states) == 4
    # Row-stochastic after Laplace smoothing.
    P = rc.chain.transition_matrix
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert (P > 0).all()


def test_renewable_chain_requires_wind_and_solar(synth_dir: Path) -> None:
    ref = load_fleet_capability_month(synth_dir / "PUB_GenOutputCapabilityMonth_202406.csv").data
    bad = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "fuel": ["GAS"],
            "output_mwh": [100.0],
        }
    )
    with pytest.raises(ValueError, match="WIND and SOLAR"):
        build_renewable_chain(bad, ref)


def test_outage_proxy_counts_zero_output_hours(synth_dir: Path) -> None:
    cap = load_fleet_capability_month(synth_dir / "PUB_GenOutputCapabilityMonth_202406.csv").data
    ests = estimate_outage_rates(cap, fuels=("NUCLEAR", "GAS", "HYDRO"))
    by_fuel = {e.fuel: e for e in ests}
    assert by_fuel["GAS"].n_hours_available == 24
    assert by_fuel["GAS"].n_hours_zero_output == 24
    assert by_fuel["GAS"].outage_rate > 0.9
    assert by_fuel["NUCLEAR"].n_hours_zero_output == 0
    assert by_fuel["NUCLEAR"].outage_rate < 0.05
    assert by_fuel["HYDRO"].outage_rate < 0.05


def test_outage_proxy_returns_zero_for_missing_fuel(synth_dir: Path) -> None:
    cap = load_fleet_capability_month(synth_dir / "PUB_GenOutputCapabilityMonth_202406.csv").data
    ests = estimate_outage_rates(cap, fuels=("COAL",))
    assert ests[0].outage_rate == 0.0 and ests[0].n_hours_available == 0


# ---------------- Ontario orchestrator test (synthetic) ----------------


def test_build_ontario_scenario_synthetic_structure(synth_dir: Path) -> None:
    from capgame.calibration.ontario import build_ontario_scenario
    from capgame.experiments.scenarios import run_scenario

    cal = build_ontario_scenario(
        year=2024,
        raw_dir=synth_dir,
        reference_month=6,
        elasticity=-0.1,
        gas_peaker_cf_threshold=0.5,  # force all GAS to peaker since our synthetic gas is idle
    )
    # Scenario was constructed.
    sc = cal.scenario
    assert len(sc.firms) >= 3
    assert sc.renewable_chain is not None
    assert sc.renewable_chain.transition_matrix.shape == (4, 4)
    # Renewables piped through.
    assert sc.wind_capacity_mw == 50.0
    assert sc.solar_capacity_mw == 20.0
    # Outage rates per firm match the literature defaults (we did not override).
    assert sc.outage_rates is not None
    assert len(sc.outage_rates) == len(sc.firms)
    # And actually runs.
    res = run_scenario(sc)
    assert res.expected_price >= 0.0
    assert res.expected_quantity >= 0.0
    assert res.adequacy.total_capacity_mw > 0.0


def test_build_ontario_allows_cost_overrides(synth_dir: Path) -> None:
    from capgame.calibration.ontario import build_ontario_scenario

    cal = build_ontario_scenario(
        year=2024,
        raw_dir=synth_dir,
        marginal_costs={"NUCLEAR": 1.0},
        fixed_costs={"NUCLEAR": 99.0},
        outage_rates={"NUCLEAR": 0.5},
    )
    tc = {c.name: c for c in cal.technology_classes}
    assert tc["NUCLEAR"].marginal_cost == 1.0
    assert tc["NUCLEAR"].fixed_cost == 99.0
    assert tc["NUCLEAR"].outage_rate == 0.5


# ---------------- real-data tests (gated by file existence) ----------------


@skip_no_data
def test_real_demand_shapes() -> None:
    d = load_demand(RAW / "PUB_Demand_2024.csv")
    assert 8700 <= d.n <= 8800  # 8784 leap-year hours
    assert 10_000 <= d.ontario_demand_mw.mean() <= 20_000
    assert 18_000 <= d.ontario_demand_mw.max() <= 30_000


@skip_no_data
def test_real_ontario_scenario_smoke() -> None:
    from capgame.calibration.ontario import build_ontario_scenario
    from capgame.experiments.scenarios import run_scenario

    cal = build_ontario_scenario(year=2024)
    # Ontario fleet nameplate should be in the tens of GW.
    total = sum(c.capacity_mw for c in cal.technology_classes)
    assert 20_000 < total < 45_000
    # Nuclear ought to dominate and have the lowest marginal cost.
    nuc = next(c for c in cal.technology_classes if c.name == "NUCLEAR")
    assert nuc.capacity_mw > 8_000
    # Equilibrium runs and is in a plausible range.
    res = run_scenario(cal.scenario)
    assert 30 < res.expected_price < 300
    assert 5_000 < res.expected_quantity < 25_000
