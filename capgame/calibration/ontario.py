"""Orchestrator: IESO public files -> :class:`ScenarioConfig` for Ontario.

This module is the only calibration entry point most callers should
need. It composes the loaders in :mod:`.ieso_loaders` with the
estimators in :mod:`.demand`, :mod:`.renewables_cf`, and :mod:`.outages`
to produce a ready-to-run :class:`~capgame.experiments.scenarios.ScenarioConfig`.

Thermal fleet abstraction
-------------------------
Ontario's wholesale market has dozens of generators but a handful of
distinct **technology classes**, and only those classes are
strategically meaningful in a Cournot model. We aggregate by fuel:

* ``NUCLEAR``   - very low marginal cost, essentially must-run baseload.
* ``HYDRO``     - low marginal cost, hourly-dispatchable.
* ``GAS_CCGT``  - mid-merit combined-cycle gas.
* ``GAS_PEAKER`` - high-marginal-cost peaker / simple-cycle gas.
* ``BIOFUEL``   - a small residual class.

Nameplate capacity per class is taken as the **sum of the maximum
observed capability** by generator over a representative month (we pass
June to avoid winter refurbs biasing nuclear). Gas is split by
unit-level capacity factor: units with annual CF below
``gas_peaker_cf_threshold`` are tagged as peakers.

Marginal and fixed costs are **literature-based defaults**; forced
outage rates for thermal fuels default to NERC GADS class averages
because the IESO "declared-available-but-zero" proxy is dispatch-driven
for gas/hydro and therefore not a true FOR. Every one of these numbers
is exposed as a parameter so downstream studies can run sensitivity
analysis without re-running the loaders.

Uncertainty & stochastic pieces
-------------------------------
* Renewable availability: empirical 4-state Markov chain estimated from
  the 2024 hourly fuel series (``build_renewable_chain``).
* Forced outages: per-fuel rates used for the adequacy layer (LOLE,
  EUE); defaulted to NERC, overridable with the empirical proxy.
* Demand: calibrated from elasticity (see :mod:`.demand`). The OLS
  diagnostic is surfaced in the returned report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capgame.calibration.demand import DemandFit, fit_linear_demand
from capgame.calibration.ieso_loaders import (
    load_demand,
    load_fleet_capability_month,
    load_fuel_hourly_xml,
    load_hoep,
)
from capgame.calibration.outages import OutageEstimate, estimate_outage_rates
from capgame.calibration.renewables_cf import RenewableCalibration, build_renewable_chain
from capgame.experiments.scenarios import ScenarioConfig
from capgame.game.cournot import Firm
from capgame.mechanisms.base import Mechanism
from capgame.mechanisms.energy_only import EnergyOnly

__all__ = [
    "DEFAULT_RAW_DIR",
    "LITERATURE_FIXED_COST",
    "LITERATURE_MARGINAL_COST",
    "LITERATURE_OUTAGE_RATES",
    "OntarioCalibration",
    "TechnologyClass",
    "build_ontario_scenario",
]

DEFAULT_RAW_DIR = Path("data/ieso/raw")

# NERC GADS-style unplanned-outage rates (approximate 2015-2022 North
# American averages). These are the rates the adequacy layer uses.
LITERATURE_OUTAGE_RATES: dict[str, float] = {
    "NUCLEAR": 0.03,
    "HYDRO": 0.02,
    "GAS_CCGT": 0.05,
    "GAS_PEAKER": 0.08,
    "BIOFUEL": 0.10,
}

# Short-run marginal cost ($/MWh), fuel + variable O&M. Uses typical 2024
# Ontario prices for nuclear fuel (~$8), hydro opportunity cost (~$3),
# and a $3.50/MMBtu gas price with 7 / 10 heat rates.
LITERATURE_MARGINAL_COST: dict[str, float] = {
    "NUCLEAR": 8.0,
    "HYDRO": 3.0,
    "GAS_CCGT": 45.0,
    "GAS_PEAKER": 85.0,
    "BIOFUEL": 95.0,
}

# Annualized fixed cost per MW of capacity ($/kW-yr * 1000 = $/MW-yr).
# Based on NREL ATB 2023 nominal fixed O&M for existing fleet; these are
# the values the capacity-payment and RO mechanisms anchor on.
LITERATURE_FIXED_COST: dict[str, float] = {
    "NUCLEAR": 120_000.0,
    "HYDRO": 40_000.0,
    "GAS_CCGT": 60_000.0,
    "GAS_PEAKER": 45_000.0,
    "BIOFUEL": 80_000.0,
}


@dataclass(frozen=True)
class TechnologyClass:
    """A calibrated thermal technology class (one CapGame firm)."""

    name: str
    capacity_mw: float
    marginal_cost: float
    fixed_cost: float
    outage_rate: float

    def to_firm(self) -> Firm:
        return Firm(
            marginal_cost=self.marginal_cost,
            capacity=self.capacity_mw,
            fixed_cost=self.fixed_cost,
            outage_rate=self.outage_rate,
            name=self.name,
        )


@dataclass(frozen=True)
class OntarioCalibration:
    """Full calibration bundle: the :class:`ScenarioConfig` and provenance.

    Attributes
    ----------
    scenario
        Ready for :func:`capgame.experiments.scenarios.run_scenario`.
    demand_fit
        The :class:`DemandFit` used, including the OLS diagnostic.
    renewable_calibration
        Nameplate capacities and state counts used for the Markov chain.
    technology_classes
        The per-class calibrated firms.
    empirical_outage_rates
        Per-fuel FOR estimates from the IESO capability proxy (for
        comparison with the literature defaults the scenario actually
        uses).
    peak_load_mw
        Annual peak load from the ``PUB_Demand`` series.
    year
        Calendar year the inputs correspond to.
    """

    scenario: ScenarioConfig
    demand_fit: DemandFit
    renewable_calibration: RenewableCalibration
    technology_classes: tuple[TechnologyClass, ...]
    empirical_outage_rates: tuple[OutageEstimate, ...]
    peak_load_mw: float
    year: int


def _classify_gas(capability_month: pd.DataFrame, threshold: float) -> tuple[float, float]:
    """Split gas nameplate into (CCGT, peaker) by each unit's CF.

    Unit-level CF = mean(output) / mean(capability) over the month.
    Units below ``threshold`` are counted as peakers; the rest as CCGT.
    """
    gas = capability_month[capability_month["fuel_type"].str.upper() == "GAS"].copy()
    if gas.empty:
        return 0.0, 0.0
    grouped = gas.groupby("generator").agg(
        cap=("capability_mw", "max"),
        mean_out=("output_mw", "mean"),
        mean_cap=("capability_mw", "mean"),
    )
    grouped["cf"] = (grouped["mean_out"] / grouped["mean_cap"]).fillna(0.0)
    peaker_mask = grouped["cf"] < threshold
    peaker_mw = float(grouped.loc[peaker_mask, "cap"].sum())
    ccgt_mw = float(grouped.loc[~peaker_mask, "cap"].sum())
    return ccgt_mw, peaker_mw


def _nameplate(capability_month: pd.DataFrame, fuel: str) -> float:
    sub = capability_month[capability_month["fuel_type"].str.upper() == fuel.upper()]
    if sub.empty:
        return 0.0
    by_gen = sub.groupby("generator")["capability_mw"].max().fillna(0.0)
    return float(by_gen.sum())


def build_ontario_scenario(
    year: int = 2024,
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    reference_month: int = 6,
    elasticity: float = -0.1,
    gas_peaker_cf_threshold: float = 0.15,
    marginal_costs: dict[str, float] | None = None,
    fixed_costs: dict[str, float] | None = None,
    outage_rates: dict[str, float] | None = None,
    mechanism: Mechanism | None = None,
    target_reserve_margin: float = 0.18,
    wind_capacity_override_mw: float | None = None,
    solar_capacity_override_mw: float | None = None,
) -> OntarioCalibration:
    """Calibrate a full Ontario scenario from IESO files on disk.

    Parameters
    ----------
    year
        Calendar year; determines which files are read.
    raw_dir
        Directory that contains ``PUB_Demand_{year}.csv``, the HOEP
        file, the hourly fuel XML, and the 12 monthly generator
        capability CSVs. (Use :func:`scripts.fetch_ieso.fetch_year` to
        populate it.)
    reference_month
        Month (1-12) used to size the fleet. June avoids winter nuclear
        refurbs and summer grid-scale solar commissioning spikes.
    elasticity
        Short-run price elasticity of demand for :func:`fit_linear_demand`.
    gas_peaker_cf_threshold
        Capacity-factor cutoff separating CCGT from peaker units.
    marginal_costs, fixed_costs, outage_rates
        Optional overrides for the literature defaults. Keys must match
        the ``TechnologyClass`` names (``NUCLEAR``, ``HYDRO``,
        ``GAS_CCGT``, ``GAS_PEAKER``, ``BIOFUEL``).
    mechanism
        Capacity mechanism under test; defaults to :class:`EnergyOnly`.
    target_reserve_margin
        NPCC reference for Ontario is ~17-18%.
    wind_capacity_override_mw, solar_capacity_override_mw
        Override the renewable-chain nameplate capacities (useful for
        scenario analysis like "what if wind doubles").
    """
    raw_dir = Path(raw_dir)

    demand_ts = load_demand(raw_dir / f"PUB_Demand_{year}.csv")
    price_ts = load_hoep(raw_dir / f"PUB_PriceHOEPPredispOR_{year}.csv")
    demand_fit = fit_linear_demand(
        demand_ts.ontario_demand_mw, price_ts.hoep, elasticity=elasticity
    )

    ref_path = raw_dir / f"PUB_GenOutputCapabilityMonth_{year}{reference_month:02d}.csv"
    ref_cap = load_fleet_capability_month(ref_path).data

    months = []
    for m in range(1, 13):
        p = raw_dir / f"PUB_GenOutputCapabilityMonth_{year}{m:02d}.csv"
        if p.exists():
            months.append(load_fleet_capability_month(p).data)
    annual_cap = pd.concat(months, ignore_index=True) if months else ref_cap

    fuel_hourly = load_fuel_hourly_xml(raw_dir / f"PUB_GenOutputbyFuelHourly_{year}.xml")
    renewable_cal = build_renewable_chain(fuel_hourly, ref_cap)

    ccgt_mw, peaker_mw = _classify_gas(ref_cap, gas_peaker_cf_threshold)
    nuclear_mw = _nameplate(ref_cap, "NUCLEAR")
    hydro_mw = _nameplate(ref_cap, "HYDRO")
    biofuel_mw = _nameplate(ref_cap, "BIOFUEL")

    mc = {**LITERATURE_MARGINAL_COST, **(marginal_costs or {})}
    fc = {**LITERATURE_FIXED_COST, **(fixed_costs or {})}
    fo = {**LITERATURE_OUTAGE_RATES, **(outage_rates or {})}

    classes = (
        TechnologyClass(
            name="NUCLEAR",
            capacity_mw=nuclear_mw,
            marginal_cost=mc["NUCLEAR"],
            fixed_cost=fc["NUCLEAR"],
            outage_rate=fo["NUCLEAR"],
        ),
        TechnologyClass(
            name="HYDRO",
            capacity_mw=hydro_mw,
            marginal_cost=mc["HYDRO"],
            fixed_cost=fc["HYDRO"],
            outage_rate=fo["HYDRO"],
        ),
        TechnologyClass(
            name="GAS_CCGT",
            capacity_mw=ccgt_mw,
            marginal_cost=mc["GAS_CCGT"],
            fixed_cost=fc["GAS_CCGT"],
            outage_rate=fo["GAS_CCGT"],
        ),
        TechnologyClass(
            name="GAS_PEAKER",
            capacity_mw=peaker_mw,
            marginal_cost=mc["GAS_PEAKER"],
            fixed_cost=fc["GAS_PEAKER"],
            outage_rate=fo["GAS_PEAKER"],
        ),
        TechnologyClass(
            name="BIOFUEL",
            capacity_mw=biofuel_mw,
            marginal_cost=mc["BIOFUEL"],
            fixed_cost=fc["BIOFUEL"],
            outage_rate=fo["BIOFUEL"],
        ),
    )
    # Drop any zero-capacity classes so the Cournot solver does not waste
    # work on trivially empty firms.
    classes = tuple(c for c in classes if c.capacity_mw > 0)
    firms = tuple(c.to_firm() for c in classes)
    fleet_outages = tuple(c.outage_rate for c in classes)

    wind_mw = (
        wind_capacity_override_mw
        if wind_capacity_override_mw is not None
        else renewable_cal.wind_capacity_mw
    )
    solar_mw = (
        solar_capacity_override_mw
        if solar_capacity_override_mw is not None
        else renewable_cal.solar_capacity_mw
    )

    empirical = tuple(estimate_outage_rates(annual_cap))

    scenario = ScenarioConfig(
        demand=demand_fit.demand,
        firms=firms,
        mechanism=mechanism if mechanism is not None else EnergyOnly(),
        market_structure="oligopoly",
        renewable_chain=renewable_cal.chain,
        wind_capacity_mw=wind_mw,
        solar_capacity_mw=solar_mw,
        outage_rates=fleet_outages,
        target_reserve_margin=target_reserve_margin,
        hours_per_period=1.0,
    )

    return OntarioCalibration(
        scenario=scenario,
        demand_fit=demand_fit,
        renewable_calibration=renewable_cal,
        technology_classes=classes,
        empirical_outage_rates=empirical,
        peak_load_mw=float(demand_ts.ontario_demand_mw.max()),
        year=year,
    )
