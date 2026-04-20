"""Parameter calibration for :class:`ScenarioConfig`.

This subpackage turns **data on disk** into a calibrated
:class:`~capgame.experiments.scenarios.ScenarioConfig`. It has two layers:

* :mod:`capgame.calibration.ieso_loaders` — pure-IO readers that parse
  raw IESO public-report files (CSVs and XMLs) into typed numpy / pandas
  structures. They do not do any statistics.
* Estimators (:mod:`.demand`, :mod:`.renewables_cf`, :mod:`.outages`) —
  turn those raw series into the numerical parameters the game model
  consumes (inverse-demand slope and intercept, renewable Markov chain,
  per-firm forced outage rates, etc.).
* :mod:`capgame.calibration.ontario` — orchestrator that wires everything
  together and returns a ready-to-run :class:`ScenarioConfig`.

All network access lives in ``scripts/fetch_ieso.py``; modules here read
from local files only, so the calibration pipeline is reproducible and
offline-friendly.
"""

from __future__ import annotations

from capgame.calibration.demand import (
    DemandFit,
    calibrate_demand_from_elasticity,
    fit_linear_demand,
    ols_slope,
)
from capgame.calibration.ieso_loaders import (
    FleetCapability,
    HourlyDemand,
    HourlyPrice,
    load_demand,
    load_fleet_capability_month,
    load_fuel_hourly_xml,
    load_hoep,
)
from capgame.calibration.ontario import (
    OntarioCalibration,
    TechnologyClass,
    build_ontario_scenario,
)
from capgame.calibration.outages import OutageEstimate, estimate_outage_rates
from capgame.calibration.renewables_cf import (
    RenewableCalibration,
    build_renewable_chain,
)

__all__ = [
    "DemandFit",
    "FleetCapability",
    "HourlyDemand",
    "HourlyPrice",
    "OntarioCalibration",
    "OutageEstimate",
    "RenewableCalibration",
    "TechnologyClass",
    "build_ontario_scenario",
    "build_renewable_chain",
    "calibrate_demand_from_elasticity",
    "estimate_outage_rates",
    "fit_linear_demand",
    "load_demand",
    "load_fleet_capability_month",
    "load_fuel_hourly_xml",
    "load_hoep",
    "ols_slope",
]
