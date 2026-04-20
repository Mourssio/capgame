"""Multi-decade forecasting layer for Ontario (and by extension, any
calibrated jurisdiction).

Turns a calibrated base year (2024) into a sequence of
:class:`~capgame.experiments.scenarios.ScenarioConfig` objects, one per
forecast year, respecting:

* **Load growth** (demand intercept / slope evolving with peak and
  average MW, holding elasticity fixed by default).
* **Fleet roll-forward**: nuclear refurbishments and retirements,
  small-modular-reactor commissioning, gas additions and retirements,
  renewable buildout (wind/solar), battery storage phased in.
* **Fuel prices**: gas-price trajectory drives CCGT/peaker marginal
  costs through an exposed heat rate + variable O&M.
* **Capex / fixed-cost decline curves** (NREL ATB-style) for wind,
  solar, storage; nuclear and gas stay flat in real terms.

Public surface
--------------
- :class:`Pathway`, :class:`CapacityTrajectory`, :class:`FuelPriceTrajectory`,
  :class:`FixedCostTrajectory` -- pathway primitives.
- :func:`default_ontario_pathway` -- IESO-APO-Reference-like deterministic
  anchor points through 2050.
- :class:`YearlyScenario`, :func:`build_trajectory` -- assemble the year-by-year
  :class:`ScenarioConfig`.
- :func:`run_trajectory` -- compute per-year equilibrium + missing money +
  adequacy metrics under a given mechanism.
- :class:`MonteCarloConfig`, :func:`run_monte_carlo` -- stochastic
  uncertainty bands over pathway parameters.
"""

from __future__ import annotations

from capgame.forecast.monte_carlo import (
    MonteCarloConfig,
    run_monte_carlo,
    summarize_paths,
)
from capgame.forecast.pathways import (
    CapacityTrajectory,
    FixedCostTrajectory,
    FuelPriceTrajectory,
    Pathway,
    default_ontario_pathway,
)
from capgame.forecast.trajectory import (
    YearlyScenario,
    build_trajectory,
    run_mechanism_matrix_trajectory,
    run_trajectory,
)

__all__ = [
    "CapacityTrajectory",
    "FixedCostTrajectory",
    "FuelPriceTrajectory",
    "MonteCarloConfig",
    "Pathway",
    "YearlyScenario",
    "build_trajectory",
    "default_ontario_pathway",
    "run_mechanism_matrix_trajectory",
    "run_monte_carlo",
    "run_trajectory",
    "summarize_paths",
]
