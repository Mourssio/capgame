"""Per-year scenario construction and evaluation.

Given a :class:`Pathway` and a base calibration (for the renewable
Markov chain and forced outage rates, which don't come from the
pathway), build a sequence of :class:`~capgame.experiments.scenarios.ScenarioConfig`
objects -- one per year -- and evaluate each under a chosen mechanism.

Design notes
------------
* **Renewable chain**: we *re-use* the calibrated chain from the base
  year (fixed CF distributions) but scale nameplate via
  :attr:`ScenarioConfig.wind_capacity_mw` / ``solar_capacity_mw``.
  This is a first-order approximation: new wind sites will have
  slightly different CF distributions than existing ones. For a
  richer forecaster, ship replacement chains per year via an
  optional ``chain_at(year)`` callable on the pathway.
* **Demand curve**: re-anchored yearly. The linear inverse-demand
  slope ``b`` is set from the (peak, mean, elasticity) triple such
  that the curve passes through the mean point with the specified
  elasticity. Intercept ``a`` follows from ``a = b * mean + p_mean``
  where ``p_mean`` is a fixed real reference price (default $35/MWh,
  close to 2024 Ontario average HOEP). Without a price anchor
  trajectory the intercept would otherwise drift unboundedly.
* **Storage** is implemented as a quasi-firm with zero marginal cost
  and its own nameplate; this is a stand-in for a proper
  inter-temporal model and should only be trusted for "what does
  growing storage do to scarcity-hour dispatch" style questions.
  Callers who do not want this approximation can pass
  ``include_storage=False``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from capgame.calibration.ontario import OntarioCalibration
from capgame.experiments.ontario_study import run_mechanism_matrix
from capgame.experiments.scenarios import (
    MissingMoneyReport,
    ScenarioConfig,
    missing_money,
    run_scenario,
)
from capgame.forecast.pathways import Pathway
from capgame.game.cournot import Firm, LinearDemand
from capgame.mechanisms.base import Mechanism
from capgame.mechanisms.energy_only import EnergyOnly

__all__ = [
    "YearlyScenario",
    "build_trajectory",
    "run_mechanism_matrix_trajectory",
    "run_trajectory",
]

HOURS_PER_YEAR = 8760.0
# Price anchor used to pin the intercept of the inverse-demand curve in
# forward years. $2024/MWh; kept constant in real terms.
_PRICE_ANCHOR_PER_MWH = 35.0


@dataclass(frozen=True)
class YearlyScenario:
    """One year of forecast, with full provenance for reporting."""

    year: int
    scenario: ScenarioConfig
    fleet_mw: dict[str, float]
    peak_mw: float
    mean_mw: float
    gas_price_per_mmbtu: float
    marginal_costs: dict[str, float]
    fixed_costs: dict[str, float]


def _build_demand(mean_mw: float, elasticity: float) -> LinearDemand:
    """Linear inverse demand that passes through (mean_mw, price_anchor).

    The choice to anchor at the *mean* (not the peak) is deliberate:
    in a static Cournot snapshot the equilibrium quantity tracks
    average load more closely than the peak. Using the peak would
    produce equilibria far out on the choke-quantity end of the curve.
    """
    if mean_mw <= 0:
        raise ValueError(f"mean_mw must be > 0, got {mean_mw}")
    b = -_PRICE_ANCHOR_PER_MWH / (elasticity * mean_mw)
    a = _PRICE_ANCHOR_PER_MWH + b * mean_mw
    return LinearDemand(a=a, b=b)


def _build_firms(
    pathway: Pathway,
    year: int,
    base_outage_rates: dict[str, float],
    include_storage: bool,
) -> tuple[tuple[Firm, ...], dict[str, float], dict[str, float], dict[str, float]]:
    fleet = pathway.fleet_mw_at(year)
    firms: list[Firm] = []
    fcs: dict[str, float] = {}
    mcs: dict[str, float] = {}
    caps: dict[str, float] = {}
    for tech, mw in fleet.items():
        if not include_storage and tech == "STORAGE":
            continue
        if tech in ("WIND", "SOLAR"):
            # renewables are handled via wind_capacity_mw/solar_capacity_mw;
            # they are not Cournot firms in this model.
            continue
        if mw <= 0:
            continue
        mc = pathway.marginal_cost_at(tech, year)
        fc = pathway.fixed_cost_at(tech, year)
        outage = base_outage_rates.get(tech, 0.05)
        firms.append(
            Firm(
                marginal_cost=mc,
                capacity=mw,
                fixed_cost=fc,
                outage_rate=outage,
                name=tech,
            )
        )
        mcs[tech] = mc
        fcs[tech] = fc
        caps[tech] = mw
    return tuple(firms), caps, mcs, fcs


def build_trajectory(
    base_cal: OntarioCalibration,
    pathway: Pathway,
    years: Sequence[int],
    mechanism: Mechanism | None = None,
    market_structure: str = "oligopoly",
    include_storage: bool = True,
) -> list[YearlyScenario]:
    """Assemble per-year :class:`ScenarioConfig` objects along a pathway.

    Parameters
    ----------
    base_cal
        Calibrated base-year bundle (supplies the renewable Markov
        chain, outage rates by technology, and ``hours_per_period``).
    pathway
        :class:`Pathway` specifying fleet / demand / fuel / fixed-cost
        trajectories.
    years
        Iterable of calendar years, e.g. ``range(2024, 2051)``.
    mechanism
        Capacity mechanism applied identically every year; if
        ``None``, defaults to :class:`EnergyOnly`.
    market_structure
        Passed to every yearly ``ScenarioConfig``.
    include_storage
        Whether ``STORAGE`` appears as a zero-MC quasi-firm (default
        ``True``). See module docstring for caveats.
    """
    if mechanism is None:
        mechanism = EnergyOnly()
    # Map technology class -> outage rate from the base calibration.
    base_outage_rates = {c.name: c.outage_rate for c in base_cal.technology_classes}
    # Storage inherits gas-ccgt-like outage as a placeholder.
    base_outage_rates.setdefault("STORAGE", 0.02)

    out: list[YearlyScenario] = []
    for y in years:
        firms, caps, mcs, fcs = _build_firms(pathway, y, base_outage_rates, include_storage)
        if not firms:
            raise ValueError(f"year {y}: no firms with positive capacity; check pathway fleet.")
        demand = _build_demand(pathway.mean_mw_at(y), pathway.elasticity)
        scenario = ScenarioConfig(
            demand=demand,
            firms=firms,
            mechanism=mechanism,
            market_structure=market_structure,
            renewable_chain=base_cal.scenario.renewable_chain,
            wind_capacity_mw=pathway.fleet_mw_at(y).get("WIND", 0.0),
            solar_capacity_mw=pathway.fleet_mw_at(y).get("SOLAR", 0.0),
            outage_rates=tuple(f.outage_rate for f in firms),
            target_reserve_margin=base_cal.scenario.target_reserve_margin,
            hours_per_period=base_cal.scenario.hours_per_period,
        )
        out.append(
            YearlyScenario(
                year=y,
                scenario=scenario,
                fleet_mw=caps,
                peak_mw=pathway.peak_mw_at(y),
                mean_mw=pathway.mean_mw_at(y),
                gas_price_per_mmbtu=pathway.gas_price.price_at(y),
                marginal_costs=mcs,
                fixed_costs=fcs,
            )
        )
    return out


def _annualize(per_period: float, hours_per_period: float) -> float:
    periods_per_year = HOURS_PER_YEAR / hours_per_period
    return float(per_period * periods_per_year)


def run_trajectory(
    trajectory: Sequence[YearlyScenario],
    include_per_firm: bool = False,
) -> pd.DataFrame:
    """Evaluate each year's scenario and return a tidy DataFrame.

    Columns:
      ``year, expected_price, expected_quantity_mw, annual_welfare,
      annual_consumer_cost_for_capacity, fleet_missing_money_per_year,
      fraction_firms_short, reserve_margin, lole_hours_per_year,
      eue_mwh_per_year, total_fleet_mw, gas_price_per_mmbtu``.

    If ``include_per_firm`` is true, a second ``per_firm`` attribute
    is returned via a wrapping ``(aggregates, per_firm)`` tuple --
    this is an easy way to feed the notebook's per-technology
    plotting without re-running the scenario.
    """
    rows: list[dict] = []
    per_firm_rows: list[dict] = []
    for ys in trajectory:
        res = run_scenario(ys.scenario)
        mm: MissingMoneyReport = missing_money(res)
        h = ys.scenario.hours_per_period
        rows.append(
            {
                "year": ys.year,
                "expected_price": res.expected_price,
                "expected_quantity_mw": res.expected_quantity,
                "annual_consumer_surplus": _annualize(res.expected_consumer_surplus, h),
                "annual_producer_surplus": _annualize(res.expected_producer_surplus, h),
                "annual_welfare": _annualize(res.expected_welfare, h),
                "annual_consumer_cost_for_capacity": _annualize(
                    res.expected_consumer_payment_for_capacity, h
                ),
                "fleet_missing_money_per_year": mm.fleet_gap_per_year,
                "fraction_firms_short": mm.fraction_firms_short,
                "reserve_margin": res.adequacy.reserve_margin,
                "lole_hours_per_year": res.adequacy.lole_hours_per_year,
                "eue_mwh_per_year": res.adequacy.eue_mwh_per_year,
                "total_fleet_mw": float(sum(ys.fleet_mw.values())),
                "peak_mw": ys.peak_mw,
                "mean_mw": ys.mean_mw,
                "gas_price_per_mmbtu": ys.gas_price_per_mmbtu,
                "wind_mw": ys.scenario.wind_capacity_mw,
                "solar_mw": ys.scenario.solar_capacity_mw,
            }
        )
        if include_per_firm:
            for i, f in enumerate(ys.scenario.firms):
                per_firm_rows.append(
                    {
                        "year": ys.year,
                        "firm": f.name or f"firm_{i}",
                        "capacity_mw": f.capacity,
                        "marginal_cost": f.marginal_cost,
                        "fixed_cost": f.fixed_cost,
                        "annual_net_revenue": float(mm.per_firm_annual_net_revenue[i]),
                        "annual_fixed_requirement": float(mm.per_firm_annual_fixed_requirement[i]),
                        "gap_per_mw_year": float(mm.per_firm_gap_per_mw_year[i]),
                        "gap_per_year": float(mm.per_firm_gap_per_mw_year[i] * f.capacity),
                        "short": bool(mm.per_firm_gap_per_mw_year[i] < 0),
                    }
                )
    aggregates = pd.DataFrame(rows)
    if include_per_firm:
        # Attach per-firm df as an attribute so a single return value is preserved.
        aggregates.attrs["per_firm"] = pd.DataFrame(per_firm_rows)
    return aggregates


def run_mechanism_matrix_trajectory(
    trajectory: Sequence[YearlyScenario],
    mechanisms: dict[str, Mechanism] | None = None,
    structures: Sequence[str] = ("oligopoly",),
) -> pd.DataFrame:
    """Run the 4x3 mechanism matrix at every year of a trajectory.

    Returns a long DataFrame indexed by (year, mechanism, structure).
    Useful for showing mechanism ranking stability across time.
    """
    frames: list[pd.DataFrame] = []
    for ys in trajectory:
        # Fake an OntarioCalibration just enough to satisfy run_mechanism_matrix;
        # the real function only uses ``cal.scenario``.
        fake_cal = _FakeCalibrationForMatrix(scenario=ys.scenario)
        df = run_mechanism_matrix(
            fake_cal,
            mechanisms=mechanisms,
            structures=tuple(structures),
        )
        df.insert(0, "year", ys.year)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class _FakeCalibrationForMatrix:
    """Minimal duck-typed stand-in for :class:`OntarioCalibration`.

    :func:`run_mechanism_matrix` only touches ``cal.scenario``; no need
    to construct a full :class:`OntarioCalibration` just to call it.
    """

    scenario: ScenarioConfig
