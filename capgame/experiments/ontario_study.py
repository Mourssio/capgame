"""Ontario applied study: missing money + optimal reliability-option strike.

This module is the driver for the portfolio / policy deliverable
described in :mod:`capgame`'s README. It composes the Ontario
calibration from :mod:`capgame.calibration.ontario` with the four
capacity mechanisms and the three market structures to answer, with a
single script, three linked research questions:

1.  **Baseline missing money.** Under Ontario's de-facto energy-only
    clearing (plus the calibrated market power from a five-technology
    fleet), which firms fall short of their fixed-cost revenue
    requirement, and by how much?
2.  **Mechanism ranking.** Across the 4 x 3 grid of {energy-only,
    capacity payment, forward capacity auction, reliability options}
    x {oligopoly, cartel, monopoly}, which combinations close the gap
    with the lowest consumer cost?
3.  **Optimal RO strike.** Holding the mechanism fixed at a reliability
    option, what strike ``K`` maximizes welfare net of consumer cost
    over the 4-state renewable Markov chain? How robust is the answer
    to doubling the wind fleet (the change Ontario is currently
    procuring)?

The outputs are all tidy ``pandas.DataFrame`` objects so the notebook
layer does nothing but plotting + prose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd

from capgame.calibration.ontario import OntarioCalibration, build_ontario_scenario
from capgame.experiments.scenarios import (
    MissingMoneyReport,
    ScenarioConfig,
    ScenarioResult,
    missing_money,
    run_scenario,
)
from capgame.game.market_structure import MarketStructure
from capgame.mechanisms.base import Mechanism
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from capgame.mechanisms.forward_capacity import ForwardCapacityMarket, ProcurementCurve
from capgame.mechanisms.reliability_options import ReliabilityOption

__all__ = [
    "MatrixRow",
    "StrikeSearchResult",
    "default_ontario_mechanisms",
    "find_optimal_strike",
    "run_mechanism_matrix",
    "run_sensitivity_sweep",
    "summarize_missing_money",
]

HOURS_PER_YEAR = 8760.0


def annual_to_per_period(value_per_mw_year: float, hours_per_period: float = 1.0) -> float:
    """Convert $/MW-yr to the per-period units mechanisms expect.

    Mechanisms like :class:`CapacityPayment` and :class:`ReliabilityOption`
    parameterize payments **per period** (a period is one hour in our
    Ontario setup). Policy analysts reason in $/MW-yr, so this helper
    makes the conversion explicit and testable.
    """
    if hours_per_period <= 0:
        raise ValueError(f"hours_per_period must be > 0, got {hours_per_period}")
    periods_per_year = HOURS_PER_YEAR / hours_per_period
    return value_per_mw_year / periods_per_year


def default_ontario_mechanisms(
    capacity_payment_rho_per_mw_year: float = 60_000.0,
    fcm_target_mw: float = 22_000.0,
    fcm_slope: float = 500.0,
    ro_premium_per_mw_year: float = 55_000.0,
    ro_strike: float = 60.0,
    hours_per_period: float = 1.0,
) -> dict[str, Mechanism]:
    """Four mechanisms calibrated to Ontario scale, parameterized in $/MW-yr.

    Inputs are in **annual** units (the natural unit for policy), and
    are converted to the per-period units the mechanism classes consume.

    Defaults:

    * ``rho = $60/kW-yr`` is roughly the IESO 2024 Capacity Auction
      clearing price for existing gas.
    * FCM target = 22 GW is Ontario's planning reserve; slope set so
      the demand curve is neither vertical nor flat.
    * RO premium = $55/kW-yr matches the capacity payment to make the
      comparison an apples-to-apples "pay the same, demand different
      risk" test. ``strike = $60/MWh`` is well below the peaker
      marginal cost so the option binds in scarcity hours.
    """
    rho_per_period = annual_to_per_period(capacity_payment_rho_per_mw_year, hours_per_period)
    premium_per_period = annual_to_per_period(ro_premium_per_mw_year, hours_per_period)
    return {
        "Energy-only": EnergyOnly(),
        "Capacity payment": CapacityPayment(rho=rho_per_period),
        "Forward capacity": ForwardCapacityMarket(
            curve=ProcurementCurve(cap_target=fcm_target_mw, slope=fcm_slope)
        ),
        "Reliability options": ReliabilityOption(
            premium=premium_per_period,
            strike_price=ro_strike,
            hours_per_period=hours_per_period,
        ),
    }


@dataclass(frozen=True)
class MatrixRow:
    """One cell of the mechanism x structure grid, plus diagnostics."""

    mechanism: str
    structure: MarketStructure
    expected_price: float
    expected_quantity_mw: float
    annual_consumer_surplus: float
    annual_producer_surplus: float
    annual_consumer_cost_for_capacity: float
    annual_welfare: float
    fleet_missing_money_per_year: float
    fraction_firms_short: float
    reserve_margin: float
    lole_hours_per_year: float | None
    eue_mwh_per_year: float | None


def _annualize(per_period: float, periods_per_year: float = HOURS_PER_YEAR) -> float:
    return float(per_period * periods_per_year)


def _row_from_result(
    mechanism_name: str,
    structure: MarketStructure,
    result: ScenarioResult,
    mm: MissingMoneyReport,
    periods_per_year: float = HOURS_PER_YEAR,
) -> MatrixRow:
    return MatrixRow(
        mechanism=mechanism_name,
        structure=structure,
        expected_price=result.expected_price,
        expected_quantity_mw=result.expected_quantity,
        annual_consumer_surplus=_annualize(result.expected_consumer_surplus, periods_per_year),
        annual_producer_surplus=_annualize(result.expected_producer_surplus, periods_per_year),
        annual_consumer_cost_for_capacity=_annualize(
            result.expected_consumer_payment_for_capacity, periods_per_year
        ),
        annual_welfare=_annualize(result.expected_welfare, periods_per_year),
        fleet_missing_money_per_year=mm.fleet_gap_per_year,
        fraction_firms_short=mm.fraction_firms_short,
        reserve_margin=result.adequacy.reserve_margin,
        lole_hours_per_year=result.adequacy.lole_hours_per_year,
        eue_mwh_per_year=result.adequacy.eue_mwh_per_year,
    )


def run_mechanism_matrix(
    cal: OntarioCalibration,
    mechanisms: Mapping[str, Mechanism] | None = None,
    structures: Sequence[MarketStructure] = ("oligopoly", "cartel", "monopoly"),
) -> pd.DataFrame:
    """Evaluate every (mechanism, structure) pair against the Ontario calibration.

    Returns a long-format DataFrame with one row per combination. The
    ``annual_*`` columns are in dollars-per-year (we multiply the
    per-period expectations by 8760). Missing money is likewise
    annualized and directly comparable to the consumer-cost column.
    """
    mechs = dict(mechanisms) if mechanisms is not None else default_ontario_mechanisms()
    rows: list[MatrixRow] = []
    for structure in structures:
        for name, mech in mechs.items():
            cfg = replace(cal.scenario, mechanism=mech, market_structure=structure)
            res = run_scenario(cfg)
            mm = missing_money(res)
            rows.append(_row_from_result(name, structure, res, mm))
    return pd.DataFrame([row.__dict__ for row in rows])


def summarize_missing_money(cal: OntarioCalibration, mechanism: Mechanism) -> pd.DataFrame:
    """Per-technology missing-money breakdown under one mechanism.

    Returns one row per firm with columns
    ``[name, capacity_mw, annual_net_revenue, annual_fixed_requirement,
    gap_per_mw_year, gap_per_year, short]``.
    """
    cfg = replace(cal.scenario, mechanism=mechanism)
    res = run_scenario(cfg)
    mm = missing_money(res)
    firms = cal.scenario.firms
    rows = []
    for i, f in enumerate(firms):
        gap = float(mm.per_firm_gap_per_mw_year[i])
        rows.append(
            {
                "name": f.name or f"firm_{i}",
                "capacity_mw": f.capacity,
                "annual_net_revenue": float(mm.per_firm_annual_net_revenue[i]),
                "annual_fixed_requirement": float(mm.per_firm_annual_fixed_requirement[i]),
                "gap_per_mw_year": gap,
                "gap_per_year": float(gap * f.capacity),
                "short": bool(gap < 0.0),
            }
        )
    return pd.DataFrame(rows)


# ---------------- endogenous strike over the renewable chain ----------------


@dataclass(frozen=True)
class StrikeSearchResult:
    """Result of a welfare-optimal strike search on the Ontario scenario."""

    strike_grid: np.ndarray = field(repr=False)
    annual_welfare: np.ndarray = field(repr=False)
    annual_consumer_cost: np.ndarray = field(repr=False)
    annual_producer_surplus: np.ndarray = field(repr=False)
    annual_missing_money: np.ndarray = field(repr=False)
    optimal_strike: float
    optimal_welfare_annual: float
    optimal_missing_money_annual: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "strike_per_mwh": self.strike_grid,
                "annual_welfare": self.annual_welfare,
                "annual_consumer_cost": self.annual_consumer_cost,
                "annual_producer_surplus": self.annual_producer_surplus,
                "annual_missing_money": self.annual_missing_money,
            }
        )


def find_optimal_strike(
    cal: OntarioCalibration,
    premium_per_mw_year: float = 55_000.0,
    coverage: float = 1.0,
    strike_bounds: tuple[float, float] = (0.0, 300.0),
    n_grid: int = 31,
    objective: Literal[
        "welfare", "missing_money_closure", "min_consumer_cost"
    ] = "missing_money_closure",
) -> StrikeSearchResult:
    """Search for the best reliability-option strike on Ontario.

    Unlike the generic :func:`capgame.game.bilevel.solve_endogenous_strike`
    this searcher runs the **full renewable-Markov-chain scenario** at
    every grid point, so the objective correctly integrates over
    (wind, solar) states where the claw-back does or does not bind.

    **Why the default objective is not welfare.** Under the standard RO
    assumption (option refunds are a transfer, not a dispatch
    distortion), the Cournot subgame is independent of ``K``, and so
    is welfare = CS + PS - ConsumerCost. Total welfare is therefore
    **constant across the strike grid** -- a finding in itself, which
    the notebook should highlight. What the strike actually controls
    is the **split** between consumers and producers. Two policy-
    relevant objectives emerge:

    * ``"min_consumer_cost"`` (default): pick the strike that
      minimizes annual consumer cost while the premium is held fixed.
      At large ``K``, refunds vanish so consumers pay the full
      premium; at small ``K``, refunds exceed the premium and the
      mechanism effectively writes consumers a cheque. The minimum
      is at the lower end of ``strike_bounds`` -- useful as a
      worst-case bound.
    * ``"missing_money_closure"``: pick the strike that brings the
      fleet's missing-money closest to zero. This is the
      reliability-focused objective: "just enough capacity payment
      to keep marginal technology indifferent between staying and
      exiting."
    * ``"welfare"``: kept for completeness; will typically be
      degenerate (flat objective).

    Parameters
    ----------
    premium_per_mw_year
        Fixed per-MW-yr premium in $/MW-yr. Converted internally.
    coverage
        Fraction of each firm's capacity under option.
    strike_bounds
        Search range for ``K`` in $/MWh.
    n_grid
        Grid size; 31 gives ~$10/MWh resolution on the default bounds.
    """
    lo, hi = strike_bounds
    if lo < 0 or hi <= lo:
        raise ValueError(f"strike_bounds must satisfy 0 <= lo < hi, got {strike_bounds}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")

    grid = np.linspace(lo, hi, n_grid)
    welfare = np.empty(n_grid)
    cons_cost = np.empty(n_grid)
    prod_surplus = np.empty(n_grid)
    mm_total = np.empty(n_grid)

    premium_per_period = annual_to_per_period(premium_per_mw_year, cal.scenario.hours_per_period)
    for i, k in enumerate(grid):
        mech = ReliabilityOption(
            premium=premium_per_period,
            strike_price=float(k),
            coverage=coverage,
            hours_per_period=cal.scenario.hours_per_period,
        )
        cfg = replace(cal.scenario, mechanism=mech)
        res = run_scenario(cfg)
        mm = missing_money(res)
        welfare[i] = _annualize(res.expected_welfare)
        cons_cost[i] = _annualize(res.expected_consumer_payment_for_capacity)
        prod_surplus[i] = _annualize(res.expected_producer_surplus)
        mm_total[i] = mm.fleet_gap_per_year

    if objective == "welfare":
        best = int(np.argmax(welfare))
    elif objective == "missing_money_closure":
        best = int(np.argmin(np.abs(mm_total)))
    elif objective == "min_consumer_cost":
        best = int(np.argmin(cons_cost))
    else:
        raise ValueError(f"unknown objective: {objective}")

    return StrikeSearchResult(
        strike_grid=grid,
        annual_welfare=welfare,
        annual_consumer_cost=cons_cost,
        annual_producer_surplus=prod_surplus,
        annual_missing_money=mm_total,
        optimal_strike=float(grid[best]),
        optimal_welfare_annual=float(welfare[best]),
        optimal_missing_money_annual=float(mm_total[best]),
    )


# ---------------- sensitivity ----------------


def run_sensitivity_sweep(
    year: int = 2024,
    raw_dir=None,
    elasticities: Sequence[float] = (-0.05, -0.1, -0.2),
    wind_multipliers: Sequence[float] = (1.0, 2.0),
    gas_cost_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    mechanism_factory=None,
) -> pd.DataFrame:
    """Sweep the three parameters most likely to flip the mechanism ranking.

    For every combination of (elasticity, wind multiplier, gas-cost
    multiplier), recalibrate Ontario, apply the given mechanism
    (default: reliability options with $60/MWh strike), and record
    price, welfare, and fleet missing money. The resulting DataFrame
    is the input to the paper's tornado chart and to the robustness
    checks in the notebook.

    The recalibration *is* done fresh per elasticity because
    elasticity changes the intercept and slope of the linear demand
    curve, which ripples through every equilibrium value.
    """
    factory = mechanism_factory or (
        lambda: ReliabilityOption(
            premium=annual_to_per_period(55_000.0, 1.0),
            strike_price=60.0,
            hours_per_period=1.0,
        )
    )
    rows = []
    kwargs = {"year": year}
    if raw_dir is not None:
        kwargs["raw_dir"] = raw_dir
    for eps in elasticities:
        base_cal = build_ontario_scenario(elasticity=eps, **kwargs)
        base_scenario: ScenarioConfig = base_cal.scenario
        base_mc = {c.name: c.marginal_cost for c in base_cal.technology_classes}
        for wmul in wind_multipliers:
            for gmul in gas_cost_multipliers:
                # Apply gas marginal cost multiplier via firm replacement.
                new_firms = []
                for f in base_scenario.firms:
                    if f.name in ("GAS_CCGT", "GAS_PEAKER") and f.name in base_mc:
                        new_firms.append(
                            type(f)(
                                marginal_cost=base_mc[f.name] * gmul,
                                capacity=f.capacity,
                                fixed_cost=f.fixed_cost,
                                outage_rate=f.outage_rate,
                                name=f.name,
                            )
                        )
                    else:
                        new_firms.append(f)
                cfg = replace(
                    base_scenario,
                    firms=tuple(new_firms),
                    wind_capacity_mw=base_scenario.wind_capacity_mw * wmul,
                    mechanism=factory(),
                )
                res = run_scenario(cfg)
                mm = missing_money(res)
                rows.append(
                    {
                        "elasticity": eps,
                        "wind_multiplier": wmul,
                        "gas_cost_multiplier": gmul,
                        "expected_price": res.expected_price,
                        "expected_quantity_mw": res.expected_quantity,
                        "annual_welfare": _annualize(res.expected_welfare),
                        "annual_consumer_cost": _annualize(
                            res.expected_consumer_payment_for_capacity
                        ),
                        "fleet_missing_money": mm.fleet_gap_per_year,
                        "fraction_firms_short": mm.fraction_firms_short,
                    }
                )
    return pd.DataFrame(rows)
