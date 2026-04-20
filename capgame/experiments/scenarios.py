"""Single-entry-point scenario runner.

One ``ScenarioConfig`` in, one ``ScenarioResult`` out. This is the function
the Streamlit UI, the notebooks, and external callers use; it is the
contract that makes the rest of the package swappable. Every downstream
surface (charts, tables, policy comparisons) reads a ``ScenarioResult``
rather than reaching into the six lower-level modules individually.

Scope (v0.1)
------------
* Static, one-period Cournot subgame under a chosen market structure.
* Optional renewable-availability Markov chain: the scenario is solved in
  each renewable state and aggregated under the stationary distribution.
* Optional fleet-level forced-outage rates for adequacy metrics.
* Any ``Mechanism`` object satisfying :class:`capgame.mechanisms.base.Mechanism`.

Deferred to v0.2 (SDP integration)
----------------------------------
* Multi-period dynamics and investment decisions -- handled by
  :mod:`capgame.optimization.sdp` today, but not yet wired into a
  ``ScenarioConfig`` field. Will become ``config.horizon`` once calibration
  lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from capgame.adequacy.eue import expected_unserved_energy
from capgame.adequacy.lole import loss_of_load_expectation
from capgame.adequacy.reserve_margin import capacity_required, reserve_margin
from capgame.game.cournot import (
    CournotEquilibrium,
    Firm,
    LinearDemand,
    consumer_surplus,
)
from capgame.game.market_structure import MarketStructure, solve_market
from capgame.mechanisms.base import Mechanism, MechanismOutcome
from capgame.stochastic.demand import MarkovChain
from capgame.stochastic.renewables import RenewableState

AdequacyCriterion = Literal["reserve_margin", "lole", "eue"]


@dataclass(frozen=True)
class ScenarioConfig:
    """Everything needed to run one scenario.

    Parameters
    ----------
    demand
        Per-period inverse demand. When a renewable chain is supplied, this
        represents gross demand; the scenario runner subtracts renewable
        output to form residual demand in each renewable state.
    firms
        Dispatchable (thermal) fleet.
    mechanism
        Any object satisfying the :class:`Mechanism` protocol.
    market_structure
        ``"oligopoly"`` (default), ``"cartel"``, or ``"monopoly"``.
    renewable_chain
        Optional :class:`MarkovChain` ``[RenewableState]``; when provided the
        scenario is resolved in each state and aggregated under the
        stationary distribution.
    wind_capacity_mw, solar_capacity_mw
        Installed renewable nameplate capacity (MW). Ignored when
        ``renewable_chain`` is ``None``.
    outage_rates
        Per-firm forced-outage rates in ``[0, 1)``; when provided, adequacy
        metrics (LOLE, EUE) are populated.
    target_reserve_margin
        Reserve-margin target (e.g. ``0.15`` for 15%) used to annotate the
        adequacy result with the corresponding required capacity.
    hours_per_period
        Period length in hours; forwarded to mechanisms that need it.
    """

    demand: LinearDemand
    firms: tuple[Firm, ...]
    mechanism: Mechanism
    market_structure: MarketStructure = "oligopoly"
    renewable_chain: MarkovChain[RenewableState] | None = None
    wind_capacity_mw: float = 0.0
    solar_capacity_mw: float = 0.0
    outage_rates: tuple[float, ...] | None = None
    target_reserve_margin: float = 0.15
    hours_per_period: float = 1.0

    def __post_init__(self) -> None:
        if len(self.firms) == 0:
            raise ValueError("At least one firm is required.")
        if self.wind_capacity_mw < 0 or self.solar_capacity_mw < 0:
            raise ValueError("Renewable capacities must be non-negative.")
        if self.outage_rates is not None and len(self.outage_rates) != len(self.firms):
            raise ValueError(
                f"outage_rates length {len(self.outage_rates)} does not match "
                f"firms length {len(self.firms)}"
            )
        if self.hours_per_period <= 0:
            raise ValueError("hours_per_period must be > 0.")


@dataclass(frozen=True)
class StateOutcome:
    """Per-(renewable) state snapshot inside a :class:`ScenarioResult`."""

    label: str
    probability: float
    residual_demand: LinearDemand
    equilibrium: CournotEquilibrium
    mechanism_outcome: MechanismOutcome
    consumer_surplus: float
    producer_surplus: float
    consumer_payment_for_capacity: float
    welfare: float


@dataclass(frozen=True)
class AdequacyReport:
    total_capacity_mw: float
    peak_load_mw: float
    reserve_margin: float
    target_reserve_margin: float
    capacity_required_mw: float
    lole_hours_per_year: float | None
    eue_mwh_per_year: float | None


@dataclass(frozen=True)
class ScenarioResult:
    """Aggregate answer for one scenario, ready for the UI or a notebook."""

    config: ScenarioConfig
    states: tuple[StateOutcome, ...]
    expected_price: float
    expected_quantity: float
    expected_consumer_surplus: float
    expected_producer_surplus: float
    expected_consumer_payment_for_capacity: float
    expected_welfare: float
    adequacy: AdequacyReport
    per_firm_expected_quantity: np.ndarray = field(repr=False)
    per_firm_expected_net_profit: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class MissingMoneyReport:
    """Per-firm and fleet annualized missing-money diagnostic.

    Missing money measures the gap between a firm's annualized net
    revenue under the current mechanism and its annualized fixed-cost
    revenue requirement. Negative values mean the firm cannot justify
    its fixed cost from market revenues alone and would exit in the
    long run; positive values mean the mechanism is paying a
    rent above break-even.

    Attributes
    ----------
    per_firm_gap_per_mw_year
        ``(net_annual_revenue - fixed_cost * capacity) / capacity`` in
        $/MW-yr, one entry per firm in the same order as
        ``config.firms``. Negative = missing money.
    per_firm_annual_net_revenue
        Expected net revenue (energy margin + capacity payments -
        refunds) annualized to $/yr.
    per_firm_annual_fixed_requirement
        ``fixed_cost * capacity`` in $/yr.
    fleet_gap_per_year
        Sum of per-firm gaps weighted by capacity; total fleet
        missing money in $/yr. Negative = fleet is underpaid.
    fraction_firms_short
        Share of firms with a negative gap.
    largest_deficit_firm
        Name of the firm with the most negative gap (``None`` if all
        non-negative).
    """

    per_firm_gap_per_mw_year: np.ndarray = field(repr=False)
    per_firm_annual_net_revenue: np.ndarray = field(repr=False)
    per_firm_annual_fixed_requirement: np.ndarray = field(repr=False)
    fleet_gap_per_year: float
    fraction_firms_short: float
    largest_deficit_firm: str | None


def missing_money(
    result: ScenarioResult,
    periods_per_year: float = 8760.0,
) -> MissingMoneyReport:
    """Compute annualized missing money for every firm in a scenario.

    ``per_firm_expected_net_profit`` from :func:`run_scenario` is the
    per-period expected net profit in dollars. We annualize by
    multiplying by ``periods_per_year`` (default 8760 for hourly
    snapshots) and compare against each firm's fixed-cost revenue
    requirement ``fixed_cost * capacity``.

    The result is the object that makes mechanism comparison concrete:
    a mechanism that closes a large missing-money gap with a small
    consumer-cost impact is Pareto-preferable to one that overpays.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")

    firms = result.config.firms
    net_annual = np.asarray(result.per_firm_expected_net_profit, dtype=float) * periods_per_year
    fixed_req = np.array([f.fixed_cost * f.capacity for f in firms], dtype=float)
    caps = np.array([f.capacity for f in firms], dtype=float)

    gap_total = net_annual - fixed_req
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_per_mw = np.where(caps > 0, gap_total / caps, 0.0)

    short_mask = gap_total < 0.0
    fraction_short = float(short_mask.mean()) if short_mask.size else 0.0
    if short_mask.any():
        worst_idx = int(np.argmin(gap_total))
        worst_name = firms[worst_idx].name or f"firm_{worst_idx}"
    else:
        worst_name = None

    return MissingMoneyReport(
        per_firm_gap_per_mw_year=gap_per_mw,
        per_firm_annual_net_revenue=net_annual,
        per_firm_annual_fixed_requirement=fixed_req,
        fleet_gap_per_year=float(gap_total.sum()),
        fraction_firms_short=fraction_short,
        largest_deficit_firm=worst_name,
    )


def _build_residual_demand(
    demand: LinearDemand,
    renewable_state: RenewableState | None,
    wind_mw: float,
    solar_mw: float,
) -> LinearDemand:
    """Subtract renewable must-take output from demand by shifting intercept.

    With gross inverse demand ``P(Q_gross) = a - b*Q_gross`` and
    must-take renewable output ``R``, thermal dispatch satisfies
    ``Q_thermal = Q_gross - R`` so the residual curve is
    ``P(Q_thermal) = a - b*(Q_thermal + R) = (a - b*R) - b*Q_thermal``.
    If ``R`` exceeds the choke quantity the residual intercept goes
    non-positive; we clamp at a tiny positive value so the solver does not
    divide by zero.
    """
    if renewable_state is None:
        return demand
    r = renewable_state.available_output(wind_mw, solar_mw)
    new_intercept = demand.a - demand.b * r
    if new_intercept <= 0.0:
        new_intercept = 1e-9
    return LinearDemand(a=new_intercept, b=demand.b)


def _evaluate_single_state(
    demand: LinearDemand,
    firms: Sequence[Firm],
    mechanism: Mechanism,
    structure: MarketStructure,
    label: str,
    probability: float,
) -> StateOutcome:
    eq = solve_market(demand, firms, structure)
    caps = [f.capacity for f in firms]
    outcome = mechanism.apply(eq, caps)

    cs = consumer_surplus(demand, eq)
    ps = float(np.asarray(outcome.net_profits).sum())
    consumer_pay = float(outcome.consumer_cost)
    welfare = cs + ps - consumer_pay
    return StateOutcome(
        label=label,
        probability=probability,
        residual_demand=demand,
        equilibrium=eq,
        mechanism_outcome=outcome,
        consumer_surplus=cs,
        producer_surplus=ps,
        consumer_payment_for_capacity=consumer_pay,
        welfare=welfare,
    )


def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Execute one scenario end-to-end and return a typed result."""
    if config.renewable_chain is None:
        states_and_probs: list[tuple[str, float, RenewableState | None]] = [
            ("deterministic", 1.0, None)
        ]
    else:
        pi = config.renewable_chain.stationary_distribution()
        states_and_probs = [
            (s.name, float(p), s) for s, p in zip(config.renewable_chain.states, pi, strict=True)
        ]

    state_outcomes: list[StateOutcome] = []
    n = len(config.firms)
    qsum = np.zeros(n, dtype=float)
    profit_sum = np.zeros(n, dtype=float)
    price = 0.0
    quantity = 0.0
    cs_sum = 0.0
    ps_sum = 0.0
    pay_sum = 0.0
    welfare_sum = 0.0

    for label, prob, rs in states_and_probs:
        residual = _build_residual_demand(
            config.demand, rs, config.wind_capacity_mw, config.solar_capacity_mw
        )
        outcome = _evaluate_single_state(
            residual,
            config.firms,
            config.mechanism,
            config.market_structure,
            label,
            prob,
        )
        state_outcomes.append(outcome)
        qsum += prob * np.asarray(outcome.equilibrium.quantities)
        profit_sum += prob * np.asarray(outcome.mechanism_outcome.net_profits)
        price += prob * outcome.equilibrium.price
        quantity += prob * outcome.equilibrium.total_quantity
        cs_sum += prob * outcome.consumer_surplus
        ps_sum += prob * outcome.producer_surplus
        pay_sum += prob * outcome.consumer_payment_for_capacity
        welfare_sum += prob * outcome.welfare

    total_cap = float(sum(f.capacity for f in config.firms))
    peak_load = float(config.demand.a / config.demand.b)
    rm = reserve_margin(total_cap, peak_load)
    cap_req = capacity_required(peak_load, config.target_reserve_margin)

    lole_val: float | None = None
    eue_val: float | None = None
    if config.outage_rates is not None and peak_load > 0.0:
        caps = [f.capacity for f in config.firms]
        lole_val = loss_of_load_expectation(
            caps,
            list(config.outage_rates),
            demand_distribution=[(peak_load, 1.0)],
            periods_per_year=8760.0,
        )
        eue_val = expected_unserved_energy(
            caps,
            list(config.outage_rates),
            demand_distribution=[(peak_load, 1.0)],
            periods_per_year=8760.0,
        )

    return ScenarioResult(
        config=config,
        states=tuple(state_outcomes),
        expected_price=price,
        expected_quantity=quantity,
        expected_consumer_surplus=cs_sum,
        expected_producer_surplus=ps_sum,
        expected_consumer_payment_for_capacity=pay_sum,
        expected_welfare=welfare_sum,
        adequacy=AdequacyReport(
            total_capacity_mw=total_cap,
            peak_load_mw=peak_load,
            reserve_margin=rm,
            target_reserve_margin=config.target_reserve_margin,
            capacity_required_mw=cap_req,
            lole_hours_per_year=lole_val,
            eue_mwh_per_year=eue_val,
        ),
        per_firm_expected_quantity=qsum,
        per_firm_expected_net_profit=profit_sum,
    )
