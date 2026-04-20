"""Bilevel / Stackelberg equilibrium with a regulator leader (RQ4).

Upper level: the regulator chooses a mechanism design parameter (e.g.
strike price ``K``). Lower level: firms play the Cournot subgame under the
mechanism, and their equilibrium feeds back into the regulator's welfare
objective.

We solve by grid search on the leader's action. For the small problems
needed by the v0.1 UI (one scalar design variable, a few dozen grid points,
closed-form Cournot) this is fast and unambiguously correct; the trade-off
is only accuracy at the grid resolution, which we document on the output.

``solve_endogenous_strike`` is a convenience specialization for the
reliability-options use case in the proposal's RQ4; it is the canonical
entry point the UI calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from capgame.game.cournot import (
    CournotEquilibrium,
    Firm,
    LinearDemand,
    consumer_surplus,
    solve,
)
from capgame.mechanisms.reliability_options import ReliabilityOption


@dataclass(frozen=True)
class BilevelSolution:
    """Result of a one-dimensional Stackelberg grid search.

    Attributes
    ----------
    leader_action
        The grid-optimal leader action (e.g. strike price in $/MWh).
    follower_equilibrium
        The follower Cournot equilibrium at ``leader_action``.
    leader_objective
        The leader's realized objective value at the optimum.
    grid
        The full grid of actions evaluated, in the order searched.
    objective_values
        ``leader_objective`` evaluated at every grid point; aligned 1:1
        with ``grid``. Exposed so the UI can plot the objective curve.
    """

    leader_action: float
    follower_equilibrium: CournotEquilibrium
    leader_objective: float
    grid: np.ndarray
    objective_values: np.ndarray


def solve_bilevel(
    leader_action_bounds: tuple[float, float],
    leader_objective: Callable[[float, CournotEquilibrium], float],
    follower_solver: Callable[[float], CournotEquilibrium],
    n_grid: int = 41,
) -> BilevelSolution:
    """Generic one-dimensional Stackelberg solver.

    Parameters
    ----------
    leader_action_bounds
        ``(low, high)`` admissible range for the leader's decision.
    leader_objective
        Function ``(action, follower_eq) -> leader_utility``. The
        regulator typically maximizes welfare net of consumer payments.
    follower_solver
        Maps an action to the resulting follower equilibrium.
    n_grid
        Number of grid points (inclusive of endpoints).
    """
    lo, hi = leader_action_bounds
    if lo >= hi:
        raise ValueError(f"leader_action_bounds must satisfy lo < hi, got {(lo, hi)}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    grid = np.linspace(lo, hi, n_grid)
    values = np.empty(n_grid, dtype=float)
    equilibria: list[CournotEquilibrium] = []

    for idx, action in enumerate(grid):
        eq = follower_solver(float(action))
        equilibria.append(eq)
        values[idx] = leader_objective(float(action), eq)

    best_idx = int(np.argmax(values))
    return BilevelSolution(
        leader_action=float(grid[best_idx]),
        follower_equilibrium=equilibria[best_idx],
        leader_objective=float(values[best_idx]),
        grid=grid,
        objective_values=values,
    )


def solve_endogenous_strike(
    demand: LinearDemand,
    firms: Sequence[Firm],
    premium: float,
    coverage: float = 1.0,
    hours_per_period: float = 1.0,
    strike_bounds: tuple[float, float] = (0.0, 200.0),
    n_grid: int = 41,
) -> BilevelSolution:
    """Regulator chooses strike ``K`` to maximize welfare net of consumer cost.

    Specialized to the reliability-options mechanism: the regulator's
    objective is

        W(K) = CS(Q*(K)) + PS(Q*(K)) - ConsumerCost(K)

    where the Cournot subgame ``Q*(K)`` is independent of ``K`` under the
    standard RO assumption that the option refund is a transfer, not a
    dispatch distortion. We nevertheless evaluate the subgame once per
    grid point so the API stays correct when that assumption is relaxed.

    Parameters
    ----------
    premium
        Fixed per-period premium in $/MW used by the regulator; the
        leader's only choice variable here is ``K``.
    strike_bounds
        Admissible strike range in $/MWh.
    """
    caps = [f.capacity for f in firms]

    def follower(_strike: float) -> CournotEquilibrium:
        return solve(demand, firms)

    def objective(strike: float, eq: CournotEquilibrium) -> float:
        option = ReliabilityOption(
            premium=premium,
            strike_price=strike,
            coverage=coverage,
            hours_per_period=hours_per_period,
        )
        outcome = option.apply(eq, caps)
        cs = consumer_surplus(demand, eq)
        ps = float(np.asarray(outcome.net_profits).sum())
        return cs + ps - float(outcome.consumer_cost)

    return solve_bilevel(
        leader_action_bounds=strike_bounds,
        leader_objective=objective,
        follower_solver=follower,
        n_grid=n_grid,
    )
