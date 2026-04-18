"""Bilevel / Stackelberg equilibrium with a regulator leader.

Used by the endogenous-strike-price extension (RQ4). The upper level is the
regulator choosing a mechanism parameter (strike price, procurement target)
subject to a reliability constraint; the lower level is the firms' Cournot
subgame. Full implementation is deferred.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from capgame.game.cournot import CournotEquilibrium, Firm, LinearDemand


@dataclass(frozen=True)
class BilevelSolution:
    leader_action: float
    follower_equilibrium: CournotEquilibrium
    leader_objective: float


def solve_bilevel(
    demand: LinearDemand,
    firms: Sequence[Firm],
    leader_action_bounds: tuple[float, float],
    leader_objective: Callable[[float, CournotEquilibrium], float],
    follower_solver: Callable[[float], CournotEquilibrium],
    n_grid: int = 41,
) -> BilevelSolution:
    """Solve a bilevel problem by grid search on the leader's action.

    A minimal but correct Stackelberg solver: enumerate a uniform grid of
    leader actions, solve the follower subgame at each, and select the
    leader-optimal point.

    Parameters
    ----------
    leader_action_bounds
        (low, high) admissible range for the leader's decision.
    leader_objective
        Function ``(action, follower_eq) -> leader_utility``. The regulator
        typically maximizes some welfare-minus-payment objective.
    follower_solver
        Maps a leader action to a follower Cournot equilibrium. This is
        typically a closure over a mechanism.
    """
    import numpy as np

    lo, hi = leader_action_bounds
    if lo >= hi:
        raise ValueError(f"leader_action_bounds must satisfy lo < hi, got {(lo, hi)}")
    grid = np.linspace(lo, hi, n_grid)

    best_action = lo
    best_eq = follower_solver(lo)
    best_obj = leader_objective(lo, best_eq)
    for action in grid[1:]:
        eq = follower_solver(float(action))
        obj = leader_objective(float(action), eq)
        if obj > best_obj:
            best_action = float(action)
            best_eq = eq
            best_obj = obj

    return BilevelSolution(
        leader_action=best_action,
        follower_equilibrium=best_eq,
        leader_objective=best_obj,
    )
