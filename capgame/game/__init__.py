"""L1 equilibrium solvers.

Each solver in this layer accepts data contracts defined in :mod:`capgame.stochastic`
and returns a market outcome (prices, quantities, profits). Solvers must be pure
functions of their inputs: no global state, no I/O.
"""

from __future__ import annotations

from capgame.game.cournot import (
    CournotEquilibrium,
    Firm,
    LinearDemand,
    solve_constrained,
    solve_unconstrained,
)

__all__ = [
    "CournotEquilibrium",
    "Firm",
    "LinearDemand",
    "solve_constrained",
    "solve_unconstrained",
]
