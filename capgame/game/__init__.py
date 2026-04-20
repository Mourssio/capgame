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
    consumer_surplus,
    solve,
    solve_constrained,
    solve_unconstrained,
)
from capgame.game.market_structure import (
    MarketStructure,
    solve_cartel,
    solve_market,
    solve_monopoly,
)

__all__ = [
    "CournotEquilibrium",
    "Firm",
    "LinearDemand",
    "MarketStructure",
    "consumer_surplus",
    "solve",
    "solve_cartel",
    "solve_constrained",
    "solve_market",
    "solve_monopoly",
    "solve_unconstrained",
]
