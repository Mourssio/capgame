"""Mixed-complementarity-problem wrapper around the Cournot KKT system.

The linear-demand, linear-cost Cournot problem admits a closed-form best
response, so :mod:`capgame.game.cournot` solves the MCP directly by
iteration. This module exposes the solution under an "MCP" name to keep the
research interface close to the Khalfallah paper and to make the eventual
switch to a general-purpose MCP solver (e.g. PATH via ``pyomo.mpec``)
localized to a single file.
"""

from __future__ import annotations

from collections.abc import Sequence

from capgame.game.cournot import (
    CournotEquilibrium,
    Firm,
    LinearDemand,
    solve_constrained,
)


def solve_cournot_mcp(
    demand: LinearDemand,
    firms: Sequence[Firm],
    **kwargs: float,
) -> CournotEquilibrium:
    """Solve the Cournot KKT mixed-complementarity problem.

    Currently delegates to the best-response iteration in
    :func:`capgame.game.cournot.solve_constrained`, which is provably
    equivalent for this problem class. A future release may swap in
    ``pyomo.mpec`` or direct PATH bindings.
    """
    return solve_constrained(demand, firms, **kwargs)
