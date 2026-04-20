"""Supply-function equilibrium (SFE) solver -- placeholder.

Supply-function equilibrium (Klemperer-Meyer, 1989; Green-Newbery, 1992)
models firms bidding linear supply schedules rather than scalar quantities,
yielding a richer family of equilibria than Cournot. The SFE solver is
scheduled for Phase 2b (see ``docs/ROADMAP.md``); the module is kept as a
deliberate placeholder so that public imports like
``from capgame.game import solve_sfe`` surface an informative error at
call time rather than an ``ImportError``.

Callers should not rely on this module existing in its current form. The
first real implementation will be a linear-SFE iteration over a discretized
price grid, following Baldick-Hogan-Newberry (2000).
"""

from __future__ import annotations

from collections.abc import Sequence

from capgame.game.cournot import CournotEquilibrium, Firm, LinearDemand

__all__ = ["solve_sfe"]


def solve_sfe(
    demand: LinearDemand,
    firms: Sequence[Firm],
) -> CournotEquilibrium:
    """Solve a linear supply-function equilibrium. Not yet implemented.

    Raises
    ------
    NotImplementedError
        Always. The function exists so that the import succeeds and static
        type-checkers see a typed public API; the body is deferred.
    """
    raise NotImplementedError(
        "solve_sfe is scheduled for Phase 2b. Use "
        "capgame.game.cournot.solve (oligopoly), "
        "capgame.game.market_structure.solve_cartel (collusive), "
        "or capgame.game.market_structure.solve_monopoly for the cases "
        "available today. Track progress in docs/ROADMAP.md."
    )
