"""Supply-function equilibrium (SFE) solver.

Placeholder for Phase 2b / research extension. Supply-function equilibrium
(Klemperer-Meyer, 1989; Green-Newbery, 1992) models firms bidding linear
supply schedules rather than quantities, yielding a richer family of
equilibria than Cournot. Full implementation is deferred to a future phase.
"""

from __future__ import annotations

from collections.abc import Sequence

from capgame.game.cournot import CournotEquilibrium, Firm, LinearDemand


def solve_sfe(
    demand: LinearDemand,
    firms: Sequence[Firm],
) -> CournotEquilibrium:
    """Solve a linear supply-function equilibrium. Not yet implemented."""
    raise NotImplementedError("SFE solver is planned for a future phase. See docs/ROADMAP.md.")
