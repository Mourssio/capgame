"""Markov-perfect equilibrium (MPE) scaffolding.

In the closed-loop dynamic game, strategies depend only on the current Markov
state (capacity vector and demand realization). Computing MPE requires a
fixed-point iteration over per-firm value functions; in Khalfallah (2011) this
is operationalized by backward induction coupled with a per-period Cournot MCP.

The MPE computation is driven by :mod:`capgame.optimization.sdp` — this module
exists for symbolic completeness and to collect MPE-specific utilities.
"""

from __future__ import annotations

from capgame.optimization.sdp import backward_induction

__all__ = ["backward_induction"]
