"""L5 numerical methods."""

from __future__ import annotations

from capgame.optimization.mcp_solver import solve_cournot_mcp
from capgame.optimization.sdp import SDPResult, backward_induction

__all__ = [
    "SDPResult",
    "backward_induction",
    "solve_cournot_mcp",
]
