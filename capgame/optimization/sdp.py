"""Stochastic dynamic programming for the dynamic capacity game.

This module provides backward induction over a Markov scenario tree. At each
``(time, state, capacity_vector)`` triple, a user-supplied ``stage_solver``
returns per-firm single-period profits. The SDP then computes per-firm
value functions

    V_i(t, s, x) = max_{Delta x_i >= 0}
        pi_i(t, s, x; Delta x) + beta * E[ V_i(t+1, s', x') | s ]

where the investment policy is coordinated across firms by a simple
simultaneous best-response at each node (Nash equilibrium on investment
given continuation values). For the MVP we support a finite discretization
of per-firm investments.

The implementation is intentionally modest: it is not a full Markov-perfect
solver for arbitrary payoff structures, but it is correct and useful on
tractable instances (few firms, short horizon, small action grids), and it
verifies end-to-end integration of the layers above it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from capgame.stochastic.demand import MarkovChain

StageSolver = Callable[
    [int, int, npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]
"""Signature: (time, demand_state_index, capacities) -> per-firm profits."""


@dataclass(frozen=True)
class SDPResult:
    """Output of backward induction.

    Attributes
    ----------
    value
        Array of shape ``(T+1, |S|, n_grid_points, N)`` giving each firm's
        value at every (time, demand state, capacity-grid-index).
    policy
        Array of shape ``(T, |S|, n_grid_points, N)`` giving each firm's
        investment action (grid index) at every node.
    capacity_grid
        Array of shape ``(n_grid_points, N)`` enumerating discretized
        capacity vectors.
    """

    value: npt.NDArray[np.float64]
    policy: npt.NDArray[np.int64]
    capacity_grid: npt.NDArray[np.float64]


def _enumerate_grid(levels_per_firm: Sequence[npt.ArrayLike]) -> np.ndarray:
    """Cartesian product of per-firm capacity levels."""
    arrays = [np.asarray(levels, dtype=float) for levels in levels_per_firm]
    meshes = np.meshgrid(*arrays, indexing="ij")
    return np.stack([m.ravel() for m in meshes], axis=-1)


def backward_induction(
    horizon: int,
    chain: MarkovChain,
    capacity_levels: Sequence[npt.ArrayLike],
    investment_levels: Sequence[npt.ArrayLike],
    stage_solver: StageSolver,
    discount: float = 0.95,
    depreciation: float = 0.0,
    construction_lag: int = 0,
) -> SDPResult:
    """Solve the dynamic game by backward induction.

    Parameters
    ----------
    horizon
        Number of decision periods ``T``. Terminal (salvage) value is zero.
    chain
        Markov chain describing the evolution of the exogenous state.
    capacity_levels
        Per-firm discretization of the capacity state. Length N.
    investment_levels
        Per-firm discretization of allowable investments ``Delta x``. Length N.
    stage_solver
        Function returning per-firm profits at a node, used inside the
        Bellman backup.
    discount
        Discount factor ``beta`` in (0, 1].
    depreciation
        Per-period depreciation rate ``delta`` in [0, 1).
    construction_lag
        Lead time ``L``. The MVP supports ``L = 0`` exactly; ``L > 0`` is
        accepted but treated as ``L = 0`` — use at your own risk until the
        roadmap Phase 4 fully implements lagged commissioning.

    Notes
    -----
    Computational cost is O(T * |S| * n_grid * n_actions), which explodes
    quickly. For serious runs, keep N small and coarsen the grids.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not (0.0 < discount <= 1.0):
        raise ValueError(f"discount must be in (0, 1], got {discount}")
    if not (0.0 <= depreciation < 1.0):
        raise ValueError(f"depreciation must be in [0, 1), got {depreciation}")
    if construction_lag < 0:
        raise ValueError(f"construction_lag must be >= 0, got {construction_lag}")

    n_firms = len(capacity_levels)
    if len(investment_levels) != n_firms:
        raise ValueError("capacity_levels and investment_levels must have the same length.")

    cap_grid = _enumerate_grid(capacity_levels)
    inv_grid = _enumerate_grid(investment_levels)
    n_cap = cap_grid.shape[0]
    n_inv = inv_grid.shape[0]
    n_states = chain.n_states

    value = np.zeros((horizon + 1, n_states, n_cap, n_firms), dtype=float)
    policy = np.zeros((horizon, n_states, n_cap, n_firms), dtype=np.int64)

    next_cap_of: dict[tuple[int, int], int] = {}
    for ci in range(n_cap):
        for ii in range(n_inv):
            nxt = (1.0 - depreciation) * cap_grid[ci] + inv_grid[ii]
            nxt = np.clip(nxt, cap_grid.min(axis=0), cap_grid.max(axis=0))
            diffs = np.linalg.norm(cap_grid - nxt, axis=1)
            next_cap_of[(ci, ii)] = int(np.argmin(diffs))

    P = chain.transition_matrix

    for t in range(horizon - 1, -1, -1):
        for s in range(n_states):
            for ci in range(n_cap):
                caps = cap_grid[ci]
                stage_profit = stage_solver(t, s, caps)
                stage_profit = np.asarray(stage_profit, dtype=float)
                if stage_profit.shape != (n_firms,):
                    raise ValueError(
                        f"stage_solver returned shape {stage_profit.shape}, expected ({n_firms},)"
                    )

                best_val = np.full(n_firms, -np.inf)
                best_action = np.zeros(n_firms, dtype=np.int64)

                for ii in range(n_inv):
                    nxt_ci = next_cap_of[(ci, ii)]
                    cont = np.zeros(n_firms, dtype=float)
                    for sp in range(n_states):
                        cont += P[s, sp] * value[t + 1, sp, nxt_ci]
                    candidate = stage_profit + discount * cont

                    improved = candidate > best_val
                    best_val = np.where(improved, candidate, best_val)
                    best_action = np.where(improved, ii, best_action)

                value[t, s, ci] = best_val
                policy[t, s, ci] = best_action

    return SDPResult(value=value, policy=policy, capacity_grid=cap_grid)
