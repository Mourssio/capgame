"""Static Nash-Cournot equilibrium with capacity constraints.

Formulation
-----------
Firms i = 1..N choose quantities q_i >= 0, q_i <= cap_i, to maximize profit:

    pi_i(q_i; q_{-i}) = P(Q) * q_i - C_i(q_i)

where Q = sum_j q_j, P(Q) = a - b*Q is inverse demand (a > 0, b > 0),
and C_i(q_i) = c_i * q_i is linear variable cost.

First-order conditions (KKT) give the mixed complementarity problem (MCP):

    0 <= q_i     perp    c_i - P(Q) + b*q_i + mu_i >= 0
    0 <= mu_i    perp    cap_i - q_i              >= 0

where mu_i is the dual on the capacity constraint.

For the unconstrained interior solution (all mu_i = 0), the classic closed form:

    q_i* = (a - N*c_i + sum_{j != i} c_j) / (b * (N + 1))

This module solves the general constrained problem via iterative best-response
(which is a contraction mapping under our assumptions), and provides the
closed-form solver for validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LinearDemand:
    """Inverse demand P(Q) = max(0, a - b*Q), with a, b > 0."""

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"Demand intercept a must be positive, got {self.a}")
        if self.b <= 0:
            raise ValueError(f"Demand slope b must be positive, got {self.b}")

    def price(self, Q: float) -> float:
        """Inverse demand, clipped at zero (no negative prices)."""
        return max(0.0, self.a - self.b * float(Q))

    def inverse(self, p: float) -> float:
        """Given a target price, return total quantity that clears demand."""
        return max(0.0, (self.a - p) / self.b)


@dataclass(frozen=True)
class Firm:
    """Single-technology firm with marginal cost and capacity cap.

    Parameters
    ----------
    marginal_cost
        Constant marginal production cost in $/MWh.
    capacity
        Installed capacity in MW. Zero-capacity firms are allowed (they
        simply do not produce).
    fixed_cost
        Annualized fixed cost per MW of capacity (used downstream by
        capacity mechanisms and investment models). Not used in the static
        Cournot subgame.
    outage_rate
        Forced-outage rate in [0, 1). Used by the adequacy layer, not the
        Cournot subgame.
    name
        Optional label, useful for plotting and reporting.
    """

    marginal_cost: float
    capacity: float
    fixed_cost: float = 0.0
    outage_rate: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.marginal_cost < 0:
            raise ValueError(f"marginal_cost must be >= 0, got {self.marginal_cost}")
        if self.capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {self.capacity}")
        if self.fixed_cost < 0:
            raise ValueError(f"fixed_cost must be >= 0, got {self.fixed_cost}")
        if not (0.0 <= self.outage_rate < 1.0):
            raise ValueError(f"outage_rate must be in [0, 1), got {self.outage_rate}")


@dataclass(frozen=True)
class CournotEquilibrium:
    """Solution to a static Cournot game."""

    quantities: npt.NDArray[np.float64]
    price: float
    profits: npt.NDArray[np.float64]
    binding: npt.NDArray[np.bool_]
    iterations: int = 0
    converged: bool = True

    @property
    def total_quantity(self) -> float:
        return float(self.quantities.sum())

    @property
    def hhi(self) -> float:
        """Herfindahl index on output shares, in [0, 10000]."""
        Q = self.total_quantity
        if Q <= 0.0:
            return 0.0
        shares = self.quantities / Q
        return float(10_000 * (shares**2).sum())

    @property
    def consumer_surplus(self) -> float:
        """Consumer surplus under linear demand P(Q) = a - b*Q.

        Computed as the area of the triangle between the demand curve and the
        equilibrium price, i.e. 0.5 * b * Q^2.  This property is populated by
        callers that have access to the demand parameters; use
        :func:`consumer_surplus` instead when demand is known.
        """
        return float("nan")


def consumer_surplus(demand: LinearDemand, eq: CournotEquilibrium) -> float:
    """Consumer surplus = 0.5 * b * Q^2 under linear demand."""
    Q = eq.total_quantity
    return 0.5 * demand.b * Q * Q


def _coerce_firms(firms: Sequence[Firm]) -> tuple[np.ndarray, np.ndarray, int]:
    if len(firms) == 0:
        raise ValueError("At least one firm is required.")
    c = np.array([f.marginal_cost for f in firms], dtype=float)
    cap = np.array([f.capacity for f in firms], dtype=float)
    return c, cap, len(firms)


def solve_unconstrained(demand: LinearDemand, firms: Sequence[Firm]) -> CournotEquilibrium:
    """Closed-form Cournot-Nash equilibrium ignoring capacity caps.

    Used both as a baseline and for validating the constrained solver in tests
    where the caps are known not to bind.

    Quantities are clipped at zero; if any firm has marginal cost exceeding
    the choke price in the resulting equilibrium, it drops out and the
    equilibrium of the remaining firms is computed.
    """
    c, _, N = _coerce_firms(firms)
    a, b = demand.a, demand.b

    sum_c = c.sum()
    q = (a - N * c + (sum_c - c)) / (b * (N + 1))

    if np.any(q < 0):
        active = q >= 0
        if not active.any():
            q = np.zeros_like(c)
        else:
            active_idx = np.where(active)[0]
            sub_firms = [firms[i] for i in active_idx]
            sub_eq = solve_unconstrained(demand, sub_firms)
            q = np.zeros_like(c)
            q[active_idx] = sub_eq.quantities

    Q = float(q.sum())
    p = demand.price(Q)
    profits = (p - c) * q
    binding = np.zeros(N, dtype=bool)
    return CournotEquilibrium(
        quantities=q,
        price=p,
        profits=profits,
        binding=binding,
        iterations=0,
        converged=True,
    )


def solve_constrained(
    demand: LinearDemand,
    firms: Sequence[Firm],
    tol: float = 1e-10,
    max_iter: int = 500,
    damping: float = 1.0,
) -> CournotEquilibrium:
    """Cournot-Nash equilibrium with capacity constraints.

    Uses Gauss-Seidel best-response iteration. For linear demand and linear
    costs, the best response given q_{-i} is:

        q_i = clip( (a - b * Q_{-i} - c_i) / (2b), 0, cap_i )

    This is a contraction mapping on the compact strategy set, and converges
    geometrically.

    Parameters
    ----------
    damping
        Convex combination weight in (0, 1] on the new iterate.
        ``damping=1`` is pure best-response; smaller values add stability in
        weakly-diagonal cases.
    """
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")

    c, cap, N = _coerce_firms(firms)
    a, b = demand.a, demand.b

    q = np.minimum(cap, np.maximum(0.0, (a - c) / (2.0 * b * N)))

    converged = False
    iteration = 0
    for k in range(1, max_iter + 1):
        iteration = k
        q_prev = q.copy()
        for i in range(N):
            Q_others = q.sum() - q[i]
            br = (a - b * Q_others - c[i]) / (2.0 * b)
            br_clipped = float(np.clip(br, 0.0, cap[i]))
            q[i] = (1.0 - damping) * q[i] + damping * br_clipped
        if np.max(np.abs(q - q_prev)) < tol:
            converged = True
            break

    Q = float(q.sum())
    p = demand.price(Q)
    profits = (p - c) * q
    binding = np.isclose(q, cap, atol=1e-6) & (cap > 0.0)
    return CournotEquilibrium(
        quantities=q,
        price=p,
        profits=profits,
        binding=binding,
        iterations=iteration,
        converged=converged,
    )


def solve(
    demand: LinearDemand,
    firms: Sequence[Firm],
    **kwargs: float,
) -> CournotEquilibrium:
    """Solve a Cournot subgame.

    Convenience wrapper: if the unconstrained equilibrium respects all caps,
    return it directly; otherwise invoke the constrained solver.
    """
    eq = solve_unconstrained(demand, firms)
    caps = np.array([f.capacity for f in firms])
    if np.all(eq.quantities <= caps + 1e-9):
        return eq
    return solve_constrained(demand, firms, **kwargs)
