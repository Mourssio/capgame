"""Market-structure variants of the static game (RQ3).

RQ3 in the project proposal asks whether mechanism rankings are robust to
changes in the competitive structure of the wholesale market. We implement
three structures; each returns a :class:`CournotEquilibrium` so the
downstream mechanism and welfare code does not need to know which structure
produced it.

* **Oligopoly** -- standard Cournot-Nash equilibrium (the existing solver).
* **Cartel** -- the firms coordinate to maximize joint profit. Dispatch is
  in merit order; the industry clearing price and total quantity are
  identical to the monopoly outcome, but profits are attributed per-firm by
  actual dispatch so the adequacy / mechanism code sees a fleet, not a
  single firm.
* **Monopoly** -- as a limiting case of cartel; kept as an explicit option
  because it's a standard benchmark and makes the UI language cleaner.

Formulation (cartel / monopoly)
-------------------------------
With linear demand ``P(Q) = a - b*Q`` and linear merit-order costs, joint
profit ``P(Q) * Q - sum_i c_i q_i`` with ``sum q_i = Q`` and ``q_i <= cap_i``
is maximized by dispatching cheapest units first up to the quantity ``Q*``
at which marginal revenue equals the marginal unit's cost:

    Q* = (a - c_marginal) / (2 b)

provided the implied ``Q*`` lies within the marginal tier's cumulative
capacity window. Otherwise we bind at a tier boundary and the marginal
unit's MC changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from capgame.game.cournot import CournotEquilibrium, Firm, LinearDemand, solve

MarketStructure = Literal["oligopoly", "cartel", "monopoly"]


def solve_joint_profit_max(
    demand: LinearDemand,
    firms: Sequence[Firm],
) -> CournotEquilibrium:
    """Merit-order dispatch that maximizes joint industry profit.

    Used for both the cartel and monopoly structures: they produce the
    same aggregate ``(Q, P)``; the per-firm split differs only in the
    monopoly limit where one firm owns the whole fleet.
    """
    if len(firms) == 0:
        raise ValueError("At least one firm is required.")

    order = sorted(range(len(firms)), key=lambda i: firms[i].marginal_cost)
    caps_sorted = np.array([firms[i].capacity for i in order], dtype=float)
    costs_sorted = np.array([firms[i].marginal_cost for i in order], dtype=float)
    total_cap = float(caps_sorted.sum())

    q_sorted = np.zeros(len(firms), dtype=float)
    cum = 0.0
    assigned = False

    for k, (cap_k, cost_k) in enumerate(zip(caps_sorted, costs_sorted, strict=True)):
        q_star = (demand.a - cost_k) / (2.0 * demand.b)
        lower = cum
        upper = cum + cap_k
        if q_star <= lower:
            break
        if q_star <= upper:
            for j in range(k):
                q_sorted[j] = caps_sorted[j]
            q_sorted[k] = max(0.0, q_star - cum)
            assigned = True
            break
        cum = upper

    if not assigned:
        price_at_full = demand.a - demand.b * total_cap
        if price_at_full >= costs_sorted[-1]:
            q_sorted = caps_sorted.copy()

    q = np.zeros(len(firms), dtype=float)
    for pos, i in enumerate(order):
        q[i] = q_sorted[pos]

    Q = float(q.sum())
    price = demand.price(Q)
    costs = np.array([f.marginal_cost for f in firms], dtype=float)
    caps = np.array([f.capacity for f in firms], dtype=float)
    profits = (price - costs) * q
    binding = np.isclose(q, caps, atol=1e-6) & (caps > 0.0)
    return CournotEquilibrium(
        quantities=q,
        price=price,
        profits=profits,
        binding=binding,
        iterations=0,
        converged=True,
    )


def solve_monopoly(
    demand: LinearDemand,
    firms: Sequence[Firm],
) -> CournotEquilibrium:
    """Single owner of the entire fleet; dispatch is merit-order.

    Mathematically identical to :func:`solve_joint_profit_max`; exposed as
    its own name for UI clarity and for future extensions where the
    monopoly case may diverge (e.g. different IC constraints).
    """
    return solve_joint_profit_max(demand, firms)


def solve_cartel(
    demand: LinearDemand,
    firms: Sequence[Firm],
) -> CournotEquilibrium:
    """Coordinated joint-profit-maximizing fleet."""
    return solve_joint_profit_max(demand, firms)


def solve_market(
    demand: LinearDemand,
    firms: Sequence[Firm],
    structure: MarketStructure = "oligopoly",
) -> CournotEquilibrium:
    """Dispatch to the appropriate static solver for ``structure``."""
    if structure == "oligopoly":
        return solve(demand, firms)
    if structure == "cartel":
        return solve_cartel(demand, firms)
    if structure == "monopoly":
        return solve_monopoly(demand, firms)
    raise ValueError(
        f"Unknown market structure {structure!r}; expected one of "
        "'oligopoly', 'cartel', 'monopoly'."
    )
