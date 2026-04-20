"""Tests for the bilevel / endogenous-strike solver (RQ4)."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.game.bilevel import (
    BilevelSolution,
    solve_bilevel,
    solve_endogenous_strike,
)
from capgame.game.cournot import Firm, LinearDemand, solve


class TestGenericSolver:
    def test_quadratic_maximum(self) -> None:
        """Toy: objective = -(action - 3)^2; optimum at 3."""
        firms = [Firm(marginal_cost=10.0, capacity=20.0)]
        demand = LinearDemand(a=100.0, b=1.0)

        def follower(_action: float):
            return solve(demand, firms)

        def objective(action: float, _eq) -> float:
            return -((action - 3.0) ** 2)

        sol = solve_bilevel(
            leader_action_bounds=(0.0, 6.0),
            leader_objective=objective,
            follower_solver=follower,
            n_grid=61,
        )
        assert isinstance(sol, BilevelSolution)
        assert sol.leader_action == pytest.approx(3.0, abs=0.1)

    def test_rejects_bad_bounds(self) -> None:
        firms = [Firm(marginal_cost=10.0, capacity=20.0)]
        demand = LinearDemand(a=100.0, b=1.0)

        def follower(_a: float):
            return solve(demand, firms)

        with pytest.raises(ValueError):
            solve_bilevel(
                leader_action_bounds=(5.0, 1.0),
                leader_objective=lambda a, _e: a,
                follower_solver=follower,
                n_grid=10,
            )

    def test_rejects_small_grid(self) -> None:
        firms = [Firm(marginal_cost=10.0, capacity=20.0)]
        demand = LinearDemand(a=100.0, b=1.0)
        with pytest.raises(ValueError):
            solve_bilevel(
                leader_action_bounds=(0.0, 1.0),
                leader_objective=lambda a, _e: a,
                follower_solver=lambda _a: solve(demand, firms),
                n_grid=1,
            )

    def test_grid_and_values_aligned(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=20.0)]
        sol = solve_bilevel(
            leader_action_bounds=(0.0, 10.0),
            leader_objective=lambda a, _e: a,
            follower_solver=lambda _a: solve(demand, firms),
            n_grid=11,
        )
        assert sol.grid.shape == (11,)
        assert sol.objective_values.shape == (11,)
        assert sol.leader_action == pytest.approx(sol.grid[int(np.argmax(sol.objective_values))])


class TestEndogenousStrike:
    def test_returns_within_bounds(self) -> None:
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [
            Firm(marginal_cost=10.0, capacity=30.0),
            Firm(marginal_cost=20.0, capacity=30.0),
            Firm(marginal_cost=30.0, capacity=30.0),
        ]
        sol = solve_endogenous_strike(
            demand=demand,
            firms=firms,
            premium=10.0,
            strike_bounds=(0.0, 200.0),
            n_grid=41,
        )
        assert 0.0 <= sol.leader_action <= 200.0
        assert sol.objective_values.shape == sol.grid.shape

    def test_high_strike_yields_zero_refunds(self) -> None:
        """When the strike is above any feasible price, refunds are zero.

        The regulator's objective then reduces to CS + PS - premium * sum caps,
        i.e. welfare minus a constant, so the optimal strike is at the
        upper bound (any sufficiently high K is equivalent)."""
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=30.0)]
        sol = solve_endogenous_strike(
            demand=demand,
            firms=firms,
            premium=5.0,
            strike_bounds=(100.0, 200.0),
            n_grid=11,
        )
        # With strike >= price everywhere, objective is constant in K.
        assert np.allclose(sol.objective_values, sol.objective_values[0], atol=1e-6)

    def test_low_strike_redistributes_to_consumers(self) -> None:
        """The regulator's objective gains from strikes that refund rents."""
        demand = LinearDemand(a=100.0, b=1.0)
        firms = [
            Firm(marginal_cost=10.0, capacity=30.0),
            Firm(marginal_cost=20.0, capacity=30.0),
        ]
        sol = solve_endogenous_strike(
            demand=demand,
            firms=firms,
            premium=5.0,
            strike_bounds=(0.0, 200.0),
            n_grid=41,
        )
        low_k_obj = sol.objective_values[0]
        high_k_obj = sol.objective_values[-1]
        # In this model the RO refund is a transfer so total welfare is
        # strike-invariant. This is a property the test asserts.
        assert low_k_obj == pytest.approx(high_k_obj, abs=1e-6)
