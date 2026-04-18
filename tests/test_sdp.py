"""Tests for stochastic dynamic programming (Phase 3)."""

from __future__ import annotations

import numpy as np

from capgame.optimization.sdp import backward_induction
from capgame.stochastic.demand import three_state_chain


def test_trivial_single_firm_no_investment() -> None:
    """With no investment options available, value is the discounted stream."""
    chain = three_state_chain()
    horizon = 3
    discount = 0.9

    def stage(t: int, s: int, caps: np.ndarray) -> np.ndarray:
        return np.array([float(s) + 1.0])

    result = backward_induction(
        horizon=horizon,
        chain=chain,
        capacity_levels=[[10.0]],
        investment_levels=[[0.0]],
        stage_solver=stage,
        discount=discount,
    )

    for s in range(chain.n_states):
        stage_profit = float(s) + 1.0
        v = result.value[0, s, 0, 0]
        assert v >= stage_profit - 1e-9

    max_stage = 3.0
    upper_bound = max_stage * horizon
    assert np.all(result.value[0, :, 0, 0] <= upper_bound + 1e-6)

    values = result.value[0, :, 0, 0]
    assert values[0] < values[1] < values[2]


def test_policy_prefers_positive_investment_when_profitable() -> None:
    """Firm should invest if higher capacity strictly raises continuation value."""
    chain = three_state_chain()
    horizon = 2

    def stage(t: int, s: int, caps: np.ndarray) -> np.ndarray:
        return caps.astype(float).copy()

    result = backward_induction(
        horizon=horizon,
        chain=chain,
        capacity_levels=[np.array([0.0, 10.0, 20.0])],
        investment_levels=[np.array([0.0, 10.0, 20.0])],
        stage_solver=stage,
        discount=0.9,
    )

    action_at_low_cap = result.policy[0, 0, 0, 0]
    assert action_at_low_cap > 0


def test_value_is_monotone_in_capacity() -> None:
    """Larger capacity cannot yield lower value when stage payoff is increasing in cap."""
    chain = three_state_chain()

    def stage(t: int, s: int, caps: np.ndarray) -> np.ndarray:
        return caps.astype(float).copy()

    levels = [np.array([0.0, 5.0, 10.0, 15.0])]
    result = backward_induction(
        horizon=3,
        chain=chain,
        capacity_levels=levels,
        investment_levels=[np.array([0.0, 5.0])],
        stage_solver=stage,
        discount=0.95,
    )

    for s in range(chain.n_states):
        v = result.value[0, s, :, 0]
        assert np.all(np.diff(v) >= -1e-8)
