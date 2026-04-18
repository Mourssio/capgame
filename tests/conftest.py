"""Shared test fixtures."""

from __future__ import annotations

import pytest

from capgame.game.cournot import Firm, LinearDemand


@pytest.fixture
def symmetric_duopoly() -> tuple[LinearDemand, list[Firm]]:
    demand = LinearDemand(a=100.0, b=1.0)
    firms = [Firm(marginal_cost=10.0, capacity=100.0), Firm(marginal_cost=10.0, capacity=100.0)]
    return demand, firms


@pytest.fixture
def asymmetric_triopoly() -> tuple[LinearDemand, list[Firm]]:
    demand = LinearDemand(a=120.0, b=1.0)
    firms = [
        Firm(marginal_cost=10.0, capacity=60.0),
        Firm(marginal_cost=20.0, capacity=40.0),
        Firm(marginal_cost=30.0, capacity=30.0),
    ]
    return demand, firms


@pytest.fixture
def tight_capacity_duopoly() -> tuple[LinearDemand, list[Firm]]:
    demand = LinearDemand(a=100.0, b=1.0)
    firms = [Firm(marginal_cost=10.0, capacity=20.0), Firm(marginal_cost=10.0, capacity=20.0)]
    return demand, firms
