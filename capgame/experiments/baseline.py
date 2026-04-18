"""Static mechanism comparison experiment.

A single-period snapshot comparing the four capacity mechanisms on an
identical three-firm fleet. Returns a pandas DataFrame suitable for
display in the dashboard or for inclusion in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capgame.game.cournot import Firm, LinearDemand, consumer_surplus, solve
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from capgame.mechanisms.forward_capacity import ForwardCapacityMarket, ProcurementCurve
from capgame.mechanisms.reliability_options import ReliabilityOption


@dataclass(frozen=True)
class BaselineConfig:
    demand_intercept: float = 100.0
    demand_slope: float = 1.0
    firms: tuple[Firm, ...] = (
        Firm(marginal_cost=10.0, capacity=30.0, fixed_cost=20.0, name="Baseload"),
        Firm(marginal_cost=25.0, capacity=25.0, fixed_cost=12.0, name="Midmerit"),
        Firm(marginal_cost=50.0, capacity=20.0, fixed_cost=4.0, name="Peaker"),
    )
    capacity_payment_rho: float = 8.0
    procurement_target: float = 80.0
    procurement_slope: float = 1.5
    option_premium: float = 10.0
    option_strike: float = 45.0


def run_static_mechanism_comparison(config: BaselineConfig | None = None) -> pd.DataFrame:
    """Evaluate all four mechanisms on a single Cournot outcome."""
    cfg = config or BaselineConfig()
    demand = LinearDemand(a=cfg.demand_intercept, b=cfg.demand_slope)
    firms = list(cfg.firms)
    caps = [f.capacity for f in firms]

    eq = solve(demand, firms)

    mechanisms = {
        "Energy-only": EnergyOnly().apply(eq, caps),
        "Capacity payment": CapacityPayment(rho=cfg.capacity_payment_rho).apply(eq, caps),
        "Forward capacity": ForwardCapacityMarket(
            curve=ProcurementCurve(cap_target=cfg.procurement_target, slope=cfg.procurement_slope)
        ).apply(eq, caps),
        "Reliability options": ReliabilityOption(
            premium=cfg.option_premium, strike_price=cfg.option_strike
        ).apply(eq, caps),
    }

    cs = consumer_surplus(demand, eq)
    rows = []
    for name, outcome in mechanisms.items():
        ps = float(np.asarray(outcome.net_profits).sum())
        welfare = float(cs + ps - outcome.consumer_cost)
        rows.append(
            {
                "mechanism": name,
                "total_quantity": float(eq.total_quantity),
                "price": float(eq.price),
                "hhi": float(eq.hhi),
                "producer_surplus": ps,
                "consumer_surplus": float(cs),
                "consumer_payment_for_capacity": float(outcome.consumer_cost),
                "welfare": welfare,
            }
        )
    return pd.DataFrame(rows)
