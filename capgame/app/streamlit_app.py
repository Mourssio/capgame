"""CapGame Streamlit dashboard.

Four tabs matching the proposal (section 6.8):

1. Baseline Reproduction -- static Cournot outcome under default parameters.
2. Mechanism Comparator -- side-by-side outcomes across the four mechanisms.
3. Scenario Explorer -- interactive sliders over demand, costs, and capacities.
4. About -- project description and citations.

Run with:  streamlit run capgame/app/streamlit_app.py
Or use the console entry point:  capgame-app
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_streamlit_here() -> None:
    """Launch Streamlit targeting this very file."""
    here = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(here)]
    sys.exit(subprocess.call(cmd))


def main() -> None:
    """Console entry point registered in pyproject as ``capgame-app``."""
    _run_streamlit_here()


def _build_ui() -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from capgame.adequacy.lole import loss_of_load_expectation
    from capgame.adequacy.reserve_margin import reserve_margin
    from capgame.experiments.baseline import BaselineConfig, run_static_mechanism_comparison
    from capgame.game.cournot import Firm, LinearDemand, consumer_surplus, solve
    from capgame.mechanisms.capacity_payment import CapacityPayment
    from capgame.mechanisms.energy_only import EnergyOnly
    from capgame.mechanisms.forward_capacity import ForwardCapacityMarket, ProcurementCurve
    from capgame.mechanisms.reliability_options import ReliabilityOption

    st.set_page_config(page_title="CapGame", page_icon="⚡", layout="wide")
    st.title("CapGame: Strategic Capacity Market Simulator")
    st.caption(
        "Game-theoretic, stochastic comparison of four electricity-market "
        "capacity mechanisms. Research prototype -- see docs/ROADMAP.md."
    )

    tab_baseline, tab_compare, tab_explore, tab_about = st.tabs(
        [
            "Baseline Reproduction",
            "Mechanism Comparator",
            "Scenario Explorer",
            "About",
        ]
    )

    with tab_baseline:
        st.header("Baseline Reproduction")
        st.write(
            "A single-period Cournot equilibrium with three asymmetric firms, "
            "evaluated under each mechanism at default Khalfallah-style parameters."
        )
        df = run_static_mechanism_comparison()
        st.dataframe(df.style.format(precision=2), use_container_width=True)
        st.bar_chart(
            df.set_index("mechanism")[
                ["producer_surplus", "consumer_payment_for_capacity", "welfare"]
            ]
        )

    with tab_compare:
        st.header("Mechanism Comparator")
        col_params, col_out = st.columns([1, 2])
        with col_params:
            st.subheader("Parameters")
            rho = st.slider("Capacity payment rho ($/MW)", 0.0, 30.0, 8.0, step=0.5)
            proc = st.slider("FCM procurement target (MW)", 10.0, 200.0, 80.0, step=5.0)
            proc_slope = st.slider("FCM curve slope", 0.1, 5.0, 1.5, step=0.1)
            premium = st.slider("RO premium ($/MW)", 0.0, 30.0, 10.0, step=0.5)
            strike = st.slider("RO strike ($/MWh)", 0.0, 100.0, 45.0, step=1.0)

        cfg = BaselineConfig(
            capacity_payment_rho=rho,
            procurement_target=proc,
            procurement_slope=proc_slope,
            option_premium=premium,
            option_strike=strike,
        )
        with col_out:
            df = run_static_mechanism_comparison(cfg)
            st.dataframe(df.style.format(precision=2), use_container_width=True)
            st.line_chart(
                df.set_index("mechanism")[["producer_surplus", "welfare"]]
            )

    with tab_explore:
        st.header("Scenario Explorer")
        a = st.slider("Demand intercept a", 50.0, 200.0, 100.0)
        b = st.slider("Demand slope b", 0.1, 3.0, 1.0)
        n_firms = st.slider("Number of firms", 1, 6, 3)
        base_cost = st.slider("Base marginal cost ($/MWh)", 5.0, 60.0, 10.0)
        cost_step = st.slider("Cost step between firms ($/MWh)", 0.0, 30.0, 10.0)
        base_cap = st.slider("Base capacity (MW)", 5.0, 100.0, 30.0)
        cap_step = st.slider("Capacity step between firms (MW)", -30.0, 30.0, -5.0)

        demand = LinearDemand(a=a, b=b)
        firms = [
            Firm(
                marginal_cost=base_cost + i * cost_step,
                capacity=max(0.0, base_cap + i * cap_step),
                name=f"Firm {i + 1}",
            )
            for i in range(n_firms)
        ]
        eq = solve(demand, firms)
        caps = np.array([f.capacity for f in firms])

        outcomes = {
            "Energy-only": EnergyOnly().apply(eq, caps),
            "Capacity payment": CapacityPayment(rho=8.0).apply(eq, caps),
            "Forward capacity": ForwardCapacityMarket(
                curve=ProcurementCurve(cap_target=80.0, slope=1.5)
            ).apply(eq, caps),
            "Reliability options": ReliabilityOption(
                premium=10.0, strike_price=45.0
            ).apply(eq, caps),
        }

        st.subheader("Equilibrium")
        st.write(
            f"**Price**: ${eq.price:.2f}/MWh  |  "
            f"**Total Q**: {eq.total_quantity:.1f} MW  |  "
            f"**HHI**: {eq.hhi:.0f}  |  "
            f"**Consumer surplus**: ${consumer_surplus(demand, eq):,.0f}"
        )
        by_firm = pd.DataFrame(
            {
                "firm": [f.name for f in firms],
                "capacity_MW": caps,
                "quantity_MW": eq.quantities,
                "marginal_cost": [f.marginal_cost for f in firms],
                "binding_cap": eq.binding,
            }
        )
        st.dataframe(by_firm.style.format(precision=2), use_container_width=True)

        st.subheader("Mechanism outcomes")
        rows = []
        for name, outcome in outcomes.items():
            rows.append(
                {
                    "mechanism": name,
                    "sum_net_profit": float(outcome.net_profits.sum()),
                    "sum_capacity_payment": float(outcome.capacity_payments.sum()),
                    "sum_refund": float(outcome.refunds.sum()),
                    "consumer_cost_of_capacity": float(outcome.consumer_cost),
                }
            )
        st.dataframe(pd.DataFrame(rows).style.format(precision=2), use_container_width=True)

        st.subheader("Adequacy")
        total_cap = float(caps.sum())
        peak = a / b
        rm = reserve_margin(total_cap, peak)
        outage_rates = np.full(len(firms), 0.05)
        lole = loss_of_load_expectation(
            caps.tolist(),
            outage_rates.tolist(),
            demand_distribution=[(peak, 1.0)],
        )
        st.write(
            f"**Reserve margin**: {rm * 100:.1f}%  |  "
            f"**LOLP** (1 period, FOR=5%): {lole:.4f}"
        )

    with tab_about:
        st.header("About CapGame")
        st.markdown(
            """
CapGame is an open-source research framework for analyzing long-term generation
capacity adequacy in restructured electricity markets. It reproduces and extends
the three-stage dynamic Cournot game of **Khalfallah (2011)**, providing a
unified environment in which four major capacity remuneration mechanisms
(energy-only, capacity payment, forward capacity market, reliability options)
can be compared under stochastic demand and imperfect competition.

**Version 0.1.0** implements the static Cournot equilibrium, all four
mechanisms as composable functions, a Markov demand process, and adequacy
metrics. Dynamic programming and the renewable / endogenous-strike
extensions are in progress -- see `docs/ROADMAP.md`.

Author: Omar Mourssi, University of Toronto ECE.
License: MIT.
"""
        )


try:
    import streamlit as _st

    if _st.runtime.exists():
        _build_ui()
except Exception:
    pass
