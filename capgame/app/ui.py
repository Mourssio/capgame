"""CapGame Streamlit UI.

Structured around the four research questions in the proposal, not around
generic "tabs". Every page:

* States the question it answers at the top.
* Uses a single primary visualization to answer it.
* Exposes only the controls relevant to that question via the sidebar.
* Explains what the user is looking at in a short callout.

The UI is a thin wrapper over :func:`capgame.experiments.scenarios.run_scenario`
and over the market-structure, bilevel, and adequacy primitives. It does not
reach into low-level modules directly; that coupling is what made the old
app fragile.

Run with:
    streamlit run capgame/app/ui.py

or via the console launcher:
    capgame-app
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from capgame.adequacy.reserve_margin import capacity_required
from capgame.experiments.scenarios import (
    ScenarioConfig,
    ScenarioResult,
    run_scenario,
)
from capgame.game.bilevel import solve_endogenous_strike
from capgame.game.cournot import Firm, LinearDemand, consumer_surplus
from capgame.mechanisms.base import Mechanism
from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from capgame.mechanisms.forward_capacity import ForwardCapacityMarket, ProcurementCurve
from capgame.mechanisms.reliability_options import ReliabilityOption
from capgame.stochastic.renewables import simple_two_state_renewables

MECHANISM_ORDER = [
    "Energy-only",
    "Capacity payment",
    "Forward capacity market",
    "Reliability options",
]
MECHANISM_COLORS = {
    "Energy-only": "#4c78a8",
    "Capacity payment": "#f58518",
    "Forward capacity market": "#54a24b",
    "Reliability options": "#e45756",
}
STRUCTURE_LABELS = {
    "oligopoly": "Cournot oligopoly",
    "cartel": "Cartel (joint profit max)",
    "monopoly": "Single monopolist",
}


@dataclass
class UIParameters:
    """Sidebar-exposed parameters shared across pages.

    Named in domain language. Anything Greek-lettered in the math lives in
    tooltips, not in slider labels.
    """

    peak_load_mw: float
    price_sensitivity: float
    n_firms: int
    base_mc: float
    cost_step: float
    base_cap_mw: float
    cap_step_mw: float
    forced_outage_rate: float
    target_reserve_margin: float

    wind_capacity_mw: float
    solar_capacity_mw: float
    low_wind_cf: float
    high_wind_cf: float
    low_solar_cf: float
    high_solar_cf: float
    renewable_correlation: float

    cap_payment_rho: float
    fcm_target_mw: float
    fcm_slope: float
    ro_premium: float
    ro_strike: float
    ro_coverage: float
    hours_per_period: float

    def build_demand(self) -> LinearDemand:
        b = self.price_sensitivity
        a = b * self.peak_load_mw
        return LinearDemand(a=a, b=b)

    def build_firms(self) -> tuple[Firm, ...]:
        firms = tuple(
            Firm(
                marginal_cost=max(0.0, self.base_mc + i * self.cost_step),
                capacity=max(0.0, self.base_cap_mw + i * self.cap_step_mw),
                outage_rate=self.forced_outage_rate,
                name=f"Firm {i + 1}",
            )
            for i in range(self.n_firms)
        )
        return firms

    def outage_rates_tuple(self) -> tuple[float, ...]:
        return tuple(self.forced_outage_rate for _ in range(self.n_firms))


DEFAULTS: dict[str, Any] = {
    "peak_load_mw": 100.0,
    "price_sensitivity": 1.0,
    "n_firms": 3,
    "base_mc": 10.0,
    "cost_step": 10.0,
    "base_cap_mw": 30.0,
    "cap_step_mw": -5.0,
    "forced_outage_rate": 0.05,
    "target_reserve_margin": 0.15,
    "wind_capacity_mw": 0.0,
    "solar_capacity_mw": 0.0,
    "low_wind_cf": 0.15,
    "high_wind_cf": 0.45,
    "low_solar_cf": 0.10,
    "high_solar_cf": 0.30,
    "renewable_correlation": 0.30,
    "cap_payment_rho": 8.0,
    "fcm_target_mw": 80.0,
    "fcm_slope": 1.5,
    "ro_premium": 10.0,
    "ro_strike": 45.0,
    "ro_coverage": 1.0,
    "hours_per_period": 1.0,
    "page": "Home",
}


def _coerce(kind: str, raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    try:
        if kind == "int":
            return int(float(raw))
        if kind == "float":
            return float(raw)
        return raw
    except (TypeError, ValueError):
        return default


def _hydrate_from_query_params() -> None:
    """Read URL query params into ``st.session_state`` on first run."""
    if st.session_state.get("_hydrated"):
        return
    qp = st.query_params
    types: dict[str, str] = {
        "peak_load_mw": "float",
        "price_sensitivity": "float",
        "n_firms": "int",
        "base_mc": "float",
        "cost_step": "float",
        "base_cap_mw": "float",
        "cap_step_mw": "float",
        "forced_outage_rate": "float",
        "target_reserve_margin": "float",
        "wind_capacity_mw": "float",
        "solar_capacity_mw": "float",
        "low_wind_cf": "float",
        "high_wind_cf": "float",
        "low_solar_cf": "float",
        "high_solar_cf": "float",
        "renewable_correlation": "float",
        "cap_payment_rho": "float",
        "fcm_target_mw": "float",
        "fcm_slope": "float",
        "ro_premium": "float",
        "ro_strike": "float",
        "ro_coverage": "float",
        "hours_per_period": "float",
        "page": "str",
    }
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            raw = qp.get(key)
            st.session_state[key] = _coerce(types.get(key, "str"), raw, default)
    st.session_state["_hydrated"] = True


def _push_to_query_params(keys: Sequence[str]) -> None:
    """Mirror selected session-state keys into the URL so it is shareable."""
    for key in keys:
        val = st.session_state.get(key)
        if val is None:
            continue
        st.query_params[key] = str(val)


def _sidebar() -> UIParameters:
    st.sidebar.title("CapGame")
    st.sidebar.caption("Strategic Capacity Market Simulator")

    page = st.sidebar.radio(
        "View",
        options=[
            "Home",
            "RQ1 · Adequacy criterion",
            "RQ2 · Renewable uncertainty",
            "RQ3 · Market structure",
            "RQ4 · Endogenous strike",
            "Methodology",
        ],
        key="page",
    )
    del page

    with st.sidebar.expander("Demand & fleet", expanded=True):
        st.slider(
            "Peak load (MW)",
            min_value=20.0,
            max_value=300.0,
            step=5.0,
            key="peak_load_mw",
            help="System peak demand in MW.",
        )
        st.slider(
            "Demand price sensitivity ($/MWh per MW)",
            min_value=0.1,
            max_value=3.0,
            step=0.1,
            key="price_sensitivity",
            help=(
                "How much the market price falls per additional MW of quantity "
                "demanded. Technically the slope b of the linear inverse demand "
                "curve P(Q) = a - b·Q."
            ),
        )
        st.slider(
            "Number of firms",
            min_value=1,
            max_value=6,
            step=1,
            key="n_firms",
            help="Dispatchable thermal firms in the market.",
        )
        st.slider(
            "Base marginal cost ($/MWh)",
            min_value=5.0,
            max_value=80.0,
            step=1.0,
            key="base_mc",
        )
        st.slider(
            "Cost step across firms ($/MWh)",
            min_value=0.0,
            max_value=30.0,
            step=1.0,
            key="cost_step",
            help="Firm i has marginal cost base + i · step.",
        )
        st.slider(
            "Base capacity (MW)",
            min_value=5.0,
            max_value=120.0,
            step=1.0,
            key="base_cap_mw",
        )
        st.slider(
            "Capacity step across firms (MW)",
            min_value=-30.0,
            max_value=30.0,
            step=1.0,
            key="cap_step_mw",
            help="Firm i has capacity base + i · step.",
        )
        st.slider(
            "Forced-outage rate per firm",
            min_value=0.0,
            max_value=0.20,
            step=0.01,
            key="forced_outage_rate",
            help="Probability any one thermal unit is unavailable.",
        )
        st.slider(
            "Target reserve margin",
            min_value=0.0,
            max_value=0.40,
            step=0.01,
            key="target_reserve_margin",
            help="Regulator's required margin above peak demand (e.g. 0.15 = 15%).",
        )

    with st.sidebar.expander("Renewables", expanded=False):
        st.slider(
            "Installed wind capacity (MW)",
            min_value=0.0,
            max_value=200.0,
            step=5.0,
            key="wind_capacity_mw",
        )
        st.slider(
            "Installed solar capacity (MW)",
            min_value=0.0,
            max_value=200.0,
            step=5.0,
            key="solar_capacity_mw",
        )
        st.slider(
            "Low wind capacity factor",
            0.0,
            0.5,
            step=0.01,
            key="low_wind_cf",
        )
        st.slider(
            "High wind capacity factor",
            0.0,
            1.0,
            step=0.01,
            key="high_wind_cf",
        )
        st.slider(
            "Low solar capacity factor",
            0.0,
            0.5,
            step=0.01,
            key="low_solar_cf",
        )
        st.slider(
            "High solar capacity factor",
            0.0,
            1.0,
            step=0.01,
            key="high_solar_cf",
        )
        st.slider(
            "Wind-solar persistence",
            min_value=-0.75,
            max_value=0.75,
            step=0.05,
            key="renewable_correlation",
            help=(
                "Biases the diagonal of the 4-state Markov transition matrix. "
                "Positive values make joint (high,high)/(low,low) states more "
                "persistent. Magnitudes above ~0.75 are clipped by the ad-hoc "
                "construction."
            ),
        )

    with st.sidebar.expander("Capacity mechanism parameters", expanded=False):
        st.slider(
            "Administered capacity payment ($/MW · period)",
            min_value=0.0,
            max_value=40.0,
            step=0.5,
            key="cap_payment_rho",
            help="Flat payment per MW installed (mechanism: Capacity Payment).",
        )
        st.slider(
            "FCM procurement target (MW)",
            min_value=10.0,
            max_value=250.0,
            step=5.0,
            key="fcm_target_mw",
            help="Regulator's zero-price procurement intercept.",
        )
        st.slider(
            "FCM demand-curve slope (MW per $/MW)",
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            key="fcm_slope",
            help=(
                "How much the regulator cuts procurement per $/MW of rising "
                "clearing price. Higher = more elastic demand curve."
            ),
        )
        st.slider(
            "RO premium ($/MW · period)",
            min_value=0.0,
            max_value=40.0,
            step=0.5,
            key="ro_premium",
            help=(
                "Option premium paid per MW of covered capacity per period. "
                "Units are $/MW · period, where the period length in hours is "
                "set below."
            ),
        )
        st.slider(
            "RO strike price ($/MWh)",
            min_value=0.0,
            max_value=150.0,
            step=1.0,
            key="ro_strike",
            help="Spot-price threshold above which the option refunds.",
        )
        st.slider(
            "RO coverage fraction",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="ro_coverage",
            help="Fraction of each firm's capacity sold under options.",
        )
        st.slider(
            "Period length (hours)",
            min_value=1.0,
            max_value=24.0,
            step=1.0,
            key="hours_per_period",
            help="Used to convert $/MWh spreads into $/period refunds.",
        )

    if st.sidebar.button("Reset parameters"):
        for key, default in DEFAULTS.items():
            st.session_state[key] = default
        st.rerun()

    _push_to_query_params(list(DEFAULTS.keys()))

    return UIParameters(
        peak_load_mw=st.session_state["peak_load_mw"],
        price_sensitivity=st.session_state["price_sensitivity"],
        n_firms=int(st.session_state["n_firms"]),
        base_mc=st.session_state["base_mc"],
        cost_step=st.session_state["cost_step"],
        base_cap_mw=st.session_state["base_cap_mw"],
        cap_step_mw=st.session_state["cap_step_mw"],
        forced_outage_rate=st.session_state["forced_outage_rate"],
        target_reserve_margin=st.session_state["target_reserve_margin"],
        wind_capacity_mw=st.session_state["wind_capacity_mw"],
        solar_capacity_mw=st.session_state["solar_capacity_mw"],
        low_wind_cf=st.session_state["low_wind_cf"],
        high_wind_cf=st.session_state["high_wind_cf"],
        low_solar_cf=st.session_state["low_solar_cf"],
        high_solar_cf=st.session_state["high_solar_cf"],
        renewable_correlation=st.session_state["renewable_correlation"],
        cap_payment_rho=st.session_state["cap_payment_rho"],
        fcm_target_mw=st.session_state["fcm_target_mw"],
        fcm_slope=st.session_state["fcm_slope"],
        ro_premium=st.session_state["ro_premium"],
        ro_strike=st.session_state["ro_strike"],
        ro_coverage=st.session_state["ro_coverage"],
        hours_per_period=st.session_state["hours_per_period"],
    )


def _build_mechanism_map(params: UIParameters) -> dict[str, Mechanism]:
    return {
        "Energy-only": EnergyOnly(),
        "Capacity payment": CapacityPayment(rho=params.cap_payment_rho),
        "Forward capacity market": ForwardCapacityMarket(
            curve=ProcurementCurve(
                cap_target=params.fcm_target_mw,
                slope=params.fcm_slope,
            )
        ),
        "Reliability options": ReliabilityOption(
            premium=params.ro_premium,
            strike_price=params.ro_strike,
            coverage=params.ro_coverage,
            hours_per_period=params.hours_per_period,
        ),
    }


def _all_mechanism_results(
    params: UIParameters,
    structure: str = "oligopoly",
    include_renewables: bool = False,
) -> dict[str, ScenarioResult]:
    demand = params.build_demand()
    firms = params.build_firms()
    chain = None
    if include_renewables and (params.wind_capacity_mw > 0 or params.solar_capacity_mw > 0):
        chain = simple_two_state_renewables(
            low_wind=params.low_wind_cf,
            high_wind=params.high_wind_cf,
            low_solar=params.low_solar_cf,
            high_solar=params.high_solar_cf,
            correlation=params.renewable_correlation,
        )
    results: dict[str, ScenarioResult] = {}
    for name, mech in _build_mechanism_map(params).items():
        cfg = ScenarioConfig(
            demand=demand,
            firms=firms,
            mechanism=mech,
            market_structure=structure,  # type: ignore[arg-type]
            renewable_chain=chain,
            wind_capacity_mw=params.wind_capacity_mw,
            solar_capacity_mw=params.solar_capacity_mw,
            outage_rates=params.outage_rates_tuple(),
            target_reserve_margin=params.target_reserve_margin,
            hours_per_period=params.hours_per_period,
        )
        results[name] = run_scenario(cfg)
    return results


def _mechanism_summary_df(results: dict[str, ScenarioResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Mechanism": name,
                "Expected price ($/MWh)": r.expected_price,
                "Expected quantity (MW)": r.expected_quantity,
                "Producer surplus ($)": r.expected_producer_surplus,
                "Consumer surplus ($)": r.expected_consumer_surplus,
                "Consumer capacity cost ($)": r.expected_consumer_payment_for_capacity,
                "Welfare ($)": r.expected_welfare,
                "Reserve margin": r.adequacy.reserve_margin,
                "LOLE (h/yr)": r.adequacy.lole_hours_per_year,
                "EUE (MWh/yr)": r.adequacy.eue_mwh_per_year,
            }
        )
    df = pd.DataFrame(rows)
    df["Mechanism"] = pd.Categorical(df["Mechanism"], categories=MECHANISM_ORDER, ordered=True)
    return df.sort_values("Mechanism").reset_index(drop=True)


def _share_button(label: str = "Copy shareable link") -> None:
    qp_items = "&".join(f"{k}={v}" for k, v in st.query_params.to_dict().items())
    url_snippet = f"?{qp_items}" if qp_items else "?"
    st.code(url_snippet, language="text")
    st.caption(label)


def _download_button(df: pd.DataFrame, filename: str, key: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_home(params: UIParameters) -> None:
    st.title("CapGame")
    st.caption(
        "A game-theoretic, stochastic comparison of four electricity "
        "capacity-remuneration mechanisms."
    )
    st.markdown(
        "**What this tool is for.** CapGame is a research prototype that "
        "reproduces and extends the dynamic Cournot capacity-adequacy game "
        "of Khalfallah (2011). Use it to answer the four research questions "
        "in the proposal: how adequacy criteria change mechanism rankings "
        "(RQ1), how renewable uncertainty shifts the investment mix (RQ2), "
        "how market structure interacts with mechanism choice (RQ3), and "
        "how an endogenously chosen strike price re-prices reliability "
        "options (RQ4)."
    )

    results = _all_mechanism_results(params, structure="oligopoly", include_renewables=False)

    st.subheader("Headline numbers under the current parameters")
    c1, c2, c3, c4 = st.columns(4)
    baseline = results["Energy-only"]
    c1.metric("Clearing price", f"${baseline.expected_price:,.2f}/MWh")
    c2.metric("Total quantity", f"{baseline.expected_quantity:,.1f} MW")
    c3.metric(
        "Reserve margin",
        f"{baseline.adequacy.reserve_margin * 100:.1f}%",
        delta=f"target {params.target_reserve_margin * 100:.0f}%",
    )
    c4.metric(
        "HHI",
        f"{baseline.states[0].equilibrium.hhi:,.0f}",
        help="Herfindahl on output shares; 10000 = monopoly.",
    )

    st.subheader("All four mechanisms, current parameters")
    df = _mechanism_summary_df(results)
    fig = px.bar(
        df,
        x="Mechanism",
        y=["Producer surplus ($)", "Consumer surplus ($)"],
        barmode="stack",
        color_discrete_sequence=["#4c78a8", "#54a24b"],
        title="Welfare decomposition under each mechanism",
    )
    fig.add_bar(
        x=df["Mechanism"],
        y=-df["Consumer capacity cost ($)"],
        name="Consumer capacity cost (-)",
        marker_color="#e45756",
    )
    fig.update_layout(yaxis_title="$ (per period)", height=380)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Details table"):
        st.dataframe(df, use_container_width=True)
        _download_button(df, "home_summary.csv", "dl_home")

    with st.expander("Assumptions & limitations"):
        st.markdown(
            "* Single-period Cournot subgame; dynamic investment is handled "
            "by the SDP module but not yet wired into the UI.\n"
            "* Linear inverse demand and linear marginal costs.\n"
            "* Forced outages are independent across firms.\n"
            "* Renewables (if enabled) enter as an exogenous Markov process "
            "with calibration deferred to Phase 5b."
        )


def _page_rq1(params: UIParameters) -> None:
    st.header("RQ1 · Do mechanism rankings change with the adequacy criterion?")
    st.info(
        "**What you're looking at.** Each dot is one mechanism. The x-axis "
        "is total welfare (higher is better). The color encodes how that "
        "mechanism scores against two different adequacy criteria: the "
        "deterministic reserve margin and the probabilistic LOLE "
        "(hours/year). If the mechanism ordering flips between the two "
        "criteria, the regulator's choice of reliability standard is "
        "not a detail — it's a policy lever."
    )

    results = _all_mechanism_results(params, structure="oligopoly", include_renewables=False)
    df = _mechanism_summary_df(results)

    rm_target = params.target_reserve_margin
    cap_req = capacity_required(params.peak_load_mw, rm_target)
    total_cap = results["Energy-only"].adequacy.total_capacity_mw
    df["Meets reserve-margin target"] = total_cap >= cap_req
    df["LOLE (h/yr)"] = df["LOLE (h/yr)"].fillna(0.0)
    df["Meets LOLE ≤ 2.4 h/yr"] = df["LOLE (h/yr)"] <= 2.4

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for _, row in df.iterrows():
            mech = str(row["Mechanism"])
            fig.add_trace(
                go.Bar(
                    x=[mech],
                    y=[row["Welfare ($)"]],
                    name=mech,
                    marker_color=MECHANISM_COLORS[mech],
                    showlegend=False,
                )
            )
        fig.update_layout(
            title="Total welfare by mechanism",
            yaxis_title="$ / period",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(
            df,
            x="Mechanism",
            y="LOLE (h/yr)",
            color="Mechanism",
            color_discrete_map=MECHANISM_COLORS,
            title="LOLE (hours/year) — exogenous fleet, uniform FOR",
        )
        fig2.add_hline(y=2.4, line_dash="dot", annotation_text="reference 2.4 h/yr")
        fig2.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Criterion-by-criterion pass / fail")
    st.dataframe(
        df[
            [
                "Mechanism",
                "Welfare ($)",
                "Reserve margin",
                "Meets reserve-margin target",
                "LOLE (h/yr)",
                "Meets LOLE ≤ 2.4 h/yr",
            ]
        ],
        use_container_width=True,
    )

    with st.expander("Interpretation"):
        st.markdown(
            "* **Reserve margin** is deterministic: it compares total "
            f"installed capacity ({total_cap:,.1f} MW) to the target "
            f"{cap_req:,.1f} MW (={(1 + rm_target) * 100 - 100:.0f}% above "
            "peak load).\n"
            "* **LOLE** integrates the forced-outage distribution of each "
            "unit and the demand distribution; it's the probabilistic "
            "counterpart and is what NERC / ENTSO-E standards are written "
            "against.\n"
            "* **Why RQ1 matters.** In this model the four mechanisms move "
            "the *distribution* of profits but do not move *installed* "
            "capacity (no investment yet). So the adequacy numbers are "
            "identical across mechanisms — which itself is RQ1's null "
            "finding. Connect this to the SDP module to see rankings "
            "diverge once investment responds."
        )

    with st.expander("Details table"):
        st.dataframe(df, use_container_width=True)
        _download_button(df, "rq1_adequacy.csv", "dl_rq1")
    _share_button()


def _page_rq2(params: UIParameters) -> None:
    st.header("RQ2 · How does renewable uncertainty shift the mix?")
    st.info(
        "**What you're looking at.** The 4-state wind/solar Markov chain "
        "is resolved in each state, the thermal fleet re-dispatches "
        "against the residual demand, and each mechanism is applied. The "
        "bars show expected per-firm thermal dispatch, weighted by the "
        "chain's stationary distribution. Sweep the wind or solar sliders "
        "in the sidebar to see how scarcity and crowding-out change the "
        "dispatch mix and mechanism revenues."
    )
    if params.wind_capacity_mw == 0.0 and params.solar_capacity_mw == 0.0:
        st.warning(
            "No renewable capacity installed — set wind or solar in the "
            "sidebar to see this page do real work."
        )

    results = _all_mechanism_results(params, structure="oligopoly", include_renewables=True)

    mech_pick = st.selectbox("Mechanism to plot per-state", MECHANISM_ORDER, key="rq2_mech")
    r = results[mech_pick]

    rows = []
    for state in r.states:
        for i, f in enumerate(params.build_firms()):
            rows.append(
                {
                    "State": state.label,
                    "Probability": state.probability,
                    "Firm": f.name,
                    "Quantity (MW)": float(state.equilibrium.quantities[i]),
                    "Price ($/MWh)": state.equilibrium.price,
                }
            )
    per_state_df = pd.DataFrame(rows)

    fig = px.bar(
        per_state_df,
        x="State",
        y="Quantity (MW)",
        color="Firm",
        title=f"Per-firm thermal dispatch by renewable state — {mech_pick}",
    )
    fig.update_layout(barmode="stack", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Expected dispatch across renewable states, by mechanism")
    mix_rows = []
    firms = params.build_firms()
    for name, rr in results.items():
        for i, f in enumerate(firms):
            mix_rows.append(
                {
                    "Mechanism": name,
                    "Firm": f.name,
                    "Expected quantity (MW)": float(rr.per_firm_expected_quantity[i]),
                }
            )
    mix_df = pd.DataFrame(mix_rows)
    fig2 = px.bar(
        mix_df,
        x="Mechanism",
        y="Expected quantity (MW)",
        color="Firm",
        title="Expected thermal mix under each mechanism",
    )
    fig2.update_layout(barmode="stack", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Per-state table"):
        st.dataframe(per_state_df, use_container_width=True)
        _download_button(per_state_df, "rq2_per_state.csv", "dl_rq2a")
    _share_button()


def _page_rq3(params: UIParameters) -> None:
    st.header("RQ3 · Under which market structures is each mechanism robust?")
    st.info(
        "**What you're looking at.** The same fleet is solved under three "
        "competitive structures: Cournot oligopoly, joint-profit cartel, "
        "and single monopoly. A mechanism is **robust** if its welfare / "
        "consumer-cost answer does not swing wildly across structures."
    )

    frames: dict[str, dict[str, ScenarioResult]] = {}
    for structure in ("oligopoly", "cartel", "monopoly"):
        frames[structure] = _all_mechanism_results(
            params, structure=structure, include_renewables=False
        )

    rows = []
    for structure, res in frames.items():
        for name, r in res.items():
            rows.append(
                {
                    "Structure": STRUCTURE_LABELS[structure],
                    "Mechanism": name,
                    "Price ($/MWh)": r.expected_price,
                    "Total quantity (MW)": r.expected_quantity,
                    "Producer surplus ($)": r.expected_producer_surplus,
                    "Consumer surplus ($)": r.expected_consumer_surplus,
                    "Consumer capacity cost ($)": r.expected_consumer_payment_for_capacity,
                    "Welfare ($)": r.expected_welfare,
                }
            )
    df = pd.DataFrame(rows)
    df["Mechanism"] = pd.Categorical(df["Mechanism"], categories=MECHANISM_ORDER, ordered=True)

    fig = px.bar(
        df,
        x="Structure",
        y="Welfare ($)",
        color="Mechanism",
        barmode="group",
        color_discrete_map=MECHANISM_COLORS,
        title="Welfare under each structure x mechanism",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        df,
        x="Structure",
        y="Consumer capacity cost ($)",
        color="Mechanism",
        barmode="group",
        color_discrete_map=MECHANISM_COLORS,
        title="Consumer payment for capacity under each structure",
    )
    fig2.update_layout(height=360)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Full table"):
        st.dataframe(df, use_container_width=True)
        _download_button(df, "rq3_structures.csv", "dl_rq3")
    with st.expander("What each structure means"):
        st.markdown(
            "* **Oligopoly.** Each firm chooses its own output taking "
            "others' output as given (Nash-Cournot).\n"
            "* **Cartel.** All firms coordinate to maximize joint profit; "
            "dispatch is merit-order up to the point where marginal "
            "revenue equals the marginal unit's cost.\n"
            "* **Monopoly.** One decision-maker owns the whole fleet; the "
            "dispatch is identical to the cartel case in this model."
        )
    _share_button()


def _page_rq4(params: UIParameters) -> None:
    st.header("RQ4 · Does endogenous strike-price selection improve reliability options?")
    st.info(
        "**What you're looking at.** The regulator chooses the strike "
        "price K to maximize total welfare net of consumer capacity cost. "
        "The blue curve is the welfare objective across the strike grid; "
        "the red marker is the grid optimum. The slider in the sidebar "
        "sets the premium; the bilevel solver returns the optimal K."
    )

    demand = params.build_demand()
    firms = params.build_firms()
    caps = [f.capacity for f in firms]

    bi = solve_endogenous_strike(
        demand=demand,
        firms=firms,
        premium=params.ro_premium,
        coverage=params.ro_coverage,
        hours_per_period=params.hours_per_period,
        strike_bounds=(0.0, 200.0),
        n_grid=81,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bi.grid,
            y=bi.objective_values,
            mode="lines",
            name="Welfare - consumer cost",
            line=dict(color="#4c78a8", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[bi.leader_action],
            y=[bi.leader_objective],
            mode="markers+text",
            name="Grid optimum",
            marker=dict(color="#e45756", size=12),
            text=[f"K* = ${bi.leader_action:.1f}/MWh"],
            textposition="top center",
        )
    )
    fig.add_vline(
        x=params.ro_strike,
        line_dash="dot",
        annotation_text=f"current K = ${params.ro_strike:.1f}",
    )
    fig.update_layout(
        xaxis_title="Strike price K ($/MWh)",
        yaxis_title="Welfare ($)",
        title="Regulator's objective over strike price",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    eq = bi.follower_equilibrium
    option_at_optimum = ReliabilityOption(
        premium=params.ro_premium,
        strike_price=bi.leader_action,
        coverage=params.ro_coverage,
        hours_per_period=params.hours_per_period,
    )
    outcome = option_at_optimum.apply(eq, caps)
    cs = consumer_surplus(demand, eq)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal strike K*", f"${bi.leader_action:,.2f}/MWh")
    c2.metric("Welfare at K*", f"${bi.leader_objective:,.0f}")
    c3.metric("Consumer surplus", f"${cs:,.0f}")
    c4.metric("Refund at K*", f"${float(outcome.refunds.sum()):,.0f}")

    with st.expander("How this is computed"):
        st.markdown(
            "* The follower Cournot subgame is solved once per grid point "
            "(the RO refund is a transfer, so in this model it does not "
            "affect dispatch — evaluating per-point keeps the API correct "
            "if that assumption is ever relaxed).\n"
            "* The grid is uniform on the strike interval "
            "`[0, 200] $/MWh` with 81 points. Resolution controls "
            "accuracy; the solver returns the full objective curve so the "
            "quantization is visible."
        )
    _share_button()


def _page_methodology() -> None:
    st.header("Methodology & about")
    st.markdown(
        "CapGame is a seven-layer research prototype implementing the dynamic "
        "capacity-adequacy game of **Khalfallah (2011)** and three proposed "
        "extensions (stochastic adequacy, renewable uncertainty, endogenous "
        "strike). Each layer is a leaf of the package and has a single "
        "responsibility."
    )

    st.subheader("Architecture")
    st.code(
        "capgame/\n"
        "├── game/              # Cournot, market structure, SFE stub, bilevel\n"
        "├── stochastic/        # Markov chains, outages, renewables\n"
        "├── mechanisms/        # Energy-only, cap pay, FCM, reliability options\n"
        "├── adequacy/          # Reserve margin, LOLE, EUE\n"
        "├── optimization/      # SDP backward induction, MCP, calibration\n"
        "├── experiments/       # ScenarioConfig → ScenarioResult, baselines\n"
        "└── app/               # cli launcher + this Streamlit UI\n",
        language="text",
    )

    st.subheader("Key formulas")
    st.markdown("**Linear inverse demand.**")
    st.latex(r"P(Q) = \max(0,\; a - b\,Q)")
    st.markdown("**Cournot best response.**")
    st.latex(
        r"q_i^\star(q_{-i}) = \mathrm{clip}\!\left("
        r"\frac{a - b\,Q_{-i} - c_i}{2\,b},\ 0,\ \bar q_i\right)"
    )
    st.markdown("**Reliability option refund** (period length $H$ hours).")
    st.latex(
        r"r_i = \big(P - K\big)_{+}\,\cdot\,x_i\,\cdot\,H "
        r"\qquad x_i = \text{coverage}\cdot \bar q_i"
    )
    st.markdown("**LOLE** with COPT $g_C$ and demand pmf $f_D$.")
    st.latex(
        r"\mathrm{LOLE}"
        r" = \text{periods\_per\_year}\ \cdot\ "
        r"\sum_{d}\ f_D(d)\ \sum_{c < d}\ g_C(c)"
    )

    st.subheader("Assumptions & limitations")
    st.markdown(
        "* Single-period Cournot; investment dynamics are in the SDP "
        "module but not yet wired into the UI.\n"
        "* Forced outages are independent across firms.\n"
        "* Renewables are an exogenous Markov process with ad-hoc "
        "calibration (see `TODO(phase-5b)` in `renewables.py`).\n"
        "* Reliability-option refund treated as a transfer, so dispatch "
        "does not react to strike. RQ4 page evaluates the subgame at every "
        "grid point anyway so the API stays correct when that assumption "
        "is relaxed."
    )

    st.subheader("Version & reproducibility")
    try:
        from capgame import __version__ as cap_version
    except ImportError:
        cap_version = "unknown"
    st.write(f"Package version: `{cap_version}`")
    st.write(
        "Every parameter on every page is encoded in the URL; copy the link "
        "from the share block at the bottom of a page to reproduce that "
        "view exactly."
    )
    st.markdown(
        "**Author.** Omar Mourssi, University of Toronto ECE.  \n"
        "**License.** MIT.  \n"
        "**Citation.** Khalfallah, M. (2011). *Generation adequacy and "
        "capacity mechanisms in the European electricity market.* Energy "
        "Policy."
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="CapGame",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _hydrate_from_query_params()
    params = _sidebar()

    page = st.session_state.get("page", "Home")
    if page == "Home":
        page_home(params)
    elif page.startswith("RQ1"):
        _page_rq1(params)
    elif page.startswith("RQ2"):
        _page_rq2(params)
    elif page.startswith("RQ3"):
        _page_rq3(params)
    elif page.startswith("RQ4"):
        _page_rq4(params)
    elif page == "Methodology":
        _page_methodology()
    else:
        page_home(params)


if __name__ == "__main__":
    main()
