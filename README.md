# CapGame

**Strategic Capacity Market Simulator** — a game-theoretic, stochastic, and
optimization-based framework for comparing electricity market capacity mechanisms.

CapGame reproduces and extends the three-stage dynamic Cournot game of
Khalfallah (2011), providing a unified environment in which four major capacity
remuneration mechanisms can be compared under stochastic demand, imperfect
competition, and alternative market structures.

> Status: **v0.1.0 — alpha**. Phases 1–3 of the roadmap are implemented with
> working scaffolding for the rest. See `docs/ROADMAP.md` for the full plan.

## Features

- **Nash–Cournot solver** (closed-form + capacity-constrained best-response iteration)
- **Market-structure variants**: oligopoly, cartel, monopoly (used by RQ3)
- **Four capacity mechanisms**: energy-only, capacity payment, forward capacity
  auction, reliability options — all satisfying a single `Mechanism` Protocol
- **Stochastic demand** as a finite-state Markov chain
- **Correlated renewables** as `MarkovChain[RenewableState]`
- **Stochastic dynamic programming** via backward induction
- **Bilevel / endogenous strike** solver for RQ4
- **Adequacy metrics**: reserve margin, LOLE, EUE (with forced outages)
- **One-entry-point scenario runner**: `ScenarioConfig → ScenarioResult`
- **Interactive Streamlit dashboard** organized around the four research
  questions, with URL-shareable parameter state
- Research-grade project hygiene: typed code, 149 tests, pre-commit, CI

## Installation

```bash
# with uv (recommended)
uv sync

# or plain pip
python -m pip install -e ".[dev,app]"
```

Requires Python >= 3.10.

## Quick start

The single-entry-point scenario runner:

```python
from capgame.experiments import ScenarioConfig, run_scenario
from capgame.game.cournot import Firm, LinearDemand
from capgame.mechanisms.reliability_options import ReliabilityOption

cfg = ScenarioConfig(
    demand=LinearDemand(a=100.0, b=1.0),
    firms=(
        Firm(marginal_cost=10.0, capacity=30.0, name="Baseload"),
        Firm(marginal_cost=25.0, capacity=25.0, name="Midmerit"),
        Firm(marginal_cost=50.0, capacity=20.0, name="Peaker"),
    ),
    mechanism=ReliabilityOption(premium=8.0, strike_price=50.0),
    outage_rates=(0.05, 0.05, 0.05),
    target_reserve_margin=0.15,
)

result = run_scenario(cfg)
print(f"Price     = ${result.expected_price:.2f}/MWh")
print(f"Welfare   = ${result.expected_welfare:,.0f}")
print(f"LOLE      = {result.adequacy.lole_hours_per_year:.2f} h/yr")
print(f"Reserve margin {result.adequacy.reserve_margin*100:.1f}% "
      f"vs target {cfg.target_reserve_margin*100:.0f}%")
```

Run the dashboard:

```bash
streamlit run capgame/app/ui.py
# or
capgame-app
```

The dashboard is organized around the four research questions (RQ1–RQ4)
and the methodology page. Every slider is URL-synchronized, so you can
share a configured scenario as a link.

## Repository layout

```
capgame/
├── capgame/
│   ├── game/            # L1 equilibrium solvers (cournot, market_structure, bilevel, sfe stub, mpe)
│   ├── stochastic/      # L2 uncertainty processes (demand, renewables, outages)
│   ├── mechanisms/      # L3 market designs + Mechanism Protocol
│   ├── adequacy/        # L4 reliability metrics (reserve_margin, LOLE, EUE)
│   ├── optimization/    # L5 numerical methods (MCP, SDP, calibration)
│   ├── experiments/     # L6 ScenarioConfig → ScenarioResult + baselines
│   └── app/             # L7 cli.py (launcher) + ui.py (Streamlit UI)
├── tests/               # mirrors package structure, 149 tests
├── docs/                # math notes, roadmap
└── data/                # calibration inputs
```

## Development

```bash
# install dev dependencies
python -m pip install -e ".[dev]"

# run tests
pytest

# lint + format
ruff check capgame tests
black capgame tests

# pre-commit
pre-commit install
pre-commit run --all-files
```

## Citing

If you use CapGame in academic work, please cite:

```bibtex
@software{mourssi_capgame_2026,
  author = {Mourssi, Omar},
  title  = {CapGame: Strategic Capacity Market Simulator},
  year   = {2026},
  url    = {https://github.com/Mourssio/capgame}
}
```

Underlying model:

> Khalfallah, M. H. (2011). *A Game Theoretic Model for Generation Capacity
> Adequacy: Comparison Between Investment Incentive Mechanisms in Electricity
> Markets.* The Energy Journal, 32(4), 117–157.

## License

MIT — see `LICENSE`.
