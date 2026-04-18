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
- **Four capacity mechanisms**: energy-only, capacity payment, forward capacity
  auction, reliability options
- **Stochastic demand** as a finite-state Markov chain
- **Stochastic dynamic programming** via backward induction on a scenario tree
- **Adequacy metrics**: deterministic reserve margin, LOLE, EUE (with forced outages)
- **Interactive Streamlit dashboard** for side-by-side mechanism comparison
- Research-grade project hygiene: typed code, tests, pre-commit, CI

## Installation

```bash
# with uv (recommended)
uv sync

# or plain pip
python -m pip install -e ".[dev,app]"
```

Requires Python >= 3.10.

## Quick start

```python
from capgame.game.cournot import Firm, LinearDemand, solve_constrained
from capgame.mechanisms.reliability_options import ReliabilityOption

demand = LinearDemand(a=100.0, b=1.0)
firms = [
    Firm(marginal_cost=10.0, capacity=30.0),
    Firm(marginal_cost=12.0, capacity=25.0),
    Firm(marginal_cost=15.0, capacity=40.0),
]

eq = solve_constrained(demand, firms)
print(f"Price={eq.price:.2f}  HHI={eq.hhi:.0f}")

option = ReliabilityOption(premium=8.0, strike_price=50.0, hours_per_period=1.0)
outcome = option.apply(eq, capacities=[f.capacity for f in firms])
print(outcome.net_profits)
```

Run the dashboard:

```bash
streamlit run capgame/app/streamlit_app.py
# or
capgame-app
```

## Repository layout

```
capgame/
├── capgame/
│   ├── game/            # L1 equilibrium solvers (cournot, sfe, bilevel, mpe)
│   ├── stochastic/      # L2 uncertainty processes (demand, renewables, outages)
│   ├── mechanisms/      # L3 market designs (energy_only, cap_payment, FCM, RO)
│   ├── adequacy/        # L4 reliability metrics (reserve_margin, LOLE, EUE)
│   ├── optimization/    # L5 numerical methods (MCP, SDP, calibration)
│   ├── experiments/     # L6 reproducible scripts
│   └── app/             # L7 Streamlit dashboard
├── tests/               # mirrors package structure
├── docs/                # math notes, review, roadmap
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
