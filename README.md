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
- **Ontario calibration pipeline** from IESO public reports: linear
  demand (elasticity-anchored with OLS diagnostics), four-state
  renewable Markov chain with Laplace-smoothed empirical transitions,
  empirical outage proxy, and a ready-to-run `ScenarioConfig`
- **Applied Ontario study**: missing-money diagnostic, 4x3
  mechanism-x-structure matrix, welfare-/missing-money-optimal
  reliability-option strike search, and a tornado-style sensitivity
  sweep over elasticity, wind capacity, and gas prices
- **2024 -> 2050 forecaster**: pathway-based (IESO APO Reference)
  year-by-year `ScenarioConfig` trajectory with fleet roll-forward
  (Pickering/Bruce/SMRs/LT1/renewables), gas-price and capex-decline
  curves, and a Monte Carlo envelope yielding P10/P50/P90 bands on
  price, welfare, missing money, and reserve margin
- Research-grade project hygiene: typed code, 219 tests, pre-commit, CI

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

### Calibrate to Ontario

```bash
python scripts/fetch_ieso.py --year 2024       # downloads ~22 MB of CSVs/XML
python scripts/validate_ontario.py              # prints a model-vs-IESO report
```

Programmatic access:

```python
from capgame.calibration import build_ontario_scenario
from capgame.experiments import run_scenario

cal = build_ontario_scenario(year=2024, elasticity=-0.1)
res = run_scenario(cal.scenario)
print(f"Cournot price = ${res.expected_price:.0f}/MWh "
      f"(observed HOEP = ${cal.demand_fit.reference_price:.0f}/MWh)")
```

### Ontario applied study: missing money and optimal RO strike

End-to-end policy analysis:

```python
from capgame.calibration.ontario import build_ontario_scenario
from capgame.experiments.ontario_study import (
    find_optimal_strike,
    run_mechanism_matrix,
    summarize_missing_money,
)
from capgame.mechanisms.energy_only import EnergyOnly

cal = build_ontario_scenario(year=2024)

# Q1: per-technology missing money under the current regime
print(summarize_missing_money(cal, EnergyOnly()))

# Q2: full 4x3 mechanism-x-structure matrix, annualized in $/yr
matrix = run_mechanism_matrix(cal)

# Q3: reliability-option strike that minimizes |fleet missing money|
ss = find_optimal_strike(cal, premium_per_mw_year=55_000.0)
print(f"Optimal K = ${ss.optimal_strike:.1f}/MWh")
```

Full narrative and charts: [`notebooks/ontario_missing_money.ipynb`](notebooks/ontario_missing_money.ipynb).

### 2024 -> 2050 forecaster

Project the calibrated base year forward along a declared pathway
(default: IESO Annual Planning Outlook Reference), optionally with
Monte Carlo uncertainty:

```python
from capgame.calibration.ontario import build_ontario_scenario
from capgame.forecast import (
    MonteCarloConfig,
    build_trajectory,
    default_ontario_pathway,
    run_monte_carlo,
    run_trajectory,
    summarize_paths,
)

cal = build_ontario_scenario(year=2024)
pathway = default_ontario_pathway()
years = range(2024, 2051)

# Deterministic year-by-year forecast under energy-only oligopoly
traj = build_trajectory(cal, pathway, years)
df = run_trajectory(traj)               # tidy per-year DataFrame

# Uncertainty bands over fleet, demand, gas, SMR schedule
mc = run_monte_carlo(cal, years, config=MonteCarloConfig(n_paths=200))
bands = summarize_paths(mc)             # P10/P50/P90 by year
```

Or run the complete forecast as a CLI that emits CSVs and a summary:

```bash
python scripts/forecast_ontario.py --out reports/forecast --monte-carlo 200
```

See section 5 of `notebooks/ontario_missing_money.ipynb` for the full
forecast narrative (fleet chart, fan charts, per-technology missing
money, mechanism ranking over time).

## Repository layout

```
capgame/
├── capgame/
│   ├── game/            # L1 equilibrium solvers (cournot, market_structure, bilevel, sfe stub, mpe)
│   ├── stochastic/      # L2 uncertainty processes (demand, renewables, outages)
│   ├── mechanisms/      # L3 market designs + Mechanism Protocol
│   ├── adequacy/        # L4 reliability metrics (reserve_margin, LOLE, EUE)
│   ├── optimization/    # L5 numerical methods (MCP, SDP, calibration)
│   ├── experiments/     # L6 ScenarioConfig → ScenarioResult + missing_money + ontario_study
│   ├── calibration/     # L7 IESO loaders + estimators + ontario.py orchestrator
│   ├── forecast/        # L8 pathways, trajectory, monte_carlo (2024 -> 2050)
│   └── app/             # L9 cli.py (launcher) + ui.py (Streamlit UI)
├── notebooks/           # ontario_missing_money.ipynb applied study
├── scripts/             # fetch_ieso.py, validate_ontario.py, forecast_ontario.py
├── tests/               # mirrors package structure, 219 tests
├── docs/                # math notes, roadmap
└── data/ieso/raw/       # downloaded IESO public reports (git-ignored)
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

## Data sources & attribution

The Ontario calibration pipeline (`capgame/calibration/ontario.py` and
`scripts/fetch_ieso.py`) downloads public reports from the Independent
Electricity System Operator (IESO) of Ontario at
[`reports-public.ieso.ca`](https://reports-public.ieso.ca/public/). Those
files are published under section 30 of the *Electricity Act, 1998*
(Ontario) and are used here under the
[IESO Terms of Use](https://www.ieso.ca/en/Terms-of-Use), which permit
non-commercial research use with attribution.

If you publish results derived from the calibrated model, please include:

> Data source: IESO public market reports, retrieved YYYY-MM-DD.
> © Independent Electricity System Operator. Used under the IESO Terms
> of Use.

## License

MIT — see `LICENSE`. The IESO data is **not** relicensed by this
project; consult the IESO Terms of Use before redistributing raw
files.
