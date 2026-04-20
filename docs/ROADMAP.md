# CapGame Implementation Roadmap

This document tracks implementation status against the eight-phase plan from
the project proposal (Section 6).

## Legend

- [x] Complete
- [~] Partial / scaffolding in place
- [ ] Not started

## Phase 1 -- Foundations (Week 1)

- [x] Repository with `pyproject.toml`, pre-commit, CI
- [x] `game.cournot` with `LinearDemand`, `Firm`, `CournotEquilibrium`
- [x] `solve_unconstrained`, `solve_constrained`, `solve`
- [x] Validation tests (monopoly, duopoly, asymmetric triopoly, binding-cap, slack-cap, HHI bounds)

## Phase 2 -- Static Mechanisms (Weeks 2-3)

- [x] `mechanisms.energy_only`
- [x] `mechanisms.capacity_payment`
- [x] `mechanisms.forward_capacity` with procurement curve and auction clearer
- [x] `mechanisms.reliability_options` with premium, strike, coverage
- [x] Static experiment (`experiments.baseline`)

## Phase 3 -- Stochastic Dynamics (Weeks 4-5)

- [x] `stochastic.demand.MarkovChain` with stationary distribution, sampling, t-step distribution
- [x] `optimization.sdp.backward_induction` over capacity grid and investment grid
- [x] Integration tests for SDP monotonicity and sensible policy
- [ ] Full three-stage game (investment + commitment + production) wired as a single `solve_dynamic_game` entry point
- [ ] Memoization of Cournot subgame solutions across SDP nodes

## Phase 4 -- Baseline Reproduction (Weeks 6-7)

- [x] Ontario calibration set from IESO public reports (2024 bundle:
      hourly demand, HOEP, fuel-level output, monthly generator
      capability), via `scripts/fetch_ieso.py` and
      `capgame/calibration/ontario.py`
- [x] `scripts/validate_ontario.py` compares the model's equilibrium
      price and fleet to published IESO numbers
- [ ] Calibrated Khalfallah parameter set in `data/` (for strict
      replication of the original paper)
- [ ] 16-year run for each of {energy-only, capacity payment, FCM, RO} x {oligopoly, cartel, monopoly}
- [ ] Comparison plots (capacity evolution, reserve margin, prices, welfare)
- [ ] `docs/baseline_reproduction.md` writeup

## Phase 5a -- Probabilistic Adequacy (Weeks 8-9)

- [x] `adequacy.reserve_margin`
- [x] `adequacy.lole` with exact convolution and demand pmf
- [x] `adequacy.eue` exact + Monte-Carlo
- [x] `stochastic.outages` with per-unit Bernoulli availability
- [ ] LOLE as a state-dependent constraint in the regulator's mechanism-parameter choice
- [ ] Ranking-comparison notebook

## Phase 5b -- Renewable Uncertainty (Weeks 10-12)

- [x] `stochastic.renewables.MarkovChain[RenewableState]` with toy four-state chain
- [x] Empirical calibration of a four-state renewable chain from IESO
      `GenOutputbyFuelHourly` data with Laplace-smoothed transitions
      (`capgame.calibration.renewables_cf.build_renewable_chain`)
- [ ] Calibration to NREL WIND Toolkit / NSRDB (latent-factor model)
- [ ] Augmented state (demand, renewables) in SDP
- [ ] Investment-mix comparison notebook

## Phase 6 -- Dashboard (Week 13)

- [x] Streamlit app organized around the four research questions (Home, RQ1-RQ4, Methodology)
- [x] Sidebar parameters with tooltips, session-state persistence, and URL deep-linking
- [x] Plotly figures (no dataframes in narrative pages; tables in expanders)
- [x] Split `capgame/app/cli.py` (launcher) and `capgame/app/ui.py` (UI)
- [x] Single-entry-point `ScenarioConfig -> ScenarioResult` consumed by the UI
- [ ] Streamlit Cloud deployment URL in README

## Phase 7 -- Paper and Release (Weeks 14-16)

- [x] Applied Ontario study: missing-money diagnostic, 4x3 mechanism-x-structure matrix, optimal-strike search, sensitivity sweep
- [x] Executable narrative notebook `notebooks/ontario_missing_money.ipynb`
- [x] 2024 -> 2050 forecaster (`capgame.forecast`) with Pathway /
      CapacityTrajectory / FuelPriceTrajectory / FixedCostTrajectory
      primitives and default IESO-APO-Reference pathway
- [x] `build_trajectory` + `run_trajectory` per-year scenario evaluation
      with missing-money and adequacy metrics
- [x] Monte Carlo uncertainty envelope (`run_monte_carlo`,
      `summarize_paths`) with P10/P50/P90 bands over demand, gas price,
      renewable build-out, and SMR schedule
- [x] `scripts/forecast_ontario.py` CLI emitting tidy CSVs + summary
- [x] Forecast section (5.1-5.6) appended to the Ontario notebook
- [ ] Draft working paper (10-12 pages)
- [ ] arXiv upload
- [ ] Tag v1.0 release, Zenodo DOI

## Phase 8 -- Research Extensions (post-v1)

- [ ] Supply-function equilibrium (`game.sfe`) for RQ beyond Cournot (stub raises informative error)
- [x] Grid-search bilevel solver with `solve_endogenous_strike` convenience (RQ4)
- [x] Market-structure variants `solve_oligopoly / solve_cartel / solve_monopoly` (RQ3)
- [ ] Full MPCC / KKT bilevel with interior-point (upgrade path from grid search)
- [ ] Storage and demand-response as first-class capacity technologies
  (current storage handling is a zero-MC quasi-firm placeholder in the forecaster)
- [ ] Pathway-conditional renewable Markov chains (re-calibrate CF
  distributions as wind/solar siting diversifies)
