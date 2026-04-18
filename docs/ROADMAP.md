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

- [ ] Calibrated Khalfallah parameter set in `data/`
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

- [~] `stochastic.renewables` with toy two-state chain (placeholder)
- [ ] Calibration to NREL / Environment Canada data
- [ ] Augmented state (demand, renewables) in SDP
- [ ] Investment-mix comparison notebook

## Phase 6 -- Dashboard (Week 13)

- [~] Streamlit app with four tabs (Baseline, Comparator, Explorer, About)
- [ ] Plotly/Altair figures with consistent styling
- [ ] Streamlit Cloud deployment URL in README

## Phase 7 -- Paper and Release (Weeks 14-16)

- [ ] Draft working paper (10-12 pages)
- [ ] arXiv upload
- [ ] Tag v1.0 release, Zenodo DOI

## Phase 8 -- Research Extensions (post-v1)

- [ ] Supply-function equilibrium (`game.sfe`) for RQ beyond Cournot
- [ ] Full bilevel solver with endogenous strike (`game.bilevel`) for RQ4
- [ ] Storage and demand-response as first-class capacity technologies
