"""Stochastic envelope over pathway uncertainty.

The deterministic :func:`default_ontario_pathway` fixes every anchor
year. Policy forecasts out to 2050 are dominated by uncertainty in a
handful of drivers; the Monte Carlo layer samples over those drivers
and re-runs the full trajectory per draw. Output is a long DataFrame
suitable for fan-chart-style summaries.

Uncertainty sources sampled
---------------------------
* **Peak demand growth**: multiplicative shifter applied to every
  ``peak_demand`` anchor from a given year onward; drawn from a
  lognormal with mean 1.0 and configurable sigma.
* **Mean demand growth**: same structure as peak; correlated with
  peak via a simple rank-shared shock (fast electrification scenarios
  shift both together).
* **Gas price trajectory**: multiplicative shifter on every gas-price
  anchor from a shock year onward (captures a persistent fuel-price
  regime).
* **Wind / solar / storage build-out speed**: multiplicative shifter
  on renewable and storage anchors from a shock year onward.
* **Nuclear availability**: multiplicative shifter on post-2028
  nuclear capacity (captures SMR schedule risk).

Each shock is drawn once per path and applied uniformly from the
shock year to the horizon. This is a two-scenario structural
uncertainty model (not a stochastic-process model); it captures the
kind of "how bad could this be" envelope policy analysts care about
without pretending to a fine-grained process that we don't have data
to estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capgame.calibration.ontario import OntarioCalibration
from capgame.forecast.pathways import (
    CapacityTrajectory,
    FuelPriceTrajectory,
    Pathway,
    default_ontario_pathway,
)
from capgame.forecast.trajectory import build_trajectory, run_trajectory
from capgame.mechanisms.base import Mechanism

__all__ = ["MonteCarloConfig", "run_monte_carlo", "summarize_paths"]


@dataclass(frozen=True)
class MonteCarloConfig:
    """Parameters for a Monte Carlo forecast ensemble.

    Shocks are lognormal multipliers with mean 1.0 and log-sigma as
    configured; they are applied from ``shock_year`` to the horizon.
    The ``base_pathway`` is evaluated once per draw with shocked
    anchor tables; the base year (2024) is always left unshocked
    because the calibration is data-anchored there.
    """

    n_paths: int = 100
    shock_year: int = 2030
    seed: int = 42
    demand_sigma: float = 0.08
    gas_sigma: float = 0.20
    renewable_sigma: float = 0.15
    nuclear_sigma: float = 0.10
    correlate_peak_and_mean: bool = True


def _apply_multiplier(
    traj: CapacityTrajectory,
    multiplier: float,
    shock_year: int,
) -> CapacityTrajectory:
    new_anchors = {y: mw if y < shock_year else mw * multiplier for y, mw in traj.anchors.items()}
    return CapacityTrajectory(name=traj.name, anchors=new_anchors)


def _apply_price_multiplier(
    traj: FuelPriceTrajectory,
    multiplier: float,
    shock_year: int,
) -> FuelPriceTrajectory:
    new_anchors = {y: p if y < shock_year else p * multiplier for y, p in traj.anchors.items()}
    return FuelPriceTrajectory(anchors=new_anchors)


def _shock_pathway(base: Pathway, rng: np.random.Generator, cfg: MonteCarloConfig) -> Pathway:
    """Return a copy of ``base`` with multiplicative shocks applied."""

    # Lognormal mean-1.0: exp(N(-sigma^2/2, sigma^2)).
    def ln_shock(sigma: float) -> float:
        if sigma <= 0:
            return 1.0
        return float(np.exp(rng.normal(-0.5 * sigma * sigma, sigma)))

    peak_mult = ln_shock(cfg.demand_sigma)
    mean_mult = peak_mult if cfg.correlate_peak_and_mean else ln_shock(cfg.demand_sigma)
    gas_mult = ln_shock(cfg.gas_sigma)
    renew_mult = ln_shock(cfg.renewable_sigma)
    storage_mult = ln_shock(cfg.renewable_sigma)
    nuc_mult = ln_shock(cfg.nuclear_sigma)

    shocked_fleet: dict[str, CapacityTrajectory] = {}
    for name, traj in base.fleet.items():
        if name == "NUCLEAR":
            shocked_fleet[name] = _apply_multiplier(traj, nuc_mult, max(2028, cfg.shock_year))
        elif name in ("WIND", "SOLAR"):
            shocked_fleet[name] = _apply_multiplier(traj, renew_mult, cfg.shock_year)
        elif name == "STORAGE":
            shocked_fleet[name] = _apply_multiplier(traj, storage_mult, cfg.shock_year)
        else:
            shocked_fleet[name] = traj
    shocked_peak = _apply_multiplier(base.peak_demand, peak_mult, cfg.shock_year)
    shocked_mean = (
        _apply_multiplier(base.mean_demand, mean_mult, cfg.shock_year)
        if base.mean_demand is not None
        else None
    )
    shocked_gas = _apply_price_multiplier(base.gas_price, gas_mult, cfg.shock_year)

    return Pathway(
        name=f"{base.name}+MC",
        fleet=shocked_fleet,
        peak_demand=shocked_peak,
        mean_demand=shocked_mean,
        load_factor=base.load_factor,
        gas_price=shocked_gas,
        fixed_costs=base.fixed_costs,
        heat_rates=base.heat_rates,
        variable_om=base.variable_om,
        elasticity=base.elasticity,
    )


def run_monte_carlo(
    base_cal: OntarioCalibration,
    years: Sequence[int],
    mechanism: Mechanism | None = None,
    market_structure: str = "oligopoly",
    include_storage: bool = True,
    base_pathway: Pathway | None = None,
    config: MonteCarloConfig | None = None,
) -> pd.DataFrame:
    """Run ``n_paths`` shocked trajectories and return a long-format DataFrame.

    The DataFrame has one row per (path, year) with the same columns
    as :func:`run_trajectory`, plus a ``path`` integer identifier.
    Use :func:`summarize_paths` to turn this into P10/P50/P90 bands.
    """
    cfg = config or MonteCarloConfig()
    base = base_pathway or default_ontario_pathway()
    rng = np.random.default_rng(cfg.seed)

    frames: list[pd.DataFrame] = []
    for path_idx in range(cfg.n_paths):
        shocked = _shock_pathway(base, rng, cfg)
        traj = build_trajectory(
            base_cal=base_cal,
            pathway=shocked,
            years=years,
            mechanism=mechanism,
            market_structure=market_structure,
            include_storage=include_storage,
        )
        df = run_trajectory(traj)
        df.insert(0, "path", path_idx)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_paths(
    mc_df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    columns: tuple[str, ...] = (
        "expected_price",
        "annual_welfare",
        "fleet_missing_money_per_year",
        "reserve_margin",
    ),
) -> pd.DataFrame:
    """Reduce a Monte Carlo long-frame to quantile bands by year.

    Returns a wide DataFrame indexed by year, with one column per
    (metric, quantile) combination named ``{metric}_q{int(q*100):02d}``.
    """
    if "path" not in mc_df.columns or "year" not in mc_df.columns:
        raise ValueError("mc_df must have 'path' and 'year' columns (see run_monte_carlo).")
    out = mc_df.groupby("year")[list(columns)].quantile(list(quantiles)).unstack(level=1)
    # Flatten column multi-index and rename quantile levels.
    out.columns = [f"{metric}_q{int(q * 100):02d}" for metric, q in out.columns]
    return out.reset_index()
