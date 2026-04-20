"""Calibrate a renewable-availability Markov chain from IESO data.

We discretize the joint (wind, solar) capacity-factor process into a
small number of states and estimate the empirical transition matrix
with Laplace smoothing. The output is a
:class:`MarkovChain[RenewableState]` ready to plug into
:class:`~capgame.experiments.scenarios.ScenarioConfig`.

Estimation
----------
1.  Compute per-fuel hourly capacity factors from the ``GenOutputbyFuel``
    series divided by the fleet-wide nameplate capacity estimated from
    the monthly capability reports.
2.  Split each factor into ``low``/``high`` by its median → 4 joint
    states (``wL_sL``, ``wL_sH``, ``wH_sL``, ``wH_sH``).
3.  Count transitions ``N_{ij}`` between consecutive hourly states and
    form a row-stochastic transition matrix

    .. math::
        P_{ij} = \\frac{N_{ij} + \\alpha}{\\sum_k (N_{ik} + \\alpha)}

    with Laplace prior ``\\alpha`` (default ``1.0``) so no row is
    degenerate even for rarely visited states.
4.  Set each state's ``wind_cf`` / ``solar_cf`` to the mean factor
    observed in that bin (so the chain is a faithful reduction of the
    hourly series).

The procedure is a classical empirical-Bayes estimator for a discrete
Markov chain. Its stationary distribution is guaranteed to exist (the
Laplace prior makes ``P`` strictly positive, hence irreducible and
aperiodic).

Limitations
-----------
* The binning is axis-aligned; a k-means discretization on the joint
  (wind, solar) factor would capture diurnal solar patterns more
  faithfully. Deferred until the project has a need for more than four
  states (the scenario-runner cost scales linearly with state count, but
  four is usually enough for policy comparison).
* We use hourly-to-hourly transitions, which over-estimates persistence
  for long-horizon planning. A coarser (e.g. 6-hourly) aggregation is
  trivial to add via a ``resample`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from capgame.stochastic.demand import MarkovChain
from capgame.stochastic.renewables import RenewableState

__all__ = ["RenewableCalibration", "build_renewable_chain"]


@dataclass(frozen=True)
class RenewableCalibration:
    """Container for the calibrated renewable chain and diagnostics."""

    chain: MarkovChain[RenewableState]
    wind_capacity_mw: float
    solar_capacity_mw: float
    state_counts: dict[str, int]
    mean_wind_cf: float
    mean_solar_cf: float


def _estimate_nameplate(capability_df: pd.DataFrame, fuel: str) -> float:
    """Largest observed ``capability_mw`` (or available capacity) by fuel."""
    sub = capability_df[capability_df["fuel_type"].str.upper() == fuel.upper()]
    if sub.empty:
        return 0.0
    cap = sub["capability_mw"].fillna(sub["available_capacity_mw"])
    cap_by_gen = cap.groupby(sub["generator"]).max()
    return float(cap_by_gen.sum())


def build_renewable_chain(
    fuel_hourly: pd.DataFrame,
    capability_month: pd.DataFrame,
    *,
    laplace_alpha: float = 1.0,
) -> RenewableCalibration:
    """Fit a four-state :class:`MarkovChain` ``[RenewableState]``.

    Parameters
    ----------
    fuel_hourly
        DataFrame from :func:`load_fuel_hourly_xml` with columns
        ``[timestamp, fuel, output_mwh]``.
    capability_month
        DataFrame from :func:`load_fleet_capability_month` (one month is
        enough to size the fleet).
    laplace_alpha
        Dirichlet prior on transition rows. Positive values regularize
        toward uniform transitions and guarantee irreducibility.
    """
    required = {"timestamp", "fuel", "output_mwh"}
    if not required.issubset(fuel_hourly.columns):
        raise ValueError(f"fuel_hourly must have columns {required}")

    wind_cap = _estimate_nameplate(capability_month, "WIND")
    solar_cap = _estimate_nameplate(capability_month, "SOLAR")
    if wind_cap <= 0 or solar_cap <= 0:
        raise ValueError(
            f"Need positive wind and solar nameplate in the capability month: "
            f"wind={wind_cap}, solar={solar_cap}"
        )

    wide = fuel_hourly.pivot_table(
        index="timestamp", columns="fuel", values="output_mwh", aggfunc="sum"
    )
    if "WIND" not in wide.columns or "SOLAR" not in wide.columns:
        raise ValueError(f"fuel_hourly must contain WIND and SOLAR; got {list(wide.columns)}")
    wide = wide.sort_index()
    wind_cf = (wide["WIND"] / wind_cap).clip(lower=0.0, upper=1.0)
    solar_cf = (wide["SOLAR"] / solar_cap).clip(lower=0.0, upper=1.0)

    w_med = float(wind_cf.median())
    s_med = float(solar_cf.median())
    w_high = (wind_cf >= w_med).to_numpy()
    s_high = (solar_cf >= s_med).to_numpy()
    # State order:  0=wL_sL, 1=wL_sH, 2=wH_sL, 3=wH_sH.
    idx = (w_high.astype(int) << 1) | s_high.astype(int)

    n_states = 4
    counts = np.zeros((n_states, n_states), dtype=float)
    for a, b in pairwise(idx):
        counts[a, b] += 1.0
    counts += laplace_alpha
    P = counts / counts.sum(axis=1, keepdims=True)

    names = ["wL_sL", "wL_sH", "wH_sL", "wH_sH"]
    # Fallback anchors when a quadrant is empty (happens on constant
    # CF series in tests, and occasionally on short samples).
    overall_wind = float(np.clip(wind_cf.mean(), 0.0, 1.0))
    overall_solar = float(np.clip(solar_cf.mean(), 0.0, 1.0))
    mean_wind = np.empty(n_states)
    mean_solar = np.empty(n_states)
    state_counts: dict[str, int] = {}
    for k, name in enumerate(names):
        mask = idx == k
        state_counts[name] = int(mask.sum())
        if mask.any():
            mean_wind[k] = float(np.clip(wind_cf.to_numpy()[mask].mean(), 0.0, 1.0))
            mean_solar[k] = float(np.clip(solar_cf.to_numpy()[mask].mean(), 0.0, 1.0))
        else:
            mean_wind[k] = overall_wind
            mean_solar[k] = overall_solar

    states = [
        RenewableState(name=names[k], wind_cf=mean_wind[k], solar_cf=mean_solar[k])
        for k in range(n_states)
    ]
    chain = MarkovChain(states=states, transition_matrix=P)
    return RenewableCalibration(
        chain=chain,
        wind_capacity_mw=wind_cap,
        solar_capacity_mw=solar_cap,
        state_counts=state_counts,
        mean_wind_cf=float(wind_cf.mean()),
        mean_solar_cf=float(solar_cf.mean()),
    )
