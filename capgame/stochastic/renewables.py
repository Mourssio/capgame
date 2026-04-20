"""Correlated renewable availability as a finite-state Markov process.

Initial implementation: a two-factor discretized chain over (wind, solar)
capacity factors with a shared latent driver. Calibration to NREL or
Environment Canada capacity-factor histories is deferred to Phase 5b
(see ``docs/ROADMAP.md``).

The chain is :class:`MarkovChain[RenewableState]`, so it shares all
machinery (stationary distribution, sampling, ``distribution_at``) with the
demand chain in :mod:`capgame.stochastic.demand` without any semantic
overload of :class:`~capgame.stochastic.demand.DemandState`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capgame.stochastic.demand import MarkovChain

__all__ = ["RenewableState", "simple_two_state_renewables"]


@dataclass(frozen=True)
class RenewableState:
    """Joint wind / solar capacity factor for a single Markov state.

    Parameters
    ----------
    name
        Human-readable label, e.g. ``"wL_sH"`` for low wind, high solar.
    wind_cf, solar_cf
        Per-resource capacity factors in ``[0, 1]``.
    """

    name: str
    wind_cf: float
    solar_cf: float

    def __post_init__(self) -> None:
        for label, v in (("wind_cf", self.wind_cf), ("solar_cf", self.solar_cf)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{label} must be in [0, 1], got {v}")

    def available_output(self, wind_capacity_mw: float, solar_capacity_mw: float) -> float:
        """Expected MW output given installed nameplate capacities."""
        if wind_capacity_mw < 0 or solar_capacity_mw < 0:
            raise ValueError("Installed capacities must be non-negative.")
        return self.wind_cf * wind_capacity_mw + self.solar_cf * solar_capacity_mw


def simple_two_state_renewables(
    low_wind: float = 0.15,
    high_wind: float = 0.45,
    low_solar: float = 0.10,
    high_solar: float = 0.30,
    correlation: float = 0.3,
) -> MarkovChain[RenewableState]:
    """Build a toy four-state chain for (wind, solar) capacity factors.

    ``correlation`` in ``(-1, 1)`` biases the diagonal of the transition
    matrix: positive values make (high, high) and (low, low) more
    persistent. **Effective range** after the post-construction renormalization
    described below is roughly ``|correlation| <= 0.75``; magnitudes beyond
    that are clipped by the ``np.clip(P, 1e-6, None)`` floor and begin to
    produce near-uniform transitions rather than stronger co-movement. This
    is a limitation of the ad-hoc construction, not a property of real
    wind-solar dynamics.

    Returns
    -------
    MarkovChain[RenewableState]
        Four-state chain with payloads ``{wL_sL, wL_sH, wH_sL, wH_sH}``.
    """
    if not (-1.0 < correlation < 1.0):
        raise ValueError(f"correlation must be in (-1, 1), got {correlation}")

    # TODO(phase-5b): replace this ad-hoc construction with a calibrated
    # transition matrix fit to historical wind/solar capacity-factor time
    # series (NREL WIND Toolkit + NSRDB, or Environment Canada HCDCS).
    # The right object to calibrate is a latent-factor model whose
    # discretization yields this 4x4 chain, not the 4x4 directly.
    base = 0.25 * np.ones((4, 4))
    nudge = 0.05 * correlation
    diag_boost = np.eye(4) * nudge
    off = -nudge / 3.0
    P = base + diag_boost + (np.ones((4, 4)) - np.eye(4)) * off
    P = np.clip(P, 1e-6, None)
    P = P / P.sum(axis=1, keepdims=True)

    states = [
        RenewableState(name="wL_sL", wind_cf=low_wind, solar_cf=low_solar),
        RenewableState(name="wL_sH", wind_cf=low_wind, solar_cf=high_solar),
        RenewableState(name="wH_sL", wind_cf=high_wind, solar_cf=low_solar),
        RenewableState(name="wH_sH", wind_cf=high_wind, solar_cf=high_solar),
    ]
    return MarkovChain(states=states, transition_matrix=P)
