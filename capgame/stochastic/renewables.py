"""Correlated renewable availability as a finite-state Markov process.

Initial implementation: a two-factor discretized chain over (wind, solar)
capacity factors with a shared latent driver. Calibration to NREL or
Environment Canada capacity-factor histories is deferred to Phase 5b
(see ``docs/ROADMAP.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capgame.stochastic.demand import MarkovChain


@dataclass(frozen=True)
class RenewableState:
    """Joint renewable capacity factor."""

    name: str
    wind_cf: float
    solar_cf: float

    def __post_init__(self) -> None:
        for label, v in (("wind_cf", self.wind_cf), ("solar_cf", self.solar_cf)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{label} must be in [0, 1], got {v}")


def simple_two_state_renewables(
    low_wind: float = 0.15,
    high_wind: float = 0.45,
    low_solar: float = 0.10,
    high_solar: float = 0.30,
    correlation: float = 0.3,
) -> MarkovChain:
    """Build a toy four-state chain for (wind, solar) capacity factors.

    ``correlation`` in (-1, 1) controls how strongly the two factors co-move:
    positive values make (high, high) and (low, low) more persistent.

    The chain is intended for smoke-testing only and should be replaced with
    a calibrated process before any quantitative claim.
    """
    if not (-1.0 < correlation < 1.0):
        raise ValueError(f"correlation must be in (-1, 1), got {correlation}")

    from capgame.stochastic.demand import DemandState

    base = 0.25 * np.ones((4, 4))
    nudge = 0.05 * correlation
    diag_boost = np.eye(4) * nudge
    off = -nudge / 3.0
    P = base + diag_boost + (np.ones((4, 4)) - np.eye(4)) * off
    P = np.clip(P, 1e-6, None)
    P = P / P.sum(axis=1, keepdims=True)

    states = [
        DemandState(name="wL_sL", intercept=low_wind + low_solar + 1.0),
        DemandState(name="wL_sH", intercept=low_wind + high_solar + 1.0),
        DemandState(name="wH_sL", intercept=high_wind + low_solar + 1.0),
        DemandState(name="wH_sH", intercept=high_wind + high_solar + 1.0),
    ]
    return MarkovChain(states=states, transition_matrix=P)
