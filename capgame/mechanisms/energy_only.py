"""Energy-only market (baseline).

Firms recover fixed costs solely from the energy market. No capacity payment,
no refund, no forward procurement. This is the baseline against which every
other mechanism is compared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from capgame.game.cournot import CournotEquilibrium
from capgame.mechanisms.base import MechanismOutcome


@dataclass(frozen=True)
class EnergyOnly:
    """Parameterless marker mechanism."""

    def apply(
        self,
        equilibrium: CournotEquilibrium,
        capacities: Sequence[float] | None = None,
    ) -> MechanismOutcome:
        del capacities
        energy_profits = np.asarray(equilibrium.profits, dtype=float).copy()
        zeros = np.zeros_like(energy_profits)
        return MechanismOutcome(
            energy_profits=energy_profits,
            capacity_payments=zeros,
            refunds=zeros.copy(),
            net_profits=energy_profits.copy(),
            consumer_cost=0.0,
        )


def apply(
    equilibrium: CournotEquilibrium,
    capacities: Sequence[float] | None = None,
) -> MechanismOutcome:
    """Function-form entry point matching the other mechanisms."""
    return EnergyOnly().apply(equilibrium, capacities)
