"""L3 capacity-remuneration mechanisms.

Each mechanism in this layer is a pure function (or small dataclass plus an
``apply`` function) that takes a Cournot equilibrium and returns the firms'
net profits under the mechanism, along with any consumer payments.
"""

from __future__ import annotations

from capgame.mechanisms.capacity_payment import CapacityPayment
from capgame.mechanisms.energy_only import EnergyOnly
from capgame.mechanisms.forward_capacity import ForwardCapacityMarket, clear_auction
from capgame.mechanisms.reliability_options import ReliabilityOption

__all__ = [
    "CapacityPayment",
    "EnergyOnly",
    "ForwardCapacityMarket",
    "ReliabilityOption",
    "clear_auction",
]
