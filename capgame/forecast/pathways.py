"""Pathway primitives: piecewise-linear trajectories of nameplate,
demand, fuel price, and fixed cost.

The design principle here is **anchor years**. A small dict
``{year: value}`` specifies the pathway at key reference points; any
intermediate year is linearly interpolated, and years outside the
anchor range extrapolate from the nearest endpoint (constant-edge
extrapolation, which is safer for long-horizon forecasts than linear
extrapolation of noisy tails).

The default Ontario pathway below mirrors the 2024 IESO *Annual
Planning Outlook (APO)* Reference scenario and its associated
procurement commitments (LT1 gas, Pickering refurbishment, Darlington
SMR), extended to 2050 with proportional extrapolation consistent with
Ontario's 2050 net-zero electricity roadmap. Every number is exposed
as a keyword argument so a user can swap in their own pathway without
rewriting any code.

References
----------
* IESO, *Annual Planning Outlook 2024*.
* OPG, *Darlington SMR Project* (first unit commissioning 2028-2030).
* IESO LT1 / LT2 procurement results (wind and gas additions).
* NREL *Annual Technology Baseline 2023* (capex decline curves).

Units convention
----------------
All MW values are nameplate at the start of the year. Fuel prices are
real ($2024/MMBtu). Fixed costs are real ($2024/MW-yr).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "CapacityTrajectory",
    "FixedCostTrajectory",
    "FuelPriceTrajectory",
    "Pathway",
    "default_ontario_pathway",
]

DEFAULT_HEAT_RATES: Mapping[str, float] = {
    "GAS_CCGT": 7.0,
    "GAS_PEAKER": 10.0,
}
DEFAULT_VOM: Mapping[str, float] = {
    "GAS_CCGT": 3.5,
    "GAS_PEAKER": 8.0,
    "NUCLEAR": 2.5,
    "HYDRO": 1.0,
    "BIOFUEL": 15.0,
}


@dataclass(frozen=True)
class CapacityTrajectory:
    """Piecewise-linear nameplate trajectory for one technology class.

    ``anchors`` is a ``{year: MW}`` mapping. Interior years linearly
    interpolate between their two bracketing anchors; years outside
    the anchor range clamp to the nearest endpoint. A trajectory with
    a single anchor is constant.
    """

    name: str
    anchors: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError(f"{self.name}: at least one anchor is required.")
        for y, mw in self.anchors.items():
            if mw < 0:
                raise ValueError(f"{self.name} @ {y}: MW must be >= 0, got {mw}")

    @property
    def years(self) -> list[int]:
        return sorted(self.anchors.keys())

    def mw_at(self, year: int) -> float:
        years = self.years
        if year <= years[0]:
            return float(self.anchors[years[0]])
        if year >= years[-1]:
            return float(self.anchors[years[-1]])
        # linear interpolation between bracketing anchors
        return float(
            np.interp(
                year,
                years,
                [self.anchors[y] for y in years],
            )
        )


@dataclass(frozen=True)
class FuelPriceTrajectory:
    """$/MMBtu by year (real), same anchor semantics as capacity."""

    anchors: dict[int, float]

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError("FuelPriceTrajectory: at least one anchor required.")
        for y, p in self.anchors.items():
            if p < 0:
                raise ValueError(f"Fuel price @ {y}: must be >= 0, got {p}")

    def price_at(self, year: int) -> float:
        years = sorted(self.anchors.keys())
        if year <= years[0]:
            return float(self.anchors[years[0]])
        if year >= years[-1]:
            return float(self.anchors[years[-1]])
        return float(np.interp(year, years, [self.anchors[y] for y in years]))


@dataclass(frozen=True)
class FixedCostTrajectory:
    """$/MW-yr fixed-cost trajectory.

    Supports two specification modes:

    * ``"anchors"``: explicit ``{year: cost}`` dict.
    * ``"decline"``: geometric decline from ``base_year`` with annual
      rate ``annual_decline`` (e.g. 0.01 = 1%/yr fall). Positive
      declines correspond to falling costs.
    """

    name: str
    mode: Literal["anchors", "decline"]
    base_year: int = 2024
    base_value: float = 0.0
    annual_decline: float = 0.0
    anchors: dict[int, float] = field(default_factory=dict)

    def cost_at(self, year: int) -> float:
        if self.mode == "anchors":
            if not self.anchors:
                raise ValueError(f"{self.name}: anchors mode requires anchors.")
            years = sorted(self.anchors.keys())
            if year <= years[0]:
                return float(self.anchors[years[0]])
            if year >= years[-1]:
                return float(self.anchors[years[-1]])
            return float(np.interp(year, years, [self.anchors[y] for y in years]))
        # decline mode: cost = base * (1 - r)^(year - base_year) clamped at 0.
        dt = max(0, year - self.base_year)
        val = self.base_value * (1.0 - self.annual_decline) ** dt
        return float(max(0.0, val))


@dataclass(frozen=True)
class Pathway:
    """Complete Ontario pathway specification.

    ``fleet`` maps technology class (``NUCLEAR``, ``HYDRO``,
    ``GAS_CCGT``, ``GAS_PEAKER``, ``BIOFUEL``, ``WIND``, ``SOLAR``,
    ``STORAGE``) to a :class:`CapacityTrajectory`. Missing entries are
    treated as zero capacity.

    ``peak_demand`` is a :class:`CapacityTrajectory` with the reserved
    "name" ``"PEAK"``. The average-load trajectory
    ``mean_demand`` is used to re-anchor the linear inverse-demand
    curve each year; if omitted, it defaults to a fixed
    ``load_factor`` times the peak.

    ``gas_price`` is the trajectory for the gas marginal-cost
    calculation. ``fixed_costs`` is keyed by technology class.

    ``heat_rates`` and ``variable_om`` are exposed so downstream users
    can override the defaults in :data:`DEFAULT_HEAT_RATES` /
    :data:`DEFAULT_VOM`.
    """

    name: str
    fleet: dict[str, CapacityTrajectory]
    peak_demand: CapacityTrajectory
    mean_demand: CapacityTrajectory | None = None
    load_factor: float = 0.60  # falls back if mean_demand is None
    gas_price: FuelPriceTrajectory = field(
        default_factory=lambda: FuelPriceTrajectory(anchors={2024: 3.5})
    )
    fixed_costs: dict[str, FixedCostTrajectory] = field(default_factory=dict)
    heat_rates: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_HEAT_RATES))
    variable_om: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_VOM))
    elasticity: float = -0.1

    def fleet_mw_at(self, year: int) -> dict[str, float]:
        return {name: traj.mw_at(year) for name, traj in self.fleet.items()}

    def peak_mw_at(self, year: int) -> float:
        return self.peak_demand.mw_at(year)

    def mean_mw_at(self, year: int) -> float:
        if self.mean_demand is not None:
            return self.mean_demand.mw_at(year)
        return self.peak_mw_at(year) * self.load_factor

    def marginal_cost_at(self, tech: str, year: int) -> float:
        """$/MWh marginal cost for a technology at a given year."""
        vom = self.variable_om.get(tech, 0.0)
        if tech in self.heat_rates:
            return float(self.heat_rates[tech] * self.gas_price.price_at(year) + vom)
        return float(vom)

    def fixed_cost_at(self, tech: str, year: int) -> float:
        if tech in self.fixed_costs:
            return self.fixed_costs[tech].cost_at(year)
        # defaults if no trajectory provided
        flat_defaults = {
            "NUCLEAR": 120_000.0,
            "HYDRO": 40_000.0,
            "GAS_CCGT": 60_000.0,
            "GAS_PEAKER": 45_000.0,
            "BIOFUEL": 80_000.0,
            "WIND": 45_000.0,
            "SOLAR": 25_000.0,
            "STORAGE": 35_000.0,
        }
        return float(flat_defaults.get(tech, 50_000.0))


def default_ontario_pathway() -> Pathway:
    """IESO-APO-Reference-like deterministic pathway, 2024 -> 2050.

    Fleet anchors reflect publicly announced Ontario commitments:

    * Pickering B refurbishment (2025-2028, short gap then return).
    * OPG Darlington SMR (first unit 2028-2030; fleet of 4 by 2035).
    * Bruce refurbishment ongoing; net positive by 2033.
    * LT1 gas procurement adds ~1.7 GW CCGT by 2028, declining
      post-2035 as per APO retirement schedule.
    * IESO renewable targets: ~12 GW wind by 2035, 20 GW by 2050;
      ~3 GW solar by 2035, 8 GW by 2050; batteries reach ~3 GW by
      2030 and ~10 GW by 2050.

    Demand follows the APO Reference: modest growth through 2030,
    electrification-driven acceleration post-2030 peaking near 48 GW
    by 2050.

    Fuel prices rise modestly in real terms: $3.5/MMBtu in 2024,
    $5.0/MMBtu by 2040, $6.0/MMBtu by 2050.

    Technology-specific capex declines follow NREL ATB 2023 trends:
    wind -1%/yr, solar -2%/yr, storage -3%/yr (from 2024 baselines).
    """
    fleet: dict[str, CapacityTrajectory] = {
        "NUCLEAR": CapacityTrajectory(
            name="NUCLEAR",
            anchors={
                2024: 12_000,
                2026: 10_500,  # Pickering A (Units 1 & 4) retired
                2028: 9_500,  # Pickering B temporary outage during refurb
                2030: 10_800,  # Pickering B Unit 5 back; Darlington SMR #1 online
                2033: 12_500,  # SMRs #2-3 + Bruce refurb complete
                2035: 13_500,  # SMR fleet complete
                2040: 14_500,
                2050: 15_000,
            },
        ),
        "HYDRO": CapacityTrajectory(
            name="HYDRO",
            anchors={
                2024: 8_800,
                2035: 9_100,  # modest incremental additions
                2050: 9_400,
            },
        ),
        "GAS_CCGT": CapacityTrajectory(
            name="GAS_CCGT",
            anchors={
                2024: 6_600,
                2028: 8_300,  # LT1 procurement online (~1.7 GW)
                2035: 7_800,  # first retirements
                2040: 5_500,
                2050: 3_000,  # net-zero trajectory
            },
        ),
        "GAS_PEAKER": CapacityTrajectory(
            name="GAS_PEAKER",
            anchors={
                2024: 3_800,
                2030: 3_800,
                2040: 2_500,
                2050: 1_000,
            },
        ),
        "BIOFUEL": CapacityTrajectory(
            name="BIOFUEL",
            anchors={2024: 100, 2050: 150},
        ),
        "WIND": CapacityTrajectory(
            name="WIND",
            anchors={
                2024: 5_000,
                2028: 6_600,  # LT2 procurement online
                2035: 12_000,
                2040: 15_500,
                2050: 20_000,
            },
        ),
        "SOLAR": CapacityTrajectory(
            name="SOLAR",
            anchors={
                2024: 500,
                2028: 1_500,
                2035: 3_500,
                2040: 5_500,
                2050: 8_000,
            },
        ),
        "STORAGE": CapacityTrajectory(
            name="STORAGE",
            anchors={
                2024: 250,  # LT1 ES procurement
                2028: 2_800,  # IESO ES procurement
                2030: 3_500,
                2035: 5_500,
                2040: 7_500,
                2050: 10_000,
            },
        ),
    }

    peak = CapacityTrajectory(
        name="PEAK",
        anchors={
            2024: 26_000,
            2030: 28_000,
            2035: 32_000,
            2040: 38_000,
            2045: 43_000,
            2050: 48_000,
        },
    )
    mean = CapacityTrajectory(
        name="MEAN",
        anchors={
            2024: 14_500,
            2030: 16_000,
            2035: 18_500,
            2040: 22_000,
            2045: 25_500,
            2050: 29_000,
        },
    )

    gas = FuelPriceTrajectory(
        anchors={
            2024: 3.5,
            2030: 4.0,
            2040: 5.0,
            2050: 6.0,
        }
    )

    fixed_costs: dict[str, FixedCostTrajectory] = {
        "NUCLEAR": FixedCostTrajectory(
            name="NUCLEAR", mode="decline", base_year=2024, base_value=120_000.0, annual_decline=0.0
        ),
        "HYDRO": FixedCostTrajectory(
            name="HYDRO", mode="decline", base_year=2024, base_value=40_000.0, annual_decline=0.0
        ),
        "GAS_CCGT": FixedCostTrajectory(
            name="GAS_CCGT",
            mode="decline",
            base_year=2024,
            base_value=60_000.0,
            annual_decline=0.0,
        ),
        "GAS_PEAKER": FixedCostTrajectory(
            name="GAS_PEAKER",
            mode="decline",
            base_year=2024,
            base_value=45_000.0,
            annual_decline=0.0,
        ),
        "BIOFUEL": FixedCostTrajectory(
            name="BIOFUEL",
            mode="decline",
            base_year=2024,
            base_value=80_000.0,
            annual_decline=0.0,
        ),
        "WIND": FixedCostTrajectory(
            name="WIND",
            mode="decline",
            base_year=2024,
            base_value=45_000.0,
            annual_decline=0.01,
        ),
        "SOLAR": FixedCostTrajectory(
            name="SOLAR",
            mode="decline",
            base_year=2024,
            base_value=25_000.0,
            annual_decline=0.02,
        ),
        "STORAGE": FixedCostTrajectory(
            name="STORAGE",
            mode="decline",
            base_year=2024,
            base_value=35_000.0,
            annual_decline=0.03,
        ),
    }

    return Pathway(
        name="IESO-APO-Reference-2050",
        fleet=fleet,
        peak_demand=peak,
        mean_demand=mean,
        gas_price=gas,
        fixed_costs=fixed_costs,
    )
