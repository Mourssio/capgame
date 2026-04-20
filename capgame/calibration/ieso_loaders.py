"""Raw-file loaders for IESO public reports.

Each loader takes a path to a file that ``scripts/fetch_ieso.py`` has
already placed on disk and returns a typed, pandas-free-when-reasonable
structure. Keeping IO isolated here makes the estimators pure functions
of arrays and lets us unit-test them against hand-crafted inputs.

Schemas (as of 2024 reports)
----------------------------
``PUB_Demand_{YYYY}.csv``
    3 comment rows prefixed with ``\\`` (backslash), then header row
    ``Date,Hour,Market Demand,Ontario Demand`` and 8784 (leap-year)
    hourly rows.

``PUB_PriceHOEPPredispOR_{YYYY}.csv``
    Same comment-then-header layout. Columns:
    ``Date,Hour,HOEP,Hour 1 Predispatch,Hour 2 Predispatch,
    Hour 3 Predispatch,OR 10 Min Sync,OR 10 Min non-sync,OR 30 Min``.

``PUB_GenOutputCapabilityMonth_{YYYYMM}.csv``
    3 comment rows, then header row
    ``Delivery Date,Generator,Fuel Type,Measurement,Hour 1..Hour 24``.
    ``Measurement`` is one of ``Capability``, ``Output``,
    ``Available Capacity``, ``Forecast``. One row per
    (generator, measurement, delivery_date).

``PUB_GenOutputbyFuelHourly_{YYYY}.xml``
    Nested XML: ``Document/DocBody/DailyData/HourlyData/FuelTotal``
    with ``Fuel`` and ``EnergyValue/Output`` leaves (MWh).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "FleetCapability",
    "HourlyDemand",
    "HourlyPrice",
    "load_demand",
    "load_fleet_capability_month",
    "load_fuel_hourly_xml",
    "load_hoep",
]

# IESO prefixes comment rows with a single backslash character.
_COMMENT = "\\"


@dataclass(frozen=True)
class HourlyDemand:
    """Hourly market and Ontario demand (MW)."""

    timestamps: pd.DatetimeIndex
    market_demand_mw: np.ndarray
    ontario_demand_mw: np.ndarray

    @property
    def n(self) -> int:
        return int(self.market_demand_mw.size)


@dataclass(frozen=True)
class HourlyPrice:
    """Hourly Ontario energy price (HOEP) and operating-reserve prices ($/MWh)."""

    timestamps: pd.DatetimeIndex
    hoep: np.ndarray
    or_10_sync: np.ndarray
    or_10_nonsync: np.ndarray
    or_30: np.ndarray

    @property
    def n(self) -> int:
        return int(self.hoep.size)


@dataclass(frozen=True)
class FleetCapability:
    """One month of generator-level capability and output.

    Stored in long format: each row is
    (delivery_date, generator, fuel_type, hour, capability_mw, output_mw,
    available_capacity_mw, forecast_mw).
    Missing measurement/hour combinations become NaN.
    """

    data: pd.DataFrame

    @property
    def generators(self) -> list[str]:
        return sorted(self.data["generator"].unique().tolist())

    @property
    def fuel_types(self) -> list[str]:
        return sorted(self.data["fuel_type"].unique().tolist())


def _read_ieso_csv(path: Path) -> pd.DataFrame:
    """Read an IESO public CSV, skipping the backslash comment header.

    ``index_col=False`` is required because the monthly generator file
    has a trailing comma on every data row (implying a 29th "column"
    that the header doesn't declare); without it pandas silently
    reinterprets the first data column as a row index.
    """
    return pd.read_csv(path, comment=_COMMENT, index_col=False)


def _timestamps_from_date_hour(dates: pd.Series, hours: pd.Series) -> pd.DatetimeIndex:
    """Combine ``Date`` and 1..24 ``Hour`` columns into a DatetimeIndex.

    IESO uses hour-ending convention (``Hour 1`` ends at 01:00), so we
    subtract one to place the timestamp at the start of the hour.
    """
    return pd.to_datetime(dates) + pd.to_timedelta(hours.astype(int) - 1, unit="h")


def load_demand(path: str | Path) -> HourlyDemand:
    """Load ``PUB_Demand_{YYYY}.csv`` into a :class:`HourlyDemand`."""
    df = _read_ieso_csv(Path(path))
    ts = _timestamps_from_date_hour(df["Date"], df["Hour"])
    return HourlyDemand(
        timestamps=pd.DatetimeIndex(ts),
        market_demand_mw=df["Market Demand"].to_numpy(dtype=float),
        ontario_demand_mw=df["Ontario Demand"].to_numpy(dtype=float),
    )


def load_hoep(path: str | Path) -> HourlyPrice:
    """Load ``PUB_PriceHOEPPredispOR_{YYYY}.csv`` into a :class:`HourlyPrice`."""
    df = _read_ieso_csv(Path(path))
    ts = _timestamps_from_date_hour(df["Date"], df["Hour"])
    return HourlyPrice(
        timestamps=pd.DatetimeIndex(ts),
        hoep=df["HOEP"].to_numpy(dtype=float),
        or_10_sync=df["OR 10 Min Sync"].to_numpy(dtype=float),
        or_10_nonsync=df["OR 10 Min non-sync"].to_numpy(dtype=float),
        or_30=df["OR 30 Min"].to_numpy(dtype=float),
    )


def load_fleet_capability_month(path: str | Path) -> FleetCapability:
    """Load one ``PUB_GenOutputCapabilityMonth_{YYYYMM}.csv`` file.

    Returns a long-format dataframe pivoted on the ``Measurement`` column
    so each row carries all four measurements for one (generator, date,
    hour). Fuels that do not report a given measurement simply leave it
    as NaN (e.g. thermal units typically have ``Capability`` but not
    ``Forecast``).
    """
    df = _read_ieso_csv(Path(path))
    df = df.rename(
        columns={
            "Delivery Date": "delivery_date",
            "Generator": "generator",
            "Fuel Type": "fuel_type",
            "Measurement": "measurement",
        }
    )
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], format="%Y-%m-%d")
    hour_cols = [c for c in df.columns if c.startswith("Hour ")]
    long = df.melt(
        id_vars=["delivery_date", "generator", "fuel_type", "measurement"],
        value_vars=hour_cols,
        var_name="hour",
        value_name="mw",
    )
    long["hour"] = long["hour"].str.replace("Hour ", "", regex=False).astype(int)
    long["mw"] = pd.to_numeric(long["mw"], errors="coerce")
    rename_map = {
        "Capability": "capability_mw",
        "Output": "output_mw",
        "Available Capacity": "available_capacity_mw",
        "Forecast": "forecast_mw",
    }
    long["measurement"] = long["measurement"].map(rename_map).fillna(long["measurement"])
    # Pivot measurement into columns so every row is one (generator, ts).
    wide = (
        long.set_index(["delivery_date", "generator", "fuel_type", "hour", "measurement"])["mw"]
        .unstack("measurement")
        .reset_index()
    )
    wide.columns.name = None
    for col in rename_map.values():
        if col not in wide.columns:
            wide[col] = np.nan
    return FleetCapability(data=wide)


def load_fuel_hourly_xml(path: str | Path) -> pd.DataFrame:
    """Parse ``PUB_GenOutputbyFuelHourly_{YYYY}.xml`` into a tidy DataFrame.

    Returns columns ``[timestamp, fuel, output_mwh]``. One row per
    (hour, fuel) combination. Namespaces are stripped so the caller does
    not have to know about the IESO schema URL.
    """
    tree = ET.parse(Path(path))
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[root.tag.find("{") + 1 : root.tag.find("}")]
    tag = (lambda name: f"{{{ns}}}{name}") if ns else (lambda name: name)
    rows: list[tuple[pd.Timestamp, str, float]] = []
    body = root.find(tag("DocBody"))
    if body is None:
        raise ValueError(f"{path}: missing DocBody element")
    for daily in body.findall(tag("DailyData")):
        day_elem = daily.find(tag("Day"))
        if day_elem is None or day_elem.text is None:
            continue
        day = pd.to_datetime(day_elem.text)
        for hourly in daily.findall(tag("HourlyData")):
            hour_elem = hourly.find(tag("Hour"))
            if hour_elem is None or hour_elem.text is None:
                continue
            hour = int(hour_elem.text)
            ts = day + pd.Timedelta(hours=hour - 1)
            for ft in hourly.findall(tag("FuelTotal")):
                fuel_elem = ft.find(tag("Fuel"))
                output_elem = ft.find(f"{tag('EnergyValue')}/{tag('Output')}")
                if fuel_elem is None or output_elem is None or output_elem.text is None:
                    continue
                try:
                    mwh = float(output_elem.text)
                except ValueError:
                    continue
                rows.append((ts, fuel_elem.text or "UNKNOWN", mwh))
    return pd.DataFrame(rows, columns=["timestamp", "fuel", "output_mwh"])
