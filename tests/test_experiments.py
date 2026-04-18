"""Smoke tests for experiment entry points."""

from __future__ import annotations

import pandas as pd

from capgame.experiments.baseline import BaselineConfig, run_static_mechanism_comparison


def test_run_static_mechanism_comparison_returns_dataframe() -> None:
    df = run_static_mechanism_comparison()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(df["mechanism"]) == {
        "Energy-only",
        "Capacity payment",
        "Forward capacity",
        "Reliability options",
    }
    for col in [
        "total_quantity",
        "price",
        "hhi",
        "producer_surplus",
        "consumer_surplus",
        "consumer_payment_for_capacity",
        "welfare",
    ]:
        assert col in df.columns


def test_run_accepts_custom_config() -> None:
    cfg = BaselineConfig(capacity_payment_rho=20.0)
    df = run_static_mechanism_comparison(cfg)
    cp_row = df[df["mechanism"] == "Capacity payment"].iloc[0]
    eo_row = df[df["mechanism"] == "Energy-only"].iloc[0]
    assert cp_row["producer_surplus"] > eo_row["producer_surplus"]
