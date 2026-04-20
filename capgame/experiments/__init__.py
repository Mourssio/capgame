"""L6 reproducible experiments.

Scripts in this package reproduce the baseline Khalfallah runs and the
extension studies. Each entry point is callable from the CLI and from the
Streamlit dashboard.
"""

from __future__ import annotations

from capgame.experiments.baseline import run_static_mechanism_comparison
from capgame.experiments.scenarios import (
    AdequacyReport,
    ScenarioConfig,
    ScenarioResult,
    StateOutcome,
    run_scenario,
)

__all__ = [
    "AdequacyReport",
    "ScenarioConfig",
    "ScenarioResult",
    "StateOutcome",
    "run_scenario",
    "run_static_mechanism_comparison",
]
