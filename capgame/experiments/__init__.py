"""L6 reproducible experiments.

Scripts in this package reproduce the baseline Khalfallah runs and the
extension studies. Each entry point is callable from the CLI and from the
Streamlit dashboard.
"""

from __future__ import annotations

from capgame.experiments.baseline import run_static_mechanism_comparison

# NB: ``capgame.experiments.ontario_study`` is intentionally NOT imported
# here. It depends on ``capgame.calibration.ontario``, which in turn
# depends on ``capgame.experiments.scenarios`` -- re-exporting the
# orchestrator would create an import cycle. Import it explicitly when
# needed: ``from capgame.experiments.ontario_study import ...``.
from capgame.experiments.scenarios import (
    AdequacyReport,
    MissingMoneyReport,
    ScenarioConfig,
    ScenarioResult,
    StateOutcome,
    missing_money,
    run_scenario,
)

__all__ = [
    "AdequacyReport",
    "MissingMoneyReport",
    "ScenarioConfig",
    "ScenarioResult",
    "StateOutcome",
    "missing_money",
    "run_scenario",
    "run_static_mechanism_comparison",
]
