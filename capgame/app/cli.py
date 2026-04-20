"""Console launcher for the Streamlit UI.

Importing :mod:`capgame.app` or this module does not start Streamlit; the
``capgame-app`` console script calls :func:`main`, which shells out to
``streamlit run`` targeting :mod:`capgame.app.ui`. Keeping the launcher and
the UI in separate modules means the UI file is safe to import from
notebooks and from tests without the subprocess side-effects.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ui_path() -> Path:
    """Absolute path to the Streamlit UI entry file."""
    return Path(__file__).resolve().parent / "ui.py"


def main() -> None:
    """Launch the Streamlit UI in a subprocess.

    Registered in ``pyproject.toml`` as the ``capgame-app`` console script.
    Returns the exit code of the Streamlit process.
    """
    cmd = [sys.executable, "-m", "streamlit", "run", str(_ui_path())]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
