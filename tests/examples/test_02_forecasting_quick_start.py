"""Smoke test for examples/02_forecasting_quick_start.ipynb via jupyter nbconvert.

Per PS505: notebook smoke tests must invoke ``jupyter nbconvert --execute``
or ``pytest --nbval[-lax]``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbconvert")

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_forecasting_quick_start.ipynb"
)


def test_notebook_executes(tmp_path):
    """Run the forecasting notebook with jupyter nbconvert --execute."""
    assert NOTEBOOK.is_file(), f"missing notebook: {NOTEBOOK}"
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=180",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, (
        f"nbconvert failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
