"""Smoke test for examples/01_detection_quick_start.ipynb via jupyter nbconvert.

Per PS505: notebook smoke tests must invoke ``jupyter nbconvert --execute``
or ``pytest --nbval[-lax]``. nbconvert is the canonical SciTeX choice.
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
    Path(__file__).resolve().parents[2] / "examples" / "01_detection_quick_start.ipynb"
)

if not NOTEBOOK.is_file():
    pytest.skip(f"missing notebook: {NOTEBOOK}", allow_module_level=True)


def test_detection_quick_start_notebook_executes_cleanly(tmp_path):
    """Run the detection notebook with jupyter nbconvert --execute."""
    # Arrange
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    # Act
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
    # Assert
    assert proc.returncode == 0, (
        f"nbconvert failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
