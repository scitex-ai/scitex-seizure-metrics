"""Smoke test for examples/quick_start_forecasting.py — runs main()."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples"))


def test_quick_start_forecasting_runs():
    import quick_start_forecasting as m
    m.main()
