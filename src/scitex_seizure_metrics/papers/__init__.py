"""Paper-replica shims — evaluate any model under the exact metric set
of a published paper, so cross-paper comparison becomes one-line.

Each module exposes one function `metrics(...)` returning the metrics
the paper reports, in the paper's preferred units. Internally each
shim composes scitex_seizure_metrics primitives (detection.evaluate /
forecasting.evaluate_stream / calibration.calibration_report /
surrogates) — no re-implementation of the math.

Available shims:
- cook2013     — first-in-man trial Snyder traffic-light (Lancet Neurology).
- karoly2017   — circadian + iEEG forecasting (Brain).
- kuhlmann2018 — Epilepsy Ecosystem clip-level AUC (Brain).
- maturana2020 — critical-slowing biomarker (Nat Commun).
- proix2021    — multi-day-horizon multidien forecasting (Lancet Neurology).
- stirling2021 — sub-scalp 4-channel forecasting (Front Neurol).
- andrade2024  — sample- vs alarm-based side-by-side (Front Neurosci).
"""
from __future__ import annotations

from . import (
    andrade2024, cook2013, karoly2017, kuhlmann2018, maturana2020,
    proix2021, stirling2021,
)

__all__ = [
    "andrade2024", "cook2013", "karoly2017", "kuhlmann2018",
    "maturana2020", "proix2021", "stirling2021",
]
