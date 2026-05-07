"""Paper-replica shims — evaluate any model under the exact metric set
of a published paper, so cross-paper comparison becomes one-line.

Each module exposes one function `metrics(...)` returning the metrics
the paper reports, in the paper's preferred units. Internally each
shim composes epileval primitives (detection.evaluate /
forecasting.evaluate_stream / calibration.calibration_report /
surrogates) — no re-implementation of the math.

Available shims:
- karoly2017  — circadian + iEEG forecasting (Brain).
- maturana2020 — critical-slowing biomarker (Nat Commun).
- kuhlmann2018 — Epilepsy Ecosystem clip-level AUC (Brain).
- andrade2024  — sample- vs alarm-based comparison (Front Neurosci).
"""
from __future__ import annotations

from . import andrade2024, karoly2017, kuhlmann2018, maturana2020

__all__ = ["andrade2024", "karoly2017", "kuhlmann2018", "maturana2020"]
