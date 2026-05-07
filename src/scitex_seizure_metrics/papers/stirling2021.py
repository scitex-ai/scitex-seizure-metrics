"""Stirling et al. 2021 (Frontiers in Neurology) replica metrics.

Paper: 'Seizure Forecasting Using a Novel Sub-Scalp Ultra-Long Term
EEG Monitoring System'. Reports detection AUC + cycle-based forecasting
IoC + time-in-warning on a 4-channel sub-scalp device cohort.

Citation: Stirling RE et al., Front Neurol 2021; 12: 713794.
doi:10.3389/fneur.2021.713794
"""
from __future__ import annotations

import numpy as np

from .. import detection, forecasting
from ..policy import AlarmPolicy


def metrics(*, y_true, y_proba, times_seconds=None, seizure_times=None,
            sph_seconds: float = 60 * 60,
            sop_seconds: float = 60 * 60,
            n_surrogate: int = 500,
            name: str = "stirling2021") -> dict:
    """Reproduce the detection-AUC + forecasting-IoC panel.

    Args:
        y_true: per-window binary detection labels.
        y_proba: per-window predictions.
        times_seconds: timestamps (optional; for forecasting metrics).
        seizure_times: seizure onsets (optional).
        sph_seconds / sop_seconds: forecasting horizon (default 1h/1h).
        n_surrogate: surrogate count.
        name: identifier.

    Returns:
        dict with detection_auc, alarm_sensitivity (if forecasting
        inputs given), ioc, fp_per_hour, time_in_warning_frac.
    """
    out = {"paper": "stirling2021", "name": name}
    rep = detection.evaluate(y_true, y_proba, name=name)
    out["detection_auc"] = rep.roc_auc

    if times_seconds is not None and seizure_times is not None:
        cadence = float(np.median(np.diff(times_seconds))) if len(times_seconds) > 1 else 60.0
        pol = AlarmPolicy(
            sph_seconds=sph_seconds, sop_seconds=sop_seconds,
            cadence_seconds=cadence, refractory_seconds=sop_seconds,
            alarm_threshold=0.5, fp_denominator="interictal",
        )
        frep = forecasting.evaluate_stream(
            y_proba, times_seconds, seizure_times, pol,
            n_surrogate=n_surrogate, surrogate="circadian", name=name,
        )
        out["alarm_sensitivity"] = frep.sensitivity
        out["ioc_vs_circadian"] = frep.ioc
        out["fp_per_hour"] = frep.fp_per_hour
        out["time_in_warning_frac"] = frep.time_in_warning_frac

    return out
