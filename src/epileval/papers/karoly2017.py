"""Karoly et al. 2017 (Brain) replica metrics.

Paper: 'The circadian profile of epilepsy improves seizure forecasting'.
Reports: AUROC + Brier score + reliability + improvement-over-circadian
baseline. Per-window evaluation; SPH 30–60 min.

Citation: Karoly PJ et al., Brain 2017; 140: 2169–2182.
doi:10.1093/brain/awx173
"""
from __future__ import annotations

import numpy as np

from .. import calibration, detection, forecasting
from ..policy import AlarmPolicy


def metrics(*, y_true, y_proba, times_seconds=None, seizure_times=None,
            sph_seconds: float = 30 * 60,
            sop_seconds: float = 30 * 60,
            n_surrogate: int = 500,
            name: str = "karoly2017") -> dict:
    """Reproduce the metric panel from Karoly 2017.

    Args:
        y_true: per-window binary labels.
        y_proba: per-window predicted probabilities.
        times_seconds: optional per-window timestamps; required if
            forecasting metrics are wanted (alarm-based).
        seizure_times: optional seizure onset timestamps; required if
            forecasting metrics are wanted.
        sph_seconds: prediction horizon (Karoly: 30–60 min).
        sop_seconds: occurrence period.
        n_surrogate: surrogates for IoC.
        name: identifier carried into the report.

    Returns:
        dict with keys: auroc, brier, reliability, resolution,
        ece, alarm_sensitivity (if forecasting inputs given),
        ioc_vs_circadian, ioc_vs_poisson.
    """
    out = {"paper": "karoly2017", "name": name}
    rep = detection.evaluate(y_true, y_proba, name=name)
    out["auroc"] = rep.roc_auc
    out["brier"] = rep.brier

    cal = calibration.calibration_report(y_true, y_proba, n_bins=10)
    out["reliability"] = cal.reliability
    out["resolution"] = cal.resolution
    out["ece"] = cal.expected_calibration_error

    if times_seconds is not None and seizure_times is not None:
        policy = AlarmPolicy(
            sph_seconds=sph_seconds, sop_seconds=sop_seconds,
            cadence_seconds=float(np.median(np.diff(times_seconds))),
            refractory_seconds=sph_seconds + sop_seconds,
        )
        rep_circadian = forecasting.evaluate_stream(
            y_proba, times_seconds, seizure_times, policy,
            n_surrogate=n_surrogate, surrogate="circadian", name=name,
        )
        rep_poisson = forecasting.evaluate_stream(
            y_proba, times_seconds, seizure_times, policy,
            n_surrogate=n_surrogate, surrogate="poisson", name=name,
        )
        out["alarm_sensitivity"] = rep_circadian.sensitivity
        out["ioc_vs_circadian"] = rep_circadian.ioc
        out["ioc_vs_poisson"] = rep_poisson.ioc
        out["fp_per_hour"] = rep_circadian.fp_per_hour
        out["time_in_warning_frac"] = rep_circadian.time_in_warning_frac
    return out
