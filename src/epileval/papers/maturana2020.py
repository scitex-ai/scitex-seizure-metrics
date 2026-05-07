"""Maturana et al. 2020 (Nat Commun) replica metrics.

Paper: 'Critical slowing down as a biomarker for seizure susceptibility'.
Reports: per-patient AUROC, IoC, time-in-warning. Long-window (hours-
to-days) features, causal filtering.

Citation: Maturana MI et al., Nat Commun 2020; 11: 2172.
doi:10.1038/s41467-020-15908-3
"""
from __future__ import annotations

import numpy as np

from .. import detection, forecasting
from ..policy import AlarmPolicy


def metrics(*, y_true, y_proba, times_seconds=None, seizure_times=None,
            sph_seconds: float = 0.0,
            sop_seconds: float = 60 * 60,
            n_surrogate: int = 500,
            name: str = "maturana2020") -> dict:
    """Reproduce the metric panel from Maturana 2020.

    Differs from Karoly 2017: shorter SPH (≈ 0, immediate forecast),
    longer SOP (1 h default), Poisson surrogate (no circadian baseline
    in the paper).

    Returns:
        dict with auroc, alarm_sensitivity, ioc, fp_per_hour,
        time_in_warning_frac, sensitivity_range_per_patient (if 2-D).
    """
    out = {"paper": "maturana2020", "name": name}
    rep = detection.evaluate(y_true, y_proba, name=name)
    out["auroc"] = rep.roc_auc

    if times_seconds is not None and seizure_times is not None:
        policy = AlarmPolicy(
            sph_seconds=sph_seconds, sop_seconds=sop_seconds,
            cadence_seconds=float(np.median(np.diff(times_seconds))),
            refractory_seconds=sop_seconds,
        )
        frep = forecasting.evaluate_stream(
            y_proba, times_seconds, seizure_times, policy,
            n_surrogate=n_surrogate, surrogate="poisson", name=name,
        )
        out["alarm_sensitivity"] = frep.sensitivity
        out["ioc"] = frep.ioc
        out["fp_per_hour"] = frep.fp_per_hour
        out["time_in_warning_frac"] = frep.time_in_warning_frac
    return out
