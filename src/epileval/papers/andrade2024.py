"""Andrade et al. 2024 (Front Neurosci) replica metrics.

Paper: 'On the performance of seizure prediction machine learning
methods across different databases: the sample and alarm-based
perspectives'. Reports both regimes side-by-side and a surrogate-
predictor chance baseline.

Citation: Andrade I, Teixeira C, Pinto M; Front Neurosci 2024;
doi:10.3389/fnins.2024.1417748
"""
from __future__ import annotations

import numpy as np

from .. import detection, forecasting
from ..policy import AlarmPolicy


def metrics(*, y_true, y_proba, times_seconds, seizure_times,
            sph_seconds: float = 5 * 60, sop_seconds: float = 30 * 60,
            n_surrogate: int = 1000,
            name: str = "andrade2024") -> dict:
    """Reproduce the side-by-side sample- + alarm-based panel.

    Returns:
        dict with sample_* and alarm_* metric pairs, plus the
        improvement-over-surrogate flag (analogous to the paper's
        '6/46 patients beat chance under alarm-based' framing).
    """
    out = {"paper": "andrade2024", "name": name}

    # Sample-based
    rep_s = detection.evaluate(y_true, y_proba, threshold=0.5, name=name)
    out["sample_auroc"] = rep_s.roc_auc
    out["sample_pr_auc"] = rep_s.pr_auc
    out["sample_sensitivity"] = rep_s.sensitivity
    out["sample_specificity"] = (
        None if rep_s.precision is None
        else 1.0 - rep_s.fp_per_day / 86400.0
    )
    out["sample_mcc"] = rep_s.mcc

    # Alarm-based
    cadence = float(np.median(np.diff(times_seconds))) if len(times_seconds) > 1 else 60.0
    policy = AlarmPolicy(
        sph_seconds=sph_seconds, sop_seconds=sop_seconds,
        cadence_seconds=cadence, refractory_seconds=sop_seconds,
        alarm_threshold=0.5, fp_denominator="interictal",
    )
    rep_a = forecasting.evaluate_stream(
        y_proba, times_seconds, seizure_times, policy,
        n_surrogate=n_surrogate, surrogate="poisson", name=name,
    )
    out["alarm_sensitivity"] = rep_a.sensitivity
    out["alarm_fp_per_hour"] = rep_a.fp_per_hour
    out["alarm_time_in_warning_frac"] = rep_a.time_in_warning_frac
    out["alarm_ioc"] = rep_a.ioc
    out["beats_chance_alarm"] = bool(rep_a.ioc > 0)

    return out
