"""Proix et al. 2021 (Lancet Neurology) replica metrics.

Paper: 'Forecasting seizure risk in adults with focal epilepsy: a
development and validation study'. Multi-day-horizon forecasting using
multidien IEA cycles on RNS data. Reports:
- IoC (Improvement-over-Chance) at horizons 1 h, 1 d, 3 d.
- AUC of sensitivity vs proportion-time-in-warning.
- Brier skill score.

Citation: Proix T et al., Lancet Neurol 2021; 20: 127–135.
doi:10.1016/S1474-4422(20)30396-3
"""
from __future__ import annotations

import numpy as np

from .. import forecasting
from ..policy import AlarmPolicy

HORIZONS_DEFAULT = {
    "1h": 3600.0,
    "1d": 86400.0,
    "3d": 3 * 86400.0,
}


def metrics(*, y_proba, times_seconds, seizure_times,
            horizons_seconds: dict | None = None,
            n_surrogate: int = 500,
            name: str = "proix2021") -> dict:
    """Reproduce the multi-horizon IoC panel.

    Args:
        y_proba: per-window predictions.
        times_seconds: matching timestamps.
        seizure_times: seizure onsets.
        horizons_seconds: dict mapping horizon-label to SOP-seconds; the
            forecasting horizon. Defaults to 1h / 1d / 3d.
        n_surrogate: surrogate count.
        name: identifier.

    Returns:
        dict with per-horizon: sensitivity, ioc, fp_per_hour,
        time_in_warning_frac. Plus a multidien_baseline IoC vs
        7-day-period surrogate.
    """
    if horizons_seconds is None:
        horizons_seconds = HORIZONS_DEFAULT
    cadence = float(np.median(np.diff(times_seconds))) if len(times_seconds) > 1 else 3600.0

    out = {"paper": "proix2021", "name": name, "horizons": {}}
    for label, sop in horizons_seconds.items():
        pol = AlarmPolicy(
            sph_seconds=0.0, sop_seconds=sop,
            cadence_seconds=cadence, refractory_seconds=sop,
            alarm_threshold=0.5, fp_denominator="interictal",
        )
        rep_p = forecasting.evaluate_stream(
            y_proba, times_seconds, seizure_times, pol,
            n_surrogate=n_surrogate, surrogate="poisson", name=name,
        )
        rep_md = forecasting.evaluate_stream(
            y_proba, times_seconds, seizure_times, pol,
            n_surrogate=n_surrogate, surrogate="multidien", name=name,
        )
        out["horizons"][label] = {
            "sensitivity": rep_p.sensitivity,
            "ioc_vs_poisson": rep_p.ioc,
            "ioc_vs_multidien": rep_md.ioc,
            "fp_per_hour": rep_p.fp_per_hour,
            "time_in_warning_frac": rep_p.time_in_warning_frac,
        }
    return out
