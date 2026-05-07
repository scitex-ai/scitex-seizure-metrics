"""Cook et al. 2013 (Lancet Neurology) replica metrics.

Paper: 'Prediction of seizure likelihood with a long-term, implanted
seizure advisory system in patients with drug-resistant epilepsy: a
first-in-man study'. The clinical-trial seminal paper. Reports:
- per-patient sensitivity at high-likelihood warning windows.
- proportion of time in high / moderate / low warning (Snyder scheme).
- no AUROC / AUPRC / FP/hr — clinical, not ML, framing.

Citation: Cook MJ et al., Lancet Neurol 2013; 12: 563–571.
doi:10.1016/S1474-4422(13)70075-9
"""
from __future__ import annotations

import numpy as np

from .. import forecasting
from ..policy import AlarmPolicy


def metrics(*, y_proba, times_seconds, seizure_times,
            sph_seconds: float = 0.0,
            sop_seconds: float = 60 * 60,
            high_threshold: float = 0.8,
            moderate_threshold: float = 0.5,
            n_surrogate: int = 200,
            name: str = "cook2013") -> dict:
    """Reproduce the Snyder-scheme time-in-warning panel.

    Args:
        y_proba: per-window predicted probabilities.
        times_seconds: matching timestamps.
        seizure_times: seizure onset times.
        sph_seconds: prediction horizon.
        sop_seconds: occurrence period.
        high_threshold / moderate_threshold: lights cut-offs.
        n_surrogate: surrogates for IoC.
        name: identifier.

    Returns:
        dict with sensitivity_high (the warning that flagged the seizure),
        time_in_high_frac, time_in_moderate_frac, time_in_low_frac,
        ioc_high — mirroring the Cook 2013 traffic-light reporting.
    """
    y_proba = np.asarray(y_proba, dtype=float)
    times_seconds = np.asarray(times_seconds, dtype=float)

    cadence = float(np.median(np.diff(times_seconds))) if len(times_seconds) > 1 else 60.0

    # High-likelihood evaluation
    pol_high = AlarmPolicy(
        sph_seconds=sph_seconds, sop_seconds=sop_seconds,
        cadence_seconds=cadence, refractory_seconds=sop_seconds,
        alarm_threshold=high_threshold, fp_denominator="interictal",
    )
    rep_high = forecasting.evaluate_stream(
        y_proba, times_seconds, seizure_times, pol_high,
        n_surrogate=n_surrogate, surrogate="poisson", name=name,
    )

    # Time-in-each-band fractions (recording-time-weighted)
    above_high = (y_proba >= high_threshold).mean()
    above_moderate = (
        (y_proba >= moderate_threshold) & (y_proba < high_threshold)
    ).mean()
    below_moderate = (y_proba < moderate_threshold).mean()

    return {
        "paper": "cook2013",
        "name": name,
        "sensitivity_high": rep_high.sensitivity,
        "ioc_high": rep_high.ioc,
        "fp_per_hour_high": rep_high.fp_per_hour,
        "time_in_high_frac": float(above_high),
        "time_in_moderate_frac": float(above_moderate),
        "time_in_low_frac": float(below_moderate),
    }
