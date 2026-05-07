"""Cross-paper bridge — analytic bounds between sample- and alarm-based
metrics under a declared AlarmPolicy.

Useful when comparing a published paper that reported only sample-based
AUC against a paper that reported only alarm-based sensitivity + FP/hr.

Derivations (informal):

Let
- s = per-window sample sensitivity = P(yhat=1 | y=1)
- α = per-window false-positive rate = 1 - specificity = P(yhat=1 | y=0)
- π = pre-ictal-window prevalence = P(y=1)
- K = ⌈SOP / cadence⌉ = number of independent prediction windows whose
      "above-threshold" event would catch a seizure under the alarm
      semantics.
- R = refractory_seconds (minimum gap between alarms after merging).
- T = total observation duration (seconds).

ALARM SENSITIVITY (per-seizure detection probability):

Upper bound (independent errors, perfect coverage):
    alarm_sens_upper = 1 - (1 - s) ** K

Lower bound (fully clustered errors — if any one of the K windows is
correctly above threshold, all K are):
    alarm_sens_lower = s

Prevalence-adjusted upper (when prevalence is very low, even 'perfect'
sample-sens may not give K independent chances because there may not
be K positive-labelled windows in the SOP):
    K_eff = min(K, max(1, int(round(SOP * π / cadence))))
    alarm_sens_upper_with_prevalence = 1 - (1 - s) ** K_eff

FP/hr:

Naive (no refractory, independent errors):
    fp_hr_naive = α * (3600 / cadence) * (1 - π)

Refractory cap (no two alarms within R, regardless of α):
    fp_hr_cap = 3600 / R

Upper bound:
    fp_hr_upper = min(fp_hr_naive, fp_hr_cap)

Lower bound (under maximal correlation, FP-clustering, alarm count
collapses; conservative non-trivial lower bound depends on the
correlation length we cannot infer from sample metrics alone):
    fp_hr_lower = 0.0   (we report 0 by convention; calibrated lower
                         bounds require an autocorr-proxy parameter).

References:
- Andrade et al. 2024 — sample- vs alarm-based perspectives.
- Mormann et al. 2007 — definition of false-prediction rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleToAlarmBounds:
    """Analytic bounds on alarm-based metrics derived from sample-based.

    Attrs:
        alarm_sensitivity_upper: 1 - (1 - s) ** K_eff
        alarm_sensitivity_lower: s (worst-case clustering)
        fp_per_hour_lower: 0.0 by convention (correlation-dependent)
        fp_per_hour_upper: min(α * preds_per_hour * (1 - π), 3600 / R)
        K_effective: number of independent chances actually used
        notes: free-form list of pertinent caveats
    """
    alarm_sensitivity_upper: float
    alarm_sensitivity_lower: float
    fp_per_hour_lower: float
    fp_per_hour_upper: float
    K_effective: int = 1
    notes: tuple[str, ...] = ()


def sample_to_alarm(*, sample_sensitivity: float,
                    sample_specificity: float,
                    sop_seconds: float,
                    cadence_seconds: float,
                    refractory_seconds: float = 0.0,
                    prevalence: float = 0.5) -> SampleToAlarmBounds:
    """Bound alarm-based metrics from sample-based metrics + AlarmPolicy.

    Args:
        sample_sensitivity: per-window sensitivity (true-positive rate).
        sample_specificity: per-window specificity (1 - FPR).
        sop_seconds: Seizure Occurrence Period.
        cadence_seconds: time step between predictions.
        refractory_seconds: minimum gap between alarms.
        prevalence: per-window prior probability of pre-ictal class.
            Lower prevalence reduces K_effective (the number of
            independent chances to detect each seizure) AND reduces
            FP/hr (because the per-hour count of negative windows
            scales with 1-π).

    Returns:
        SampleToAlarmBounds with four numbers and K_effective.
    """
    if cadence_seconds <= 0 or sop_seconds <= 0:
        raise ValueError("cadence_seconds and sop_seconds must be > 0")
    for label, val in [("sensitivity", sample_sensitivity),
                       ("specificity", sample_specificity),
                       ("prevalence", prevalence)]:
        if not (0 <= val <= 1):
            raise ValueError(f"{label} must be in [0, 1]; got {val}")

    s = sample_sensitivity
    alpha = 1.0 - sample_specificity
    K = max(1, int(np.ceil(sop_seconds / cadence_seconds)))
    # Prevalence-adjusted effective K: very-low-prevalence streams
    # cannot offer K independent chances because there may not be K
    # pre-ictal-labelled windows inside the SOP.
    K_eff = max(1, int(round(K * prevalence))) if prevalence < 1.0 else K
    K_eff = min(K, K_eff if K_eff > 0 else 1)

    alarm_sens_upper = 1.0 - (1.0 - s) ** K_eff
    alarm_sens_lower = s

    preds_per_hour = 3600.0 / cadence_seconds
    naive_fph = alpha * preds_per_hour * (1.0 - prevalence)
    if refractory_seconds > 0:
        fph_upper = min(naive_fph, 3600.0 / refractory_seconds)
    else:
        fph_upper = naive_fph

    notes = []
    if prevalence < 0.05:
        notes.append("very low prevalence — K_eff reduced; bounds wider")
    if refractory_seconds <= 0:
        notes.append("refractory=0 — fp_per_hour_upper not capped")

    return SampleToAlarmBounds(
        alarm_sensitivity_upper=float(alarm_sens_upper),
        alarm_sensitivity_lower=float(alarm_sens_lower),
        fp_per_hour_lower=0.0,
        fp_per_hour_upper=float(fph_upper),
        K_effective=int(K_eff),
        notes=tuple(notes),
    )


def alarm_to_sample(*, alarm_sensitivity: float, fp_per_hour: float,
                    sop_seconds: float, cadence_seconds: float,
                    refractory_seconds: float = 0.0,
                    prevalence: float = 0.5) -> dict:
    """Reverse-bound: feasible sample-metric ranges from alarm metrics.

    Returns dict with sample_sensitivity_lower / upper and
    sample_specificity_lower / upper.
    """
    if cadence_seconds <= 0 or sop_seconds <= 0:
        raise ValueError("cadence_seconds and sop_seconds must be > 0")
    if not (0 <= alarm_sensitivity <= 1):
        raise ValueError("alarm_sensitivity must be in [0, 1]")

    K = max(1, int(np.ceil(sop_seconds / cadence_seconds)))
    K_eff = max(1, int(round(K * prevalence))) if prevalence < 1.0 else K
    K_eff = min(K, K_eff if K_eff > 0 else 1)

    if alarm_sensitivity <= 0:
        sens_lower = 0.0
    else:
        sens_lower = 1.0 - (1.0 - alarm_sensitivity) ** (1.0 / K_eff)
    sens_upper = float(min(1.0, alarm_sensitivity))

    preds_per_hour = 3600.0 / cadence_seconds
    if refractory_seconds > 0:
        fp_per_hour_eff = min(fp_per_hour, 3600.0 / refractory_seconds)
    else:
        fp_per_hour_eff = fp_per_hour
    # Invert naive_fph = α * preds_per_hour * (1 - π)
    denom = preds_per_hour * max(1e-12, 1.0 - prevalence)
    alpha_min = float(fp_per_hour_eff / denom)
    return {
        "sample_sensitivity_lower": float(sens_lower),
        "sample_sensitivity_upper": float(sens_upper),
        "sample_specificity_lower": 0.0,
        "sample_specificity_upper": float(max(0.0, 1.0 - alpha_min)),
        "K_effective": int(K_eff),
    }
