r"""Cross-paper bridge — analytic bounds between sample- and alarm-based
metrics under a declared AlarmPolicy.

Useful when comparing a published paper that reported only sample-based
AUC against a paper that reported only alarm-based sensitivity + FP/hr.

Symbols
-------

- :math:`s` — per-window sample sensitivity, :math:`\Pr(\hat{y} = 1 \mid y = 1)`.
- :math:`\alpha` — per-window false-positive rate,
  :math:`1 - \text{specificity} = \Pr(\hat{y} = 1 \mid y = 0)`.
- :math:`\pi` — pre-ictal-window prevalence, :math:`\Pr(y = 1)`.
- :math:`\Delta` — ``cadence_seconds`` (seconds between prediction windows).
- :math:`\text{SOP}` — Seizure Occurrence Period (seconds).
- :math:`R` — ``refractory_seconds`` (minimum gap between alarms).

Windows per SOP (K)
-------------------

The number of prediction windows inside one Seizure Occurrence Period —
i.e. the number of independent chances to catch a given seizure, since a
seizure is "caught" iff at least one of its SOP windows fires:

.. math::

   K = \left\lceil \frac{\text{SOP}}{\Delta} \right\rceil

By construction every seizure's SOP contains exactly :math:`K` windows,
*independent of the global prevalence* :math:`\pi`. Prevalence governs
how many windows are pre-ictal across the whole stream (and hence the
FP/hr negative-window count), not how many windows sit inside a single
SOP. We therefore use :math:`K_{\text{eff}} = K` for the per-seizure
detection bounds and do **not** shrink it by :math:`\pi`.

Alarm sensitivity (per-seizure detection probability)
-----------------------------------------------------

Upper bound (independent errors — optimistic envelope):

.. math::

   \text{alarm\_sens}_{\text{upper}} = 1 - (1 - s)^{K}

Lower bound (fully-clustered errors — if the :math:`K` windows inside one
SOP are perfectly correlated, either all fire or none does, so the
per-seizure detection probability collapses to the per-window
sensitivity; pessimistic envelope):

.. math::

   \text{alarm\_sens}_{\text{lower}} = s

Because :math:`K = \lceil \text{SOP} / \Delta \rceil`, a longer SOP now
correctly widens the detection band (more chances per seizure) — the
earlier prevalence-adjustment made SOP = 15 s and SOP = 60 s degenerate
at low :math:`\pi`, which was wrong.

FP/hr (alarms per hour)
-----------------------

Naive (no refractory, independent errors):

.. math::

   \text{FP/h}_{\text{naive}} = \alpha \cdot \frac{3600}{\Delta} \cdot (1 - \pi)

Refractory cap (no two alarms within :math:`R` seconds, regardless of
:math:`\alpha`):

.. math::

   \text{FP/h}_{\text{cap}} = \frac{3600}{R}

Upper bound:

.. math::

   \text{FP/h}_{\text{upper}} = \min\!\big(\text{FP/h}_{\text{naive}},\ \text{FP/h}_{\text{cap}}\big)

Lower bound: ``0.0`` by convention. Under maximal positive correlation
the alarm count collapses; a calibrated non-trivial lower bound requires
an autocorrelation-proxy parameter that sample-only metrics do not
identify.

References
----------

- Andrade et al. 2024 — sample- vs alarm-based perspectives.
- Mormann et al. 2007 — definition of false-prediction rate.
- ``docs/math/sample_to_alarm.md`` — paper-ready derivation including
  the inverse direction (alarm → sample) and a worked example.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleToAlarmBounds:
    """Analytic bounds on alarm-based metrics derived from sample-based.

    Attrs:
        alarm_sensitivity_upper: 1 - (1 - s) ** K
        alarm_sensitivity_lower: s (worst-case clustering)
        fp_per_hour_lower: 0.0 by convention (correlation-dependent)
        fp_per_hour_upper: min(α * preds_per_hour * (1 - π), 3600 / R)
        K_effective: windows per SOP, K = ceil(SOP / cadence) — the
            number of independent chances to detect each seizure
        notes: free-form list of pertinent caveats
    """

    alarm_sensitivity_upper: float
    alarm_sensitivity_lower: float
    fp_per_hour_lower: float
    fp_per_hour_upper: float
    K_effective: int = 1
    notes: tuple[str, ...] = ()


def sample_to_alarm(
    *,
    sample_sensitivity: float,
    sample_specificity: float,
    sop_seconds: float,
    cadence_seconds: float,
    refractory_seconds: float = 0.0,
    prevalence: float = 0.5,
) -> SampleToAlarmBounds:
    """Bound alarm-based metrics from sample-based metrics + AlarmPolicy.

    Args:
        sample_sensitivity: per-window sensitivity (true-positive rate).
        sample_specificity: per-window specificity (1 - FPR).
        sop_seconds: Seizure Occurrence Period.
        cadence_seconds: time step between predictions.
        refractory_seconds: minimum gap between alarms.
        prevalence: per-window prior probability of pre-ictal class.
            Affects ONLY FP/hr: the per-hour count of negative windows
            scales with 1-π. It does NOT shrink the per-seizure
            detection bounds — every seizure's SOP holds K windows by
            construction, regardless of the global prevalence.

    Returns:
        SampleToAlarmBounds with four numbers and K_effective (= K).
    """
    if cadence_seconds <= 0 or sop_seconds <= 0:
        raise ValueError("cadence_seconds and sop_seconds must be > 0")
    for label, val in [
        ("sensitivity", sample_sensitivity),
        ("specificity", sample_specificity),
        ("prevalence", prevalence),
    ]:
        if not (0 <= val <= 1):
            raise ValueError(f"{label} must be in [0, 1]; got {val}")

    s = sample_sensitivity
    alpha = 1.0 - sample_specificity
    # Windows per SOP = independent chances per seizure. Each SOP holds
    # K windows by construction, independent of the GLOBAL prevalence π
    # (π governs the per-hour negative-window count for FP/hr, not how
    # many windows live inside one SOP), so K_eff = K — no shrink by π.
    K = max(1, int(np.ceil(sop_seconds / cadence_seconds)))
    K_eff = K

    alarm_sens_upper = 1.0 - (1.0 - s) ** K_eff
    alarm_sens_lower = s

    preds_per_hour = 3600.0 / cadence_seconds
    naive_fph = alpha * preds_per_hour * (1.0 - prevalence)
    if refractory_seconds > 0:
        fph_upper = min(naive_fph, 3600.0 / refractory_seconds)
    else:
        fph_upper = naive_fph

    notes = []
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


def alarm_to_sample(
    *,
    alarm_sensitivity: float,
    fp_per_hour: float,
    sop_seconds: float,
    cadence_seconds: float,
    refractory_seconds: float = 0.0,
    prevalence: float = 0.5,
) -> dict:
    """Reverse-bound: feasible sample-metric ranges from alarm metrics.

    Returns dict with sample_sensitivity_lower / upper and
    sample_specificity_lower / upper.
    """
    if cadence_seconds <= 0 or sop_seconds <= 0:
        raise ValueError("cadence_seconds and sop_seconds must be > 0")
    if not (0 <= alarm_sensitivity <= 1):
        raise ValueError("alarm_sensitivity must be in [0, 1]")

    # Mirror sample_to_alarm: K windows per SOP, independent of π.
    K = max(1, int(np.ceil(sop_seconds / cadence_seconds)))
    K_eff = K

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
