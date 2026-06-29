"""Forecasting-regime classification primitives.

The forecasting (alarm/seizure-prediction) regime already counts
true positives (caught seizures) and false positives (alarms that
catch nothing) in ``forecasting.evaluate`` via ``_alarm.alarm_match``.
This module turns those raw counts into the standard binary-classifier
scores (specificity, PPV, NPV, F1) and the observed lead/warning time,
under one explicit, documented convention.

Confusion-matrix convention (alarm / prediction-opportunity basis)
------------------------------------------------------------------
Counts are defined on a *mixed* basis that mirrors how the seizure-
prediction literature (Snyder 2008; Schelter/Winterhalder 2006;
Mormann 2007) scores an alarm system:

- ``TP`` — number of seizures *caught*: a seizure at ``t_s`` is a TP iff
  some alarm fires at ``t_a`` with ``t_a + sph <= t_s <= t_a + sph + sop``
  (this is ``_alarm.alarm_match`` ``seizure_caught.sum()``).
- ``FN`` — seizures *not* caught = ``n_seizures - TP``.
- ``FP`` — alarms that catch no seizure = ``alarm_useful`` False count.
- ``TN`` — interictal "prediction opportunities" in which the system
  correctly stayed silent. The interictal time (the FP/hr denominator,
  with each seizure's ``[t_s - sop - sph, t_s + sop]`` window removed)
  is partitioned into non-overlapping SOP-length opportunities:
  ``n_opportunities = floor(interictal_seconds / sop)``. Each opportunity
  that did *not* contain a false alarm is a TN, so
  ``TN = max(0, n_opportunities - FP)``.

From these:

- ``sensitivity`` (recall) = TP / (TP + FN)                  [per-seizure]
- ``specificity``          = TN / (TN + FP)                  [per-opportunity]
- ``ppv`` (alarm precision) = TP / (TP + FP)
- ``npv``                  = TN / (TN + FN)
- ``f1``                   = 2·TP / (2·TP + FP + FN)

Caveats a reviewer should confirm
---------------------------------
"True negative" is genuinely convention-dependent for an alarm system —
there is no canonical unit of "a correctly-quiet interictal moment".
The SOP-length-opportunity convention is the most defensible packaged
choice (it matches the time unit the alarm validity window uses and the
``interictal`` FP denominator), but specificity/NPV scale with the
chosen opportunity length. They are therefore reported alongside the raw
``n_opportunities`` (in ``extras``) so the denominator is always visible.
``ppv`` and ``f1`` do not depend on the TN convention.

Observed lead time
-------------------
Distinct from the SPH *constraint*: for each caught seizure, the
observed lead time is ``t_s - t_a_earliest``, where ``t_a_earliest`` is
the earliest alarm whose validity window covers that seizure. By
construction it is ``>= sph``. The per-seizure array plus mean/median
summaries describe how much warning the system actually delivered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlarmClassification:
    """Forecasting-regime confusion matrix and derived classifier scores.

    All fields use the convention documented at module level. ``tn`` and
    the metrics that depend on it (``specificity``, ``npv``) are computed
    from interictal SOP-length opportunities; ``n_opportunities`` exposes
    that denominator.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    n_opportunities: int
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float


def _safe_ratio(num: float, den: float) -> float:
    """num/den, or NaN when the denominator is 0 (fail-loud, no silent 0)."""
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def alarm_classification(
    n_tp: int,
    n_fp: int,
    n_seizures: int,
    interictal_seconds: float,
    sop_seconds: float,
) -> AlarmClassification:
    """Build the alarm-regime confusion matrix and classifier scores.

    Args:
        n_tp: number of caught seizures (``alarm_match`` seizure_caught sum).
        n_fp: number of alarms that caught no seizure.
        n_seizures: total reference seizures.
        interictal_seconds: interictal duration (FP/hr denominator).
        sop_seconds: Seizure Occurrence Period — the opportunity length.

    Returns:
        AlarmClassification with TP/FP/FN/TN, n_opportunities, and the
        sensitivity/specificity/PPV/NPV/F1 scores.
    """
    if n_tp < 0 or n_fp < 0 or n_seizures < 0:
        raise ValueError("counts must be >= 0")
    if n_tp > n_seizures:
        raise ValueError(f"n_tp ({n_tp}) cannot exceed n_seizures ({n_seizures})")
    if sop_seconds <= 0:
        raise ValueError("sop_seconds must be > 0")
    if interictal_seconds < 0:
        raise ValueError("interictal_seconds must be >= 0")

    tp = int(n_tp)
    fp = int(n_fp)
    fn = int(n_seizures) - tp

    n_opportunities = int(math.floor(interictal_seconds / sop_seconds))
    tn = max(0, n_opportunities - fp)

    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    ppv = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)

    return AlarmClassification(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_opportunities=n_opportunities,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        f1=f1,
    )


def observed_lead_times(
    alarms: np.ndarray,
    seizures: np.ndarray,
    sph: float,
    sop: float,
) -> np.ndarray:
    """Observed lead time per *caught* seizure.

    For each seizure that is caught, the observed lead time is
    ``t_s - t_a_earliest``, where ``t_a_earliest`` is the earliest alarm
    whose validity window ``[t_a + sph, t_a + sph + sop]`` covers the
    seizure. Uncaught seizures contribute no entry.

    By construction every returned value is ``>= sph`` and ``<= sph + sop``.

    Args:
        alarms: alarm onset times (seconds).
        seizures: seizure onset times (seconds).
        sph: Seizure Prediction Horizon (seconds).
        sop: Seizure Occurrence Period (seconds).

    Returns:
        1-D float array of observed lead times, one per caught seizure
        (empty array if no seizure is caught).
    """
    alarms = np.sort(np.asarray(alarms, dtype=float).ravel())
    seizures = np.asarray(seizures, dtype=float).ravel()
    leads: list[float] = []
    for t_s in seizures:
        lo = t_s - sph - sop
        hi = t_s - sph
        covering = alarms[(alarms >= lo) & (alarms <= hi)]
        if covering.size > 0:
            earliest = float(covering.min())
            leads.append(float(t_s - earliest))
    return np.asarray(leads, dtype=float)


def lead_time_summary(lead_times: np.ndarray) -> dict:
    """Mean/median/min/max summary of an observed-lead-time array.

    Returns NaN summaries (not silent zeros) when no seizure is caught,
    so an empty result is never mistaken for "0 s lead".
    """
    lead_times = np.asarray(lead_times, dtype=float).ravel()
    if lead_times.size == 0:
        return {
            "lead_time_mean": float("nan"),
            "lead_time_median": float("nan"),
            "lead_time_min": float("nan"),
            "lead_time_max": float("nan"),
            "n_caught": 0,
        }
    return {
        "lead_time_mean": float(np.mean(lead_times)),
        "lead_time_median": float(np.median(lead_times)),
        "lead_time_min": float(np.min(lead_times)),
        "lead_time_max": float(np.max(lead_times)),
        "n_caught": int(lead_times.size),
    }
