"""Correctness tests for the forecasting-regime classification primitives.

Pins down the alarm-vs-interictal-opportunity confusion convention
(specificity / PPV / NPV / F1) and the observed-lead-time computation so
refactors can't drift, with explicit edge cases: no alarms, all-caught,
no seizures, and ties.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scitex_seizure_metrics._classification import (
    alarm_classification,
    lead_time_summary,
    observed_lead_times,
)

# ---------- alarm_classification: confusion matrix ----------------------


def test_alarm_classification_confusion_tp():
    # Arrange
    # 3 seizures, 2 caught, 1 false alarm; interictal 3600 s, sop 600 s
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.tp == 2


def test_alarm_classification_confusion_fp():
    # Arrange
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.fp == 1


def test_alarm_classification_confusion_fn():
    # Arrange
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.fn == 1


def test_alarm_classification_confusion_tn():
    # Arrange
    # n_opportunities = floor(3600/600) = 6; tn = 6 - 1 = 5
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.tn == 5


def test_alarm_classification_n_opportunities_floor():
    # Arrange
    # 3650 / 600 = 6.08 -> 6 opportunities
    # Act
    clf = alarm_classification(
        n_tp=0,
        n_fp=0,
        n_seizures=1,
        interictal_seconds=3650.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.n_opportunities == 6


def test_alarm_classification_sensitivity():
    # Arrange
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=4,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.sensitivity == pytest.approx(2 / 4)


def test_alarm_classification_specificity():
    # Arrange
    # tn=5, fp=1 -> specificity = 5/6
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.specificity == pytest.approx(5 / 6)


def test_alarm_classification_ppv():
    # Arrange
    # tp=2, fp=1 -> ppv = 2/3
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.ppv == pytest.approx(2 / 3)


def test_alarm_classification_npv():
    # Arrange
    # tn=5, fn=1 -> npv = 5/6
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.npv == pytest.approx(5 / 6)


def test_alarm_classification_f1():
    # Arrange
    # tp=2, fp=1, fn=1 -> f1 = 2*2/(2*2+1+1) = 4/6
    # Act
    clf = alarm_classification(
        n_tp=2,
        n_fp=1,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.f1 == pytest.approx(4 / 6)


# ---------- edge case: no alarms ----------------------------------------


def test_alarm_classification_no_alarms_sensitivity_zero():
    # Arrange
    # 3 seizures, none caught, no false alarms
    # Act
    clf = alarm_classification(
        n_tp=0,
        n_fp=0,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.sensitivity == 0.0


def test_alarm_classification_no_alarms_specificity_perfect():
    # Arrange
    # no FP -> all opportunities are TN -> specificity = 1.0
    # Act
    clf = alarm_classification(
        n_tp=0,
        n_fp=0,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.specificity == 1.0


def test_alarm_classification_no_alarms_ppv_is_nan():
    # Arrange
    # tp+fp = 0 -> ppv undefined -> NaN (fail-loud, not silent 0)
    # Act
    clf = alarm_classification(
        n_tp=0,
        n_fp=0,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert math.isnan(clf.ppv)


# ---------- edge case: all caught ---------------------------------------


def test_alarm_classification_all_caught_sensitivity_one():
    # Arrange
    # Act
    clf = alarm_classification(
        n_tp=3,
        n_fp=0,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.sensitivity == 1.0


def test_alarm_classification_all_caught_f1_one():
    # Arrange
    # Act
    clf = alarm_classification(
        n_tp=3,
        n_fp=0,
        n_seizures=3,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.f1 == 1.0


# ---------- edge case: no seizures --------------------------------------


def test_alarm_classification_no_seizures_sensitivity_nan():
    # Arrange
    # tp+fn = 0 -> recall undefined -> NaN
    # Act
    clf = alarm_classification(
        n_tp=0,
        n_fp=2,
        n_seizures=0,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert math.isnan(clf.sensitivity)


def test_alarm_classification_no_seizures_specificity_defined():
    # Arrange
    # 6 opportunities, 2 FP -> tn=4 -> specificity = 4/6
    # Act
    clf = alarm_classification(
        n_tp=0,
        n_fp=2,
        n_seizures=0,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.specificity == pytest.approx(4 / 6)


# ---------- edge case: more FP than opportunities -----------------------


def test_alarm_classification_tn_clipped_at_zero():
    # Arrange
    # 6 opportunities, 10 FP -> tn clipped to 0
    # Act
    clf = alarm_classification(
        n_tp=1,
        n_fp=10,
        n_seizures=2,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.tn == 0


def test_alarm_classification_tn_clipped_specificity_zero():
    # Arrange
    # tn=0 -> specificity = 0
    # Act
    clf = alarm_classification(
        n_tp=1,
        n_fp=10,
        n_seizures=2,
        interictal_seconds=3600.0,
        sop_seconds=600.0,
    )
    # Assert
    assert clf.specificity == 0.0


# ---------- validation --------------------------------------------------


def test_alarm_classification_tp_exceeds_seizures_raises():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        alarm_classification(
            n_tp=5,
            n_fp=0,
            n_seizures=3,
            interictal_seconds=3600.0,
            sop_seconds=600.0,
        )


def test_alarm_classification_bad_sop_raises():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        alarm_classification(
            n_tp=1,
            n_fp=0,
            n_seizures=1,
            interictal_seconds=3600.0,
            sop_seconds=0.0,
        )


# ---------- observed_lead_times -----------------------------------------


def test_observed_lead_times_single_catch():
    # Arrange
    # seizure at 3600, alarm at 3000, sph=300 sop=600
    # window [2700, 3300]; alarm 3000 covers it; lead = 600
    # Act
    leads = observed_lead_times(
        alarms=np.array([3000.0]),
        seizures=np.array([3600.0]),
        sph=300,
        sop=600,
    )
    # Assert
    assert leads.tolist() == [600.0]


def test_observed_lead_times_earliest_alarm_wins():
    # Arrange
    # two covering alarms -> earliest (3000) gives the longest lead
    # Act
    leads = observed_lead_times(
        alarms=np.array([3200.0, 3000.0]),
        seizures=np.array([3600.0]),
        sph=300,
        sop=600,
    )
    # Assert
    assert leads.tolist() == [600.0]


def test_observed_lead_times_uncaught_excluded():
    # Arrange
    # alarm too late (after window) -> no lead entry
    # Act
    leads = observed_lead_times(
        alarms=np.array([3700.0]),
        seizures=np.array([3600.0]),
        sph=300,
        sop=600,
    )
    # Assert
    assert leads.size == 0


def test_observed_lead_times_respects_sph_lower_bound():
    # Arrange
    # window [2700, 3300]; alarm 3290 covers; lead = 310 >= sph 300
    # Act
    leads = observed_lead_times(
        alarms=np.array([3290.0]),
        seizures=np.array([3600.0]),
        sph=300,
        sop=600,
    )
    # Assert
    assert leads[0] >= 300


def test_observed_lead_times_tie_two_seizures():
    # Arrange
    # one alarm covering both seizures' windows
    # Act
    leads = observed_lead_times(
        alarms=np.array([3000.0]),
        seizures=np.array([3600.0, 3500.0]),
        sph=300,
        sop=600,
    )
    # Assert
    assert sorted(leads.tolist()) == [500.0, 600.0]


# ---------- lead_time_summary -------------------------------------------


def test_lead_time_summary_mean():
    # Arrange
    # Act
    s = lead_time_summary(np.array([600.0, 500.0, 400.0]))
    # Assert
    assert s["lead_time_mean"] == pytest.approx(500.0)


def test_lead_time_summary_median():
    # Arrange
    # Act
    s = lead_time_summary(np.array([600.0, 500.0, 400.0]))
    # Assert
    assert s["lead_time_median"] == pytest.approx(500.0)


def test_lead_time_summary_n_caught():
    # Arrange
    # Act
    s = lead_time_summary(np.array([600.0, 500.0, 400.0]))
    # Assert
    assert s["n_caught"] == 3


def test_lead_time_summary_empty_mean_is_nan():
    # Arrange
    # Act
    s = lead_time_summary(np.array([]))
    # Assert
    assert math.isnan(s["lead_time_mean"])


def test_lead_time_summary_empty_n_caught_zero():
    # Arrange
    # Act
    s = lead_time_summary(np.array([]))
    # Assert
    assert s["n_caught"] == 0
