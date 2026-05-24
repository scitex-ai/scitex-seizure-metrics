"""Internal-correctness tests for the alarm-matching engine.

These tests pin down the SPH/SOP semantics, refractory rule, merging
rule, and interictal-denominator computation so refactors can't drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics._alarm import (
    alarm_match, interictal_seconds, proba_stream_to_alarms, union_length,
)


# ---------- proba_stream_to_alarms ---------------------------------------

def test_stream_no_threshold_crossings():
    # Arrange
    proba = np.array([0.1, 0.2, 0.3])
    # Act
    times = np.array([0., 60., 120.])
    # Assert
    assert proba_stream_to_alarms(proba, times, 0.5, 0).size == 0


def test_stream_merge_consecutive():
    # Arrange
    proba = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.9])
    times = np.arange(6.0) * 60.0
    # Act
    fires = proba_stream_to_alarms(proba, times, 0.5,
                                   refractory_seconds=0,
                                   merge_consecutive=True)
    # First contiguous run starts at 60s; next isolated at 300s
    # Assert
    assert list(fires) == [60.0, 300.0]


def test_stream_no_merge():
    # Arrange
    proba = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.9])
    times = np.arange(6.0) * 60.0
    # Act
    fires = proba_stream_to_alarms(proba, times, 0.5, 0,
                                   merge_consecutive=False)
    # Assert
    assert list(fires) == [60.0, 120.0, 180.0, 300.0]


def test_stream_refractory_caps_alarms():
    # Arrange
    proba = np.ones(10) * 0.9
    times = np.arange(10.0) * 60.0   # alarms every minute
    # Act
    fires = proba_stream_to_alarms(proba, times, 0.5,
                                   refractory_seconds=180,
                                   merge_consecutive=False)
    # Refractory 180 s → first kept then every 3 ticks
    # Assert
    assert list(fires) == [0.0, 180.0, 360.0, 540.0]


# ---------- alarm_match --------------------------------------------------

def test_alarm_match_in_window_sc_tolist_true():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3000.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert sc.tolist() == [True]


def test_alarm_match_in_window_au_tolist_true():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3000.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert au.tolist() == [True]




def test_alarm_match_too_early_miss_sc_tolist_false():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([1000.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert sc.tolist() == [False]


def test_alarm_match_too_early_miss_au_tolist_false():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([1000.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert au.tolist() == [False]




def test_alarm_match_too_late_miss_sc_tolist_false():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3700.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert sc.tolist() == [False]


def test_alarm_match_too_late_miss_au_tolist_false():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3700.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert au.tolist() == [False]




def test_alarm_match_two_alarms_one_seizure_sc_tolist_true():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3000.0, 3200.0])  # both cover the seizure
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert sc.tolist() == [True]


def test_alarm_match_two_alarms_one_seizure_au_tolist_true_true():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3000.0, 3200.0])  # both cover the seizure
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # Act
    # Assert
    assert au.tolist() == [True, True]




# ---------- interictal_seconds + union_length ---------------------------

def test_interictal_seconds_no_seizures():
    # Arrange
    # Act
    # Assert
    assert interictal_seconds(3600, np.array([]), sop=600) == 3600.0


def test_interictal_seconds_one_seizure():
    # 3600 s recording, 1 seizure at t=1800, sop=600 sph=0
    # exclusion window = [1200, 2400] → 1200 s removed → 2400 interictal
    # Arrange
    # Act
    out = interictal_seconds(3600, np.array([1800.0]), sop=600, sph=0)
    # Assert
    assert pytest.approx(out, abs=1e-6) == 2400.0


def test_interictal_seconds_overlapping_exclusions():
    # Arrange
    # Act
    out = interictal_seconds(
        3600, np.array([1800.0, 2000.0]), sop=600, sph=0
    )
    # [1200, 2400] ∪ [1400, 2600] = [1200, 2600] → 1400 removed → 2200
    # Assert
    assert pytest.approx(out, abs=1e-6) == 2200.0


def test_union_length_disjoint_and_overlap():
    # Arrange
    # Act
    iv = np.array([[0, 10], [20, 30], [25, 40]])
    # Assert
    assert union_length(iv) == 30.0    # [0,10] + [20,40]
