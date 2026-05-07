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
    proba = np.array([0.1, 0.2, 0.3])
    times = np.array([0., 60., 120.])
    assert proba_stream_to_alarms(proba, times, 0.5, 0).size == 0


def test_stream_merge_consecutive():
    proba = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.9])
    times = np.arange(6.0) * 60.0
    fires = proba_stream_to_alarms(proba, times, 0.5,
                                   refractory_seconds=0,
                                   merge_consecutive=True)
    # First contiguous run starts at 60s; next isolated at 300s
    assert list(fires) == [60.0, 300.0]


def test_stream_no_merge():
    proba = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.9])
    times = np.arange(6.0) * 60.0
    fires = proba_stream_to_alarms(proba, times, 0.5, 0,
                                   merge_consecutive=False)
    assert list(fires) == [60.0, 120.0, 180.0, 300.0]


def test_stream_refractory_caps_alarms():
    proba = np.ones(10) * 0.9
    times = np.arange(10.0) * 60.0   # alarms every minute
    fires = proba_stream_to_alarms(proba, times, 0.5,
                                   refractory_seconds=180,
                                   merge_consecutive=False)
    # Refractory 180 s → first kept then every 3 ticks
    assert list(fires) == [0.0, 180.0, 360.0, 540.0]


# ---------- alarm_match --------------------------------------------------

def test_alarm_match_in_window():
    seizures = np.array([3600.0])
    alarms = np.array([3000.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # alarm at 3000 covers [3300, 3900] → seizure 3600 inside → caught
    assert sc.tolist() == [True]
    assert au.tolist() == [True]


def test_alarm_match_too_early_miss():
    seizures = np.array([3600.0])
    alarms = np.array([1000.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    # alarm covers [1300, 1900] → 3600 outside → miss
    assert sc.tolist() == [False]
    assert au.tolist() == [False]


def test_alarm_match_too_late_miss():
    seizures = np.array([3600.0])
    alarms = np.array([3700.0])
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    assert sc.tolist() == [False]
    assert au.tolist() == [False]


def test_alarm_match_two_alarms_one_seizure():
    seizures = np.array([3600.0])
    alarms = np.array([3000.0, 3200.0])  # both cover the seizure
    sc, au = alarm_match(alarms, seizures, sph=300, sop=600)
    assert sc.tolist() == [True]
    # Both are "useful" because both have the seizure in their window.
    assert au.tolist() == [True, True]


# ---------- interictal_seconds + union_length ---------------------------

def test_interictal_seconds_no_seizures():
    assert interictal_seconds(3600, np.array([]), sop=600) == 3600.0


def test_interictal_seconds_one_seizure():
    # 3600 s recording, 1 seizure at t=1800, sop=600 sph=0
    # exclusion window = [1200, 2400] → 1200 s removed → 2400 interictal
    out = interictal_seconds(3600, np.array([1800.0]), sop=600, sph=0)
    assert pytest.approx(out, abs=1e-6) == 2400.0


def test_interictal_seconds_overlapping_exclusions():
    out = interictal_seconds(
        3600, np.array([1800.0, 2000.0]), sop=600, sph=0
    )
    # [1200, 2400] ∪ [1400, 2600] = [1200, 2600] → 1400 removed → 2200
    assert pytest.approx(out, abs=1e-6) == 2200.0


def test_union_length_disjoint_and_overlap():
    iv = np.array([[0, 10], [20, 30], [25, 40]])
    assert union_length(iv) == 30.0    # [0,10] + [20,40]
