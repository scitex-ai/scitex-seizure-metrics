"""Smoke tests — exercise both regimes end-to-end on synthetic data."""
from __future__ import annotations

import numpy as np

from scitex_seizure_metrics import detection, forecasting, AlarmPolicy
from scitex_seizure_metrics.adapters import proba_to_alarms


POL = AlarmPolicy(
    sph_seconds=300, sop_seconds=600, cadence_seconds=60,
    refractory_seconds=600, alarm_threshold=0.5,
)


def test_detection_perfect_classifier():
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.001, size=y_true.size), 0, 1)
    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1, name="perfect")
    assert rep.regime == "detection"
    assert rep.roc_auc > 0.99
    assert rep.pr_auc > 0.99
    assert rep.brier < 0.01
    assert rep.sensitivity > 0.99
    assert rep.precision > 0.99


def test_detection_random_classifier():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=1000)
    y_proba = rng.uniform(0, 1, size=1000)
    rep = detection.evaluate(y_true, y_proba, name="random")
    assert 0.4 < rep.roc_auc < 0.6
    assert 0.4 < rep.pr_auc < 0.6


def test_forecasting_alarms_in_window():
    seizures = np.array([3600.0, 7200.0, 10800.0])
    alarms = seizures - 600.0
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=14400.0, n_surrogate=200, name="aligned",
    )
    assert rep.sensitivity == 1.0
    assert rep.n_tp == 3
    assert rep.ioc > 0.0


def test_forecasting_alarms_too_late_miss():
    seizures = np.array([3600.0])
    alarms = np.array([3600.0])
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=7200.0, n_surrogate=100,
    )
    assert rep.sensitivity == 0.0
    assert rep.n_tp == 0
    assert rep.n_fp == 1


def test_proba_to_alarms_refractory():
    times = np.arange(0, 100, 1.0)
    proba = np.where((times >= 10) & (times < 20), 0.9, 0.0)
    fires = proba_to_alarms(proba, times, threshold=0.5, refractory_seconds=5.0)
    assert list(fires) == [10.0, 15.0]


def test_report_to_frame_roundtrip():
    rep = detection.evaluate(np.array([0, 1, 0, 1]),
                             np.array([0.1, 0.9, 0.2, 0.8]), name="tiny")
    df = rep.to_frame()
    assert "roc_auc" in df.columns
    assert df.loc[0, "name"] == "tiny"
