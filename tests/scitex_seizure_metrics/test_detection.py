"""Smoke tests — exercise both regimes end-to-end on synthetic data."""
from __future__ import annotations

import numpy as np

from scitex_seizure_metrics import detection, forecasting, AlarmPolicy
from scitex_seizure_metrics.adapters import proba_to_alarms


POL = AlarmPolicy(
    sph_seconds=300, sop_seconds=600, cadence_seconds=60,
    refractory_seconds=600, alarm_threshold=0.5,
)


def test_detection_perfect_classifier_rep_regime_equals_detection():
    # Arrange
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.001, size=y_true.size), 0, 1)
    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1, name="perfect")
    # Act
    # Assert
    assert rep.regime == "detection"


def test_detection_perfect_classifier_rep_roc_auc_0_99():
    # Arrange
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.001, size=y_true.size), 0, 1)
    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1, name="perfect")
    # Act
    # Assert
    assert rep.roc_auc > 0.99


def test_detection_perfect_classifier_rep_pr_auc_0_99():
    # Arrange
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.001, size=y_true.size), 0, 1)
    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1, name="perfect")
    # Act
    # Assert
    assert rep.pr_auc > 0.99


def test_detection_perfect_classifier_rep_brier_0_01():
    # Arrange
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.001, size=y_true.size), 0, 1)
    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1, name="perfect")
    # Act
    # Assert
    assert rep.brier < 0.01


def test_detection_perfect_classifier_rep_sensitivity_0_99():
    # Arrange
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.001, size=y_true.size), 0, 1)
    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1, name="perfect")
    # Act
    # Assert
    assert rep.sensitivity > 0.99


def test_detection_perfect_classifier_rep_precision_0_99():
    # Arrange
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.001, size=y_true.size), 0, 1)
    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1, name="perfect")
    # Act
    # Assert
    assert rep.precision > 0.99




def test_detection_random_classifier_n_0_4_rep_roc_auc_0_6():
    # Arrange
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=1000)
    y_proba = rng.uniform(0, 1, size=1000)
    rep = detection.evaluate(y_true, y_proba, name="random")
    # Act
    # Assert
    assert 0.4 < rep.roc_auc < 0.6


def test_detection_random_classifier_n_0_4_rep_pr_auc_0_6():
    # Arrange
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=1000)
    y_proba = rng.uniform(0, 1, size=1000)
    rep = detection.evaluate(y_true, y_proba, name="random")
    # Act
    # Assert
    assert 0.4 < rep.pr_auc < 0.6




def test_forecasting_alarms_in_window_rep_sensitivity_equals_n_1_0():
    # Arrange
    seizures = np.array([3600.0, 7200.0, 10800.0])
    alarms = seizures - 600.0
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=14400.0, n_surrogate=200, name="aligned",
    )
    # Act
    # Assert
    assert rep.sensitivity == 1.0


def test_forecasting_alarms_in_window_rep_n_tp_equals_n_3():
    # Arrange
    seizures = np.array([3600.0, 7200.0, 10800.0])
    alarms = seizures - 600.0
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=14400.0, n_surrogate=200, name="aligned",
    )
    # Act
    # Assert
    assert rep.n_tp == 3


def test_forecasting_alarms_in_window_rep_ioc_0_0():
    # Arrange
    seizures = np.array([3600.0, 7200.0, 10800.0])
    alarms = seizures - 600.0
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=14400.0, n_surrogate=200, name="aligned",
    )
    # Act
    # Assert
    assert rep.ioc > 0.0




def test_forecasting_alarms_too_late_miss_rep_sensitivity_equals_n_0_0():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3600.0])
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=7200.0, n_surrogate=100,
    )
    # Act
    # Assert
    assert rep.sensitivity == 0.0


def test_forecasting_alarms_too_late_miss_rep_n_tp_equals_n_0():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3600.0])
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=7200.0, n_surrogate=100,
    )
    # Act
    # Assert
    assert rep.n_tp == 0


def test_forecasting_alarms_too_late_miss_rep_n_fp_equals_n_1():
    # Arrange
    seizures = np.array([3600.0])
    alarms = np.array([3600.0])
    rep = forecasting.evaluate(
        alarms, seizures, POL,
        total_recording_time=7200.0, n_surrogate=100,
    )
    # Act
    # Assert
    assert rep.n_fp == 1




def test_proba_to_alarms_refractory():
    # Arrange
    times = np.arange(0, 100, 1.0)
    proba = np.where((times >= 10) & (times < 20), 0.9, 0.0)
    # Act
    fires = proba_to_alarms(proba, times, threshold=0.5, refractory_seconds=5.0)
    # Assert
    assert list(fires) == [10.0, 15.0]


def test_report_to_frame_roundtrip_roc_auc_in_df_columns():
    # Arrange
    rep = detection.evaluate(np.array([0, 1, 0, 1]),
                             np.array([0.1, 0.9, 0.2, 0.8]), name="tiny")
    df = rep.to_frame()
    # Act
    # Assert
    assert "roc_auc" in df.columns


def test_report_to_frame_roundtrip_df_loc_0_name_tiny():
    # Arrange
    rep = detection.evaluate(np.array([0, 1, 0, 1]),
                             np.array([0.1, 0.9, 0.2, 0.8]), name="tiny")
    df = rep.to_frame()
    # Act
    # Assert
    assert df.loc[0, "name"] == "tiny"


