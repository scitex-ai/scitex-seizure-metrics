"""Tests for AlarmPolicy validation and end-to-end stream evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics import AlarmPolicy, forecasting


def make_policy(**overrides):
    base = dict(
        sph_seconds=300,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        alarm_threshold=0.5,
    )
    base.update(overrides)
    return AlarmPolicy(**base)


def test_policy_validation_rejects_bad_values_raises_valueerror():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        AlarmPolicy(
            sph_seconds=-1, sop_seconds=10, cadence_seconds=1, refractory_seconds=0
        )


def test_policy_validation_rejects_bad_values_raises_valueerror():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        AlarmPolicy(
            sph_seconds=0, sop_seconds=0, cadence_seconds=1, refractory_seconds=0
        )


def test_policy_validation_rejects_bad_values_raises_valueerror():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        AlarmPolicy(
            sph_seconds=0, sop_seconds=10, cadence_seconds=0, refractory_seconds=0
        )


def test_policy_validation_rejects_bad_values_raises_valueerror():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        AlarmPolicy(
            sph_seconds=0,
            sop_seconds=10,
            cadence_seconds=1,
            refractory_seconds=0,
            alarm_threshold=1.5,
        )


def test_policy_describe_roundtrip_d_sph_s_300():
    # Arrange
    p = make_policy()
    d = p.describe()
    # Act
    # Assert
    assert d["sph_s"] == 300


def test_policy_describe_roundtrip_d_sop_s_600():
    # Arrange
    p = make_policy()
    d = p.describe()
    # Act
    # Assert
    assert d["sop_s"] == 600


def test_policy_describe_roundtrip_d_cadence_s_60():
    # Arrange
    p = make_policy()
    d = p.describe()
    # Act
    # Assert
    assert d["cadence_s"] == 60


def test_policy_describe_roundtrip_d_fp_denominator_interictal():
    # Arrange
    p = make_policy()
    d = p.describe()
    # Act
    # Assert
    assert d["fp_denominator"] == "interictal"


def test_evaluate_stream_perfect_predictor_rep_sensitivity_equals_n_1_0():
    # Arrange
    seizures = np.array([3600.0, 7200.0, 10800.0])
    times = np.arange(0, 14400, 60.0)
    proba = np.zeros_like(times)
    for sz in seizures:
        mask = (times >= sz - 900) & (times < sz - 300)
        proba[mask] = 0.9
    pol = make_policy()
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=200,
    )
    # Act
    # Assert
    assert rep.sensitivity == 1.0


def test_evaluate_stream_perfect_predictor_rep_n_tp_equals_n_3():
    # Arrange
    seizures = np.array([3600.0, 7200.0, 10800.0])
    times = np.arange(0, 14400, 60.0)
    proba = np.zeros_like(times)
    for sz in seizures:
        mask = (times >= sz - 900) & (times < sz - 300)
        proba[mask] = 0.9
    pol = make_policy()
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=200,
    )
    # Act
    # Assert
    assert rep.n_tp == 3


def test_evaluate_stream_perfect_predictor_rep_fp_per_hour_equals_n_0_0():
    # Arrange
    seizures = np.array([3600.0, 7200.0, 10800.0])
    times = np.arange(0, 14400, 60.0)
    proba = np.zeros_like(times)
    for sz in seizures:
        mask = (times >= sz - 900) & (times < sz - 300)
        proba[mask] = 0.9
    pol = make_policy()
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=200,
    )
    # Act
    # Assert
    assert rep.fp_per_hour == 0.0


def test_evaluate_stream_random_baseline_low_ioc():
    # Arrange
    rng = np.random.default_rng(0)
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    proba = rng.uniform(0, 1, size=times.size)
    pol = make_policy()
    # Act
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=10800.0,
        n_surrogate=200,
        rng_seed=0,
    )
    # Random predictor's IoC should hover near 0 ± small.
    # Assert
    assert -0.5 < rep.ioc < 0.5


def test_sweep_thresholds_monotone_sensitivity():
    # Arrange
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    # Strong signal: high proba near each seizure
    proba = np.zeros_like(times)
    for sz in seizures:
        proba[(times >= sz - 1200) & (times < sz - 300)] = 0.9
    pol = make_policy()
    df = forecasting.sweep_thresholds(
        proba,
        times,
        seizures,
        pol,
        thresholds=np.linspace(0.1, 0.95, 9),
        total_recording_time=10800.0,
        n_surrogate=50,
    )
    # As threshold rises, sensitivity should be non-increasing.
    # Act
    sens = df.sort_values("threshold")["sensitivity"].values
    # Assert
    assert all(sens[i] >= sens[i + 1] - 1e-9 for i in range(len(sens) - 1))


def test_sweep_policies_cadence_ablation_len_df_is_4():
    # Arrange
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    proba = np.where((times >= 2400) & (times < 3300), 0.9, 0.0) + np.where(
        (times >= 6000) & (times < 6900), 0.9, 0.0
    )
    policies = [make_policy(cadence_seconds=c) for c in [30, 60, 120, 300]]
    df = forecasting.sweep_policies(
        proba,
        times,
        seizures,
        policies,
        total_recording_time=10800.0,
        n_surrogate=50,
    )
    # Act
    # Assert
    assert len(df) == 4


def test_sweep_policies_cadence_ablation_cadence_s_in_df_columns():
    # Arrange
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    proba = np.where((times >= 2400) & (times < 3300), 0.9, 0.0) + np.where(
        (times >= 6000) & (times < 6900), 0.9, 0.0
    )
    policies = [make_policy(cadence_seconds=c) for c in [30, 60, 120, 300]]
    df = forecasting.sweep_policies(
        proba,
        times,
        seizures,
        policies,
        total_recording_time=10800.0,
        n_surrogate=50,
    )
    # Act
    # Assert
    assert "cadence_s" in df.columns


def test_fp_denominator_interictal_vs_total():
    # Arrange
    seizures = np.array([3600.0])
    times = np.arange(0, 7200, 60.0)
    # One alarm at t=10 (way before seizure): outside SPH/SOP → FP
    proba = np.zeros_like(times)
    proba[0] = 0.9
    pol_inter = make_policy(fp_denominator="interictal")
    pol_total = make_policy(fp_denominator="total")
    r_inter = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol_inter,
        total_recording_time=7200.0,
        n_surrogate=10,
    )
    # Act
    r_total = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol_total,
        total_recording_time=7200.0,
        n_surrogate=10,
    )
    # Interictal denominator is smaller → FP/hr larger
    # Assert
    assert r_inter.fp_per_hour > r_total.fp_per_hour


def test_bootstrap_ci_basic_lo_mean_hi():
    # Arrange
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, size=200)
    mean, lo, hi = forecasting.bootstrap_ci(vals, n_boot=500, ci=0.95)
    # Act
    # Assert
    assert lo < mean < hi


def test_bootstrap_ci_basic_hi_lo_0_05():
    # Arrange
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, size=200)
    mean, lo, hi = forecasting.bootstrap_ci(vals, n_boot=500, ci=0.95)
    # Act
    # Assert
    assert hi - lo < 0.05  # tight CI for n=200


# ---------- forecasting-regime classification metrics -------------------


def _perfect_stream():
    seizures = np.array([3600.0, 7200.0, 10800.0])
    times = np.arange(0, 14400, 60.0)
    proba = np.zeros_like(times)
    for sz in seizures:
        mask = (times >= sz - 900) & (times < sz - 300)
        proba[mask] = 0.9
    return proba, times, seizures


def test_evaluate_stream_perfect_predictor_specificity_is_one():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    # Act
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Assert
    assert rep.specificity == 1.0


def test_evaluate_stream_perfect_predictor_ppv_is_one():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    # Act
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Assert
    assert rep.ppv == 1.0


def test_evaluate_stream_perfect_predictor_forecasting_f1_is_one():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    # Act
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Assert
    assert rep.forecasting_f1 == 1.0


def test_evaluate_stream_perfect_predictor_npv_is_one():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    # Act
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Assert
    assert rep.npv == 1.0


def test_evaluate_stream_perfect_predictor_lead_time_within_sop():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    # Act
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Lead time must be >= sph (300) and <= sph + sop (900).
    # Assert
    assert 300 <= rep.lead_time_mean <= 900


def test_evaluate_stream_perfect_predictor_lead_times_array_in_extras():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    # Act
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Assert
    assert len(rep.extras["lead_times_seconds"]) == 3


def test_evaluate_no_alarms_ppv_is_nan():
    # Arrange
    seizures = np.array([3600.0])
    pol = make_policy()
    # Act
    rep = forecasting.evaluate(
        np.array([]),
        seizures,
        pol,
        total_recording_time=7200.0,
        n_surrogate=10,
    )
    # Assert
    assert np.isnan(rep.ppv)


def test_evaluate_no_alarms_lead_time_mean_is_nan():
    # Arrange
    seizures = np.array([3600.0])
    pol = make_policy()
    # Act
    rep = forecasting.evaluate(
        np.array([]),
        seizures,
        pol,
        total_recording_time=7200.0,
        n_surrogate=10,
    )
    # Assert
    assert np.isnan(rep.lead_time_mean)


def test_evaluate_no_alarms_specificity_is_one():
    # Arrange
    # No alarms -> no FP -> all interictal opportunities are TN.
    seizures = np.array([3600.0])
    pol = make_policy()
    # Act
    rep = forecasting.evaluate(
        np.array([]),
        seizures,
        pol,
        total_recording_time=7200.0,
        n_surrogate=10,
    )
    # Assert
    assert rep.specificity == 1.0


def test_to_frame_includes_specificity_column():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Act
    df = rep.to_frame()
    # Assert
    assert "specificity" in df.columns


def test_to_dict_includes_lead_times_in_extras():
    # Arrange
    proba, times, seizures = _perfect_stream()
    pol = make_policy()
    rep = forecasting.evaluate_stream(
        proba,
        times,
        seizures,
        pol,
        total_recording_time=14400.0,
        n_surrogate=50,
    )
    # Act
    d = rep.to_dict()
    # Assert
    assert "lead_times_seconds" in d["extras"]


def test_sweep_thresholds_includes_specificity_column():
    # Arrange
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    proba = np.zeros_like(times)
    for sz in seizures:
        proba[(times >= sz - 1200) & (times < sz - 300)] = 0.9
    pol = make_policy()
    # Act
    df = forecasting.sweep_thresholds(
        proba,
        times,
        seizures,
        pol,
        thresholds=np.linspace(0.1, 0.95, 5),
        total_recording_time=10800.0,
        n_surrogate=20,
    )
    # Assert
    assert "specificity" in df.columns
