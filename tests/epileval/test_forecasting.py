"""Tests for AlarmPolicy validation and end-to-end stream evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from epileval import AlarmPolicy, forecasting


def make_policy(**overrides):
    base = dict(sph_seconds=300, sop_seconds=600, cadence_seconds=60,
                refractory_seconds=600, alarm_threshold=0.5)
    base.update(overrides)
    return AlarmPolicy(**base)


def test_policy_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        AlarmPolicy(sph_seconds=-1, sop_seconds=10, cadence_seconds=1,
                    refractory_seconds=0)
    with pytest.raises(ValueError):
        AlarmPolicy(sph_seconds=0, sop_seconds=0, cadence_seconds=1,
                    refractory_seconds=0)
    with pytest.raises(ValueError):
        AlarmPolicy(sph_seconds=0, sop_seconds=10, cadence_seconds=0,
                    refractory_seconds=0)
    with pytest.raises(ValueError):
        AlarmPolicy(sph_seconds=0, sop_seconds=10, cadence_seconds=1,
                    refractory_seconds=0, alarm_threshold=1.5)


def test_policy_describe_roundtrip():
    p = make_policy()
    d = p.describe()
    assert d["sph_s"] == 300
    assert d["sop_s"] == 600
    assert d["cadence_s"] == 60
    assert d["fp_denominator"] == "interictal"


def test_evaluate_stream_perfect_predictor():
    """Perfect proba in window placing alarm so [alarm+SPH, alarm+SPH+SOP]
    covers each seizure onset.

    With SPH=300, SOP=600: alarm must fire in [sz - SPH - SOP, sz - SPH] =
    [sz - 900, sz - 300]. Then merged-alarm at sz - 900 → covers [sz-600, sz].
    """
    seizures = np.array([3600.0, 7200.0, 10800.0])
    times = np.arange(0, 14400, 60.0)
    proba = np.zeros_like(times)
    for sz in seizures:
        mask = (times >= sz - 900) & (times < sz - 300)
        proba[mask] = 0.9
    pol = make_policy()
    rep = forecasting.evaluate_stream(
        proba, times, seizures, pol,
        total_recording_time=14400.0, n_surrogate=200,
    )
    assert rep.sensitivity == 1.0
    assert rep.n_tp == 3
    assert rep.fp_per_hour == 0.0


def test_evaluate_stream_random_baseline_low_ioc():
    rng = np.random.default_rng(0)
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    proba = rng.uniform(0, 1, size=times.size)
    pol = make_policy()
    rep = forecasting.evaluate_stream(
        proba, times, seizures, pol,
        total_recording_time=10800.0, n_surrogate=200, rng_seed=0,
    )
    # Random predictor's IoC should hover near 0 ± small.
    assert -0.5 < rep.ioc < 0.5


def test_sweep_thresholds_monotone_sensitivity():
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    # Strong signal: high proba near each seizure
    proba = np.zeros_like(times)
    for sz in seizures:
        proba[(times >= sz - 1200) & (times < sz - 300)] = 0.9
    pol = make_policy()
    df = forecasting.sweep_thresholds(
        proba, times, seizures, pol,
        thresholds=np.linspace(0.1, 0.95, 9),
        total_recording_time=10800.0, n_surrogate=50,
    )
    # As threshold rises, sensitivity should be non-increasing.
    sens = df.sort_values("threshold")["sensitivity"].values
    assert all(sens[i] >= sens[i+1] - 1e-9 for i in range(len(sens)-1))


def test_sweep_policies_cadence_ablation():
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 60.0)
    proba = np.where((times >= 2400) & (times < 3300), 0.9, 0.0) + \
            np.where((times >= 6000) & (times < 6900), 0.9, 0.0)
    policies = [make_policy(cadence_seconds=c) for c in [30, 60, 120, 300]]
    df = forecasting.sweep_policies(
        proba, times, seizures, policies,
        total_recording_time=10800.0, n_surrogate=50,
    )
    assert len(df) == 4
    assert "cadence_s" in df.columns


def test_fp_denominator_interictal_vs_total():
    seizures = np.array([3600.0])
    times = np.arange(0, 7200, 60.0)
    # One alarm at t=10 (way before seizure): outside SPH/SOP → FP
    proba = np.zeros_like(times)
    proba[0] = 0.9
    pol_inter = make_policy(fp_denominator="interictal")
    pol_total = make_policy(fp_denominator="total")
    r_inter = forecasting.evaluate_stream(
        proba, times, seizures, pol_inter,
        total_recording_time=7200.0, n_surrogate=10,
    )
    r_total = forecasting.evaluate_stream(
        proba, times, seizures, pol_total,
        total_recording_time=7200.0, n_surrogate=10,
    )
    # Interictal denominator is smaller → FP/hr larger
    assert r_inter.fp_per_hour > r_total.fp_per_hour


def test_bootstrap_ci_basic():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, size=200)
    mean, lo, hi = forecasting.bootstrap_ci(vals, n_boot=500, ci=0.95)
    assert lo < mean < hi
    assert hi - lo < 0.05  # tight CI for n=200
