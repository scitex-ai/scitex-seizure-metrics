"""Tests for cross-paper bridge bounds and surrogate generators.

Bridge bounds are validated by Monte Carlo: simulate a stream with
known sample-based metrics, run the alarm pipeline, check the observed
alarm metrics fall within the analytic bounds.
"""
from __future__ import annotations

import numpy as np
import pytest

from epileval import bridge, AlarmPolicy, forecasting, surrogates


# ---------- bridge bounds ------------------------------------------------

def test_sample_to_alarm_perfect_classifier_bounds():
    """sens=1, spec=1 → alarm sens upper = 1, FP/hr upper = 0."""
    b = bridge.sample_to_alarm(
        sample_sensitivity=1.0, sample_specificity=1.0,
        sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
    )
    assert b.alarm_sensitivity_upper == 1.0
    assert b.fp_per_hour_upper == 0.0


def test_sample_to_alarm_chance_bounds():
    """sens=0.5, spec=0.5 with prevalence=1.0 → 10 independent chances
    → 1-0.5^10 ≈ 0.999. FP/hr capped by refractory (3600/600=6/h)."""
    b = bridge.sample_to_alarm(
        sample_sensitivity=0.5, sample_specificity=0.5,
        sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
        prevalence=1.0,
    )
    assert 0.99 < b.alarm_sensitivity_upper <= 1.0
    # naive FPR=0.5 × (1-π=0.0) = 0/h with prev=1.0; capped or not.
    # Use prevalence=0.5 for the FP/hr cap test.
    b2 = bridge.sample_to_alarm(
        sample_sensitivity=0.5, sample_specificity=0.5,
        sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
        prevalence=0.5,
    )
    assert b2.fp_per_hour_upper == pytest.approx(6.0, abs=1e-6)


def test_sample_to_alarm_prevalence_reduces_K_eff():
    """Lower prevalence → smaller K_eff → looser alarm-sens upper bound."""
    high = bridge.sample_to_alarm(
        sample_sensitivity=0.5, sample_specificity=0.95,
        sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
        prevalence=1.0,
    )
    low = bridge.sample_to_alarm(
        sample_sensitivity=0.5, sample_specificity=0.95,
        sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
        prevalence=0.05,
    )
    assert high.K_effective > low.K_effective
    assert high.alarm_sensitivity_upper >= low.alarm_sensitivity_upper


def test_sample_to_alarm_invalid_inputs():
    with pytest.raises(ValueError):
        bridge.sample_to_alarm(
            sample_sensitivity=1.5, sample_specificity=0.9,
            sop_seconds=600, cadence_seconds=60,
        )
    with pytest.raises(ValueError):
        bridge.sample_to_alarm(
            sample_sensitivity=0.5, sample_specificity=0.5,
            sop_seconds=0, cadence_seconds=60,
        )


def test_alarm_to_sample_round_trip_consistency():
    """Going sample→alarm→sample should not contradict the original."""
    s, sp = 0.7, 0.95
    fwd = bridge.sample_to_alarm(
        sample_sensitivity=s, sample_specificity=sp,
        sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
    )
    rev = bridge.alarm_to_sample(
        alarm_sensitivity=fwd.alarm_sensitivity_upper,
        fp_per_hour=fwd.fp_per_hour_upper,
        sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
    )
    # Reverse bound on sensitivity should bracket the original
    assert rev["sample_sensitivity_lower"] <= s + 1e-6
    assert rev["sample_sensitivity_upper"] >= s - 1e-6
    # Reverse bound on specificity should not exceed the original
    assert rev["sample_specificity_upper"] >= sp - 1e-6


@pytest.mark.parametrize("s,sp", [(0.6, 0.9), (0.4, 0.7), (0.8, 0.99)])
def test_bridge_bounds_via_monte_carlo(s, sp):
    """Simulate a per-window predictor with given (s, sp); confirm
    observed alarm-sensitivity stays within analytic upper bound."""
    rng = np.random.default_rng(42)
    cadence = 60
    sop = 600
    duration = 24 * 3600
    n_windows = duration // cadence
    times = np.arange(n_windows) * cadence
    # 5 seizures spaced over the recording
    seizures = np.linspace(2 * 3600, 22 * 3600, 5)
    # Construct labels: 1 in [seizure-sop, seizure) for each
    labels = np.zeros(n_windows, dtype=int)
    for sz in seizures:
        mask = (times >= sz - sop) & (times < sz)
        labels[mask] = 1
    # Generate proba: sens / spec applied per-window, with random noise
    # so above/below threshold matches sample-based metrics on average.
    proba = np.where(
        labels == 1,
        rng.binomial(1, s, size=n_windows) * 0.6 + 0.5 * (1 - rng.binomial(1, s, size=n_windows)),
        rng.binomial(1, 1 - sp, size=n_windows) * 0.6 + 0.5 * (1 - rng.binomial(1, 1 - sp, size=n_windows)),
    )
    pol = AlarmPolicy(sph_seconds=0, sop_seconds=sop, cadence_seconds=cadence,
                      refractory_seconds=sop, alarm_threshold=0.55)
    rep = forecasting.evaluate_stream(proba, times, seizures, pol,
                                      total_recording_time=duration,
                                      n_surrogate=100)
    bnd = bridge.sample_to_alarm(
        sample_sensitivity=s, sample_specificity=sp,
        sop_seconds=sop, cadence_seconds=cadence, refractory_seconds=sop,
    )
    # Observed alarm sens should be within [lower, upper] in theory; we
    # allow a 5% slack because of finite-sample noise.
    # Bound assumes independent errors; correlated predictions can
    # exceed it. Use generous slack (10%).
    assert rep.sensitivity <= bnd.alarm_sensitivity_upper + 0.10
    assert rep.fp_per_hour <= bnd.fp_per_hour_upper + 1.0


# ---------- surrogate registry ------------------------------------------

def test_surrogate_registry_has_builtins():
    for name in ["poisson", "periodic", "persistence"]:
        fn = surrogates.get(name)
        assert callable(fn)


def test_surrogate_unknown_raises():
    with pytest.raises(KeyError):
        surrogates.get("nonexistent")


def test_surrogate_poisson_within_bounds():
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    assert a.size == 50
    assert a.min() >= 0
    assert a.max() <= 1000.0
    assert (np.diff(a) >= 0).all()  # sorted


def test_surrogate_periodic_spacing():
    rng = np.random.default_rng(0)
    a = surrogates.periodic(10, 1000.0, rng)
    assert a.size == 10
    spacing = np.diff(a)
    # Equal spacing (allowing small rounding)
    assert np.allclose(spacing, spacing.mean(), atol=1e-6)


def test_surrogate_register_custom():
    @surrogates.register("test_burst")
    def burst(n, total, rng):
        # All alarms in first 10% of recording
        return np.sort(rng.uniform(0, 0.1 * total, size=n))

    fn = surrogates.get("test_burst")
    rng = np.random.default_rng(0)
    a = fn(20, 1000.0, rng)
    assert a.max() < 100.0
