"""Tests for cross-paper bridge bounds and surrogate generators.

Bridge bounds are validated by Monte Carlo: simulate a stream with
known sample-based metrics, run the alarm pipeline, check the observed
alarm metrics fall within the analytic bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics import AlarmPolicy, bridge, forecasting, surrogates

# ---------- bridge bounds ------------------------------------------------


def test_sample_to_alarm_perfect_classifier_bounds_b_alarm_sensitivity_upper_equals_n_1_0():
    # Arrange
    b = bridge.sample_to_alarm(
        sample_sensitivity=1.0,
        sample_specificity=1.0,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    # Act
    # Assert
    assert b.alarm_sensitivity_upper == 1.0


def test_sample_to_alarm_perfect_classifier_bounds_b_fp_per_hour_upper_equals_n_0_0():
    # Arrange
    b = bridge.sample_to_alarm(
        sample_sensitivity=1.0,
        sample_specificity=1.0,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    # Act
    # Assert
    assert b.fp_per_hour_upper == 0.0


def test_sample_to_alarm_chance_bounds_n_0_99_b_alarm_sensitivity_upper_1_0():
    # Arrange
    b = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.5,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        prevalence=1.0,
    )
    # Act
    # Assert
    assert 0.99 < b.alarm_sensitivity_upper <= 1.0


def test_sample_to_alarm_chance_bounds_b2_fp_per_hour_upper_equals_pytest_approx_6_0_abs_1e_06():
    # Arrange
    # naive FPR=0.5 × (1-π=0.0) = 0/h with prev=1.0; capped or not.
    # Use prevalence=0.5 for the FP/hr cap test.
    # Act
    b2 = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.5,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        prevalence=0.5,
    )
    # Assert
    assert b2.fp_per_hour_upper == pytest.approx(6.0, abs=1e-6)


def test_sample_to_alarm_prevalence_does_not_change_K_eff_equals_K():
    # Each SOP holds K windows by construction, so K_eff = K is
    # independent of the global prevalence (K_eff soundness fix).
    # Arrange
    high = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.95,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        prevalence=1.0,
    )
    low = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.95,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        prevalence=0.05,
    )
    # Act
    # Assert — K = ceil(600 / 60) = 10, regardless of prevalence.
    assert high.K_effective == low.K_effective == 10


def test_sample_to_alarm_prevalence_does_not_change_alarm_sensitivity_upper():
    # Arrange
    high = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.95,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        prevalence=1.0,
    )
    low = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.95,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        prevalence=0.05,
    )
    # Act
    # Assert — detection band no longer collapses at low prevalence.
    assert high.alarm_sensitivity_upper == pytest.approx(low.alarm_sensitivity_upper)


def test_sample_to_alarm_longer_sop_widens_detection_band():
    # SOP now correctly drives K = ceil(SOP / cadence); the old
    # prevalence-shrink made SOP=15 and SOP=60 degenerate at low π.
    # Arrange
    short = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.95,
        sop_seconds=15,
        cadence_seconds=15,
        refractory_seconds=15,
        prevalence=0.05,
    )
    long = bridge.sample_to_alarm(
        sample_sensitivity=0.5,
        sample_specificity=0.95,
        sop_seconds=60,
        cadence_seconds=15,
        refractory_seconds=60,
        prevalence=0.05,
    )
    # Act
    # Assert
    assert long.alarm_sensitivity_upper > short.alarm_sensitivity_upper


def test_sample_to_alarm_out_of_range_sensitivity_raises_valueerror():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        bridge.sample_to_alarm(
            sample_sensitivity=1.5,
            sample_specificity=0.9,
            sop_seconds=600,
            cadence_seconds=60,
        )


def test_sample_to_alarm_nonpositive_sop_raises_valueerror():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        bridge.sample_to_alarm(
            sample_sensitivity=0.5,
            sample_specificity=0.5,
            sop_seconds=0,
            cadence_seconds=60,
        )


def test_alarm_to_sample_round_trip_consistency_rev_sample_sensitivity_lower_s_1e_06():
    # Arrange
    s, sp = 0.7, 0.95
    fwd = bridge.sample_to_alarm(
        sample_sensitivity=s,
        sample_specificity=sp,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    rev = bridge.alarm_to_sample(
        alarm_sensitivity=fwd.alarm_sensitivity_upper,
        fp_per_hour=fwd.fp_per_hour_upper,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    # Act
    # Assert
    assert rev["sample_sensitivity_lower"] <= s + 1e-6


def test_alarm_to_sample_round_trip_consistency_rev_sample_sensitivity_upper_s_1e_06():
    # Arrange
    s, sp = 0.7, 0.95
    fwd = bridge.sample_to_alarm(
        sample_sensitivity=s,
        sample_specificity=sp,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    rev = bridge.alarm_to_sample(
        alarm_sensitivity=fwd.alarm_sensitivity_upper,
        fp_per_hour=fwd.fp_per_hour_upper,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    # Act
    # Assert
    assert rev["sample_sensitivity_upper"] >= s - 1e-6


def test_alarm_to_sample_round_trip_consistency_rev_sample_specificity_upper_sp_1e_06():
    # Arrange
    s, sp = 0.7, 0.95
    fwd = bridge.sample_to_alarm(
        sample_sensitivity=s,
        sample_specificity=sp,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    rev = bridge.alarm_to_sample(
        alarm_sensitivity=fwd.alarm_sensitivity_upper,
        fp_per_hour=fwd.fp_per_hour_upper,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
    )
    # Act
    # Assert
    assert rev["sample_specificity_upper"] >= sp - 1e-6


@pytest.mark.parametrize("s,sp", [(0.6, 0.9), (0.4, 0.7), (0.8, 0.99)])
def test_bridge_bounds_via_monte_carlo_rep_sensitivity_bnd_alarm_sensitivity_upper_0_1(
    s, sp
):
    # Arrange
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
        rng.binomial(1, s, size=n_windows) * 0.6
        + 0.5 * (1 - rng.binomial(1, s, size=n_windows)),
        rng.binomial(1, 1 - sp, size=n_windows) * 0.6
        + 0.5 * (1 - rng.binomial(1, 1 - sp, size=n_windows)),
    )
    pol = AlarmPolicy(
        sph_seconds=0,
        sop_seconds=sop,
        cadence_seconds=cadence,
        refractory_seconds=sop,
        alarm_threshold=0.55,
    )
    rep = forecasting.evaluate_stream(
        proba, times, seizures, pol, total_recording_time=duration, n_surrogate=100
    )
    bnd = bridge.sample_to_alarm(
        sample_sensitivity=s,
        sample_specificity=sp,
        sop_seconds=sop,
        cadence_seconds=cadence,
        refractory_seconds=sop,
    )
    # Act
    # Assert
    assert rep.sensitivity <= bnd.alarm_sensitivity_upper + 0.10


@pytest.mark.parametrize("s,sp", [(0.6, 0.9), (0.4, 0.7), (0.8, 0.99)])
def test_bridge_bounds_via_monte_carlo_rep_fp_per_hour_bnd_fp_per_hour_upper_1_0(s, sp):
    # Arrange
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
        rng.binomial(1, s, size=n_windows) * 0.6
        + 0.5 * (1 - rng.binomial(1, s, size=n_windows)),
        rng.binomial(1, 1 - sp, size=n_windows) * 0.6
        + 0.5 * (1 - rng.binomial(1, 1 - sp, size=n_windows)),
    )
    pol = AlarmPolicy(
        sph_seconds=0,
        sop_seconds=sop,
        cadence_seconds=cadence,
        refractory_seconds=sop,
        alarm_threshold=0.55,
    )
    rep = forecasting.evaluate_stream(
        proba, times, seizures, pol, total_recording_time=duration, n_surrogate=100
    )
    bnd = bridge.sample_to_alarm(
        sample_sensitivity=s,
        sample_specificity=sp,
        sop_seconds=sop,
        cadence_seconds=cadence,
        refractory_seconds=sop,
    )
    # Act
    # Assert
    assert rep.fp_per_hour <= bnd.fp_per_hour_upper + 1.0


# ---------- surrogate registry ------------------------------------------


def test_surrogate_registry_has_builtins():
    # Arrange
    # Act
    # Assert
    for name in ["poisson", "periodic", "persistence"]:
        fn = surrogates.get(name)
        assert callable(fn)


def test_surrogate_unknown_raises():
    # Arrange
    # Act
    # Assert
    with pytest.raises(KeyError):
        surrogates.get("nonexistent")


def test_surrogate_poisson_within_bounds_a_size_equals_n_50():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert a.size == 50


def test_surrogate_poisson_within_bounds_a_min_0():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert a.min() >= 0


def test_surrogate_poisson_within_bounds_a_max_1000_0():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert a.max() <= 1000.0


def test_surrogate_poisson_within_bounds_np_diff_a_0_all():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert (np.diff(a) >= 0).all()  # sorted


def test_surrogate_periodic_spacing_a_size_equals_n_10():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.periodic(10, 1000.0, rng)
    # Act
    # Assert
    assert a.size == 10


def test_surrogate_periodic_spacing_np_allclose_spacing_spacing_mean_atol_1e_06():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.periodic(10, 1000.0, rng)
    # Act
    spacing = np.diff(a)
    # Assert
    assert np.allclose(spacing, spacing.mean(), atol=1e-6)


def test_surrogate_register_custom():
    @surrogates.register("test_burst")
    # Arrange
    def burst(n, total, rng):
        # All alarms in first 10% of recording
        return np.sort(rng.uniform(0, 0.1 * total, size=n))

    fn = surrogates.get("test_burst")
    rng = np.random.default_rng(0)
    # Act
    a = fn(20, 1000.0, rng)
    # Assert
    assert a.max() < 100.0
