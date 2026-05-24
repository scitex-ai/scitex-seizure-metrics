"""Tests for the surrogate registry (poisson/periodic/persistence + register)."""

from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics import surrogates


def test_registry_has_builtins():
    # Arrange
    # Act
    # Assert
    assert all(
        callable(surrogates.get(name))
        for name in ["poisson", "periodic", "persistence"]
    )


def test_unknown_raises_raises_keyerror():
    # Arrange
    # Act
    # Assert
    with pytest.raises(KeyError):
        surrogates.get("nonexistent")


def test_poisson_within_bounds_a_size_equals_n_50():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert a.size == 50


def test_poisson_within_bounds_a_min_0():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert a.min() >= 0


def test_poisson_within_bounds_a_max_1000_0():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert a.max() <= 1000.0


def test_poisson_within_bounds_np_diff_a_0_all():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    # Act
    # Assert
    assert (np.diff(a) >= 0).all()


def test_periodic_spacing_uniform_a_size_equals_n_10():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.periodic(10, 1000.0, rng)
    # Act
    # Assert
    assert a.size == 10


def test_periodic_spacing_uniform_np_allclose_sp_sp_mean_atol_1e_06():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.periodic(10, 1000.0, rng)
    # Act
    sp = np.diff(a)
    # Assert
    assert np.allclose(sp, sp.mean(), atol=1e-6)


def test_persistence_single_alarm():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    a = surrogates.persistence(1, 1000.0, rng)
    # Assert
    assert a.size == 1


def test_register_custom_a_max_100_0():
    @surrogates.register("test_burst")
    # Arrange
    def burst(n, total, rng):
        return np.sort(rng.uniform(0, 0.1 * total, size=n))

    rng = np.random.default_rng(0)
    # Act
    a = surrogates.get("test_burst")(20, 1000.0, rng)
    # Assert
    assert a.max() < 100.0


def test_circadian_24h_period_a_size_equals_n_7():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.circadian(7, 7 * 86400.0, rng)
    # Act
    # Assert
    assert a.size == 7


def test_circadian_24h_period_np_allclose_sp_86400_0_atol_1_0():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.circadian(7, 7 * 86400.0, rng)
    # Act
    sp = np.diff(a)
    # Assert
    assert np.allclose(sp, 86400.0, atol=1.0)


def test_multidien_default_7day_a_size_equals_n_4():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.multidien(4, 28 * 86400.0, rng)
    # Act
    # Assert
    assert a.size == 4


def test_multidien_default_7day_np_allclose_sp_7_86400_0_atol_1_0():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.multidien(4, 28 * 86400.0, rng)
    # Act
    sp = np.diff(a)
    # Assert
    assert np.allclose(sp, 7 * 86400.0, atol=1.0)


def test_from_history_periodic_extrapolation():
    # Arrange
    rng = np.random.default_rng(0)
    history = np.array([3600.0, 7200.0, 10800.0])
    # Act
    a = surrogates.from_history(2, 100000.0, rng, seizure_history_seconds=history)
    # Assert
    assert abs(a[0] - 14400.0) < 1.0


def test_from_history_falls_back_without_history_a_size_equals_n_5():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.from_history(5, 1000.0, rng, seizure_history_seconds=None)
    # Act
    # Assert
    assert a.size == 5


def test_from_history_falls_back_without_history_a_0_all_and_a_1000_0_all():
    # Arrange
    rng = np.random.default_rng(0)
    a = surrogates.from_history(5, 1000.0, rng, seizure_history_seconds=None)
    # Act
    # Assert
    assert (a >= 0).all() and (a <= 1000.0).all()
