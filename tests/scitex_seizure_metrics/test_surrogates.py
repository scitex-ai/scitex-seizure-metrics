"""Tests for the surrogate registry (poisson/periodic/persistence + register)."""
from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics import surrogates


def test_registry_has_builtins():
    for name in ["poisson", "periodic", "persistence"]:
        assert callable(surrogates.get(name))


def test_unknown_raises():
    with pytest.raises(KeyError):
        surrogates.get("nonexistent")


def test_poisson_within_bounds():
    rng = np.random.default_rng(0)
    a = surrogates.poisson(50, 1000.0, rng)
    assert a.size == 50
    assert a.min() >= 0
    assert a.max() <= 1000.0
    assert (np.diff(a) >= 0).all()


def test_periodic_spacing_uniform():
    rng = np.random.default_rng(0)
    a = surrogates.periodic(10, 1000.0, rng)
    assert a.size == 10
    sp = np.diff(a)
    assert np.allclose(sp, sp.mean(), atol=1e-6)


def test_persistence_single_alarm():
    rng = np.random.default_rng(0)
    a = surrogates.persistence(1, 1000.0, rng)
    assert a.size == 1


def test_register_custom():
    @surrogates.register("test_burst")
    def burst(n, total, rng):
        return np.sort(rng.uniform(0, 0.1 * total, size=n))
    rng = np.random.default_rng(0)
    a = surrogates.get("test_burst")(20, 1000.0, rng)
    assert a.max() < 100.0


def test_circadian_24h_period():
    rng = np.random.default_rng(0)
    a = surrogates.circadian(7, 7 * 86400.0, rng)
    assert a.size == 7
    sp = np.diff(a)
    assert np.allclose(sp, 86400.0, atol=1.0)


def test_multidien_default_7day():
    rng = np.random.default_rng(0)
    a = surrogates.multidien(4, 28 * 86400.0, rng)
    assert a.size == 4
    sp = np.diff(a)
    assert np.allclose(sp, 7 * 86400.0, atol=1.0)


def test_from_history_periodic_extrapolation():
    rng = np.random.default_rng(0)
    history = np.array([3600.0, 7200.0, 10800.0])
    a = surrogates.from_history(2, 100000.0, rng,
                                seizure_history_seconds=history)
    assert abs(a[0] - 14400.0) < 1.0


def test_from_history_falls_back_without_history():
    rng = np.random.default_rng(0)
    a = surrogates.from_history(5, 1000.0, rng,
                                seizure_history_seconds=None)
    assert a.size == 5
    assert (a >= 0).all() and (a <= 1000.0).all()
