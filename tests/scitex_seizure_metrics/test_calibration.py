"""Tests for the Brier-decomposition calibration module."""
from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics import calibration


def test_calibration_perfect_predictor_rep_brier_equals_pytest_approx_0_0_abs_1e_09():
    # Arrange
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(int)
    p = y.astype(float)
    rep = calibration.calibration_report(y, p, n_bins=5)
    # Act
    # Assert
    assert rep.brier == pytest.approx(0.0, abs=1e-9)


def test_calibration_perfect_predictor_rep_reliability_equals_pytest_approx_0_0_abs_1e_09():
    # Arrange
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(int)
    p = y.astype(float)
    rep = calibration.calibration_report(y, p, n_bins=5)
    # Act
    # Assert
    assert rep.reliability == pytest.approx(0.0, abs=1e-9)


def test_calibration_perfect_predictor_rep_expected_calibration_error_equals_pytest_approx_0_0_abs_1e_09():
    # Arrange
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(int)
    p = y.astype(float)
    rep = calibration.calibration_report(y, p, n_bins=5)
    # Act
    # Assert
    assert rep.expected_calibration_error == pytest.approx(0.0, abs=1e-9)




def test_calibration_uniform_random_predictor_n_0_2_rep_brier_0_4():
    # Arrange
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=2000)
    p = rng.uniform(0, 1, size=2000)
    rep = calibration.calibration_report(y, p, n_bins=10)
    # Act
    # Assert
    assert 0.2 < rep.brier < 0.4   # ≈ uncertainty


def test_calibration_uniform_random_predictor_rep_resolution_0_05():
    # Arrange
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=2000)
    p = rng.uniform(0, 1, size=2000)
    rep = calibration.calibration_report(y, p, n_bins=10)
    # Act
    # Assert
    assert rep.resolution < 0.05   # no discrimination




def test_calibration_decomposition_identity():
    """Brier ≈ reliability − resolution + uncertainty (Murphy 1973),
    within binning error."""
    # Arrange
    rng = np.random.default_rng(2)
    y = (rng.uniform(0, 1, size=500) < 0.3).astype(int)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.1, size=500), 0, 1)
    rep = calibration.calibration_report(y, p, n_bins=10)
    # Act
    decomp = rep.reliability - rep.resolution + rep.uncertainty
    # Binning approximation introduces small discrepancy
    # Assert
    assert abs(rep.brier - decomp) < 0.05


def test_calibration_quantile_strategy():
    # Arrange
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=300)
    p = rng.uniform(0, 1, size=300)
    rep = calibration.calibration_report(y, p, n_bins=10,
                                         strategy="quantile")
    # Each quantile bin gets approximately equal counts.
    # Act
    counts = rep.bin_counts[rep.bin_counts > 0]
    # Assert
    assert counts.std() / counts.mean() < 0.5


def test_calibration_invalid_strategy_raises():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        calibration.calibration_report([0, 1], [0.1, 0.9], strategy="bogus")
