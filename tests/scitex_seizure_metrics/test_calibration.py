"""Tests for the Brier-decomposition calibration module."""
from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics import calibration


def test_calibration_perfect_predictor():
    """Predictions exactly equal labels (binary) → reliability=0, ECE=0."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(int)
    p = y.astype(float)
    rep = calibration.calibration_report(y, p, n_bins=5)
    assert rep.brier == pytest.approx(0.0, abs=1e-9)
    assert rep.reliability == pytest.approx(0.0, abs=1e-9)
    assert rep.expected_calibration_error == pytest.approx(0.0, abs=1e-9)


def test_calibration_uniform_random_predictor():
    """Random p, random y → high Brier, near-zero resolution."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=2000)
    p = rng.uniform(0, 1, size=2000)
    rep = calibration.calibration_report(y, p, n_bins=10)
    assert 0.2 < rep.brier < 0.4   # ≈ uncertainty
    assert rep.resolution < 0.05   # no discrimination


def test_calibration_decomposition_identity():
    """Brier ≈ reliability − resolution + uncertainty (Murphy 1973),
    within binning error."""
    rng = np.random.default_rng(2)
    y = (rng.uniform(0, 1, size=500) < 0.3).astype(int)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.1, size=500), 0, 1)
    rep = calibration.calibration_report(y, p, n_bins=10)
    decomp = rep.reliability - rep.resolution + rep.uncertainty
    # Binning approximation introduces small discrepancy
    assert abs(rep.brier - decomp) < 0.05


def test_calibration_quantile_strategy():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=300)
    p = rng.uniform(0, 1, size=300)
    rep = calibration.calibration_report(y, p, n_bins=10,
                                         strategy="quantile")
    # Each quantile bin gets approximately equal counts.
    counts = rep.bin_counts[rep.bin_counts > 0]
    assert counts.std() / counts.mean() < 0.5


def test_calibration_invalid_strategy_raises():
    with pytest.raises(ValueError):
        calibration.calibration_report([0, 1], [0.1, 0.9], strategy="bogus")
