"""Tests for ``sensitivity_tiw._significance`` — above-chance significance.

Synthetic data with KNOWN ground truth:

- the chance diagonal (sensitivity == TiW) and its unit-interval clip;
- a better-than-chance predictor -> significant (binomial + surrogate);
- a random predictor -> not significant (surrogate);
- input validation (no seizures, non-positive counts).

One assertion per test, mirroring the package convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from scitex_seizure_metrics import AlarmPolicy, sensitivity_tiw

# --------------------------------------------------------------------------
# Fixtures / builders
# --------------------------------------------------------------------------

CADENCE = 60.0
SOP = 600.0
DURATION = 24 * 3600.0


def _policy(**overrides):
    base = dict(
        sph_seconds=0.0,
        sop_seconds=SOP,
        cadence_seconds=CADENCE,
        refractory_seconds=SOP,
    )
    base.update(overrides)
    return AlarmPolicy(**base)


def _grid():
    times = np.arange(0, DURATION, CADENCE)
    seizures = np.linspace(2 * 3600.0, 22 * 3600.0, 10)
    return times, seizures


def _informative_scores(times, seizures, rng, *, signal=0.85, base=0.05):
    """High score in [seizure - SOP, seizure); low elsewhere (+ noise)."""
    scores = base + 0.05 * rng.random(times.size)
    for sz in seizures:
        mask = (times >= sz - SOP) & (times < sz)
        fire = rng.random(mask.sum()) < signal
        idx = np.where(mask)[0]
        scores[idx[fire]] = 0.9 + 0.05 * rng.random(int(fire.sum()))
    return scores


# --------------------------------------------------------------------------
# Chance baseline (diagonal)
# --------------------------------------------------------------------------


def test_chance_sensitivity_is_diagonal():
    # Arrange
    # Act
    # Assert
    assert sensitivity_tiw.chance_sensitivity(0.3) == pytest.approx(0.3)


def test_chance_sensitivity_clipped_to_unit_interval():
    # Arrange
    # Act
    # Assert
    assert sensitivity_tiw.chance_sensitivity(1.5) == 1.0


# --------------------------------------------------------------------------
# Better-than-chance predictor -> significant
# --------------------------------------------------------------------------


def test_informative_surrogate_significant():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    # Act
    sig = sensitivity_tiw.surrogate_above_chance(
        scores,
        _policy(),
        threshold=0.5,
        seizure_times=seizures,
        times=times,
        n_surrogate=500,
        rng_seed=1,
    )
    # Assert
    assert sig.p_value < 0.05


def test_informative_binomial_significant():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Pick the operating point at the target TiW budget.
    mask = curve.tiw <= curve.target_tiw + 1e-12
    i = np.where(mask)[0][np.argmax(curve.sensitivity[mask])]
    n_caught = int(round(curve.sensitivity[i] * curve.n_seizures))
    # Act
    sig = sensitivity_tiw.binomial_above_chance(
        n_caught=n_caught, n_seizures=curve.n_seizures, tiw=float(curve.tiw[i])
    )
    # Assert
    assert sig.p_value < 0.05


# --------------------------------------------------------------------------
# Random predictor -> not significant
# --------------------------------------------------------------------------


def test_random_surrogate_not_significant():
    # Arrange
    rng = np.random.default_rng(7)
    times, seizures = _grid()
    scores = rng.random(times.size)
    # Act
    sig = sensitivity_tiw.surrogate_above_chance(
        scores,
        _policy(),
        threshold=0.5,
        seizure_times=seizures,
        times=times,
        n_surrogate=500,
        rng_seed=3,
    )
    # Assert
    assert sig.p_value > 0.05


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_surrogate_no_seizures_raises():
    # Arrange
    times = np.arange(0, 3600.0, CADENCE)
    scores = np.zeros(times.size)
    # Act
    # Assert
    with pytest.raises(ValueError):
        sensitivity_tiw.surrogate_above_chance(
            scores,
            _policy(),
            threshold=0.5,
            seizure_times=np.array([]),
            times=times,
            n_surrogate=10,
        )


def test_binomial_rejects_nonpositive_n():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        sensitivity_tiw.binomial_above_chance(n_caught=0, n_seizures=0, tiw=0.2)
