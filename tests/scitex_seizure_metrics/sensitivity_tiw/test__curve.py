"""Tests for ``sensitivity_tiw._curve`` — the empirical sensitivity-vs-TiW curve.

Synthetic data with KNOWN ground truth:

- a better-than-chance predictor -> curve provably above the diagonal;
- a random / constant predictor -> curve on the diagonal;
- edge cases (all-positive, all-negative, single seizure);
- the monotone upper envelope the operating-point marker reads off.

We assert the summary scalars + monotonicity, mirroring the package's
flat, one-assertion-per-test convention.
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
    """High score in [seizure - SOP, seizure); low elsewhere (+ noise).

    A genuinely predictive forecaster: it raises its score inside each
    pre-ictal window. ``signal`` controls how often it fires pre-ictally.
    """
    scores = base + 0.05 * rng.random(times.size)
    for sz in seizures:
        mask = (times >= sz - SOP) & (times < sz)
        fire = rng.random(mask.sum()) < signal
        idx = np.where(mask)[0]
        scores[idx[fire]] = 0.9 + 0.05 * rng.random(int(fire.sum()))
    return scores


def _labels_from_seizures(times, seizures):
    labels = np.zeros(times.size, dtype=int)
    for sz in seizures:
        labels[(times >= sz - SOP) & (times < sz)] = 1
    return labels


# --------------------------------------------------------------------------
# Better-than-chance predictor
# --------------------------------------------------------------------------


def test_informative_curve_improvement_over_chance_positive():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert
    assert curve.improvement_over_chance > 0.05


def test_informative_curve_lies_above_diagonal_at_low_tiw():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Act
    # At a usable time-in-warning budget the model beats the diagonal.
    sens_at = curve.sensitivity_at_target_tiw
    # Assert
    assert sens_at > curve.target_tiw + 0.1


def test_informative_curve_n_seizures_matches():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert
    assert curve.n_seizures == seizures.size


def test_informative_labels_mode_matches_seizure_times_mode():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    labels = _labels_from_seizures(times, seizures)
    c_lab = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), labels=labels, times=times
    )
    c_sz = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Act
    # Assert — label-derived onsets recover the same seizure count.
    assert c_lab.n_seizures == c_sz.n_seizures


# --------------------------------------------------------------------------
# Random / constant predictor -> on the diagonal
# --------------------------------------------------------------------------


def test_random_curve_improvement_near_zero():
    # Arrange
    rng = np.random.default_rng(7)
    times, seizures = _grid()
    scores = rng.random(times.size)
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert — random forecaster hugs the diagonal (small |area|).
    assert abs(curve.improvement_over_chance) < 0.12


def test_curve_chance_column_equals_tiw():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    df = curve.to_frame()
    # Act
    # Assert — the plotted chance line is exactly the diagonal.
    assert np.allclose(df["chance_sensitivity"].values, df["tiw"].values)


# --------------------------------------------------------------------------
# Monotonicity + ordering
# --------------------------------------------------------------------------


def test_curve_sorted_by_ascending_tiw():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Act
    # Assert
    assert np.all(np.diff(curve.tiw) >= -1e-12)


def test_sensitivity_monotone_nondecreasing_with_tiw():
    # Arrange
    rng = np.random.default_rng(0)
    times, seizures = _grid()
    scores = _informative_scores(times, seizures, rng)
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Sensitivity, read as threshold drops (TiW rises), is non-decreasing
    # on the upper envelope.
    t = curve.tiw
    s = curve.sensitivity
    uniq_t, inv = np.unique(t, return_inverse=True)
    env = np.full(uniq_t.shape, -np.inf)
    np.maximum.at(env, inv, s)
    # Act
    # Assert
    assert np.all(np.diff(env) >= -1e-9)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_all_positive_scores_top_right_corner():
    # Arrange
    times, seizures = _grid()
    scores = np.ones(times.size)  # every window above any threshold
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert — permanent warning catches everything.
    assert curve.sensitivity.max() == pytest.approx(1.0)


def test_all_positive_scores_high_tiw():
    # Arrange
    times, seizures = _grid()
    scores = np.ones(times.size)
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert
    assert curve.tiw.max() > 0.9


def test_constant_predictor_reaches_origin():
    # Arrange — a constant (uninformative) stream is swept across its
    # value, giving the silent operating point (threshold above the
    # constant fires nothing).
    times, seizures = _grid()
    scores = np.zeros(times.size)
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert — the (0, 0) corner is present.
    assert curve.tiw.min() == 0.0 and curve.sensitivity.min() == 0.0


def test_constant_predictor_lies_on_diagonal():
    # Arrange — a constant predictor carries no information, so its
    # operating points sit on the chance diagonal (no area above it).
    times, seizures = _grid()
    scores = np.zeros(times.size)
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert
    assert abs(curve.improvement_over_chance) < 1e-9


def test_single_seizure_sensitivity_binary():
    # Arrange
    times = np.arange(0, 6 * 3600.0, CADENCE)
    seizures = np.array([3 * 3600.0])
    scores = np.zeros(times.size)
    scores[(times >= 3 * 3600.0 - SOP) & (times < 3 * 3600.0)] = 0.9
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert — with one seizure, sensitivity can only be 0 or 1.
    assert set(np.unique(curve.sensitivity)).issubset({0.0, 1.0})


def test_single_seizure_caught_when_fired():
    # Arrange
    times = np.arange(0, 6 * 3600.0, CADENCE)
    seizures = np.array([3 * 3600.0])
    scores = np.zeros(times.size)
    scores[(times >= 3 * 3600.0 - SOP) & (times < 3 * 3600.0)] = 0.9
    # Act
    curve = sensitivity_tiw.sensitivity_tiw_curve(
        scores, _policy(), seizure_times=seizures, times=times
    )
    # Assert
    assert curve.sensitivity.max() == 1.0


# --------------------------------------------------------------------------
# Input validation (curve construction)
# --------------------------------------------------------------------------


def test_requires_labels_or_seizure_times():
    # Arrange
    scores = np.random.default_rng(0).random(100)
    # Act
    # Assert
    with pytest.raises(ValueError):
        sensitivity_tiw.sensitivity_tiw_curve(scores, _policy())


def test_scores_times_shape_mismatch_raises():
    # Arrange
    scores = np.zeros(100)
    times = np.arange(50.0)
    # Act
    # Assert
    with pytest.raises(ValueError):
        sensitivity_tiw.sensitivity_tiw_curve(
            scores, _policy(), seizure_times=np.array([10.0]), times=times
        )


# --------------------------------------------------------------------------
# Monotone upper envelope (operating frontier the marker reads off)
# --------------------------------------------------------------------------


def test_monotone_upper_envelope_is_nondecreasing():
    # Arrange — a non-monotone raw curve.
    tiw = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
    sens = np.array([0.0, 0.6, 0.3, 0.9, 0.4])
    # Act
    _, env = sensitivity_tiw.monotone_upper_envelope(tiw, sens)
    # Assert
    assert np.all(np.diff(env) >= -1e-12)


def test_monotone_upper_envelope_collapses_duplicate_tiw_to_max():
    # Arrange — two operating points at the same TiW.
    tiw = np.array([0.2, 0.2, 0.4])
    sens = np.array([0.3, 0.7, 0.8])
    # Act
    env_t, env_s = sensitivity_tiw.monotone_upper_envelope(tiw, sens)
    # Assert
    assert env_t.size == 2 and env_s[0] == pytest.approx(0.7)


def test_monotone_upper_envelope_value_at_target_equals_sensitivity_at_tiw():
    # Arrange — straddling curve where the best-within-budget point sits
    # strictly left of the budget (the marker-float bug geometry).
    from scitex_seizure_metrics.sensitivity_tiw import _curve

    tiw = np.array([0.0, 0.15, 0.30, 1.0])
    sens = np.array([0.0, 0.42, 0.90, 1.0])
    target = 0.20
    env_t, env_s = sensitivity_tiw.monotone_upper_envelope(tiw, sens)
    # Act — value the steps-post envelope holds at the target budget.
    held = float(env_s[env_t <= target + 1e-12].max())
    scalar = _curve.sensitivity_at_tiw(tiw, sens, target)
    # Assert — marker scalar lands exactly on the envelope staircase.
    assert held == pytest.approx(scalar)
