"""Smoke test for papers.andrade2024.metrics()."""
import numpy as np
from scitex_seizure_metrics.papers import andrade2024


def _data():
    rng = np.random.default_rng(0)
    duration = 24 * 3600
    cadence = 60
    times = np.arange(0, duration, cadence, dtype=float)
    seizures = np.array([4 * 3600, 11 * 3600, 19 * 3600], dtype=float)
    proba = rng.uniform(0, 0.3, size=times.size)
    for sz in seizures:
        m = (times >= sz - 3600) & (times < sz - 600)
        proba[m] += np.linspace(0.1, 0.7, m.sum())
    proba = np.clip(proba, 0, 1)
    y_true = np.zeros_like(times, dtype=int)
    for sz in seizures:
        y_true[(times >= sz - 1800) & (times < sz)] = 1
    return y_true, proba, times, seizures


def test_andrade2024_returns_dict_out_is_dict():
    # Arrange
    y, p, t, sz = _data()
    if "andrade2024" == "kuhlmann2018":
        out = andrade2024.metrics(y_true=y, y_proba=p)
    else:
        out = andrade2024.metrics(y_true=y, y_proba=p, times_seconds=t,
                              seizure_times=sz, n_surrogate=30)
    # Act
    # Assert
    assert isinstance(out, dict)


def test_andrade2024_returns_dict_paper_in_out():
    # Arrange
    y, p, t, sz = _data()
    if "andrade2024" == "kuhlmann2018":
        out = andrade2024.metrics(y_true=y, y_proba=p)
    else:
        out = andrade2024.metrics(y_true=y, y_proba=p, times_seconds=t,
                              seizure_times=sz, n_surrogate=30)
    # Act
    # Assert
    assert "paper" in out


