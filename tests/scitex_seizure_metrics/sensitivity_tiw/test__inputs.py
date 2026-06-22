"""Tests for ``sensitivity_tiw._inputs`` — label/seizure-time helpers.

Covers ``seizures_from_labels`` (derive onset times from per-window
binary pre-ictal labels).
"""

from __future__ import annotations

import numpy as np

from scitex_seizure_metrics import sensitivity_tiw


def test_seizures_from_labels_counts_runs():
    # Arrange
    times = np.arange(0, 100.0, 10.0)  # 10 windows
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])  # two pre-ictal runs
    # Act
    onsets = sensitivity_tiw.seizures_from_labels(labels, times)
    # Assert
    assert onsets.size == 2
