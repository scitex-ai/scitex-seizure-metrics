"""Input normalisation for the sensitivity-vs-TiW trade-off.

Two input modes are supported, mirroring the rest of the package:

- Mode A: per-window binary pre-ictal ``labels`` (+ optional ``times``).
- Mode B: explicit ``seizure_times`` + per-window ``times``.

Both collapse to a common ``(scores, times, seizures, total_T, cadence)``
tuple consumed by the curve and significance code.
"""

from __future__ import annotations

import numpy as np


def seizures_from_labels(labels, times) -> np.ndarray:
    """Derive one onset time per contiguous pre-ictal (label==1) run.

    The onset is placed at the END of the pre-ictal run (the first
    timestamp after the run) because pre-ictal windows precede the
    seizure. If a run reaches the end of the recording, the onset is
    placed one cadence after the last labelled window.

    Args:
        labels: per-window binary pre-ictal labels.
        times: matching per-window timestamps (seconds).

    Returns:
        np.ndarray of seizure onset times (seconds), sorted ascending.
    """
    labels = np.asarray(labels).astype(int).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if labels.shape != times.shape:
        raise ValueError(
            f"labels and times shape mismatch: {labels.shape} vs {times.shape}"
        )
    if labels.size == 0:
        return np.array([])
    cadence = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    onsets = []
    prev = 0
    for i, lab in enumerate(labels):
        if lab == 0 and prev == 1:
            # Pre-ictal run ended at i-1; seizure onset at this timestamp.
            onsets.append(times[i])
        prev = lab
    if prev == 1:
        # Run reached the end; place onset just past the last window.
        onsets.append(times[-1] + cadence)
    return np.sort(np.asarray(onsets, dtype=float))


def resolve_inputs(scores, *, labels, seizure_times, times):
    """Normalise the two input modes to a common tuple.

    Args:
        scores: 1-D per-window scores / probabilities.
        labels: per-window binary labels (mode A) or None.
        seizure_times: explicit onset timestamps (mode B) or None.
        times: per-window timestamps or None (defaults to unit cadence).

    Returns:
        (scores, times, seizures, total_T, cadence).

    Raises:
        ValueError: empty scores, shape mismatch, or neither mode given.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("scores must be non-empty")

    if times is None:
        times = np.arange(scores.size, dtype=float)
    else:
        times = np.asarray(times, dtype=float).ravel()
    if times.shape != scores.shape:
        raise ValueError(
            f"scores and times shape mismatch: {scores.shape} vs {times.shape}"
        )

    if seizure_times is not None:
        seizures = np.sort(np.asarray(seizure_times, dtype=float).ravel())
    elif labels is not None:
        seizures = seizures_from_labels(labels, times)
    else:
        raise ValueError(
            "provide either `labels` (per-window binary) or `seizure_times`"
        )

    cadence = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    total_T = float(times.max() - times.min()) + cadence
    return scores, times, seizures, total_T, cadence
