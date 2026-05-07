"""Internal alarm-matching primitives shared by the forecasting module.

Stays private (leading underscore): the public surface is forecasting.evaluate
and forecasting.evaluate_stream. These helpers are documented only because
they're called from tests for correctness verification.
"""
from __future__ import annotations

import numpy as np


def proba_stream_to_alarms(proba: np.ndarray, times: np.ndarray,
                           threshold: float, refractory_seconds: float,
                           merge_consecutive: bool = True) -> np.ndarray:
    """Threshold + dedupe a continuous prediction stream into alarm times.

    Args:
        proba: 1-D continuous predictions in [0, 1].
        times: matching timestamps (seconds since recording start).
        threshold: alarm-candidate fires when proba >= threshold.
        refractory_seconds: minimum gap between consecutive alarms.
        merge_consecutive: if True, contiguous above-threshold runs count
            as one alarm at the first window; subsequent suppressions are
            in addition to the refractory rule.

    Returns:
        Sorted np.ndarray of alarm times.
    """
    proba = np.asarray(proba, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if proba.shape != times.shape:
        raise ValueError(f"shape mismatch: {proba.shape} vs {times.shape}")
    above = proba >= threshold

    if merge_consecutive:
        # Find rising edges of contiguous above-threshold runs.
        if above.size == 0:
            return np.array([])
        prev = np.concatenate([[False], above[:-1]])
        edges = above & ~prev
        candidates = times[edges]
    else:
        candidates = times[above]

    if refractory_seconds > 0 and candidates.size > 1:
        kept = [candidates[0]]
        for t in candidates[1:]:
            if t - kept[-1] >= refractory_seconds:
                kept.append(t)
        candidates = np.asarray(kept)
    return candidates


def alarm_match(alarms: np.ndarray, seizures: np.ndarray,
                sph: float, sop: float):
    """Per-seizure-and-per-alarm match under SPH/SOP semantics.

    A seizure at t_s is caught iff some alarm fires at t_a with
    t_a + sph <= t_s <= t_a + sph + sop.

    Returns:
        seizure_caught (bool array, len(seizures))
        alarm_useful (bool array, len(alarms))
    """
    seizures = np.asarray(seizures, dtype=float)
    alarms = np.asarray(alarms, dtype=float)
    sc = np.zeros(seizures.size, dtype=bool)
    au = np.zeros(alarms.size, dtype=bool)
    for i, a in enumerate(alarms):
        lo, hi = a + sph, a + sph + sop
        in_window = (seizures >= lo) & (seizures <= hi)
        if np.any(in_window):
            sc[in_window] = True
            au[i] = True
    return sc, au


def interictal_seconds(total_seconds: float, seizures: np.ndarray,
                       sop: float, sph: float = 0.0) -> float:
    """Total seconds NOT inside any seizure-related exclusion window.

    The Mormann tradition: FP/hr should be normalised by interictal-only
    time, where each seizure removes [seizure - sop - sph, seizure + sop]
    from the denominator (the SPH+SOP-before margin where any alarm is
    a true positive, plus a post-seizure SOP buffer).

    Args:
        total_seconds: full recording duration.
        seizures: seizure onset times (sorted).
        sop: Seizure Occurrence Period (seconds).
        sph: Seizure Prediction Horizon (seconds).

    Returns:
        Interictal duration (seconds).
    """
    if seizures.size == 0:
        return float(total_seconds)
    ex = np.column_stack([seizures - sop - sph, seizures + sop])
    ex = ex.clip(0, total_seconds)
    ex = ex[np.argsort(ex[:, 0])]
    excluded = 0.0
    cur_lo, cur_hi = ex[0]
    for lo, hi in ex[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            excluded += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    excluded += cur_hi - cur_lo
    return float(max(0.0, total_seconds - excluded))


def union_length(intervals: np.ndarray) -> float:
    """Total length of a union of [start, end] intervals."""
    if intervals.size == 0:
        return 0.0
    iv = intervals[np.argsort(intervals[:, 0])]
    total = 0.0
    cur_lo, cur_hi = iv[0]
    for lo, hi in iv[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return float(total)
