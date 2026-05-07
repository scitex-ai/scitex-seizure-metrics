"""Adapters for data formats found in the wild.

- proba_to_alarms: convert a continuous-prediction time series + threshold
  into an alarm-time list (forecasting input).
- cv_summary_load: read one of our neurovista
  classify_pac_nested_cv_ensemble_out/P*/cv_summary/*.json files into a
  flat dict (so the existing pipeline can be re-evaluated without code
  changes upstream).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def proba_to_alarms(
    proba,
    times,
    threshold: float,
    *,
    refractory_seconds: float = 0.0,
) -> np.ndarray:
    """Threshold a probability stream → list of alarm times.

    Args:
        proba: 1-D continuous predictions.
        times: matching timestamps (seconds).
        threshold: alarm fires when proba >= threshold.
        refractory_seconds: minimum gap between consecutive alarms; the
                            classic anti-burst rule.

    Returns:
        Sorted np.ndarray of alarm times.
    """
    proba = np.asarray(proba, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if proba.shape != times.shape:
        raise ValueError(f"shape mismatch: {proba.shape} vs {times.shape}")
    fires = times[proba >= threshold]
    if refractory_seconds > 0 and fires.size > 1:
        kept = [fires[0]]
        for t in fires[1:]:
            if t - kept[-1] >= refractory_seconds:
                kept.append(t)
        fires = np.asarray(kept)
    return fires


def cv_summary_load(path: str | Path) -> dict:
    """Load one classify_pac_nested_cv_ensemble_out/P*/cv_summary/*.json file.

    The filename encodes: {metric}_mean-{m}_std-{s}_n-{n}.json.
    Returns a flat dict suitable for stacking via pd.DataFrame.

    Example:
        cv-summary_balanced-accuracy_mean-0.687_std-0.012_n-3.json
        →  {patient: 'P03', metric: 'balanced-accuracy',
            mean: 0.687, std: 0.012, n: 3, raw: {...json contents...}}
    """
    path = Path(path)
    parts = path.stem.split("_")
    # Drop the leading "cv-summary" token if present.
    parts = [p for p in parts if p and p != "cv-summary"]
    metric = parts[0] if parts else "?"
    kv = {}
    for p in parts[1:]:
        if "-" in p:
            k, v = p.split("-", 1)
            try:
                kv[k] = float(v) if k != "n" else int(v)
            except ValueError:
                kv[k] = v
    patient = path.parent.parent.name  # P03
    raw = json.loads(path.read_text()) if path.exists() else {}
    return {
        "patient": patient,
        "metric": metric,
        "mean": kv.get("mean"),
        "std": kv.get("std"),
        "n": kv.get("n"),
        "raw": raw,
    }


def stack_cv_summaries(paths: Iterable[str | Path]):
    """Stack multiple cv_summary JSONs into one DataFrame (one row per file)."""
    import pandas as pd

    rows = [cv_summary_load(p) for p in paths]
    df = pd.DataFrame(rows).drop(columns=["raw"], errors="ignore")
    return df
