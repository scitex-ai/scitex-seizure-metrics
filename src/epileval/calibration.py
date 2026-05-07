"""Calibration metrics — reliability diagrams and Brier-score
decomposition.

Brier(p, y) = mean((p - y)^2)
            = reliability − resolution + uncertainty       (Murphy 1973)

where, for K equal-frequency bins:
- reliability = sum_k (n_k/N) * (mean_p_k - mean_y_k)^2
                  → 0 when predicted probabilities match observed freqs
- resolution  = sum_k (n_k/N) * (mean_y_k - mean_y)^2
                  → high when bins differ in their observed positive rate
- uncertainty = mean_y * (1 - mean_y)
                  → property of the data alone (max 0.25 at p=0.5)

Smaller Brier is better; the decomposition tells you whether bad scores
come from miscalibration (reliability) or low discrimination (low
resolution).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReport:
    """Container for calibration metrics + per-bin curve points."""
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    expected_calibration_error: float
    bin_centers: np.ndarray            # mean predicted proba per bin
    bin_observed: np.ndarray           # observed positive rate per bin
    bin_counts: np.ndarray             # count per bin


def calibration_report(y_true, y_proba, *, n_bins: int = 10,
                       strategy: str = "uniform") -> CalibrationReport:
    """Compute Brier decomposition + reliability table.

    Args:
        y_true: 1-D binary labels.
        y_proba: 1-D continuous predictions in [0, 1].
        n_bins: number of probability bins for the reliability table.
        strategy: 'uniform' (equal-width) or 'quantile' (equal-frequency).

    Returns:
        CalibrationReport.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba shape mismatch")
    n = y_true.size
    if n == 0:
        raise ValueError("empty inputs")

    # Bin edges
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(
            np.quantile(y_proba, np.linspace(0.0, 1.0, n_bins + 1))
        )
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    bin_idx = np.clip(np.digitize(y_proba, edges[1:-1], right=False),
                      0, len(edges) - 2)

    bin_p, bin_o, bin_c = [], [], []
    for k in range(len(edges) - 1):
        mask = bin_idx == k
        if mask.any():
            bin_p.append(float(y_proba[mask].mean()))
            bin_o.append(float(y_true[mask].mean()))
            bin_c.append(int(mask.sum()))
        else:
            bin_p.append(float((edges[k] + edges[k + 1]) / 2))
            bin_o.append(0.0)
            bin_c.append(0)

    bin_p = np.asarray(bin_p)
    bin_o = np.asarray(bin_o)
    bin_c = np.asarray(bin_c)

    base_rate = float(y_true.mean())
    weight = bin_c / max(1, n)

    reliability = float(np.sum(weight * (bin_p - bin_o) ** 2))
    resolution = float(np.sum(weight * (bin_o - base_rate) ** 2))
    uncertainty = float(base_rate * (1.0 - base_rate))
    brier = float(np.mean((y_proba - y_true) ** 2))

    # ECE = sum_k (n_k/N) * |mean_p_k - mean_y_k|
    ece = float(np.sum(weight * np.abs(bin_p - bin_o)))

    return CalibrationReport(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        expected_calibration_error=ece,
        bin_centers=bin_p,
        bin_observed=bin_o,
        bin_counts=bin_c,
    )
