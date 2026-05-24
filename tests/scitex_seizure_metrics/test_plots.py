"""Smoke tests for plot generators — assert no crash + axis labels set.

We don't visually verify; we rely on the upstream calls being shape-
correct. Generated artefacts are saved to /tmp/scitex-seizure-metrics-plots/ for
manual inspection.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # no display backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scitex_seizure_metrics import AlarmPolicy, forecasting, plots

PLOT_DIR = "/tmp/scitex-seizure-metrics-plots"
os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture
def operating_curve():
    """A realistic sweep_thresholds DataFrame to feed plotters."""
    seizures = np.array([3600.0, 7200.0, 10800.0, 14400.0, 18000.0])
    times = np.arange(0, 21600, 60.0)
    proba = np.zeros_like(times)
    for sz in seizures:
        # ramp up to high proba close to seizure
        d = sz - times
        m = (d > 0) & (d < 1800)
        proba[m] = np.clip(0.9 * (1 - d[m] / 1800), 0, 1)
    pol = AlarmPolicy(
        sph_seconds=300,
        sop_seconds=600,
        cadence_seconds=60,
        refractory_seconds=600,
        alarm_threshold=0.5,
    )
    return forecasting.sweep_thresholds(
        proba,
        times,
        seizures,
        pol,
        thresholds=np.linspace(0.05, 0.95, 19),
        total_recording_time=21600.0,
        n_surrogate=200,
    )


def test_plot_sensitivity_vs_fp_per_hour_sensitivity_in_ax_get_ylabel_lower(
    operating_curve,
):
    # Arrange
    fig, ax = plots.sensitivity_vs_fp_per_hour(operating_curve)
    # Act
    # Assert
    assert "sensitivity" in ax.get_ylabel().lower()


def test_plot_sensitivity_vs_fp_per_hour_fp_in_ax_get_xlabel_lower(operating_curve):
    # Arrange
    fig, ax = plots.sensitivity_vs_fp_per_hour(operating_curve)
    # Act
    # Assert
    assert "fp" in ax.get_xlabel().lower()


def test_plot_ioc_vs_surrogate(operating_curve):
    # Arrange
    # Act
    fig, ax = plots.ioc_vs_surrogate(operating_curve)
    # Assert
    assert "threshold" in ax.get_xlabel().lower()
    fig.savefig(f"{PLOT_DIR}/ioc_vs_surrogate.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def test_plot_sample_vs_alarm_scatter():
    # Arrange
    rng = np.random.default_rng(0)
    n = 15
    df = pd.DataFrame(
        {
            "patient": [f"P{i:02d}" for i in range(1, n + 1)],
            "roc_auc": rng.uniform(0.4, 0.9, n),
            "sensitivity": rng.uniform(0.0, 0.8, n),
        }
    )
    # Act
    fig, ax = plots.sample_vs_alarm_scatter(df)
    fig.savefig(f"{PLOT_DIR}/sample_vs_alarm_scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # Assert
    assert fig is not None


def test_plot_cadence_ablation():
    # Arrange
    seizures = np.array([3600.0, 7200.0])
    times = np.arange(0, 10800, 30.0)
    proba = np.where((times >= 2400) & (times < 3300), 0.9, 0.0)
    policies = [
        AlarmPolicy(
            sph_seconds=0, sop_seconds=600, cadence_seconds=c, refractory_seconds=600
        )
        for c in [30, 60, 120, 300, 600]
    ]
    df = forecasting.sweep_policies(
        proba,
        times,
        seizures,
        policies,
        total_recording_time=10800.0,
        n_surrogate=50,
    )
    # Act
    fig, ax = plots.cadence_ablation(df)
    fig.savefig(f"{PLOT_DIR}/cadence_ablation.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # Assert
    assert fig is not None


def test_plot_reliability_diagram():
    # Arrange
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.uniform(0, 1, n) < 0.3).astype(int)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, n), 0, 1)
    from scitex_seizure_metrics import calibration

    cal = calibration.calibration_report(y, p, n_bins=10)
    # Act
    fig, ax = plots.reliability_diagram(cal)
    fig.savefig(f"{PLOT_DIR}/reliability_diagram.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # Assert
    assert fig is not None


def test_plot_metric_correlation_heatmap():
    # Arrange
    rng = np.random.default_rng(0)
    n = 30
    base = rng.uniform(0.3, 0.9, n)
    df = pd.DataFrame(
        {
            "roc_auc": base + rng.normal(0, 0.05, n),
            "pr_auc": base + rng.normal(0, 0.07, n),
            "balanced_accuracy": base + rng.normal(0, 0.06, n),
            "mcc": base * 1.5 - 0.7 + rng.normal(0, 0.1, n),
            "sensitivity": base * 0.8 + rng.normal(0, 0.1, n),
            "fp_per_hour": rng.uniform(0, 5, n),
            "ioc": (base - 0.5) + rng.normal(0, 0.1, n),
        }
    )
    # Act
    fig, ax = plots.metric_correlation_heatmap(df, method="spearman")
    fig.savefig(
        f"{PLOT_DIR}/metric_correlation_heatmap.png", dpi=110, bbox_inches="tight"
    )
    plt.close(fig)
    # Assert
    assert fig is not None
