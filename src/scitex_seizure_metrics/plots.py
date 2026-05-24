"""Plotting utilities to surface metric-to-metric relationships.

These plots make the sample-vs-alarm gap (Andrade 2024), threshold
sensitivity, cadence sensitivity, and IoC vs surrogate transparent.
All return (fig, ax); none save to disk — that's the caller's job.
"""
from __future__ import annotations

import numpy as np


def sensitivity_vs_fp_per_hour(sweep_df, *, ax=None,
                                acceptable_fp_per_hour: float | None = 0.15,
                                wearable_fp_per_hour: float | None = 0.042):
    """Operating-curve plot from forecasting.sweep_thresholds output.

    Plots sensitivity (y) vs FP/hr (x). Optional reference lines for
    Mormann's 0.15/h and the wearable target 0.042/h.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    df = sweep_df.sort_values("fp_per_hour")
    ax.plot(df["fp_per_hour"], df["sensitivity"],
            marker="o", linestyle="-", label="model")
    if acceptable_fp_per_hour is not None:
        ax.axvline(acceptable_fp_per_hour, color="grey", linestyle="--",
                   label=f"Mormann ({acceptable_fp_per_hour:g}/h)")
    if wearable_fp_per_hour is not None:
        ax.axvline(wearable_fp_per_hour, color="black", linestyle=":",
                   label=f"wearable ({wearable_fp_per_hour:g}/h)")
    ax.set_xlabel("FP per hour (interictal)")
    ax.set_ylabel("alarm-based sensitivity")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    return ax.figure, ax


def sample_vs_alarm_scatter(per_patient_df, *, ax=None,
                            x_metric: str = "roc_auc",
                            y_metric: str = "sensitivity"):
    """Reproduce the Andrade 2024 finding: per-patient sample-based AUC
    vs alarm-based sensitivity. Identity line shows the (false) hope of
    direct correspondence.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(per_patient_df[x_metric], per_patient_df[y_metric],
               s=40, alpha=0.85)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="identity")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5,
               label="chance sensitivity")
    ax.set_xlabel(f"sample-based {x_metric}")
    ax.set_ylabel(f"alarm-based {y_metric}")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    return ax.figure, ax


def cadence_ablation(sweep_df, *, ax=None, x: str = "cadence_s",
                     y: str = "fp_per_hour", logx: bool = True):
    """How does FP/hr (or any metric) move as we change the cadence?
    Input: forecasting.sweep_policies output sorted by cadence.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    df = sweep_df.sort_values(x)
    ax.plot(df[x], df[y], marker="s", linestyle="-")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax.figure, ax


def ioc_vs_surrogate(sweep_df, *, ax=None):
    """IoC (sensitivity − surrogate_sensitivity) across thresholds.
    Useful to see the threshold range where the model truly beats chance.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    df = sweep_df.sort_values("threshold")
    ax.plot(df["threshold"], df["sensitivity"], label="model sens",
            marker="o")
    ax.plot(df["threshold"], df["surrogate_sensitivity"],
            label="surrogate sens", linestyle="--", marker="x")
    ax.fill_between(df["threshold"], df["surrogate_sensitivity"],
                    df["sensitivity"], where=df["sensitivity"]
                    > df["surrogate_sensitivity"], alpha=0.2,
                    label="IoC > 0")
    ax.set_xlabel("alarm threshold")
    ax.set_ylabel("sensitivity")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.02)
    return ax.figure, ax


def reliability_diagram(cal_report, *, ax=None,
                        title: str = "Reliability diagram"):
    """Plot a reliability diagram from a CalibrationReport.

    The dashed identity line represents perfect calibration. Bin counts
    are shown via marker size.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    cnts = cal_report.bin_counts
    sizes = 30 + 200 * (cnts / max(1, cnts.max()))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="ideal")
    ax.plot(cal_report.bin_centers, cal_report.bin_observed,
            color="C0", linestyle="-", marker="o", markersize=0,
            label="model")
    ax.scatter(cal_report.bin_centers, cal_report.bin_observed,
               s=sizes, color="C0", edgecolor="white", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title)
    ece = cal_report.expected_calibration_error
    ax.text(0.05, 0.95, f"ECE={ece:.3f}\nBrier={cal_report.brier:.3f}\n"
                        f"Rel={cal_report.reliability:.3f}, "
                        f"Res={cal_report.resolution:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8)
    return ax.figure, ax


def metric_correlation_heatmap(per_patient_df, *, ax=None,
                                metrics=None, method: str = "spearman"):
    """Heatmap of metric-to-metric correlations across patients.
    Surfaces redundancy ("this metric tells us nothing new") and the
    sample-vs-alarm divergence axis.
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        candidates = ["roc_auc", "pr_auc", "balanced_accuracy", "mcc",
                      "sensitivity", "precision", "f1", "fp_per_hour",
                      "ioc", "time_in_warning_frac"]
        metrics = [m for m in candidates if m in per_patient_df.columns]
    sub = per_patient_df[metrics].select_dtypes(include="number")
    corr = sub.corr(method=method)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.figure.colorbar(im, ax=ax, label=f"{method} ρ")
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    return ax.figure, ax
