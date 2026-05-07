"""Detection regime — per-window classification metrics.

This is what most of the literature calls "sample-based" evaluation
(Andrade 2024). Built on sklearn for the threshold-free metrics
(AUROC, AUPRC, Brier) and on timescoring.SampleScoring for the
threshold-dependent confusion-matrix metrics.

Use this when:
- Your model emits one probability per fixed-length window (clip-level
  classification, e.g. 10-min preictal-vs-interictal).
- You want the imbalance-aware metrics (AUPRC + balanced accuracy +
  MCC) that no NeuroVista paper has reported (gap noted in the
  literature review).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from timescoring.annotations import Annotation
from timescoring.scoring import SampleScoring

from .report import MetricsReport


def evaluate(
    y_true,
    y_proba,
    *,
    threshold: float = 0.5,
    fs: int = 1,
    name: str = "",
) -> MetricsReport:
    """Evaluate a continuous-output classifier on per-window labels.

    Args:
        y_true: 1-D binary array of ground-truth labels (1 = pre-ictal).
        y_proba: 1-D continuous score / probability matched to y_true.
        threshold: cutoff for the binary-mask metrics (sensitivity etc.).
                   AUROC/AUPRC/Brier are threshold-free.
        fs: sampling frequency of the label vector (Hz). Defaults to 1
            (one label per second, or per window if windows are 1-s
            spaced — adjust accordingly).
        name: identifier carried into the report.

    Returns:
        MetricsReport with the detection-regime fields populated.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if y_true.shape != y_proba.shape or y_true.ndim != 1:
        raise ValueError(
            f"y_true and y_proba must be matching 1-D arrays; got "
            f"{y_true.shape} vs {y_proba.shape}"
        )

    rep = MetricsReport(name=name, regime="detection")

    # Threshold-free
    has_both = y_true.min() == 0 and y_true.max() == 1
    if has_both:
        rep.roc_auc = float(roc_auc_score(y_true, y_proba))
        rep.pr_auc = float(average_precision_score(y_true, y_proba))
        rep.brier = float(brier_score_loss(y_true, y_proba))

    # Threshold-dependent
    y_pred = (y_proba >= threshold).astype(int)
    if has_both:
        rep.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        rep.mcc = float(matthews_corrcoef(y_true, y_pred))

    # Sample-based confusion-matrix metrics via timescoring
    ref = Annotation(y_true.astype(bool), fs=fs)
    hyp = Annotation(y_pred.astype(bool), fs=fs)
    ss = SampleScoring(ref, hyp, fs=fs)
    rep.sensitivity = float(ss.sensitivity) if not np.isnan(ss.sensitivity) else None
    rep.precision = float(ss.precision) if not np.isnan(ss.precision) else None
    rep.f1 = float(ss.f1) if not np.isnan(ss.f1) else None
    rep.fp_per_day = float(ss.fpRate)
    rep.fp_per_hour = float(ss.fpRate / 24.0)
    rep.n_ref_events = int(ss.refTrue)
    rep.n_tp = int(ss.tp)
    rep.n_fp = int(ss.fp)

    return rep
