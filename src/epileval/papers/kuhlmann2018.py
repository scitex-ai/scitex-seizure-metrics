"""Kuhlmann et al. 2018 (Brain) — Epilepsy Ecosystem replica metrics.

Paper: 'Epilepsyecosystem.org: crowd-sourcing reproducible seizure
prediction with long-term human intracranial EEG'. Reports clip-level
AUROC (10-min windows preictal vs interictal); pseudo-prospective
sensitivity vs proportion-time-in-warning.

Citation: Kuhlmann L et al., Brain 2018; 141: 2619–2630.
doi:10.1093/brain/awy210
"""
from __future__ import annotations

from .. import detection


def metrics(*, y_true, y_proba, name: str = "kuhlmann2018") -> dict:
    """Reproduce the Kaggle-style metric panel from Kuhlmann 2018.

    Args:
        y_true: per-clip binary labels.
        y_proba: per-clip predicted probabilities.

    Returns:
        dict with auroc, balanced_accuracy, mcc — the headline numbers
        from Table 2 of the paper.
    """
    rep = detection.evaluate(y_true, y_proba, name=name)
    return {
        "paper": "kuhlmann2018",
        "name": name,
        "auroc": rep.roc_auc,
        "balanced_accuracy": rep.balanced_accuracy,
        "mcc": rep.mcc,
        "pr_auc": rep.pr_auc,           # not in paper — bonus
        "brier": rep.brier,             # not in paper — bonus
    }
