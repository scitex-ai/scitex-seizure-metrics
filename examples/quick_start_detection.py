"""Detection-regime quick start — per-window classification metrics.

Run:
    python examples/quick_start_detection.py
"""
from __future__ import annotations

import numpy as np

from epileval import detection


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    # Imbalanced ground truth — 5% positive rate (typical pre-ictal:inter-ictal)
    y_true = (rng.uniform(0, 1, n) < 0.05).astype(int)
    # Predictor with modest signal
    y_proba = np.clip(0.4 * y_true + rng.normal(0.3, 0.15, n), 0, 1)

    rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1,
                             name="quick_start")
    print(rep)


if __name__ == "__main__":
    main()
