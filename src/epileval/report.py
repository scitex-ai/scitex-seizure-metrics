"""MetricsReport — frozen dataclass that carries a full evaluation result.

Designed so the same object can be:
- serialised to JSON for `stx.io.save(rep.to_dict(), "metrics.json")`
- materialised as a one-row pandas DataFrame for stacking across patients/folds
- pretty-printed for terminal logs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MetricsReport:
    """Single-row evaluation result.

    Fields are deliberately flat so the report stacks cleanly across
    patients/folds via `pd.concat([r.to_frame() for r in reports])`.
    """

    # Identification
    name: str = ""                  # e.g. "P03_fold0"
    regime: str = ""                # "detection" | "forecasting"

    # Detection-style (continuous prediction → ranking metrics)
    roc_auc: float | None = None
    pr_auc: float | None = None
    brier: float | None = None
    balanced_accuracy: float | None = None
    mcc: float | None = None

    # Sample/event scoring (timescoring engine)
    sensitivity: float | None = None
    precision: float | None = None
    f1: float | None = None
    fp_per_day: float | None = None
    fp_per_hour: float | None = None
    n_ref_events: int | None = None
    n_tp: int | None = None
    n_fp: int | None = None

    # Forecasting (alarm-based)
    sph_seconds: float | None = None
    sop_seconds: float | None = None
    time_in_warning_frac: float | None = None
    ioc: float | None = None        # Improvement over Chance (vs surrogate)
    surrogate_sensitivity: float | None = None

    # Free-form extras (per-class scores, calibration table, etc.)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        d = self.to_dict()
        d.pop("extras")
        return pd.DataFrame([d])

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __str__(self) -> str:
        d = {k: v for k, v in self.to_dict().items()
             if v not in (None, {}, "")}
        lines = [f"MetricsReport({d.get('name', '?')}, {d.get('regime', '?')})"]
        for k, v in d.items():
            if k in ("name", "regime"):
                continue
            if isinstance(v, float):
                lines.append(f"  {k:<25} {v:.4f}")
            else:
                lines.append(f"  {k:<25} {v}")
        return "\n".join(lines)
