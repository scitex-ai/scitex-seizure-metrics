"""AlarmPolicy — explicit container for the alarm-generation knobs that
drive forecasting metrics. Required argument to all alarm-aware functions
in scitex_seizure_metrics; never silently defaulted, so every reported number can be
traced to the policy that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class AlarmPolicy:
    """Knobs that govern how a continuous prediction stream is converted
    into discrete alarms, and how those alarms are matched to seizures.

    Every alarm-based metric in scitex_seizure_metrics requires an explicit AlarmPolicy
    so reports are reproducible across papers.

    Args:
        sph_seconds: Seizure Prediction Horizon (lead time required between
            an alarm and the earliest valid seizure). Andrade et al. 2024.
        sop_seconds: Seizure Occurrence Period (validity window after SPH
            within which the seizure must occur).
        cadence_seconds: Time step between successive predictions in the
            continuous stream. 60 s ≈ once-per-minute, 360 s ≈ once-per-
            6-min. Defines the maximum theoretical alarm rate before
            refractory.
        refractory_seconds: Minimum gap between consecutive alarms. After
            an alarm fires, the next earliest alarm is suppressed until
            this much time has passed. Common choices: SOP, 30 min, 1 h.
        alarm_threshold: Probability threshold above which a window's
            prediction triggers an alarm-candidate.
        merge_consecutive: If True, runs of contiguous above-threshold
            windows count as one alarm (fired at the first window).
        fp_denominator: Whether FP/hr is normalised by total recording
            time or by interictal-only time (with seizure ± SOP windows
            removed). The Mormann tradition is "interictal".
    """

    sph_seconds: float
    sop_seconds: float
    cadence_seconds: float
    refractory_seconds: float
    alarm_threshold: float = 0.5
    merge_consecutive: bool = True
    fp_denominator: Literal["interictal", "total"] = "interictal"

    def __post_init__(self):
        if self.sph_seconds < 0:
            raise ValueError("sph_seconds must be >= 0")
        if self.sop_seconds <= 0:
            raise ValueError("sop_seconds must be > 0")
        if self.cadence_seconds <= 0:
            raise ValueError("cadence_seconds must be > 0")
        if self.refractory_seconds < 0:
            raise ValueError("refractory_seconds must be >= 0")
        if not (0 <= self.alarm_threshold <= 1):
            raise ValueError("alarm_threshold must be in [0, 1]")

    def describe(self) -> dict:
        """Compact dict suitable for serialising into a metric report."""
        return {
            "sph_s": self.sph_seconds,
            "sop_s": self.sop_seconds,
            "cadence_s": self.cadence_seconds,
            "refractory_s": self.refractory_seconds,
            "alarm_threshold": self.alarm_threshold,
            "merge_consecutive": self.merge_consecutive,
            "fp_denominator": self.fp_denominator,
        }
