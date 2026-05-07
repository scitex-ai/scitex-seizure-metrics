"""scitex_seizure_metrics — unified evaluation for seizure detection / forecasting.

Public API:
- detection.evaluate(y_true, y_proba, threshold, fs, name)
- forecasting.evaluate(alarms, seizures, policy, total_recording_time, ...)
- forecasting.evaluate_stream(proba, times, seizures, policy, ...)
- forecasting.sweep_thresholds(proba, times, seizures, policy, ...)
- forecasting.sweep_policies(proba, times, seizures, policies, ...)
- forecasting.bootstrap_ci(values, n_boot, ci, rng_seed)
- bridge.sample_to_alarm(...) / bridge.alarm_to_sample(...)
- surrogates.{poisson, periodic, persistence} (registered)
- plots.{sensitivity_vs_fp_per_hour, sample_vs_alarm_scatter,
         cadence_ablation, ioc_vs_surrogate, metric_correlation_heatmap}

Data classes:
- AlarmPolicy(sph_seconds, sop_seconds, cadence_seconds,
              refractory_seconds, alarm_threshold, merge_consecutive,
              fp_denominator)
- MetricsReport (frozen single-row report; .to_frame(), .to_json())
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("scitex_seizure_metrics")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

from . import (
    adapters, bridge, calibration, detection, forecasting, papers,
    plots, report, surrogates,
)
from .policy import AlarmPolicy
from .report import MetricsReport

__all__ = [
    "__version__",
    "adapters", "bridge", "calibration", "detection", "forecasting",
    "papers", "plots", "report", "surrogates", "AlarmPolicy", "MetricsReport",
]
