"""Forecasting regime — alarm-based metrics with explicit AlarmPolicy.

Two entry points:
- evaluate(alarms, seizures, policy, ...) — accepts pre-computed alarm
  times; useful when alarms come from a different pipeline.
- evaluate_stream(proba, times, seizures, policy, ...) — accepts a
  continuous prediction stream (proba per timepoint); applies the
  policy's threshold + refractory + merging to produce alarms, then
  evaluates. This is the gold-standard entry point — every reported
  number is reproducible from (proba, times, policy).

Also provides:
- sweep_thresholds — sensitivity-vs-FP/hr ROC across a threshold sweep.
- sweep_policies   — same metrics across multiple AlarmPolicy values
                     (e.g. cadence ablation).

All metrics use the AlarmPolicy.fp_denominator convention. The Mormann-
tradition default ("interictal") excludes [seizure - sop - sph,
seizure + sop] windows from the FP/hr denominator.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from . import surrogates as _surrogates
from . import _alarm
from .policy import AlarmPolicy
from .report import MetricsReport


def evaluate(alarm_times, seizure_times, policy: AlarmPolicy, *,
             total_recording_time: float, n_surrogate: int = 1000,
             surrogate: str = "poisson", rng_seed: int = 0,
             name: str = "") -> MetricsReport:
    """Alarm-based forecasting evaluation under an explicit AlarmPolicy.

    Args:
        alarm_times: seconds-since-recording-start timestamps of alarms.
        seizure_times: seconds-since-recording-start timestamps of true
                       seizure onsets.
        policy: AlarmPolicy describing SPH/SOP/cadence/refractory/etc.
        total_recording_time: total recording duration (seconds).
        n_surrogate: number of random surrogates to estimate the chance
                     baseline for IoC.
        surrogate: name of registered surrogate ("poisson", "periodic",
                   "persistence", or any user-registered).
        rng_seed: surrogate RNG seed.
        name: identifier carried into the report.

    Returns:
        MetricsReport with the forecasting-regime fields populated.
    """
    alarms = np.sort(np.asarray(alarm_times, dtype=float).ravel())
    seizures = np.sort(np.asarray(seizure_times, dtype=float).ravel())
    if total_recording_time <= 0:
        raise ValueError("total_recording_time must be > 0")

    sc, au = _alarm.alarm_match(alarms, seizures,
                                policy.sph_seconds, policy.sop_seconds)
    tp = int(sc.sum())
    fp = int((~au).sum())

    # Time-in-warning: union of [a + SPH, a + SPH + SOP] across alarms.
    if alarms.size > 0:
        tiw_seconds = _alarm.union_length(np.column_stack([
            alarms + policy.sph_seconds,
            alarms + policy.sph_seconds + policy.sop_seconds,
        ]))
    else:
        tiw_seconds = 0.0
    tiw_frac = float(tiw_seconds / total_recording_time)

    # FP/hr denominator
    if policy.fp_denominator == "interictal":
        denom_seconds = _alarm.interictal_seconds(
            total_recording_time, seizures, policy.sop_seconds,
            policy.sph_seconds)
    else:
        denom_seconds = total_recording_time
    fp_per_hour = fp / max(1e-9, denom_seconds / 3600.0)

    sens = float(tp / seizures.size) if seizures.size > 0 else float("nan")

    # Surrogate IoC
    rng = np.random.default_rng(rng_seed)
    surro_fn = _surrogates.get(surrogate)
    surrogate_sens = []
    for _ in range(n_surrogate):
        rand_alarms = surro_fn(int(alarms.size),
                               max(total_recording_time - policy.sop_seconds
                                   - policy.sph_seconds, 1.0),
                               rng)
        sc_s, _ = _alarm.alarm_match(rand_alarms, seizures,
                                     policy.sph_seconds, policy.sop_seconds)
        surrogate_sens.append(sc_s.sum() / max(1, seizures.size))
    surro_mean = float(np.mean(surrogate_sens)) if surrogate_sens else 0.0

    rep = MetricsReport(
        name=name, regime="forecasting",
        sensitivity=sens, n_ref_events=int(seizures.size),
        n_tp=tp, n_fp=fp,
        fp_per_hour=fp_per_hour,
        fp_per_day=fp_per_hour * 24.0,
        sph_seconds=policy.sph_seconds, sop_seconds=policy.sop_seconds,
        time_in_warning_frac=tiw_frac,
        ioc=sens - surro_mean, surrogate_sensitivity=surro_mean,
        extras={"policy": policy.describe(),
                "interictal_seconds": denom_seconds,
                "n_alarms": int(alarms.size)},
    )
    return rep


def evaluate_stream(proba, times, seizure_times, policy: AlarmPolicy, *,
                    total_recording_time: float | None = None,
                    n_surrogate: int = 1000,
                    surrogate: str = "poisson",
                    rng_seed: int = 0, name: str = "") -> MetricsReport:
    """Continuous-stream evaluation entry point — gold standard.

    Threshold + dedupe (per AlarmPolicy) the proba stream, then call
    evaluate() on the derived alarms. Use this when you have per-window
    predictions and want fully-reproducible alarm-based metrics.
    """
    proba = np.asarray(proba, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if total_recording_time is None:
        total_recording_time = float(times.max() - times.min()) if times.size else 0.0
    alarms = _alarm.proba_stream_to_alarms(
        proba, times,
        threshold=policy.alarm_threshold,
        refractory_seconds=policy.refractory_seconds,
        merge_consecutive=policy.merge_consecutive,
    )
    return evaluate(alarms, seizure_times, policy,
                    total_recording_time=total_recording_time,
                    n_surrogate=n_surrogate, surrogate=surrogate,
                    rng_seed=rng_seed, name=name)


def sweep_thresholds(proba, times, seizure_times, policy: AlarmPolicy, *,
                     thresholds: Iterable[float] | None = None,
                     total_recording_time: float | None = None,
                     n_surrogate: int = 200,
                     surrogate: str = "poisson", rng_seed: int = 0,
                     name: str = ""):
    """Generate the sensitivity-vs-FP/hr operating curve.

    Returns a pandas DataFrame with one row per threshold; columns
    include sensitivity, fp_per_hour, time_in_warning_frac, ioc.
    """
    import pandas as pd

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        pol = AlarmPolicy(
            sph_seconds=policy.sph_seconds, sop_seconds=policy.sop_seconds,
            cadence_seconds=policy.cadence_seconds,
            refractory_seconds=policy.refractory_seconds,
            alarm_threshold=float(t),
            merge_consecutive=policy.merge_consecutive,
            fp_denominator=policy.fp_denominator,
        )
        rep = evaluate_stream(proba, times, seizure_times, pol,
                              total_recording_time=total_recording_time,
                              n_surrogate=n_surrogate, surrogate=surrogate,
                              rng_seed=rng_seed,
                              name=f"{name}@thresh={t:.2f}")
        d = rep.to_dict()
        d.update({"threshold": float(t)})
        rows.append(d)
    return pd.DataFrame(rows)


def sweep_policies(proba, times, seizure_times,
                   policies: list[AlarmPolicy], *,
                   total_recording_time: float | None = None,
                   n_surrogate: int = 200, surrogate: str = "poisson",
                   rng_seed: int = 0, name: str = ""):
    """Same evaluation across a list of AlarmPolicy values (e.g. cadence
    or refractory ablation). Returns DataFrame with one row per policy
    plus the policy.describe() columns."""
    import pandas as pd

    rows = []
    for i, pol in enumerate(policies):
        rep = evaluate_stream(proba, times, seizure_times, pol,
                              total_recording_time=total_recording_time,
                              n_surrogate=n_surrogate, surrogate=surrogate,
                              rng_seed=rng_seed, name=f"{name}_pol{i}")
        d = rep.to_dict()
        d.update(pol.describe())
        rows.append(d)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 1000, ci: float = 0.95,
                 rng_seed: int = 0):
    """Percentile bootstrap CI of the mean for a 1-D array."""
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(values, size=values.size, replace=True).mean()
    lo = np.quantile(boots, (1 - ci) / 2)
    hi = np.quantile(boots, 1 - (1 - ci) / 2)
    return float(values.mean()), float(lo), float(hi)
