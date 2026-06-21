r"""Empirical sensitivity-vs-time-in-warning (TiW) operating curve.

This is the *empirical* complement to
:func:`scitex_seizure_metrics.bridge.sample_to_alarm` (the analytic
sample-to-alarm envelope). We take a per-window score stream plus an
:class:`AlarmPolicy`, sweep the decision threshold, and at each
threshold measure the two field-standard forecasting axes:

- **Time-in-warning (TiW)** — fraction of total recorded time the alarm
  is ON (the operational "cost" axis).
- **Sensitivity** — fraction of seizures *caught*, where a seizure is
  caught iff >=1 alarm fires inside its pre-ictal SPH/SOP window
  (seizure-level, NOT per-window).

The (TiW, sensitivity) curve is the view in Karoly et al. 2017 (Brain
140:2169) Fig 6 and Karoly et al. 2019 — a forecaster carries signal
beyond the clock only insofar as its curve sits above the chance
diagonal (sensitivity == TiW; see :mod:`._significance`).

References
----------

- Karoly PJ et al., *Brain* 2017; 140: 2169-2182 (Fig 6).
  doi:10.1093/brain/awx173
- Karoly PJ et al., *Lancet Neurology* 2019.
- Mormann F et al., *Brain* 2007; 130: 314-333.
- ``docs/math/sensitivity_tiw.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .. import _alarm
from ..policy import AlarmPolicy
from ._inputs import resolve_inputs


@dataclass
class SensitivityTiWCurve:
    """Empirical sensitivity-vs-time-in-warning operating curve.

    Arrays are ordered by ascending time-in-warning so the curve plots /
    integrates left-to-right against the chance diagonal.

    Attrs:
        thresholds: decision thresholds, aligned with the other arrays.
        tiw: time-in-warning fraction in [0, 1] at each threshold.
        sensitivity: seizure-level sensitivity in [0, 1] at each
            threshold.
        n_seizures: number of seizures (sensitivity denominator).
        improvement_over_chance: signed trapezoidal area between the
            curve and the chance diagonal, integrated over TiW. Positive
            means the forecaster is, on net, above a time-matched coin.
        sensitivity_at_target_tiw: best sensitivity achievable at
            TiW <= ``target_tiw`` (NaN if no operating point fits).
        tiw_at_target_sensitivity: smallest TiW at which the curve first
            reaches ``target_sensitivity`` (NaN if never reached).
        target_tiw: TiW operating point the scalar was read at.
        target_sensitivity: sensitivity operating point the scalar was
            read at.
        name: identifier carried from the caller.
        notes: free-form caveats.
    """

    thresholds: np.ndarray
    tiw: np.ndarray
    sensitivity: np.ndarray
    n_seizures: int
    improvement_over_chance: float
    sensitivity_at_target_tiw: float
    tiw_at_target_sensitivity: float
    target_tiw: float = 0.20
    target_sensitivity: float = 0.75
    name: str = ""
    notes: tuple[str, ...] = ()

    def to_frame(self):
        """One row per threshold as a pandas DataFrame (curve only)."""
        import pandas as pd

        return pd.DataFrame(
            {
                "name": self.name,
                "threshold": self.thresholds,
                "tiw": self.tiw,
                "sensitivity": self.sensitivity,
                "chance_sensitivity": self.tiw,  # the diagonal
            }
        )

    def summary(self) -> dict:
        """Flat dict of the summary scalars (no per-threshold arrays)."""
        return {
            "name": self.name,
            "n_seizures": int(self.n_seizures),
            "improvement_over_chance": float(self.improvement_over_chance),
            "target_tiw": float(self.target_tiw),
            "sensitivity_at_target_tiw": float(self.sensitivity_at_target_tiw),
            "target_sensitivity": float(self.target_sensitivity),
            "tiw_at_target_sensitivity": float(self.tiw_at_target_sensitivity),
        }


def _warning_intervals(alarm_starts: np.ndarray, *, policy: AlarmPolicy) -> np.ndarray:
    """[a + SPH, a + SPH + SOP] warning interval per warning onset."""
    if alarm_starts.size == 0:
        return np.empty((0, 2))
    return np.column_stack(
        [
            alarm_starts + policy.sph_seconds,
            alarm_starts + policy.sph_seconds + policy.sop_seconds,
        ]
    )


def tiw_fraction(
    alarm_starts: np.ndarray, *, policy: AlarmPolicy, total_T: float
) -> float:
    """Time-weighted fraction of the recording the alarm is ON.

    The warning state is the union of ``[a + SPH, a + SPH + SOP]`` over
    every warning onset (every above-threshold window), so a stream that
    is permanently above threshold is in warning ~100 % of the time and a
    silent stream 0 %. The union de-duplicates overlapping windows, so
    this is the field-standard "fraction of time in warning" (Karoly
    2017), not a per-alarm block count.
    """
    if alarm_starts.size == 0 or total_T <= 0:
        return 0.0
    on_seconds = _alarm.union_length(_warning_intervals(alarm_starts, policy=policy))
    return float(min(1.0, on_seconds / total_T))


def sensitivity_of(
    alarm_starts: np.ndarray, seizures: np.ndarray, *, policy: AlarmPolicy
) -> float:
    """Fraction of seizures caught (>=1 warning interval covers the onset).

    ``alarm_starts`` are warning onsets (above-threshold window times or
    deduped alarm times — both give the same caught set, since merging /
    refractory never change interval coverage).
    """
    if seizures.size == 0:
        return float("nan")
    sc, _ = _alarm.alarm_match(
        alarm_starts, seizures, policy.sph_seconds, policy.sop_seconds
    )
    return float(sc.sum() / seizures.size)


def area_above_diagonal(tiw: np.ndarray, sens: np.ndarray) -> float:
    """Signed trapezoidal area of (sensitivity - TiW) integrated over TiW.

    The curve is reduced to its upper envelope (max sensitivity per
    distinct TiW: a forecaster can always discard signal to move down,
    so the upper envelope is the meaningful frontier). Endpoints (0, 0)
    and (1, 1) are added so the integral spans the full [0, 1] TiW range
    and a curve lying exactly on the diagonal integrates to 0.
    """
    if tiw.size == 0:
        return 0.0
    order = np.argsort(tiw)
    t = tiw[order]
    s = sens[order]
    uniq_t, inv = np.unique(t, return_inverse=True)
    uniq_s = np.full(uniq_t.shape, -np.inf)
    np.maximum.at(uniq_s, inv, s)
    t_full = uniq_t
    s_full = uniq_s
    if t_full[0] > 0:
        t_full = np.concatenate([[0.0], t_full])
        s_full = np.concatenate([[0.0], s_full])
    if t_full[-1] < 1:
        t_full = np.concatenate([t_full, [1.0]])
        s_full = np.concatenate([s_full, [1.0]])
    diff = s_full - t_full
    # np.trapezoid (numpy>=2) replaced the deprecated np.trapz; fall back
    # only if trapezoid is absent, without eagerly touching the removed name.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(diff, t_full))


def monotone_upper_envelope(
    tiw: np.ndarray, sens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The achievable (TiW, sensitivity) frontier as a non-decreasing step.

    A forecaster can always discard signal to move *down* the curve, so
    the meaningful operating frontier at any time-in-warning budget is the
    running maximum of sensitivity over all operating points whose TiW
    does not exceed that budget. This collapses duplicate-TiW points to
    their best sensitivity and then takes the cumulative max, giving the
    monotone non-decreasing envelope that
    :func:`sensitivity_at_tiw` reads off (so a marker placed at
    ``(target_tiw, sensitivity_at_tiw(target_tiw))`` lands exactly on this
    envelope, never floating above or below the drawn line).

    Returns:
        ``(env_tiw, env_sens)`` sorted by ascending TiW. Empty input
        returns two empty arrays.
    """
    tiw = np.asarray(tiw, dtype=float)
    sens = np.asarray(sens, dtype=float)
    if tiw.size == 0:
        return np.empty(0), np.empty(0)
    uniq_t, inv = np.unique(tiw, return_inverse=True)
    best = np.full(uniq_t.shape, -np.inf)
    np.maximum.at(best, inv, sens)
    env = np.maximum.accumulate(best)
    return uniq_t, env


def sensitivity_at_tiw(tiw: np.ndarray, sens: np.ndarray, target_tiw: float) -> float:
    """Best sensitivity achievable at TiW <= ``target_tiw``.

    Field convention (Karoly): "what sensitivity can I get while staying
    inside my warning-time budget?" — max sensitivity over operating
    points whose TiW does not exceed the budget. This is the value of the
    :func:`monotone_upper_envelope` at ``target_tiw``.
    """
    mask = tiw <= target_tiw + 1e-12
    if not np.any(mask):
        return float("nan")
    return float(np.nanmax(sens[mask]))


def tiw_at_sensitivity(
    tiw: np.ndarray, sens: np.ndarray, target_sensitivity: float
) -> float:
    """Smallest TiW at which sensitivity first reaches the target."""
    mask = sens >= target_sensitivity - 1e-12
    if not np.any(mask):
        return float("nan")
    return float(np.nanmin(tiw[mask]))


def sensitivity_tiw_curve(
    scores,
    policy: AlarmPolicy,
    *,
    labels=None,
    seizure_times=None,
    times=None,
    thresholds: Iterable[float] | None = None,
    n_thresholds: int = 41,
    target_tiw: float = 0.20,
    target_sensitivity: float = 0.75,
    name: str = "",
) -> SensitivityTiWCurve:
    r"""Empirical sensitivity-vs-time-in-warning trade-off curve.

    Sweep the decision threshold over the score range; at each threshold
    convert the score stream into alarms (via the policy's
    refractory/merge rules) and measure time-in-warning and
    seizure-level sensitivity. The empirical complement to
    :func:`bridge.sample_to_alarm` and the curve behind Karoly 2017
    Fig 6.

    Args:
        scores: 1-D per-window predicted scores / probabilities.
        policy: :class:`AlarmPolicy` (SPH, SOP, refractory, merge rule).
            The policy's ``alarm_threshold`` is ignored — the threshold
            is what gets swept.
        labels: optional per-window binary pre-ictal labels (mode A).
        seizure_times: optional seizure onset timestamps (mode B). One
            of ``labels`` / ``seizure_times`` is required.
        times: optional per-window timestamps (seconds). Defaults to a
            unit-cadence index ``[0, 1, 2, ...]`` if omitted.
        thresholds: explicit thresholds to sweep. If None, ``n_thresholds``
            values spanning the observed score range are used.
        n_thresholds: number of thresholds when ``thresholds`` is None.
        target_tiw: TiW budget for ``sensitivity_at_target_tiw``
            (Karoly's common 0.20 / 20 %).
        target_sensitivity: target for ``tiw_at_target_sensitivity``.
        name: identifier carried into the result.

    Returns:
        :class:`SensitivityTiWCurve`.

    Raises:
        ValueError: neither labels nor seizure_times given, or shape
            mismatch.
    """
    scores, times, seizures, total_T, _cadence = resolve_inputs(
        scores, labels=labels, seizure_times=seizure_times, times=times
    )

    if thresholds is None:
        lo = float(np.nanmin(scores))
        hi = float(np.nanmax(scores))
        if hi <= lo:
            # Degenerate constant-score stream: bracket the single
            # achievable operating point with all-on / all-off.
            thr = np.array([lo - 1e-9, hi + 1e-9])
        else:
            span = hi - lo
            # Span just below min .. just above max so the extreme
            # all-on and all-off operating points are both represented.
            thr = np.linspace(lo - 1e-9 * span, hi + 1e-9 * span, n_thresholds)
    else:
        thr = np.asarray(list(thresholds), dtype=float)

    tiw_vals = np.empty(thr.size)
    sens_vals = np.empty(thr.size)
    for i, t in enumerate(thr):
        # Warning onsets = every above-threshold window. Time-in-warning
        # and the caught set are both properties of the warning *state*,
        # so they use the per-window onsets directly (independent of alarm
        # merging / refractory, which only matter for FP/alarm counts).
        onsets = times[scores >= float(t)]
        tiw_vals[i] = tiw_fraction(onsets, policy=policy, total_T=total_T)
        sens_vals[i] = sensitivity_of(onsets, seizures, policy=policy)

    order = np.argsort(tiw_vals, kind="stable")
    thr_o = thr[order]
    tiw_o = tiw_vals[order]
    sens_o = sens_vals[order]

    ioc = area_above_diagonal(tiw_o, sens_o)
    sens_at = sensitivity_at_tiw(tiw_o, sens_o, target_tiw)
    tiw_at = tiw_at_sensitivity(tiw_o, sens_o, target_sensitivity)

    notes: list[str] = []
    if seizures.size == 0:
        notes.append("no seizures — sensitivity is undefined (NaN)")
    elif seizures.size == 1:
        notes.append("single seizure — sensitivity is 0 or 1 only")
    if tiw_o.size and np.nanmax(tiw_o) < target_tiw:
        notes.append(
            f"curve never reaches target_tiw={target_tiw:g}; "
            "sensitivity_at_target_tiw is the best within budget"
        )

    return SensitivityTiWCurve(
        thresholds=thr_o,
        tiw=tiw_o,
        sensitivity=sens_o,
        n_seizures=int(seizures.size),
        improvement_over_chance=ioc,
        sensitivity_at_target_tiw=sens_at,
        tiw_at_target_sensitivity=tiw_at,
        target_tiw=float(target_tiw),
        target_sensitivity=float(target_sensitivity),
        name=name,
        notes=tuple(notes),
    )
