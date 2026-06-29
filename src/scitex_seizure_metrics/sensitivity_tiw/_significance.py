r"""Chance baseline + above-chance significance for the sensitivity-TiW view.

A time-matched random alarm that is ON for a fraction :math:`w` of the
recording catches each seizure with probability :math:`w` in
expectation, so the chance curve is the diagonal
:math:`\text{sens}_{\text{chance}}(w) = w` (Karoly 2017 Fig 6; Mormann
2007 random-predictor baseline). A forecaster is significant only if its
operating point sits above that diagonal by more than sampling noise.

Two complementary tests are provided:

- :func:`binomial_above_chance` — exact binomial test treating each
  seizure as an independent Bernoulli(``tiw``) catch under the null.
- :func:`surrogate_above_chance` — Karoly-style permutation test using
  circular time-shift surrogates (no independence assumption; holds
  time-in-warning fixed while breaking seizure phase-locking).

References
----------

- Karoly PJ et al., *Brain* 2017; 140: 2169-2182.
- Karoly PJ et al., *Lancet Neurology* 2019.
- Mormann F et al., *Brain* 2007; 130: 314-333.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..policy import AlarmPolicy
from ._curve import sensitivity_of, tiw_fraction
from ._inputs import resolve_inputs


def chance_sensitivity(tiw: float) -> float:
    r"""Analytic chance sensitivity for a time-matched random alarm.

    A random warning ON for a fraction ``tiw`` of the recording overlaps
    each seizure's pre-ictal window with probability ``tiw`` in
    expectation, so :math:`\text{sens}_{\text{chance}} = \text{tiw}`
    (the diagonal). See ``docs/math/sensitivity_tiw.md``.
    """
    return float(min(1.0, max(0.0, tiw)))


@dataclass
class TiWSignificance:
    """Result of an above-chance test at one operating point.

    Attrs:
        tiw: time-in-warning fraction of the tested operating point.
        sensitivity: observed sensitivity at that point.
        chance_sensitivity: the diagonal value (== tiw).
        n_seizures: number of seizures (binomial n).
        n_caught: number of seizures caught (binomial successes).
        p_value: one-sided p-value for H1: sensitivity > chance.
        ci_low / ci_high: Wilson interval on the true catch rate.
        method: "binomial" (exact) or "surrogate" (permutation).
        n_surrogate: number of surrogate draws (surrogate method only).
    """

    tiw: float
    sensitivity: float
    chance_sensitivity: float
    n_seizures: int
    n_caught: int
    p_value: float
    ci_low: float
    ci_high: float
    method: str = "binomial"
    n_surrogate: int = 0


def _wilson_interval(k: int, n: int, ci: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (robust at edges)."""
    if n == 0:
        return float("nan"), float("nan")
    from scipy import stats  # stx-allow: STX-I002 (standalone pkg; no stx dep)

    z = float(stats.norm.ppf(1 - (1 - ci) / 2))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def binomial_above_chance(
    *, n_caught: int, n_seizures: int, tiw: float, ci: float = 0.95
) -> TiWSignificance:
    r"""Exact binomial test that observed sensitivity beats chance.

    Under the null (a time-matched random alarm) each of ``n_seizures``
    seizures is caught independently with probability
    :math:`p_0 = \text{tiw}`. The one-sided p-value is
    :math:`\Pr(X \ge n_{\text{caught}})`, :math:`X \sim
    \text{Binom}(n, p_0)`.

    Args:
        n_caught: seizures caught at the operating point.
        n_seizures: total seizures (binomial n).
        tiw: time-in-warning fraction (the null catch probability).
        ci: confidence level for the Wilson interval on the catch rate.

    Returns:
        :class:`TiWSignificance` with ``method == "binomial"``.

    Raises:
        ValueError: if ``n_seizures <= 0`` or ``tiw`` not in [0, 1].
    """
    if n_seizures <= 0:
        raise ValueError("n_seizures must be > 0")
    if not (0.0 <= tiw <= 1.0):
        raise ValueError("tiw must be in [0, 1]")
    n_caught = int(n_caught)
    p0 = min(1.0, max(0.0, float(tiw)))
    sens = n_caught / n_seizures

    from scipy import stats  # stx-allow: STX-I002 (standalone pkg; no stx dep)

    p_value = float(
        stats.binomtest(n_caught, n_seizures, p0, alternative="greater").pvalue
    )
    lo, hi = _wilson_interval(n_caught, n_seizures, ci)
    return TiWSignificance(
        tiw=p0,
        sensitivity=float(sens),
        chance_sensitivity=p0,
        n_seizures=int(n_seizures),
        n_caught=n_caught,
        p_value=p_value,
        ci_low=lo,
        ci_high=hi,
        method="binomial",
    )


def surrogate_above_chance(
    scores,
    policy: AlarmPolicy,
    *,
    threshold: float,
    labels=None,
    seizure_times=None,
    times=None,
    n_surrogate: int = 1_000,
    rng_seed: int = 0,
    ci: float = 0.95,
) -> TiWSignificance:
    r"""Permutation test: observed sensitivity vs TiW-matched surrogates.

    Karoly-style surrogate analysis using **circular time-shift**
    surrogates of the score stream. We measure the observed operating
    point at ``threshold`` (its TiW and sensitivity), then ``n_surrogate``
    times roll the score stream by a random offset and recompute
    sensitivity at the same threshold. A circular shift preserves the
    score multiset and the warning-block geometry *exactly* — hence the
    time-in-warning is unchanged — while destroying any phase-locking of
    the scores to the seizures. The one-sided p-value is

    .. math::

       p = \frac{1 + \#\{\text{surrogate sens} \ge \text{observed sens}\}}
                {1 + n_{\text{surrogate}}} .

    Complements :func:`binomial_above_chance`: the binomial test assumes
    independent per-seizure catches at rate ``tiw``; the surrogate test
    makes no independence assumption and honours the real warning
    autocorrelation (run lengths, SOP, refractory). Because TiW is held
    fixed, a forecaster that merely warns a lot — without locking to
    seizures — is correctly judged non-significant.

    Args:
        scores: per-window scores (same as ``sensitivity_tiw_curve``).
        policy: :class:`AlarmPolicy`.
        threshold: operating-point threshold to test.
        labels / seizure_times / times: input modes.
        n_surrogate: number of circular-shift surrogates.
        rng_seed: RNG seed for reproducibility.
        ci: confidence level for the Wilson interval.

    Returns:
        :class:`TiWSignificance` with ``method == "surrogate"``.

    Raises:
        ValueError: if there are no seizures.
    """
    scores, times, seizures, total_T, _cad = resolve_inputs(
        scores, labels=labels, seizure_times=seizure_times, times=times
    )
    if seizures.size == 0:
        raise ValueError("no seizures — significance is undefined")

    # Observed operating point: warning onsets = above-threshold windows
    # (the same warning-state view the curve uses).
    thr = float(threshold)
    above = scores >= thr
    obs_tiw = tiw_fraction(times[above], policy=policy, total_T=total_T)
    obs_sens = sensitivity_of(times[above], seizures, policy=policy)
    n_caught = int(round(obs_sens * seizures.size))

    # Null: circular time-shift of the score stream. Rolling the boolean
    # "above-threshold" mask preserves the exact number and run-structure
    # of warning windows (hence TiW), only changing where they sit
    # relative to the seizures.
    n = above.size
    rng = np.random.default_rng(rng_seed)
    ge = 0
    if n > 1 and above.any() and not above.all():
        shifts = rng.integers(1, n, size=n_surrogate)
        for sh in shifts:
            rolled = np.roll(above, int(sh))
            rand_sens = sensitivity_of(times[rolled], seizures, policy=policy)
            if rand_sens >= obs_sens - 1e-12:
                ge += 1
    else:
        # Degenerate: a constant warning state cannot be shifted into a
        # different operating point, so every surrogate equals the
        # observation -> not significant.
        ge = n_surrogate
    p_value = (1.0 + ge) / (1.0 + n_surrogate)
    lo, hi = _wilson_interval(n_caught, int(seizures.size), ci)
    return TiWSignificance(
        tiw=float(obs_tiw),
        sensitivity=float(obs_sens),
        chance_sensitivity=float(obs_tiw),
        n_seizures=int(seizures.size),
        n_caught=n_caught,
        p_value=float(p_value),
        ci_low=lo,
        ci_high=hi,
        method="surrogate",
        n_surrogate=int(n_surrogate),
    )
