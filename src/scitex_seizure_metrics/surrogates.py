"""Surrogate alarm-time generators for chance-baseline comparisons (IoC).

Each surrogate takes a description of the recording and returns synthetic
alarm times. Used by forecasting metrics to estimate Improvement-over-
Chance.

Built-in surrogates:
- poisson:    uniform-random alarm times at matched rate
- periodic:   alarms on a fixed period (circadian-style)
- persistence:alarm fires every X seconds since the last seizure
"""
from __future__ import annotations

from typing import Callable

import numpy as np

SurrogateFn = Callable[[int, float, "np.random.Generator"], np.ndarray]
_REGISTRY: dict[str, SurrogateFn] = {}


def register(name: str):
    """Decorator to register a surrogate generator."""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> SurrogateFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown surrogate {name!r}; "
                       f"registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


@register("poisson")
def poisson(n_alarms: int, total_seconds: float,
            rng: np.random.Generator) -> np.ndarray:
    """Uniform-random alarm times. The classic null model."""
    if n_alarms == 0:
        return np.array([])
    return np.sort(rng.uniform(0, total_seconds, size=n_alarms))


@register("periodic")
def periodic(n_alarms: int, total_seconds: float,
             rng: np.random.Generator) -> np.ndarray:
    """Equally-spaced alarms with a random phase offset.

    Stand-in for circadian-only forecasters (Karoly 2017): if a method
    can't beat regular ticks, the forecaster has no signal beyond the
    daily clock.
    """
    if n_alarms == 0 or total_seconds <= 0:
        return np.array([])
    period = total_seconds / n_alarms
    phase = rng.uniform(0, period)
    return np.clip(phase + np.arange(n_alarms) * period, 0, total_seconds)


@register("persistence")
def persistence(n_alarms: int, total_seconds: float,
                rng: np.random.Generator) -> np.ndarray:
    """Single alarm at a random time. Models 'always predict at last
    known seizure' style baselines collapsed to one decision."""
    if n_alarms == 0 or total_seconds <= 0:
        return np.array([])
    return np.array([rng.uniform(0, total_seconds)])


SECONDS_PER_DAY = 86_400.0


@register("circadian")
def circadian(n_alarms: int, total_seconds: float,
              rng: np.random.Generator,
              period_seconds: float = SECONDS_PER_DAY) -> np.ndarray:
    """Alarms entrained to a 24-h cycle at a random phase.

    Stand-in for circadian-only forecasters (Karoly 2017): if a method
    cannot beat regular daily ticks, it has no signal beyond the clock.
    """
    if n_alarms == 0 or total_seconds <= 0:
        return np.array([])
    n_periods = max(1, int(np.ceil(total_seconds / period_seconds)))
    phase = rng.uniform(0, period_seconds)
    base = phase + np.arange(n_periods) * period_seconds
    base = base[base < total_seconds]
    if base.size >= n_alarms:
        idx = np.linspace(0, base.size - 1, n_alarms).astype(int)
        return np.sort(base[idx])
    # Pad with random alarms uniformly within the recording.
    extra = rng.uniform(0, total_seconds, size=n_alarms - base.size)
    return np.sort(np.concatenate([base, extra]))


@register("multidien")
def multidien(n_alarms: int, total_seconds: float,
              rng: np.random.Generator,
              period_seconds: float = 7 * SECONDS_PER_DAY) -> np.ndarray:
    """Alarms entrained to a multi-day cycle (default 7 days).

    Karoly 2018 / Proix 2021 baseline — a method that exploits no per-
    seizure signal but tracks the multidien rhythm.
    """
    return circadian(n_alarms, total_seconds, rng,
                     period_seconds=period_seconds)


@register("from_history")
def from_history(n_alarms: int, total_seconds: float,
                 rng: np.random.Generator,
                 seizure_history_seconds: np.ndarray | None = None
                 ) -> np.ndarray:
    """Alarm at every (last_seizure + mean_interval) — Karoly 2019 style.

    Requires `seizure_history_seconds` (the past seizure timestamps in the
    same recording). Falls back to uniform-random if no history given.
    """
    if seizure_history_seconds is None or len(seizure_history_seconds) < 2:
        return poisson(n_alarms, total_seconds, rng)
    history = np.sort(np.asarray(seizure_history_seconds, dtype=float))
    interval = float(np.mean(np.diff(history)))
    if interval <= 0:
        return poisson(n_alarms, total_seconds, rng)
    last = history[-1]
    alarms = []
    t = last + interval
    while t < total_seconds and len(alarms) < n_alarms:
        alarms.append(t)
        t += interval
    if len(alarms) < n_alarms:
        extra = rng.uniform(0, total_seconds, size=n_alarms - len(alarms))
        alarms.extend(extra.tolist())
    return np.sort(np.asarray(alarms))
