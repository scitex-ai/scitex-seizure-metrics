"""Forecasting-regime quick start — alarm-based metrics with explicit
AlarmPolicy.

Run:
    python examples/quick_start_forecasting.py
"""
from __future__ import annotations

import numpy as np

from scitex_seizure_metrics import AlarmPolicy, forecasting


def main() -> None:
    rng = np.random.default_rng(0)
    duration = 24 * 3600
    cadence = 60.0
    times = np.arange(0, duration, cadence)
    seizures = np.array([4 * 3600, 11 * 3600, 19 * 3600], dtype=float)

    # Simulated stream — noise + true preictal "ramp" 1 h before each seizure
    proba = rng.uniform(0, 0.3, size=times.size)
    for sz in seizures:
        ramp_mask = (times >= sz - 3600) & (times < sz - 600)
        proba[ramp_mask] += np.linspace(0.1, 0.7, ramp_mask.sum())
    proba = np.clip(proba, 0, 1)

    policy = AlarmPolicy(
        sph_seconds=300, sop_seconds=600, cadence_seconds=60,
        refractory_seconds=600, alarm_threshold=0.6,
    )
    rep = forecasting.evaluate_stream(
        proba, times, seizures, policy,
        total_recording_time=float(duration), n_surrogate=500,
    )
    print(rep)


if __name__ == "__main__":
    main()
