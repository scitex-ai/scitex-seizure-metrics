r"""Empirical sensitivity vs time-in-warning (TiW) trade-off.

The field-standard forecasting view (Karoly et al. 2017, *Brain*
140:2169, Fig 6; Karoly et al. 2019): sweep the decision threshold and
trace seizure-level **sensitivity** against **time-in-warning** (the
fraction of recorded time the alarm is ON). The empirical complement to
:func:`scitex_seizure_metrics.bridge.sample_to_alarm` (analytic
sample-to-alarm bounds).

Public API:

- :func:`sensitivity_tiw_curve` — the empirical (threshold, TiW,
  sensitivity) curve + summary scalars (improvement-over-chance,
  sensitivity-at-target-TiW, TiW-at-target-sensitivity).
- :class:`SensitivityTiWCurve` — the result container.
- :func:`chance_sensitivity` — the chance diagonal (sensitivity == TiW).
- :func:`binomial_above_chance` / :func:`surrogate_above_chance` —
  above-chance significance tests at an operating point.
- :class:`TiWSignificance` — the significance result container.

The plotter lives in :func:`scitex_seizure_metrics.plots.sensitivity_tiw`
(package convention: all plots in ``plots.py``).

References
----------

- Karoly PJ et al., *Brain* 2017; 140: 2169-2182. doi:10.1093/brain/awx173
- Karoly PJ et al., *Lancet Neurology* 2019.
- Mormann F et al., *Brain* 2007; 130: 314-333.
- ``docs/math/sensitivity_tiw.md``.
"""

from __future__ import annotations

from ._curve import (
    SensitivityTiWCurve,
    monotone_upper_envelope,
    sensitivity_tiw_curve,
)
from ._inputs import seizures_from_labels
from ._significance import (
    TiWSignificance,
    binomial_above_chance,
    chance_sensitivity,
    surrogate_above_chance,
)

__all__ = [
    "sensitivity_tiw_curve",
    "SensitivityTiWCurve",
    "monotone_upper_envelope",
    "chance_sensitivity",
    "binomial_above_chance",
    "surrogate_above_chance",
    "TiWSignificance",
    "seizures_from_labels",
]
