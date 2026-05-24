---
description: |
  [TOPIC] scitex-seizure-metrics — Python API
  [DETAILS] Public surface: `detection.evaluate`, `forecasting.evaluate` /
  `forecasting.evaluate_stream`, `bridge.sample_to_alarm`, and the required
  `AlarmPolicy` configuration object that pins SPH, SOP, cadence,
  refractory, and FP-denominator with no silent defaults.
tags: [scitex-seizure-metrics-python-api]
---

# Python API

## Detection (sample-based)

```python
from scitex_seizure_metrics import detection

m = detection.evaluate(
    y_true=labels,        # array of 0/1 per window
    y_proba=preds,        # predicted probability per window
    fs=256,               # sampling rate (Hz) for FP-per-hour conversion
)
print(m["sensitivity"], m["fp_per_hour"], m["auroc"], m["brier"])
```

Outputs every sample-based metric used in the literature (AUROC, AUPRC,
Brier, MCC, sensitivity at fixed specificity, ...). All values fixed
to a single window size; no implicit smoothing.

## Forecasting (alarm-based)

```python
from scitex_seizure_metrics import forecasting, AlarmPolicy

policy = AlarmPolicy(
    sph_seconds=300,       # seizure prediction horizon
    sop_seconds=600,       # seizure occurrence period
    refractory_seconds=60,
    fp_denominator="interictal_hours",
)
f = forecasting.evaluate_stream(
    proba=alarm_probs,
    times=alarm_times,
    seizures=onset_times,
    policy=policy,
)
print(f["ioc"], f["sensitivity"], f["fp_per_hour"], f["time_in_warning_pct"])
```

`AlarmPolicy` is required — there are no silent defaults. Callers
must pin every reproducibility knob explicitly.

## Bridge (sample → alarm)

When a paper reports only sample-based metrics, `bridge.sample_to_alarm`
gives analytic bounds (best/worst case) on the corresponding alarm-based
sensitivity / FP-per-hour given the chosen `AlarmPolicy`:

```python
from scitex_seizure_metrics import bridge

bounds = bridge.sample_to_alarm(
    sample_metrics=m,
    policy=policy,
    seizure_count=42,
    interictal_hours=120.0,
)
```

## Paper-replica shims

`scitex_seizure_metrics.papers.<author><year>` exposes drop-in metric
axes matching each paper's exact reporting convention:

- `cook2013`, `karoly2017`, `kuhlmann2018`, `maturana2020`,
  `proix2021`, `stirling2021`, `andrade2024`

Each shim wraps the unified `detection` / `forecasting` core with the
same denominators and label conventions the original paper used.
