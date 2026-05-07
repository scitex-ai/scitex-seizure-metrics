---
description: |
  [TOPIC] scitex_seizure_metrics Quick Start
  [DETAILS] Minimal end-to-end recipes for the three core entry points — `detection.evaluate(y_true, y_proba)` for sample-based metrics (AUROC / AUPRC / Brier / MCC), `forecasting.evaluate_stream(proba, times, seizures, AlarmPolicy(...))` for alarm-based metrics (sensitivity / FP-per-hour / IoC / time-in-warning), and `bridge.sample_to_alarm(...)` for cross-paper conversion bounds. Plus the threshold + cadence sweep helpers and the Andrade 2024 paper-replica shim. Each recipe is a 3-8 line snippet that returns a `MetricsReport`.
tags: [scitex_seizure_metrics-quick-start]
---

# Quick start

## Sample-based evaluation (per-window classification)

```python
from scitex_seizure_metrics import detection

rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1)
print(rep.roc_auc, rep.pr_auc, rep.brier, rep.mcc, rep.balanced_accuracy)
```

`MetricsReport` carries every metric the audit runs reach for in one
object — no per-metric function chains, no surprise NaNs.

## Alarm-based evaluation (continuous probability stream)

```python
from scitex_seizure_metrics import forecasting, AlarmPolicy

policy = AlarmPolicy(
    sph_seconds=300,            # seizure-prediction horizon
    sop_seconds=600,            # seizure-occurrence period
    cadence_seconds=60,         # alarm-evaluation cadence
    refractory_seconds=600,     # post-alarm refractory window
    alarm_threshold=0.5,
    fp_denominator="interictal",  # Mormann tradition; "total" also valid
)
rep = forecasting.evaluate_stream(
    proba, times, seizures, policy,
    total_recording_time=24 * 3600,
    n_surrogate=1000,
)
print(rep.sensitivity, rep.fp_per_hour, rep.ioc, rep.time_in_warning_frac)
```

`AlarmPolicy` is mandatory — there are no silent defaults for any of its
knobs. Every reported number is reproducible from the policy alone.

## Threshold sweep + cadence ablation

```python
df_thr = forecasting.sweep_thresholds(proba, times, seizures, policy)

policies = [
    AlarmPolicy(**{**policy.__dict__, "cadence_seconds": c})
    for c in [30, 60, 120, 300]
]
df_cad = forecasting.sweep_policies(proba, times, seizures, policies)
```

## Bridge: sample → alarm bounds

When a paper reports only sample-based numbers and you need an
alarm-based comparison:

```python
from scitex_seizure_metrics import bridge

bnd = bridge.sample_to_alarm(
    sample_sensitivity=0.79,
    sample_specificity=0.85,
    sop_seconds=600,
    cadence_seconds=60,
    refractory_seconds=600,
)
print(bnd.alarm_sensitivity_upper, bnd.fp_per_hour_upper)
```

## Paper-replica shim (Andrade 2024 panel)

```python
from scitex_seizure_metrics.papers import andrade2024

out = andrade2024.metrics(
    y_true=labels, y_proba=preds,
    times_seconds=times, seizure_times=onsets,
)
print(out["sample_auroc"], out["alarm_sensitivity"], out["beats_chance_alarm"])
```

Reproduces the side-by-side sample-vs-alarm table from the paper. Other
shims: `cook2013`, `karoly2017`, `kuhlmann2018`, `maturana2020`,
`proix2021`, `stirling2021`.

## Umbrella access

```python
import scitex
scitex.scitex_seizure_metrics.detection.evaluate(...)   # same object as scitex_seizure_metrics.detection
```
