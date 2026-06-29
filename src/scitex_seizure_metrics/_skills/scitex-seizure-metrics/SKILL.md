---
name: scitex-seizure-metrics
description: |
  [WHAT] Unified evaluation library for seizure detection and forecasting — sample-based metrics (AUROC, AUPRC, Brier, MCC), alarm-based metrics with a required AlarmPolicy (sensitivity, FP/hr, IoC, time-in-warning), and analytic bridge bounds between the two regimes. Paper-replica shims (Cook 2013, Karoly 2017, Kuhlmann 2018, Maturana 2020, Proix 2021, Stirling 2021, Andrade 2024) drop new methods onto each paper's exact metric axis.
  [WHEN] Use whenever you need to report a seizure-detection or seizure-forecasting method on more than one paper's metric set, audit the sample-vs-alarm regime gap (Andrade 2024), or pin every reproducibility knob (SPH, SOP, cadence, refractory, FP-denominator) with no silent defaults.
  [HOW] `from scitex_seizure_metrics import detection, forecasting, AlarmPolicy` — call `detection.evaluate(y_true, y_proba)` for sample-based, `forecasting.evaluate_stream(proba, times, seizures, AlarmPolicy(...))` for alarm-based, and `bridge.sample_to_alarm(...)` when only one regime was published.
primary_interface: python
interfaces: {python: 4, cli: 0, mcp: 0, skills: 3, hook: 0, http: 0}
tags: [scitex-seizure-metrics]
---

# scitex_seizure_metrics

`scitex_seizure_metrics` lets one method be reported on every prior paper's exact
metric set — sample-based AUROC/Brier, alarm-based sensitivity/FP-per-hour,
IoC vs surrogate, and analytic bounds between the two regimes — without
re-running anyone's pipeline.

## Sub-skills

* [01_installation.md](01_installation.md) — install scitex_seizure_metrics and its `[plots]` / `[test]` / `[dev]` / `[docs]` / `[all]` extras; umbrella `pip install scitex[scitex_seizure_metrics]`.
* [02_quick-start.md](02_quick-start.md) — end-to-end recipes for `detection.evaluate`, `forecasting.evaluate_stream` (with required `AlarmPolicy`), threshold + cadence sweeps, `bridge.sample_to_alarm`, and the Andrade 2024 paper-replica.
* [03_python-api.md](03_python-api.md) — public surface reference: `detection`, `forecasting`, `bridge`, `AlarmPolicy`, paper-replica shims.
* [04_forecasting-classification.md](04_forecasting-classification.md) — idiomatic v0.2.0 alarm-regime workflow: probabilities → `AlarmPolicy` → `forecasting.evaluate_stream`; choosing SPH/SOP; pinning an operating point by time-in-warning (`sensitivity_tiw`); reading the confusion-matrix scores + lead time; and the surrogate / IoC baseline.

## Quick reference

```python
from scitex_seizure_metrics import detection, forecasting, AlarmPolicy

# Sample-based
rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1)
rep.roc_auc, rep.pr_auc, rep.brier, rep.mcc

# Alarm-based — every reproducibility knob explicit
policy = AlarmPolicy(
    sph_seconds=300, sop_seconds=600, cadence_seconds=60,
    refractory_seconds=600, alarm_threshold=0.5,
    fp_denominator="interictal",
)
rep = forecasting.evaluate_stream(
    proba, times, seizures, policy,
    total_recording_time=24 * 3600,
)
rep.sensitivity, rep.fp_per_hour, rep.ioc, rep.time_in_warning_frac
```

## Umbrella access

```python
import scitex
scitex.scitex_seizure_metrics.detection  # same object as scitex_seizure_metrics.detection
```

## Design constants

- **`AlarmPolicy` is required** by every alarm-aware function — no silent
  defaults for SPH, SOP, cadence, refractory, alarm threshold, or
  `fp_denominator` ("total" vs "interictal"). Every reported number is
  reproducible from the policy alone.
- **`MetricsReport` carries both regimes** — same object exposes
  `roc_auc` / `brier` (sample) and `sensitivity` / `fp_per_hour` / `ioc`
  (alarm). No method needs to know which regime emitted it.
- **`bridge.sample_to_alarm` is analytic** — derives bounds (not point
  estimates) so the regime gap (Andrade 2024: 50/56 patients beat
  chance under sample, only 6/46 under alarm) is visible without
  re-running anyone's pipeline.
