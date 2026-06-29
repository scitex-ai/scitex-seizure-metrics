---
description: |
  [TOPIC] scitex-seizure-metrics — idiomatic forecasting (alarm-regime) workflow
  [DETAILS] The end-to-end v0.2.0 alarm path: per-window probabilities →
  `AlarmPolicy` (SPH/SOP/cadence/refractory/threshold/FP-denominator) →
  `forecasting.evaluate_stream`; choosing SPH/SOP; pinning an operating point
  by time-in-warning with `sensitivity_tiw`; reading the confusion-matrix
  scores (specificity/PPV/NPV/forecasting_f1 + n_tn/n_opportunities) and the
  observed lead time; and the surrogate / IoC chance baseline.
tags: [scitex-seizure-metrics-forecasting-classification]
---

# Forecasting (alarm-regime) workflow

The idiomatic path from a model's per-window scores to a reproducible,
clinically-meaningful alarm report. Everything below is reproducible from
`(proba, times, seizures, AlarmPolicy)` alone.

## 1. From per-window probabilities to alarms

`evaluate_stream` is the gold-standard entry point: it thresholds and
de-duplicates the probability stream *per the policy*, then evaluates. Do
not pre-threshold yourself — pass the raw probabilities so the policy
governs alarm derivation.

```python
import numpy as np
from scitex_seizure_metrics import AlarmPolicy, forecasting

policy = AlarmPolicy(
    sph_seconds=300,            # intervention lead time (min gap alarm→seizure)
    sop_seconds=600,            # validity window after the SPH
    cadence_seconds=60,         # spacing of the prediction stream
    refractory_seconds=600,     # min gap between consecutive alarms
    alarm_threshold=0.5,        # proba above this is an alarm candidate
    fp_denominator="interictal",  # Mormann tradition; "total" also valid
)
rep = forecasting.evaluate_stream(
    proba, times, seizures, policy,
    total_recording_time=24 * 3600,
    n_surrogate=1000,
)
```

Already have alarms from a different pipeline? Use `forecasting.evaluate`
with `alarm_times` directly — same scores, same report.

## 2. Choosing SPH and SOP

- **SPH (Seizure Prediction Horizon)** is the *clinical* constant: how
  much lead time an intervention needs. A seizure may not occur before
  `t_alarm + SPH` for the alarm to count. Pick it from the use case
  (e.g. 5 min to take fast-acting medication), not from the data.
- **SOP (Seizure Occurrence Period)** is the validity window after the
  SPH. Longer SOP → each alarm covers more time → easier to catch
  seizures but higher time-in-warning (more disruptive). It is the unit
  the true-negative count and time-in-warning are measured in.
- A seizure is **caught** iff some alarm satisfies
  `t_a + SPH <= t_s <= t_a + SPH + SOP`.

Hold SPH/SOP fixed when comparing methods — `specificity`/`NPV` scale
with SOP (see step 4).

## 3. Pinning an operating point by time-in-warning

`alarm_threshold` is the knob that trades sensitivity against burden. The
field-standard way to pick it is the **sensitivity vs time-in-warning**
curve (Karoly 2017 Fig 6), not FP/hr — refractory periods make per-hour
counts misleading. Time-in-warning (TiW) is the fraction of recording
spent under an active warning; chance is the diagonal
(sensitivity == TiW).

```python
from scitex_seizure_metrics import sensitivity_tiw

curve = sensitivity_tiw.sensitivity_tiw_curve(
    proba, policy, seizure_times=seizures, times=times, target_tiw=0.15,
)
print(curve.improvement_over_chance,      # area above the chance diagonal
      curve.sensitivity_at_target_tiw,    # sensitivity at 15% TiW
      curve.tiw_at_target_sensitivity)    # TiW needed for the target sensitivity

# Is the operating point above a time-matched coin?
sig = sensitivity_tiw.surrogate_above_chance(
    proba, policy, threshold=0.5, seizure_times=seizures, times=times,
)
print(sig.p_value, sig.ci_low, sig.ci_high)
```

Fix the threshold whose TiW you can clinically tolerate, then run
`evaluate_stream` at that threshold for the final report.

## 4. Reading the report

`evaluate_stream` returns a `MetricsReport` carrying, in the forecasting
regime:

| Field | Meaning |
| ----- | ------- |
| `sensitivity` | fraction of seizures caught (TP / n_seizures) |
| `fp_per_hour` | false alarms per hour over the chosen denominator |
| `time_in_warning_frac` | fraction of recording under active warning |
| `ioc` | improvement over chance (sensitivity − surrogate mean) |
| `specificity` | TN / (TN + FP) — **scales with SOP**, read with `n_opportunities` |
| `ppv` | alarm precision TP / (TP + FP) — convention-independent |
| `npv` | TN / (TN + FN) — scales with SOP |
| `forecasting_f1` | 2·TP / (2·TP + FP + FN) — convention-independent |
| `n_tn`, `n_opportunities` | the TN denominator, always visible |
| `lead_time_mean`, `lead_time_median` | observed warning delivered (seconds) |
| `extras["lead_times_seconds"]` | per-seizure observed lead times |
| `extras["n_fn"]`, `["n_caught"]`, `["lead_time_min"/"max"]` | extras |

The **true negative** is an interictal SOP-length "prediction
opportunity" with no false alarm (`n_opportunities = floor(interictal / SOP)`,
`TN = max(0, n_opportunities − FP)`). This is a deliberate, documented
convention — see ADR-0001 and the `_classification` module docstring.
Because TN dwarfs the other cells on a long recording, `specificity`/`NPV`
are near 1 and weak discriminators; lean on `sensitivity`, `ppv`,
`forecasting_f1`, `fp_per_hour`, and `ioc`.

**Observed lead time** is distinct from the SPH *constraint*: SPH is the
*required minimum*, lead time is what the system *actually delivered*
(always `SPH <= lead <= SPH + SOP`). Empty (no seizure caught) summarises
to **NaN**, never `0 s` — every undefined ratio in this package is NaN
(fail-loud), never a silent 0.

```python
print(rep.sensitivity, rep.fp_per_hour, rep.ioc, rep.time_in_warning_frac)
print(rep.specificity, rep.ppv, rep.npv, rep.forecasting_f1,
      rep.n_tn, rep.n_opportunities)
print(rep.lead_time_mean, rep.extras["lead_times_seconds"])

row = rep.to_frame()           # one-row DataFrame; stacks across patients/folds
```

## 5. The surrogate / IoC chance baseline

`ioc` (improvement over chance) compares the model's alarm sensitivity to
the *same statistic* recomputed under a random alarm generator that fires
the **same number of alarms** — so a model that simply warns constantly
does not look good. The default generator is Poisson; others are
registered in `surrogates` (`periodic`, `persistence`, `circadian`,
`multidien`, `from_history`). Increase `n_surrogate` for a tighter chance
estimate. `surrogate_above_chance` (step 3) gives a time-matched
permutation p-value holding TiW fixed.

```python
rep = forecasting.evaluate_stream(
    proba, times, seizures, policy,
    total_recording_time=24 * 3600,
    n_surrogate=1000, surrogate="poisson", rng_seed=0,
)
print(rep.ioc, rep.surrogate_sensitivity)
```

## 6. Sweeps

```python
# Operating curve across thresholds (carries all the v0.2.0 fields too)
df_thr = forecasting.sweep_thresholds(proba, times, seizures, policy)

# Cadence / refractory ablation
policies = [AlarmPolicy(**{**policy.__dict__, "cadence_seconds": c})
            for c in [30, 60, 120, 300]]
df_cad = forecasting.sweep_policies(proba, times, seizures, policies)
```

## Minimal end-to-end check (alarm-times entry point)

Three seizures, three alarms, the first two of which catch:

```python
import numpy as np
from scitex_seizure_metrics import AlarmPolicy, forecasting

seizures = np.array([3600.0, 7200.0, 18000.0])
alarms   = np.array([3000.0, 6900.0, 12000.0])
policy = AlarmPolicy(sph_seconds=300, sop_seconds=600,
                     cadence_seconds=60, refractory_seconds=600,
                     fp_denominator="interictal")

rep = forecasting.evaluate(alarms, seizures, policy,
                           total_recording_time=24 * 3600, n_surrogate=50)
# n_tp=2, n_fp=1, n_fn=1, n_tn=135, n_opportunities=136
# sensitivity=0.667, specificity=0.993, ppv=0.667, npv=0.993, f1=0.667
# lead_time_mean=450.0, lead_times_seconds=[600.0, 300.0]
```

See [docs/math/alarm_confusion_matrix.md](../../../../docs/math/alarm_confusion_matrix.md)
for the full derivation and the same worked example step by step.
