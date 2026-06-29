# The Alarm Confusion Matrix: Definitions and the SPH/SOP Framework

This document defines the forecasting-regime confusion matrix implemented
in `scitex_seizure_metrics._classification` (shipped in v0.2.0) and wired
into `forecasting.evaluate` / `evaluate_stream`. It explains the
SPH/SOP alarm framework the matching rule rests on, the
convention-dependent **true-negative** definition (the subject of
[ADR-0001](../adr/0001-true-negative-for-alarm-based-seizure-warning.md)),
the **observed lead time**, and **when to reach for detection vs
forecasting metrics**. A fully worked numeric example closes the page.

Where the [sample-to-alarm bridge](sample_to_alarm.md) gives the
*analytic* envelope linking the two regimes and
[sensitivity vs time-in-warning](sensitivity_tiw.md) gives the
*empirical operating curve*, this page is about the *alarm-regime
confusion matrix itself* — the four counted cells and the standard
classifier scores derived from them.

## The SPH/SOP alarm framework

A seizure-warning system does not predict an instant; it raises an alarm
and claims a seizure will follow within a bounded window. Two policy
constants (Winterhalder/Schelter 2003; Mormann 2007) define that window,
both fields of `AlarmPolicy`:

| Symbol | Name | `AlarmPolicy` field | Meaning |
| ------ | ---- | ------------------- | ------- |
| $\text{SPH}$ | Seizure Prediction Horizon | `sph_seconds` | Minimum lead time. After an alarm at $t_a$, the seizure may **not** occur before $t_a + \text{SPH}$ (intervention time). |
| $\text{SOP}$ | Seizure Occurrence Period | `sop_seconds` | Validity window after the SPH. The seizure must occur within $[t_a + \text{SPH},\ t_a + \text{SPH} + \text{SOP}]$ for the alarm to count as correct. |

### Matching rule

A seizure at onset $t_s$ is **caught** if and only if at least one alarm
$t_a$ satisfies

$$
t_a + \text{SPH} \;\le\; t_s \;\le\; t_a + \text{SPH} + \text{SOP}.
$$

This is `_alarm.alarm_match`. Equivalently, the alarm's *validity window*
$[t_a + \text{SPH},\ t_a + \text{SPH} + \text{SOP}]$ must cover the
seizure onset.

## The four cells

Three of the four confusion-matrix cells follow directly from the
matching rule and the package's existing counts:

| Cell | Definition | Basis |
| ---- | ---------- | ----- |
| **TP** | seizures *caught* (≥ 1 covering alarm) | per-seizure |
| **FN** | seizures *not* caught $= n_\text{seizures} - \text{TP}$ | per-seizure |
| **FP** | alarms that catch no seizure | per-alarm |
| **TN** | interictal SOP-length opportunities with **no** false alarm | per-opportunity |

The first three are uncontested. The fourth — **TN** — has no canonical
unit for an alarm system, because alarms are discrete events, not a
per-instant decision. The package's convention (see
[ADR-0001](../adr/0001-true-negative-for-alarm-based-seizure-warning.md))
partitions the interictal time into non-overlapping SOP-length
"prediction opportunities":

$$
n_\text{opportunities} = \left\lfloor \frac{\text{interictal\_seconds}}{\text{SOP}} \right\rfloor,
\qquad
\text{TN} = \max\!\big(0,\; n_\text{opportunities} - \text{FP}\big).
$$

The interictal time is the **same** denominator the FP/hr rate uses with
`fp_denominator="interictal"` — each seizure's
$[t_s - \text{SOP} - \text{SPH},\ t_s + \text{SOP}]$ window removed
(Mormann tradition) — so every alarm score shares one coherent time
basis.

## Derived scores

$$
\begin{align*}
\text{sensitivity (recall)} &= \frac{\text{TP}}{\text{TP} + \text{FN}} &&\text{[per-seizure]}\\[4pt]
\text{specificity} &= \frac{\text{TN}}{\text{TN} + \text{FP}} &&\text{[per-opportunity]}\\[4pt]
\text{ppv (alarm precision)} &= \frac{\text{TP}}{\text{TP} + \text{FP}}\\[4pt]
\text{npv} &= \frac{\text{TN}}{\text{TN} + \text{FN}}\\[4pt]
\text{forecasting\_f1} &= \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}
\end{align*}
$$

These populate `MetricsReport.specificity` / `ppv` / `npv` /
`forecasting_f1`, with the raw `n_tn` and `n_opportunities` carried
alongside so the TN denominator is always visible.

**Two cautions, both enforced in code:**

1. **specificity / NPV scale with the SOP-opportunity convention.** They
   depend on the chosen opportunity length, so they are read *with*
   `n_tn` / `n_opportunities`. `ppv` and `forecasting_f1` do **not**
   depend on the TN convention and are directly comparable across studies.
2. **Undefined ratios are `NaN`, never a silent `0`** (`_safe_ratio`):
   any cell with a zero denominator (e.g. no seizures, no interictal
   time) returns NaN so "undefined" is never mistaken for "perfect" or
   "zero". This is the package-wide fail-loud rule.

## Observed lead time

Distinct from the SPH **constraint** (the *minimum required* gap), the
observed lead time is *what the system actually delivered*. For each
caught seizure,

$$
\text{lead} = t_s - t_{a,\text{earliest}},
$$

where $t_{a,\text{earliest}}$ is the **earliest** alarm whose validity
window covers the seizure. By construction
$\text{SPH} \le \text{lead} \le \text{SPH} + \text{SOP}$. Uncaught
seizures contribute no entry.

`observed_lead_times(alarms, seizures, sph, sop)` returns the per-seizure
array; `lead_time_summary` reduces it to mean / median / min / max /
`n_caught`. `MetricsReport` exposes `lead_time_mean` / `lead_time_median`,
with the full per-seizure array in `extras["lead_times_seconds"]` and
`lead_time_min` / `lead_time_max` / `n_caught` in `extras`. An empty
result (no seizure caught) summarises to **NaN**, never `0 s`, so an empty
catch is never misread as "0 s lead".

## When to use detection vs forecasting metrics

| You have… | You want… | Use | Key scores |
| --------- | --------- | --- | ---------- |
| per-window labels + probabilities | threshold-free ranking quality, calibration | `detection.evaluate` | AUROC, AUPRC, Brier, MCC, balanced accuracy |
| a continuous probability stream + seizure onsets + an explicit policy | clinically meaningful warning performance | `forecasting.evaluate_stream` | sensitivity, FP/hr, IoC, time-in-warning, **specificity/PPV/NPV/F1**, lead time |
| pre-computed alarm times (alarms from a different pipeline) | the same forecasting scores | `forecasting.evaluate` | as above |
| only one regime published in a paper | the other regime's feasible range | `bridge.sample_to_alarm` / `alarm_to_sample` | analytic bounds |

The forecasting confusion-matrix scores let an alarm system be placed on
any paper's full confusion-matrix axis without re-running its pipeline.
They are provided for completeness and cross-paper matching; because TN
dwarfs the other cells on a long interictal recording, specificity and
NPV are typically near 1 and are weak discriminators — `sensitivity`,
`ppv`, `forecasting_f1`, FP/hr and `IoC` remain the load-bearing scores.

## Worked example

Three seizures at $t_s \in \{3600,\ 7200,\ 18000\}\,$s. Three alarms at
$t_a \in \{3000,\ 6900,\ 12000\}\,$s. Policy: $\text{SPH} = 300\,$s,
$\text{SOP} = 600\,$s, 24 h recording, `fp_denominator="interictal"`.

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
print(rep.n_tp, rep.n_fp, rep.extras["n_fn"], rep.n_tn, rep.n_opportunities)
# 2 1 1 135 136
print(rep.sensitivity, rep.specificity, rep.ppv, rep.npv, rep.forecasting_f1)
# 0.667 0.9926 0.667 0.9926 0.667
print(rep.lead_time_mean, rep.extras["lead_times_seconds"])
# 450.0 [600.0, 300.0]
```

Step by step:

- **Alarm at 3000 → seizure at 3600.** Validity window
  $[3000 + 300,\ 3000 + 300 + 600] = [3300,\ 3900]$ covers 3600 → caught.
  Lead $= 3600 - 3000 = 600\,$s.
- **Alarm at 6900 → seizure at 7200.** Window $[7200,\ 7800]$ covers
  7200 → caught. Lead $= 7200 - 6900 = 300\,$s.
- **Alarm at 12000 → no seizure** in $[12300,\ 12900]$ → false positive.
- **Seizure at 18000** has no covering alarm → false negative.

So $\text{TP} = 2$, $\text{FP} = 1$, $\text{FN} = 1$. With ~22.7 h of
interictal time after removing the three seizure windows,
$n_\text{opportunities} = \lfloor 81\,900 / 600 \rfloor = 136$ and
$\text{TN} = 136 - 1 = 135$. Hence sensitivity $= 2/3 = 0.667$,
specificity $= 135/136 = 0.993$, PPV $= 2/3 = 0.667$,
NPV $= 135/136 = 0.993$, F1 $= 4/6 = 0.667$. Mean observed lead time
$= (600 + 300)/2 = 450\,$s — comfortably above the 300 s SPH the policy
*required*.

(The smaller, denominator-free variant
`_classification.alarm_classification(n_tp=2, n_fp=1, n_seizures=3,
interictal_seconds=36000, sop_seconds=600)` gives the same scores with
$n_\text{opportunities} = 60$, $\text{TN} = 59$ — illustrating directly
that specificity/NPV move with the interictal denominator while
PPV/sensitivity/F1 do not.)

## References

- Winterhalder M, Schelter B et al. (2003). The seizure prediction
  characteristic. *Epilepsy Behav* — SPH/SOP validity window.
- Mormann F et al. (2007). Seizure prediction: the long and winding road.
  *Brain* 130:314. doi:10.1093/brain/awl241.
- Snyder DE et al. (2008). The statistics of a practical seizure warning
  system. *J Neural Eng* 5:392.
- Andrade I, Teixeira C, Pinto M (2024). doi:10.3389/fnins.2024.1417748.
- [ADR-0001](../adr/0001-true-negative-for-alarm-based-seizure-warning.md)
  — the TN convention this page implements.
- Code: `src/scitex_seizure_metrics/_classification.py`,
  `src/scitex_seizure_metrics/forecasting.py`.
