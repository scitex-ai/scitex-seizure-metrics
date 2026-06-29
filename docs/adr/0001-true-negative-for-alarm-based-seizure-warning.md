# ADR-0001: True negative for an alarm-based seizure-warning system — SOP-length interictal opportunities (2026-06-29)

**Status:** Accepted (codifies the convention implemented in
`src/scitex_seizure_metrics/_classification.py`, shipped in v0.2.0).

## Context

The detection (sample-based) regime has an unambiguous confusion matrix:
every prediction window is one of {TP, FP, FN, TN}, so specificity, NPV
and the rest fall out directly. The forecasting (alarm-based) regime does
not. An alarm system emits a small number of discrete warnings over a
long recording, and seizures are rare events. Three of the four cells are
already well-defined and uncontested in the seizure-prediction literature
(Mormann 2007; Snyder 2008; Schelter/Winterhalder 2006):

- **TP** — a seizure that is *caught*: a seizure at `t_s` is a TP iff some
  alarm fires at `t_a` with `t_a + SPH <= t_s <= t_a + SPH + SOP`. This is
  `_alarm.alarm_match`'s `seizure_caught.sum()` — the package's existing
  alarm-vs-seizure matching rule.
- **FN** — a seizure that is *not* caught = `n_seizures - TP`.
- **FP** — an alarm that catches no seizure.

The fourth cell, **TN**, has no canonical unit. There is no natural,
agreed-upon "a correctly-quiet interictal moment" for an alarm system,
because alarms are events, not a per-instant decision. Yet `specificity`
and `NPV` — both of which reviewers routinely ask for, and which the
detection regime reports — are undefined without one. v0.2.0 had to pick a
convention to compute them, and that choice is genuinely convention-
dependent: it must be documented, not buried.

## Decision

**A true negative is an interictal "prediction opportunity" of length SOP
in which the system correctly stayed silent.**

The interictal time — the same denominator the FP/hr rate already uses,
with each seizure's `[t_s - SOP - SPH, t_s + SOP]` window removed
(Mormann tradition, `AlarmPolicy.fp_denominator="interictal"`) — is
partitioned into non-overlapping SOP-length opportunities:

```
n_opportunities = floor(interictal_seconds / SOP)
TN              = max(0, n_opportunities - FP)
```

i.e. every SOP-length interictal slot that did *not* contain a false alarm
is one TN. From the four cells:

```
sensitivity (recall) = TP / (TP + FN)        [per-seizure]
specificity          = TN / (TN + FP)        [per-opportunity]
ppv (alarm precision) = TP / (TP + FP)
npv                  = TN / (TN + FN)
f1 (forecasting_f1)  = 2·TP / (2·TP + FP + FN)
```

Any ratio with a zero denominator returns **NaN** (fail-loud), never a
silent 0, so "undefined" is never mistaken for "0".

**`n_tn` and `n_opportunities` are always reported alongside the scores**
(`MetricsReport.n_tn` / `n_opportunities`) so the TN denominator is never
hidden. `specificity` and `npv` are read *with* their denominator;
`ppv` and `forecasting_f1` do not depend on the TN convention at all and
are safe to compare across studies regardless.

## Considered and rejected

The alternatives below are all defensible; the choice between them changes
specificity/NPV by orders of magnitude on the same data, which is exactly
why it needs an ADR.

**Per-window TN (one TN per below-threshold prediction window).** This is
the detection-regime answer transplanted onto the alarm regime. Rejected
for the alarm scores because it inflates TN by the number of windows per
opportunity (`SOP/cadence`), pushing specificity to ≈ 1.000 for any
system and making it uninformative. It is, however, still available — it
*is* `detection.evaluate`'s specificity, which is why the package keeps
both regimes side by side rather than forcing one number.

**Per-hour TN (one TN per quiet interictal hour).** Equivalent to this
decision with the opportunity length fixed at 3600 s instead of SOP.
Rejected as the default because the opportunity length would then be
divorced from the alarm validity window: a 10-minute SOP and a 30-minute
SOP would score identical specificity, hiding the very knob the policy
exists to pin. (A caller who wants per-hour units can recover them by
reading `n_tn`/`n_opportunities` and rescaling.)

**Snyder 2008 / Schelter-Winterhalder 2006 "proportion of time under
false warning" (time-based specificity).** These report
`1 − (time-in-warning / interictal-time)` rather than a counted TN cell.
Rejected as the *cell* definition because it does not yield an integer
confusion matrix that PPV/NPV/F1 can share — but the package already
reports the time-based quantity separately as
`MetricsReport.time_in_warning_frac`, so no information is lost.

**No TN at all (report only sensitivity, PPV, FP/hr).** The honest
minimal-assumption option, and what the package did before v0.2.0.
Rejected because reviewers and cross-paper comparison routinely require
specificity/NPV, and refusing to compute them pushes every user to
re-implement the convention ad hoc — the exact fragmentation this package
exists to prevent. Reporting them *with* a documented, visible denominator
is more useful than omitting them.

## Consequences

**Positive.**

- specificity/NPV are now available in the alarm regime, so a method can
  be placed on any paper's full confusion-matrix axis without re-running
  its pipeline.
- The TN time unit (SOP) matches the alarm validity window and the FP/hr
  interictal denominator, so all alarm scores share one coherent time
  basis.
- `ppv` and `forecasting_f1` are convention-independent and directly
  comparable across studies.
- Fail-loud NaN on empty denominators means a degenerate run (no seizures,
  no interictal time) never silently reports a misleading 0.

**Negative / tradeoffs.**

- **specificity and NPV scale with the chosen opportunity length (SOP).**
  Two studies with different SOP are not directly comparable on
  specificity/NPV unless the SOP is held fixed. This is mitigated, not
  removed, by always reporting `n_tn` / `n_opportunities`: the reader can
  see the denominator and rescale. It is the unavoidable cost of giving an
  alarm system a TN cell at all.
- Because TN dwarfs the other cells on a long interictal recording,
  specificity is almost always very high and NPV very high. They are
  therefore weak discriminators between methods — `ppv`, `sensitivity`,
  `forecasting_f1`, `IoC` and FP/hr remain the load-bearing scores. The
  confusion-matrix scores are provided for completeness and paper-axis
  matching, not as the headline metric.
- `TN = max(0, n_opportunities − FP)` clips at 0: a system that fires more
  false alarms than there are opportunities (pathological / mis-policed)
  reports TN = 0 rather than a negative count. This is correct but means
  specificity = 0 in that regime rather than a more granular penalty.

## Implementation

| Element | Code |
| ------- | ---- |
| TN / opportunity definition | `_classification.py::alarm_classification` (module docstring carries the full convention) |
| Matching rule (TP/FP source) | `_alarm.py::alarm_match` |
| Interictal denominator | `_alarm.py::interictal_seconds` (reused for both FP/hr and the TN partition in `forecasting.py::evaluate`) |
| Fail-loud NaN | `_classification.py::_safe_ratio` |
| Reported denominators | `report.py::MetricsReport.n_tn` / `n_opportunities` |
| Wiring into reports | `forecasting.py::evaluate` (flows through `evaluate_stream` / `sweep_thresholds` / `sweep_policies` / `to_dict` / `to_frame`) |

## References

- Mormann F et al. (2007). Seizure prediction: the long and winding road.
  *Brain* 130:314. doi:10.1093/brain/awl241 — false-prediction-rate /
  proportion-of-time conventions.
- Snyder DE et al. (2008). The statistics of a practical seizure warning
  system. *J Neural Eng* 5:392.
- Winterhalder M, Schelter B et al. (2003/2006). The seizure prediction
  characteristic. *Epilepsy Behav* — SPH/SOP validity-window framework.
- Schulze-Bonhage A et al. (2020). Performance metrics for online seizure
  prediction. [PMC7340210](https://pmc.ncbi.nlm.nih.gov/articles/PMC7340210/).
- Andrade I, Teixeira C, Pinto M (2024). Sample- and alarm-based
  perspectives. *Front Neurosci*. doi:10.3389/fnins.2024.1417748.
- Code: `src/scitex_seizure_metrics/_classification.py`,
  `src/scitex_seizure_metrics/forecasting.py`.
- Conceptual docs page: `docs/sphinx/math/alarm_confusion_matrix.md`.
