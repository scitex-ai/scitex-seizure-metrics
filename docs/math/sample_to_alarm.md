# Sample-to-Alarm Bridge: Mathematical Derivation

This document derives the analytic bounds implemented in
`scitex_seizure_metrics.bridge.sample_to_alarm` and its inverse
`bridge.alarm_to_sample`. The bounds let a per-window classification
report (sensitivity, specificity, prevalence) be translated into a
per-seizure alarm report (alarm sensitivity, false-positive rate per
hour) under an explicit alarm policy — and vice versa.

The motivation, after Andrade et al. 2024, is that the two evaluation
regimes can give very different verdicts on the same model on the same
data, and a paper that reports only one is not comparable to a paper
that reports only the other. This bridge gives the *analytic* envelope
that the other regime must live inside, given what the first regime
reported, without re-running anything.

## Setup and notation

Let a classifier emit a binary prediction $\hat{y} \in \{0, 1\}$ for
each fixed-length prediction window, with cadence $\Delta$ seconds
between consecutive predictions. Ground truth $y \in \{0, 1\}$ labels
each window as pre-ictal ($y = 1$) or non-preictal ($y = 0$).

Define the per-window classification quantities:

$$
s = \Pr(\hat{y} = 1 \mid y = 1) \quad \text{(sample sensitivity)}
$$

$$
\alpha = \Pr(\hat{y} = 1 \mid y = 0) = 1 - \text{specificity} \quad \text{(per-window FPR)}
$$

$$
\pi = \Pr(y = 1) \quad \text{(prevalence)}
$$

The alarm policy fixes:

| Symbol | Meaning | SSM field |
| ------ | ------- | --------- |
| $\Delta$ | seconds between prediction windows | `cadence_seconds` |
| $\text{SOP}$ | Seizure Occurrence Period (s) | `sop_seconds` |
| $R$ | refractory: minimum gap between alarms (s) | `refractory_seconds` |
| $T$ | total observation time (s) | (input to `evaluate`) |

We say "an alarm fires for seizure $i$" if at least one prediction
window inside the seizure's SOP is above threshold.

## Sample → Alarm

### Step 1. Number of independent prediction windows per SOP

In one SOP of duration $\text{SOP}$ seconds with prediction cadence
$\Delta$ seconds, the number of distinct prediction windows that could
fire an alarm for that seizure is

$$
K = \left\lceil \frac{\text{SOP}}{\Delta} \right\rceil .
$$

### Step 2. Prevalence-adjusted effective K

Low-prevalence streams may not actually *contain* $K$ pre-ictal-labelled
windows inside a given SOP — many of the $K$ windows could be
unlabelled as pre-ictal. The bridge therefore uses an effective $K$
that adjusts for prevalence:

$$
K_{\text{eff}} = \min\!\Big( K,\ \max(1, \operatorname{round}(K \cdot \pi)) \Big) .
$$

When $\pi = 1$ (every window inside the SOP is pre-ictal) this reduces
to $K_{\text{eff}} = K$. When $\pi$ is small (most windows inside the
SOP carry no pre-ictal label), $K_{\text{eff}}$ shrinks toward 1.

### Step 3. Alarm sensitivity bounds

Let "alarm fires for a given seizure" be the event that at least one of
the $K_{\text{eff}}$ candidate windows is above threshold. Under the
**independent-errors** assumption — the optimistic envelope — the
probability that none of the $K_{\text{eff}}$ windows fires is
$(1 - s)^{K_{\text{eff}}}$, so

$$
\boxed{\ \text{alarm\_sens}_{\text{upper}} = 1 - (1 - s)^{K_{\text{eff}}} \ }
$$

Under **fully-clustered errors** — the pessimistic envelope, where the
classifier's window-level decisions inside one SOP are perfectly
correlated, so either all $K_{\text{eff}}$ fire or none does — the
alarm probability collapses to the per-window sensitivity:

$$
\boxed{\ \text{alarm\_sens}_{\text{lower}} = s \ }
$$

These two bounds are tight envelopes: any real classifier whose
window-level errors have positive but partial correlation will fall
between them.

### Step 4. False-positive rate per hour

The naive predictions-per-hour budget is

$$
N_{\text{preds/h}} = \frac{3600}{\Delta} .
$$

Of these, a fraction $(1 - \pi)$ are non-pre-ictal-labelled, and the
classifier mis-fires on each of them with probability $\alpha$. So the
expected naive FP count per hour is

$$
\text{FP/h}_{\text{naive}} = \alpha \cdot N_{\text{preds/h}} \cdot (1 - \pi) .
$$

A refractory period $R > 0$ caps the maximum number of distinct alarms
per hour at $3600 / R$. The **upper bound** is the minimum of the two:

$$
\boxed{\ \text{FP/h}_{\text{upper}} = \min\!\Big(\, \alpha \cdot \tfrac{3600}{\Delta} \cdot (1 - \pi),\ \tfrac{3600}{R}\, \Big) \ }
$$

(When $R = 0$ the cap is dropped and the upper bound is just the naive
expression.)

The **lower bound** is reported as

$$
\boxed{\ \text{FP/h}_{\text{lower}} = 0 \ }
$$

by convention: a non-trivial lower bound requires an FP-correlation
parameter (autocorrelation length, burst statistics) that the per-window
metrics alone do not pin down.

## Alarm → Sample (inverse)

Given a published alarm-based pair $(\text{alarm\_sens}, \text{FP/h})$,
the same constraint runs in reverse to give the feasible per-window
metric ranges.

### Sensitivity bounds

Inverting Step 3's upper bound (independent errors) gives the smallest
per-window $s$ consistent with the reported alarm sensitivity:

$$
s_{\text{lower}} = 1 - (1 - \text{alarm\_sens})^{1 / K_{\text{eff}}}
$$

The trivial upper bound is obtained from Step 3's lower bound (fully
clustered errors), where $s = \text{alarm\_sens}$:

$$
s_{\text{upper}} = \min(1, \text{alarm\_sens}) .
$$

### Specificity bounds

Inverting the naive $\text{FP/h}$ expression — first applying the
refractory cap to the input —

$$
\text{FP/h}_{\text{eff}} = \min\!\Big(\, \text{FP/h},\ \tfrac{3600}{R}\, \Big)
$$

then solving for $\alpha$:

$$
\alpha_{\min} = \frac{\text{FP/h}_{\text{eff}}}{\, N_{\text{preds/h}} \cdot (1 - \pi)\, } .
$$

The upper bound on specificity is therefore

$$
\text{specificity}_{\text{upper}} = 1 - \alpha_{\min} ,
$$

and we report $\text{specificity}_{\text{lower}} = 0$ for the same
correlation-uncertainty reason as the FP/h lower bound.

## Properties and edge cases

- **Width of the alarm-sensitivity band.** The gap
  $\text{alarm\_sens}_{\text{upper}} - \text{alarm\_sens}_{\text{lower}}$
  grows with $K_{\text{eff}}$. For $K_{\text{eff}} = 1$ the bounds
  coincide at $s$ (one chance per seizure → no opportunity for
  independent retries). For large $K_{\text{eff}}$ the upper bound
  approaches 1 even for modest $s$, which is exactly the Andrade-2024
  observation that the two regimes can disagree.

- **Refractory dominance.** If $\alpha \cdot N_{\text{preds/h}} \cdot
  (1 - \pi) > 3600 / R$, the refractory cap binds and the classifier's
  $\alpha$ becomes invisible to the FP/h reporter: any further
  degradation in specificity is absorbed by the cap. The bridge
  surfaces this case via the `notes` field on
  `SampleToAlarmBounds`.

- **Prevalence assumptions.** $\pi$ is a single per-window scalar in
  this bridge. Real streams may have time-varying prevalence (e.g.,
  the diurnal seizure-rate periodicity Karoly 2017 documents). The
  bridge does not model that variation. A conservative practice is to
  evaluate the bridge under both the empirical $\pi$ and a low-$\pi$
  alternative ($\pi = 0.01$ for fully-streaming evaluation) and
  report the broader of the two envelopes.

- **`fp_per_hour_lower = 0` is a convention, not a theorem.** A
  classifier whose errors are independent will sit close to
  $\text{FP/h}_{\text{upper}}$; one whose errors are tightly clustered
  can produce far fewer distinct alarms. Without an extra parameter
  describing the clustering, "0" is the only lower bound we can
  justify analytically.

## Worked example

Suppose a PAC-based classifier reports $s = 0.6$, specificity $0.85$
($\alpha = 0.15$), prevalence $\pi = 0.5$ (balanced seizure /
interictal-control windows per ADR-0007 of the consuming project), with
$\text{SOP} = 1800\,\text{s}$, $\Delta = 30\,\text{s}$,
$R = 1800\,\text{s}$.

- $K = \lceil 1800 / 30 \rceil = 60$.
- $K_{\text{eff}} = \min(60,\ \max(1,\ \operatorname{round}(60 \cdot 0.5))) = 30$.
- $\text{alarm\_sens}_{\text{upper}} = 1 - 0.4^{30} \approx 1.000$.
- $\text{alarm\_sens}_{\text{lower}} = 0.6$.
- $N_{\text{preds/h}} = 120$.
- $\text{FP/h}_{\text{naive}} = 0.15 \cdot 120 \cdot 0.5 = 9.0$.
- $\text{FP/h}_{\text{cap}} = 3600 / 1800 = 2.0$.
- $\text{FP/h}_{\text{upper}} = \min(9.0,\ 2.0) = 2.0$ — *refractory
  dominates*.
- $\text{FP/h}_{\text{lower}} = 0$.

The bridge therefore reports: alarm sensitivity in $[0.6,\ \approx 1.0]$,
$\text{FP/h}$ in $[0,\ 2.0]$. The 40-point gap on alarm sensitivity is
the regime-disagreement region — exactly what Andrade 2024 says you
need to surface in any honest report.

## References

- Andrade et al. 2024 — *Sample- vs alarm-based perspectives on
  seizure-prediction performance.*
- Mormann et al. 2007 — *Seizure prediction: the long and winding
  road.* (false-prediction rate definition).
- Code: `src/scitex_seizure_metrics/bridge.py::sample_to_alarm` /
  `::alarm_to_sample`.
- Companion: `src/scitex_seizure_metrics/policy.py::AlarmPolicy` (the
  policy object that fixes $\Delta$, SOP, $R$, denominator
  convention).
