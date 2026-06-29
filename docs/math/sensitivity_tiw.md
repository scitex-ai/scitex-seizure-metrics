# Sensitivity vs Time-in-Warning: Mathematical Derivation

This document derives the empirical sensitivity-vs-time-in-warning
trade-off implemented in
`scitex_seizure_metrics.sensitivity_tiw.sensitivity_tiw_curve`, its
chance baseline `chance_sensitivity`, and the two above-chance tests
`binomial_above_chance` and `surrogate_above_chance`.

Where the
[sample-to-alarm bridge](sample_to_alarm.md) gives the *analytic*
envelope linking per-window classification metrics to alarm-based ones,
this view is the *empirical* operating curve: feed an actual prediction
stream and an explicit alarm policy, sweep the decision threshold, and
read off the curve. It is the field-standard forecasting figure — Karoly
et al. 2017 (*Brain* 140:2169) Fig 6 and Karoly et al. 2019 — used to
judge a forecaster against a time-matched random predictor.

## Setup and notation

Let a forecaster emit a continuous score $x_j \in \mathbb{R}$ for each
prediction window $j$ at timestamp $t_j$ (cadence $\Delta$ seconds).
Ground truth is either per-window pre-ictal labels $y_j \in \{0, 1\}$ or
a set of seizure onset times $\{\tau_i\}_{i=1}^{N}$.

The alarm policy (`AlarmPolicy`) fixes:

| Symbol | Meaning | SSM field |
| ------ | ------- | --------- |
| $\text{SPH}$ | Seizure Prediction Horizon (s) | `sph_seconds` |
| $\text{SOP}$ | Seizure Occurrence Period (s) | `sop_seconds` |
| $R$ | refractory: minimum gap between alarms (s) | `refractory_seconds` |
| $T$ | total recording time (s) | derived from `times` |

Given a threshold $\theta$, a window is *above threshold* iff
$x_j \ge \theta$. Above-threshold windows are deduped into discrete
alarm times $\{a_k(\theta)\}$ by the policy's merge + refractory rules
(`_alarm.proba_stream_to_alarms`). Each alarm $a_k$ asserts a **warning
interval**

$$
W_k = [\,a_k + \text{SPH},\ a_k + \text{SPH} + \text{SOP}\,] ,
$$

the window in which it claims the next seizure will occur.

## Time-in-warning

The **time-in-warning** at threshold $\theta$ is the fraction of the
recording covered by the union of warning intervals (the union so that
overlapping / refractory-spaced alarms never double-count time):

$$
\boxed{\ w(\theta) = \frac{\big|\, \bigcup_k W_k(\theta) \,\big|}{T}\ }
\quad\in [0, 1] .
$$

It is monotone non-increasing in $\theta$: raising the threshold can only
remove above-threshold windows, hence only shrink the warning union.

## Sensitivity (seizure-level, SOP-aware)

A seizure at onset $\tau_i$ is **caught** at threshold $\theta$ iff at
least one alarm's warning interval covers it, i.e. some $a_k$ satisfies
$a_k + \text{SPH} \le \tau_i \le a_k + \text{SPH} + \text{SOP}$. This is
the standard SPH/SOP matching (`_alarm.alarm_match`) — *not* a per-window
hit rate. The **sensitivity** is the caught fraction:

$$
\boxed{\ \text{sens}(\theta)
   = \frac{1}{N}\sum_{i=1}^{N}
     \mathbb{1}\!\left[\exists\, k:\ \tau_i \in W_k(\theta)\right]\ } .
$$

Like $w$, it is monotone non-increasing in $\theta$.

The curve is the ordered set $\{(w(\theta), \text{sens}(\theta))\}$,
returned sorted by ascending $w$.

## Chance baseline: the diagonal

Consider a **time-matched random alarm** that is ON for a fraction $w$ of
the recording, placed independently of the seizures. For one seizure, the
probability that its (fixed-length) pre-ictal window overlaps the random
warning equals the fraction of admissible placements that hit it, which
to first order is the warning duty cycle $w$. Hence each seizure is
caught with probability $w$ in expectation, and

$$
\boxed{\ \text{sens}_{\text{chance}}(w) = w\ } .
$$

So **chance is the diagonal** $\text{sens} = w$. A forecaster carries
information beyond the clock only where its curve lies *above* the
diagonal. (This is the precise sense in which Karoly 2017 Fig 6 reads a
forecaster: distance above the diagonal at a usable time-in-warning.)

### Improvement over chance (area summary)

Reduce the curve to its upper envelope $\hat{s}(w) = \max\{\text{sens} :
w(\theta) = w\}$ (a forecaster can always throw away signal to slide down
to a lower operating point, so the envelope is the achievable frontier),
anchored at the trivial points $(0, 0)$ and $(1, 1)$. The **improvement
over chance** is the signed area between the envelope and the diagonal:

$$
\boxed{\ \text{IoC}_{\text{area}}
   = \int_0^1 \big[\hat{s}(w) - w\big]\,\mathrm{d}w\ } .
$$

It is $0$ for a curve lying exactly on the diagonal, positive above it,
and bounded by $\tfrac{1}{2}$ for the perfect corner curve $(0,0) \to
(0,1) \to (1,1)$. This is an AUC-like scalar analogous to "AUC $-\ 0.5$".

### Operating-point scalars

Two readouts at fixed operating points (Karoly-style "sensitivity at
X % time-in-warning"):

- **Sensitivity at a target time-in-warning** $w^\*$ (default
  $w^\* = 0.20$): the best sensitivity within the warning-time budget,
  $\max\{\text{sens}(\theta) : w(\theta) \le w^\*\}$.
- **Time-in-warning at a target sensitivity** $s^\*$ (default
  $s^\* = 0.75$): the smallest $w$ reaching it,
  $\min\{w(\theta) : \text{sens}(\theta) \ge s^\*\}$.

## Significance: is the curve above chance?

### Binomial test (independent-catch null)

At a chosen operating point with time-in-warning $w$ and $c$ of $N$
seizures caught, the null model is a time-matched random alarm catching
each seizure independently with probability $p_0 = w$. The one-sided
$p$-value for $H_1:\text{sens} > \text{chance}$ is the upper binomial
tail

$$
\boxed{\ p = \Pr\!\big(X \ge c\big),\quad X \sim \text{Binom}(N, p_0=w)\ } .
$$

`binomial_above_chance` returns this exact tail (`scipy.stats.binomtest`,
`alternative="greater"`) plus a Wilson score interval on the true catch
rate. The independence assumption is reasonable when seizures are
well-separated relative to the SOP.

### Surrogate / permutation test (Karoly-style)

When the warning autocorrelation matters (long SOP, strong refractory,
bursty scores), the independence assumption can be too generous.
`surrogate_above_chance` instead uses **circular time-shift surrogates**
of the above-threshold mask: roll the mask by a random offset $B$ times
and recompute sensitivity at the same threshold. A circular shift leaves
the score multiset and the warning run-structure *exactly* unchanged —
so the time-in-warning $w$ is held fixed — while breaking any
phase-locking of the scores to the seizures. The rank-based $p$-value is

$$
\boxed{\ p = \frac{1 + \#\{b : \text{sens}^{(b)}_{\text{surr}}
                       \ge \text{sens}_{\text{obs}}\}}{1 + B}\ } .
$$

The $+1$ in numerator and denominator is the standard add-one correction
so $p > 0$ always. Because TiW is held fixed by construction, a
forecaster that merely *warns a lot* (high duty cycle but no
seizure-locking) is correctly judged non-significant — exactly the
failure mode a naive random-block null would miss. This is the empirical
analogue of the time-shift surrogates used throughout the seizure-
prediction literature (Andrzejak/Mormann) and of the IoC-vs-surrogate
machinery the package already uses for `forecasting.evaluate`.

## Properties and edge cases

- **Monotonicity.** Both $w(\theta)$ and $\text{sens}(\theta)$ are
  non-increasing in $\theta$; the highest threshold gives the
  lowest-left operating point and the lowest threshold the
  upper-right. The synthetic tests assert this.

- **All-positive scores** (every window above every swept threshold):
  the alarm is permanently ON, $w \to 1$, $\text{sens} \to 1$ — the
  top-right corner, exactly on the diagonal there.

- **All-negative scores** (no window ever fires): $w = 0$,
  $\text{sens} = 0$ — the origin, on the diagonal.

- **Single seizure** ($N = 1$): sensitivity is $0$ or $1$ only; the
  binomial test reduces to $p = w$ when caught (one Bernoulli trial),
  which is correctly under-powered — a single catch at high
  time-in-warning is not significant. `sensitivity_tiw_curve` flags
  this in `notes`.

- **No seizures** ($N = 0$): sensitivity is undefined (`NaN`);
  `surrogate_above_chance` raises, and the curve carries a `notes`
  caveat.

## Worked example

Take a once-per-minute forecaster ($\Delta = 60\,\text{s}$) over
$T = 24\,\text{h}$ with $N = 10$ seizures, policy $\text{SPH} = 0$,
$\text{SOP} = 600\,\text{s}$, $R = 600\,\text{s}$. Suppose at a usable
threshold the alarm is ON for $w = 0.20$ (4.8 h of warning across the
day) and catches $c = 8$ of the $10$ seizures, so
$\text{sens} = 0.80$.

- Chance at this point: $\text{sens}_{\text{chance}} = w = 0.20$.
- Binomial $p$-value: $\Pr(X \ge 8),\ X \sim \text{Binom}(10, 0.20)$.
  The mean of the null is $2$ catches, so $8$ catches is far in the
  upper tail — $p \approx 7.8\times10^{-5}$. The forecaster is highly
  significant: it sits $0.60$ above the diagonal at a $20\,\%$
  time-in-warning budget.
- A perfect forecaster would reach the top-left as quickly as its SOP
  duty cycle allows; this one's `improvement_over_chance` area is the
  shaded region between its curve and the diagonal.

This is precisely the read Karoly 2017 Fig 6 invites: a forecaster is
useful when, at a clinically tolerable time-in-warning, its sensitivity
clears the diagonal by a margin that survives the chance test.

## References

- Karoly PJ, Ung H, Grayden DB, et al. *The circadian profile of
  epilepsy improves seizure forecasting.* **Brain** 2017; 140:
  2169-2182. doi:10.1093/brain/awx173 — Fig 6 (sensitivity vs
  time-in-warning).
- Karoly PJ, Goldenholz DM, Freestone DR, et al. *Circadian and
  circaseptan rhythms in human epilepsy.* **Lancet Neurology** 2018 /
  Karoly et al. 2019 — forecasting evaluation against
  time-in-warning-matched chance.
- Mormann F, Andrzejak RG, Elger CE, Lehnertz K. *Seizure prediction:
  the long and winding road.* **Brain** 2007; 130: 314-333 —
  random-predictor baselines.
- Code: `src/scitex_seizure_metrics/sensitivity_tiw/` (`_curve.py`,
  `_significance.py`, `_inputs.py`).
- Companion: [sample_to_alarm.md](sample_to_alarm.md) (the analytic
  bridge), `policy.py::AlarmPolicy` (the policy object), and
  `plots.py::sensitivity_tiw` (the plotter).
