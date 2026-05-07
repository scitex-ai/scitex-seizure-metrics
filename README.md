# epileval

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center"><b>Unified evaluation library for seizure detection and forecasting — sample-based, alarm-based, and the bridge between them.</b></p>

<p align="center">
  <a href="https://epileval.readthedocs.io/">Full Documentation</a> · <code>pip install epileval</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/epileval/"><img src="https://img.shields.io/pypi/v/epileval.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/epileval/"><img src="https://img.shields.io/pypi/pyversions/epileval.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/epileval/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/epileval/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/epileval"><img src="https://codecov.io/gh/ywatanabe1989/epileval/graph/badge.svg" alt="Coverage"></a>
  <a href="https://epileval.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/epileval/badge/?version=latest" alt="Docs"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Problem and Solution

| # | Problem | Solution |
|---|---------|----------|
| 1 | **Cross-paper comparison is broken** — Cook 2013 reports time-in-warning, Karoly 2017 reports AUROC + Brier, Maturana 2020 reports AUROC + IoC, Kuhlmann 2018 reports AUROC, Proix 2021 reports IoC + AUC of sensitivity vs proportion-time-in-warning. No two of these can be plotted on the same axis without re-running their methods. | **One `MetricsReport` object** carries both regimes through one API; `bridge.sample_to_alarm` gives analytic bounds when only one side is reported. |
| 2 | **Sample- vs alarm-based collapse is documented but untooled** — Andrade 2024 showed that 50/56 patients beat chance under sample-based eval but **only 6/46 under alarm-based**. The community accepts the warning but has no packaged tool to apply both regimes routinely. | **`detection.evaluate` + `forecasting.evaluate_stream`** through one library; same input, both regimes side-by-side. |
| 3 | **FP/hr lacks a denominator convention** — some papers normalise by total recording time, some by interictal-only time, refractory rules vary or are unstated. | **Explicit `AlarmPolicy`** required by every alarm-aware function — no silent defaults; every reported number is reproducible. |

## Comparison with existing tools

| Tool | Language | Sample-based | Event-based | Forecasting (SPH/SOP) | IoC vs surrogate | Cross-paper convertor | Status |
|---|---|---|---|---|---|---|---|
| `timescoring` (SzCORE engine, [Dan et al. 2024](https://doi.org/10.1111/epi.18113)) | Python | ✓ | ✓ | ✗ | ✗ | ✗ | active |
| `szcore-evaluation` (BIDS wrapper) | Python | ✓ | ✓ | ✗ | ✗ | ✗ | active |
| `EPILAB` ([Direito et al. 2011](https://doi.org/10.1016/j.jneumeth.2011.06.022)) | MATLAB | ✓ | partial | ✓ | ✓ | ✗ | dead since 2018 |
| `PySeizure` ([2025](https://arxiv.org/html/2508.07253)) | Python | ✓ | ✗ | ✗ | ✗ | ✗ | very new, narrow |
| `SeizyML` ([2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11160878/)) | Python | ✓ | ✓ | ✗ | ✗ | ✗ | detection-only |
| Andrade et al. 2024 (paper) | — | ✓ | ✓ | ✓ | ✓ | ✗ | not packaged |
| **epileval** | Python | ✓ | ✓ | ✓ | ✓ | ✓ | active |

## Installation

```bash
pip install epileval
```

## Quick Start

```python
from epileval import detection, forecasting, AlarmPolicy

# Detection — per-window classification
rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1)
print(rep.roc_auc, rep.pr_auc, rep.brier, rep.mcc)

# Forecasting — continuous stream with explicit alarm policy
policy = AlarmPolicy(
    sph_seconds=300, sop_seconds=600, cadence_seconds=60,
    refractory_seconds=600, alarm_threshold=0.5,
    fp_denominator="interictal",   # Mormann tradition
)
rep = forecasting.evaluate_stream(
    proba, times, seizures, policy,
    total_recording_time=24 * 3600,
)
print(rep.sensitivity, rep.fp_per_hour, rep.ioc, rep.time_in_warning_frac)
```

See `examples/quick_start_detection.py` and `examples/quick_start_forecasting.py`.

## Demo

Two runnable scripts ship under `examples/`:

- [`quick_start_detection.py`](examples/quick_start_detection.py) — sample-based
  evaluation on a synthetic per-window probability stream; prints `roc_auc`,
  `pr_auc`, `brier`, `mcc`, `balanced_accuracy`.
- [`quick_start_forecasting.py`](examples/quick_start_forecasting.py) —
  alarm-based evaluation with an explicit `AlarmPolicy`; prints
  `sensitivity`, `fp_per_hour`, `ioc`, `time_in_warning_frac` and runs an
  IoC surrogate test.

```mermaid
flowchart LR
    Probs[per-window proba<br/>+ ground truth] --> Det[detection.evaluate]
    Probs --> StreamIn[forecasting.evaluate_stream]
    Policy[AlarmPolicy<br/>SPH · SOP · cadence · refractory · FP denom] --> StreamIn
    Det --> RepDet[MetricsReport<br/>AUROC · AUPRC · Brier · MCC]
    StreamIn --> RepFc[MetricsReport<br/>sensitivity · FP/hr · IoC · TIW]
    RepDet -. bridge.sample_to_alarm .-> RepFc
    RepFc --> Plots[plots.sensitivity_vs_fp_per_hour · ioc_vs_surrogate · cadence_ablation]
```

## Architecture

Module layout under `src/epileval/`:

```
epileval/
├── detection.py        sample-based metric pipeline (AUROC, AUPRC, Brier, MCC, ...)
├── forecasting.py      alarm-based pipeline — evaluate_stream, sweep_thresholds,
│                       sweep_policies (cadence ablation)
├── policy.py           AlarmPolicy dataclass — SPH · SOP · cadence · refractory ·
│                       fp_denominator (no silent defaults)
├── _alarm.py           internal alarm-derivation (private)
├── bridge.py           sample ↔ alarm analytic bounds (cross-paper conversion)
├── calibration.py      Brier decomposition · reliability · ECE
├── surrogates.py       IoC surrogate distribution under chance
├── report.py           MetricsReport — unifies sample + alarm in one object
├── adapters.py         I/O adapters for common dataset / score formats
├── plots.py            sensitivity-vs-FP/hr · IoC-vs-surrogate · cadence ablation ·
│                       sample-vs-alarm scatter (the Andrade 2024 figure)
└── papers/             paper-replica shims (one module per work)
    ├── andrade2024.py
    ├── cook2013.py
    ├── karoly2017.py
    ├── kuhlmann2018.py
    ├── maturana2020.py
    ├── proix2021.py
    └── stirling2021.py
```

The split mirrors how the seizure-evaluation literature itself is
organised — sample-based vs alarm-based vs the bridge — so a
paper-faithful re-implementation lives in exactly one place.
`MetricsReport` is the single object that travels between regimes;
`AlarmPolicy` is the single object that pins every reproducibility
decision an alarm-based metric requires.

## 5 Interfaces

<details open>
<summary><b><code>epileval.forecasting</code></b> — alarm-based metrics with explicit AlarmPolicy (primary)</summary>

```python
from epileval import AlarmPolicy, forecasting

policy = AlarmPolicy(
    sph_seconds=300, sop_seconds=600, cadence_seconds=60,
    refractory_seconds=600, alarm_threshold=0.5,
    fp_denominator="interictal",
)
rep = forecasting.evaluate_stream(
    proba, times, seizures, policy,
    total_recording_time=24 * 3600, n_surrogate=1000,
)
print(rep.sensitivity, rep.fp_per_hour, rep.ioc, rep.time_in_warning_frac)

# Operating curve across thresholds
df = forecasting.sweep_thresholds(proba, times, seizures, policy)

# Cadence ablation
policies = [AlarmPolicy(..., cadence_seconds=c) for c in [30, 60, 120, 300]]
df = forecasting.sweep_policies(proba, times, seizures, policies)
```

</details>

<details>
<summary><b><code>epileval.detection</code></b> — sample-based metrics (AUROC, AUPRC, Brier, MCC, ...)</summary>

```python
from epileval import detection
rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1)
print(rep.roc_auc, rep.pr_auc, rep.brier, rep.mcc, rep.balanced_accuracy)
```

</details>

<details>
<summary><b><code>epileval.bridge</code></b> — sample↔alarm analytic bounds for cross-paper comparison</summary>

```python
from epileval import bridge

bnd = bridge.sample_to_alarm(
    sample_sensitivity=0.79, sample_specificity=0.85,
    sop_seconds=600, cadence_seconds=60, refractory_seconds=600,
)
print(bnd.alarm_sensitivity_upper, bnd.fp_per_hour_upper)
```

</details>

<details>
<summary><b><code>epileval.papers</code></b> — paper-replica shims (Karoly 2017, Maturana 2020, Kuhlmann 2018, Andrade 2024)</summary>

```python
from epileval.papers import andrade2024
out = andrade2024.metrics(
    y_true=labels, y_proba=preds,
    times_seconds=times, seizure_times=onsets,
)
print(out["sample_auroc"], out["alarm_sensitivity"], out["beats_chance_alarm"])
# Reproduces the side-by-side sample-vs-alarm panel from the paper.
```

Available shims: `karoly2017`, `maturana2020`, `kuhlmann2018`, `andrade2024`. Each `metrics(...)` returns a dict in the paper's preferred metric set.

</details>

<details>
<summary><b><code>epileval.calibration</code></b> — Brier decomposition + reliability diagram</summary>

```python
from epileval import calibration, plots
cal = calibration.calibration_report(y_true, y_proba, n_bins=10)
print(cal.brier, cal.reliability, cal.resolution, cal.uncertainty,
      cal.expected_calibration_error)
plots.reliability_diagram(cal)
```

</details>

<details>
<summary><b><code>epileval.plots</code></b> — relationships between metrics</summary>

```python
from epileval import plots
plots.sensitivity_vs_fp_per_hour(sweep_df)        # operating curve
plots.ioc_vs_surrogate(sweep_df)                  # model vs chance
plots.cadence_ablation(policy_sweep_df)           # FP/hr vs cadence
plots.sample_vs_alarm_scatter(per_patient_df)     # the Andrade 2024 figure
plots.metric_correlation_heatmap(per_patient_df)  # redundancy diagnostic
```

</details>

## Part of SciTeX

`epileval` is part of [**SciTeX**](https://scitex.ai). Install via the umbrella with `pip install scitex[epileval]` and import as `scitex.epileval`.

>Four Freedoms for Research
>
>0. The freedom to **run** your research anywhere — your machine, your terms.
>1. The freedom to **study** how every step works — from raw data to final manuscript.
>2. The freedom to **redistribute** your workflows, not just your papers.
>3. The freedom to **modify** any module and share improvements with the community.
>
>AGPL-3.0 — because we believe research infrastructure deserves the same freedoms as the software it runs on.

## References

- Andrade I, Teixeira C, Pinto M (2024). On the performance of seizure prediction machine learning methods across different databases: the sample and alarm-based perspectives. *Frontiers in Neuroscience*. [doi:10.3389/fnins.2024.1417748](https://doi.org/10.3389/fnins.2024.1417748).
- Cook MJ et al. (2013). *Lancet Neurology*. [doi:10.1016/S1474-4422(13)70075-9](https://doi.org/10.1016/S1474-4422(13)70075-9).
- Dan J et al. (2024). SzCORE. *Epilepsia*. [doi:10.1111/epi.18113](https://doi.org/10.1111/epi.18113).
- Direito B et al. (2011). EPILAB. *J Neurosci Methods*. [doi:10.1016/j.jneumeth.2011.06.022](https://doi.org/10.1016/j.jneumeth.2011.06.022).
- Karoly PJ et al. (2017). *Brain*. [doi:10.1093/brain/awx173](https://doi.org/10.1093/brain/awx173).
- Kuhlmann L et al. (2018). *Brain*. [doi:10.1093/brain/awy210](https://doi.org/10.1093/brain/awy210).
- Maturana MI et al. (2020). *Nature Communications*. [doi:10.1038/s41467-020-15908-3](https://doi.org/10.1038/s41467-020-15908-3).
- Mormann F et al. (2007). Seizure prediction: the long and winding road. *Brain*. [doi:10.1093/brain/awl241](https://doi.org/10.1093/brain/awl241).
- Schulze-Bonhage A et al. (2020). Performance Metrics for Online Seizure Prediction. [PMC7340210](https://pmc.ncbi.nlm.nih.gov/articles/PMC7340210/).

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
</p>
