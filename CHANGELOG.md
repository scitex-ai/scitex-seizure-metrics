# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`scitex_seizure_metrics.sensitivity_tiw`** — empirical sensitivity-vs-time-in-warning trade-off (the field-standard forecasting view; Karoly 2017 *Brain* 140:2169 Fig 6 / Karoly 2019), the empirical complement to the analytic `bridge`:
  - `sensitivity_tiw_curve(scores, policy, labels=… | seizure_times=…, …)` — sweeps the decision threshold and returns an ordered (threshold, time-in-warning, sensitivity) curve plus summary scalars: `improvement_over_chance` (AUC-like area above the chance diagonal), `sensitivity_at_target_tiw`, `tiw_at_target_sensitivity`. Time-in-warning is the time-weighted fraction of windows above threshold; sensitivity is SOP-aware (seizure caught iff ≥1 warning covers the pre-ictal window).
  - `chance_sensitivity(tiw)` — the chance diagonal (a time-matched random alarm catches a fraction `tiw` of seizures in expectation).
  - `binomial_above_chance(...)` — exact one-sided binomial test of sensitivity vs chance at an operating point, with a Wilson interval.
  - `surrogate_above_chance(...)` — circular time-shift permutation test (holds time-in-warning fixed while breaking seizure phase-locking).
  - `SensitivityTiWCurve` / `TiWSignificance` result containers.
- **`scitex_seizure_metrics.plots.sensitivity_tiw`** — the Karoly 2017 Fig 6 plotter: sensitivity (%) vs time-in-warning (%), one curve per subject, overlaying the chance diagonal with optional operating-point markers; optional `save_path` writes png + pdf.
- **`docs/math/sensitivity_tiw.md`** — definitions, the chance-diagonal derivation, the two significance tests, and a worked example.
- `scipy` added as an explicit runtime dependency (binomial test + Wilson interval).

## [0.1.1] - 2026-05-11

### Changed

- CI: Newb workflow now forwards `NEWB_CLAUDE_CODE_CREDENTIALS_JSON` so OAuth (Claude Code Pro / Max) auth works alongside bare `NEWB_ANTHROPIC_API_KEY`.

### Removed

- Audit gate: removed three stale `skip_rules` (`PS-120` / `PS-141` / `SK-102`) — the docs refresh that introduced `## Demo`, the umbrella one-liner under `## Part of SciTeX`, and `_skills/scitex-seizure-metrics/SKILL.md` satisfied them, but the bypass tuple was never trimmed.

## [0.1.0] - 2026-05-07

First public release.

### Added

- **`scitex_seizure_metrics.detection`** — sample-based metrics: AUROC, AUPRC, Brier, balanced accuracy, MCC, F1, sensitivity, precision; built on sklearn + `timescoring.SampleScoring`.
- **`scitex_seizure_metrics.forecasting`** — alarm-based metrics with explicit `AlarmPolicy`; `evaluate(alarms, seizures, policy)`, `evaluate_stream(proba, times, seizures, policy)`, `sweep_thresholds`, `sweep_policies`, percentile `bootstrap_ci`.
- **`scitex_seizure_metrics.AlarmPolicy`** — frozen dataclass with explicit knobs (SPH, SOP, cadence, refractory, threshold, merging rule, FP-denominator). Required argument to all alarm-aware functions; no silent defaults.
- **`scitex_seizure_metrics.bridge`** — analytic cross-paper metric bounds: `sample_to_alarm` (with prevalence-aware K_effective), `alarm_to_sample`. Validated by Monte Carlo.
- **`scitex_seizure_metrics.calibration`** — Brier decomposition (reliability + resolution + uncertainty), Expected Calibration Error, per-bin reliability table.
- **`scitex_seizure_metrics.surrogates`** — registry of chance-baseline alarm generators: `poisson`, `periodic`, `persistence`, `circadian` (24-h), `multidien` (default 7-day), `from_history`. User-extensible via `surrogates.register()`.
- **`scitex_seizure_metrics.papers`** — paper-replica shims so any new method can be slotted onto each paper's exact metric axis: `cook2013`, `karoly2017`, `kuhlmann2018`, `maturana2020`, `proix2021`, `stirling2021`, `andrade2024`.
- **`scitex_seizure_metrics.plots`** — relationship plots: `sensitivity_vs_fp_per_hour` operating curve, `ioc_vs_surrogate`, `cadence_ablation`, `sample_vs_alarm_scatter` (the Andrade 2024 figure), `metric_correlation_heatmap`, `reliability_diagram`.
- **`scitex_seizure_metrics.MetricsReport`** — frozen single-row report with `.to_frame()`, `.to_json()`, `.to_dict()`.
- 69 tests across 13 test modules; full scitex-dev `audit-python-apis` + `audit-project` clean.
- Two example scripts (`examples/quick_start_detection.py`, `examples/quick_start_forecasting.py`) with smoke tests.
