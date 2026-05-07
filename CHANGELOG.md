# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-07

### Added
- `epileval.calibration` — Brier decomposition (reliability + resolution + uncertainty), expected-calibration-error, per-bin reliability table.
- `epileval.papers` subpackage — paper-replica shims for **Karoly 2017** (Brain), **Maturana 2020** (Nat Commun), **Kuhlmann 2018** (Brain / Epilepsy Ecosystem), **Andrade 2024** (Front Neurosci). Each `metrics()` returns the paper's headline metric set so any new method can be slotted onto each paper's exact axis.
- New surrogates: `circadian` (24-h period), `multidien` (default 7-day), `from_history` (Karoly 2019-style extrapolation from past seizures).
- `bridge.sample_to_alarm` now accepts `prevalence`; computes prevalence-aware `K_effective` (max independent chances) — tighter alarm-sens upper bound under low-prevalence regimes (the typical seizure-prediction setting).
- `plots.reliability_diagram(cal_report)` — per-bin reliability curve with marker-size by bin count + ECE/Brier/decomposition annotation.
- 23 new tests; tests reorganised under `tests/epileval/...` mirroring `src/epileval/...` per scitex-dev PS204 convention.

### Changed
- `tests/` reorganised to `tests/epileval/...` mirror layout.
- README rewritten to canonical SciTeX template (logo, badges, problem/solution table, `<details open>` interfaces, Four Freedoms, footer).

### Fixed
- `__init__.py` PA201/202/203/501 conformance (importlib.metadata version, `from __future__ import annotations`).

## [0.2.0] - 2026-05-07

### Added
- `AlarmPolicy` frozen dataclass with explicit knobs (SPH/SOP/cadence/refractory/threshold/merging/FP-denominator).
- `forecasting.evaluate_stream(proba, times, seizures, policy, ...)` continuous-stream entry point.
- `forecasting.sweep_thresholds`, `forecasting.sweep_policies` for operating curves and ablations.
- `forecasting.bootstrap_ci` percentile bootstrap confidence intervals.
- Surrogate registry — `surrogates.poisson`, `surrogates.periodic`, `surrogates.persistence` + `register()` decorator.
- `bridge.sample_to_alarm` / `bridge.alarm_to_sample` analytic metric bounds with Monte Carlo validation.
- `plots` module — sensitivity-vs-FP/hr operating curve, IoC vs surrogate, cadence ablation, sample-vs-alarm scatter (Andrade 2024 figure), metric-correlation heatmap.
- Interictal-only FP/hr denominator (Mormann tradition) with `AlarmPolicy.fp_denominator`.
- 43 tests across 5 test modules.

## [0.1.0] - 2026-05-07

### Added
- `detection.evaluate(y_true, y_proba, ...)` — sklearn ranking metrics + timescoring SampleScoring.
- `forecasting.evaluate(alarms, seizures, sph, sop, ...)` — basic alarm matching + surrogate IoC.
- `MetricsReport` dataclass with `.to_dict()` / `.to_frame()` / `.to_json()`.
- `adapters.proba_to_alarms` — threshold + refractory.
