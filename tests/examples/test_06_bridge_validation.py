"""Empirical bridge-validation guard for examples/06_bridge_validation.py.

CI guard for the K_eff soundness fix: a Monte-Carlo stream with KNOWN
per-window sensitivity + specificity + seizures must land inside the
analytic ``bridge.sample_to_alarm`` bands, and the reverse
``bridge.alarm_to_sample`` must recover the true per-window metrics —
across every (s, alpha, SOP, prevalence) setting, in BOTH directions.

The example module is imported directly (it is a plain, dependency-free
script) so the assertions run on the same logic that produces the README
figure + table.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # no display backend

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "06_bridge_validation.py"

if not EXAMPLE.is_file():
    pytest.skip(f"missing example: {EXAMPLE}", allow_module_level=True)


def _load_example():
    name = "ssm_bridge_validation"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve cls.__module__ under
    # `from __future__ import annotations` (Python < 3.12 quirk).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_MOD = _load_example()
_RESULTS = _MOD.run_validation()
_IDS = [f"s{r.s}_spec{r.spec}_p{r.prevalence:.2f}" for r in _RESULTS]


@pytest.mark.parametrize("r", _RESULTS, ids=_IDS)
def test_empirical_alarm_sensitivity_inside_band(r):
    # Arrange
    band = (r.alarm_sensitivity_lower, r.alarm_sensitivity_upper)
    # Act
    inside = r.alarm_sensitivity_in_band
    # Assert
    assert inside, f"emp alarm sens {r.emp_alarm_sensitivity:.3f} outside {band}"


@pytest.mark.parametrize("r", _RESULTS, ids=_IDS)
def test_empirical_fp_per_hour_inside_band(r):
    # Arrange
    band = (r.fp_per_hour_lower, r.fp_per_hour_upper)
    # Act
    inside = r.fp_per_hour_in_band
    # Assert
    assert inside, f"emp fp/hr {r.emp_fp_per_hour:.3f} outside {band}"


@pytest.mark.parametrize("r", _RESULTS, ids=_IDS)
def test_reverse_recovers_true_sample_sensitivity(r):
    # Arrange
    band = (r.rev_sample_sensitivity_lower, r.rev_sample_sensitivity_upper)
    # Act
    recovered = r.reverse_sensitivity_recovers
    # Assert
    assert recovered, f"true s={r.s} outside recovered {band}"


@pytest.mark.parametrize("r", _RESULTS, ids=_IDS)
def test_reverse_recovers_true_sample_specificity(r):
    # Arrange
    band = (r.rev_sample_specificity_lower, r.rev_sample_specificity_upper)
    # Act
    recovered = r.reverse_specificity_recovers
    # Assert
    assert recovered, f"true spec={r.spec} outside recovered {band}"


def test_low_prevalence_detection_band_does_not_collapse_to_s():
    # Arrange — the s=0.5, prevalence=0.05, SOP=600, cadence=60 setting.
    # Pre-fix the upper bound read 0.5 (== s); post-fix it is ~1.0 because
    # K = ceil(SOP / cadence) = 10 chances per seizure (K_eff soundness fix).
    r = next(x for x in _RESULTS if x.s == 0.5 and x.K == 10)
    # Act
    upper = r.alarm_sensitivity_upper
    # Assert
    assert upper > 0.95


def test_all_settings_pass_both_directions():
    # Arrange
    results = _RESULTS
    # Act
    failed = [r for r in results if not r.all_pass]
    # Assert
    assert not failed, f"{len(failed)} setting(s) violated a bridge bound"


def test_figure_writes_png(tmp_path):
    # Arrange
    base = tmp_path / "bridge_validation"
    # Act
    _MOD.make_figure(_RESULTS, save_path=base)
    # Assert
    assert (tmp_path / "bridge_validation.png").is_file()


def test_figure_writes_pdf(tmp_path):
    # Arrange
    base = tmp_path / "bridge_validation"
    # Act
    _MOD.make_figure(_RESULTS, save_path=base)
    # Assert
    assert (tmp_path / "bridge_validation.pdf").is_file()


def test_results_frame_has_one_row_per_setting():
    # Arrange
    results = _RESULTS
    # Act
    df = _MOD.results_to_frame(results)
    # Assert
    assert len(df) == len(results)
