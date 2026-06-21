#!/usr/bin/env python3
r"""Empirical validation of the sample<->alarm bridge (Monte Carlo).

Generates a long per-window stream with a KNOWN per-window sensitivity
``s`` and specificity ``1 - alpha`` plus seizures, runs the alarm policy,
measures the EMPIRICAL alarm-sensitivity + FP/hr, and checks that

1. the empirical alarm-sensitivity + FP/hr land inside the analytic
   ``bridge.sample_to_alarm`` ``[lower, upper]`` bands, and
2. the reverse ``bridge.alarm_to_sample`` recovers the true per-window
   ``s`` + specificity (both inside the returned ranges),

across several ``(s, alpha, SOP, prevalence)`` settings. This is the
empirical evidence behind the K_eff soundness fix: per-seizure detection
uses ``K = ceil(SOP / cadence)`` windows independent of the global
prevalence, so the upper detection bound is ``1 - (1 - s) ** K`` (not the
prevalence-collapsed ``s`` of the earlier release).

Run:
    python 06_bridge_validation.py            # writes the figure + table
    python 06_bridge_validation.py --help

Outputs:
    docs/bridge_validation.png / .pdf                          (README asset)
    examples/06_bridge_validation_out/FINISHED_SUCCESS/<id>/   (session run:
        bridge_validation.csv + captured logs/config)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scitex_session as stx  # standalone session lib (no umbrella drag; PS-139)

from scitex_seizure_metrics import AlarmPolicy, bridge, forecasting

# Settings spanning low/high sensitivity, specificity, SOP, and (the
# crux of the fix) realistic LOW prevalence where the old K_eff collapsed.
VALIDATION_SETTINGS: tuple[dict, ...] = (
    dict(s=0.50, spec=0.90, sop=600, cadence=60, prevalence=0.05),
    dict(s=0.30, spec=0.95, sop=900, cadence=30, prevalence=0.02),
    dict(s=0.70, spec=0.85, sop=300, cadence=60, prevalence=0.10),
    dict(s=0.60, spec=0.99, sop=1800, cadence=30, prevalence=0.01),
)

_TOL = 1e-9


@dataclass
class BridgeValidationResult:
    """One Monte-Carlo validation row: empirical values vs analytic bounds."""

    s: float
    spec: float
    sop: float
    cadence: float
    prevalence: float
    K: int
    emp_alarm_sensitivity: float
    alarm_sensitivity_lower: float
    alarm_sensitivity_upper: float
    emp_fp_per_hour: float
    fp_per_hour_lower: float
    fp_per_hour_upper: float
    rev_sample_sensitivity_lower: float
    rev_sample_sensitivity_upper: float
    rev_sample_specificity_lower: float
    rev_sample_specificity_upper: float

    @property
    def alarm_sensitivity_in_band(self) -> bool:
        return (
            self.alarm_sensitivity_lower - _TOL
            <= self.emp_alarm_sensitivity
            <= self.alarm_sensitivity_upper + _TOL
        )

    @property
    def fp_per_hour_in_band(self) -> bool:
        return (
            self.fp_per_hour_lower - _TOL
            <= self.emp_fp_per_hour
            <= self.fp_per_hour_upper + _TOL
        )

    @property
    def reverse_sensitivity_recovers(self) -> bool:
        return (
            self.rev_sample_sensitivity_lower - _TOL
            <= self.s
            <= self.rev_sample_sensitivity_upper + _TOL
        )

    @property
    def reverse_specificity_recovers(self) -> bool:
        return (
            self.rev_sample_specificity_lower - _TOL
            <= self.spec
            <= self.rev_sample_specificity_upper + _TOL
        )

    @property
    def all_pass(self) -> bool:
        return (
            self.alarm_sensitivity_in_band
            and self.fp_per_hour_in_band
            and self.reverse_sensitivity_recovers
            and self.reverse_specificity_recovers
        )


def _simulate_stream(
    *,
    s: float,
    spec: float,
    sop: float,
    cadence: float,
    prevalence: float,
    n_seizures: int = 120,
    seed: int = 0,
):
    """Long stream with known per-window s + specificity + seizures.

    Each seizure's SOP holds ``K = ceil(sop / cadence)`` pre-ictal
    windows; interictal windows are padded around them so the realized
    pre-ictal-window prevalence matches the requested ``prevalence``.
    Per-window predictions fire with probability ``s`` inside pre-ictal
    windows and ``alpha = 1 - spec`` inside interictal windows
    (independent errors), so the stream has the requested sample metrics.
    """
    rng = np.random.default_rng(seed)
    K = int(np.ceil(sop / cadence))
    n_preictal = n_seizures * K
    n_total = int(round(n_preictal / prevalence))
    n_inter = n_total - n_preictal
    gap = max(K + 1, n_inter // n_seizures)

    labels: list[int] = []
    seizures: list[float] = []
    t = 0
    for _ in range(n_seizures):
        for _ in range(gap):
            labels.append(0)
            t += 1
        for _ in range(K):
            labels.append(1)
            t += 1
        seizures.append(t * cadence)  # onset right after the SOP

    labels_arr = np.asarray(labels, dtype=int)
    times = np.arange(labels_arr.size, dtype=float) * cadence
    seizures_arr = np.asarray(seizures, dtype=float)
    realized_prevalence = float(labels_arr.mean())

    proba = np.zeros(labels_arr.size)
    pos = labels_arr == 1
    proba[pos] = (rng.random(int(pos.sum())) < s).astype(float)
    proba[~pos] = (rng.random(int((~pos).sum())) < (1.0 - spec)).astype(float)

    total_T = float(times.max() + cadence)
    policy = AlarmPolicy(
        sph_seconds=0,
        sop_seconds=sop,
        cadence_seconds=cadence,
        refractory_seconds=sop,
        alarm_threshold=0.5,
        fp_denominator="total",
    )
    rep = forecasting.evaluate_stream(
        proba, times, seizures_arr, policy, total_recording_time=total_T, n_surrogate=0
    )
    return rep, realized_prevalence, K


def validate_setting(
    *,
    s: float,
    spec: float,
    sop: float,
    cadence: float,
    prevalence: float,
    n_seizures: int = 120,
    seed: int = 0,
) -> BridgeValidationResult:
    """Run one Monte-Carlo setting and compare to both bridge directions."""
    rep, realized_prevalence, K = _simulate_stream(
        s=s,
        spec=spec,
        sop=sop,
        cadence=cadence,
        prevalence=prevalence,
        n_seizures=n_seizures,
        seed=seed,
    )
    fwd = bridge.sample_to_alarm(
        sample_sensitivity=s,
        sample_specificity=spec,
        sop_seconds=sop,
        cadence_seconds=cadence,
        refractory_seconds=sop,
        prevalence=realized_prevalence,
    )
    rev = bridge.alarm_to_sample(
        alarm_sensitivity=rep.sensitivity,
        fp_per_hour=rep.fp_per_hour,
        sop_seconds=sop,
        cadence_seconds=cadence,
        refractory_seconds=sop,
        prevalence=realized_prevalence,
    )
    return BridgeValidationResult(
        s=s,
        spec=spec,
        sop=sop,
        cadence=cadence,
        prevalence=realized_prevalence,
        K=K,
        emp_alarm_sensitivity=float(rep.sensitivity),
        alarm_sensitivity_lower=fwd.alarm_sensitivity_lower,
        alarm_sensitivity_upper=fwd.alarm_sensitivity_upper,
        emp_fp_per_hour=float(rep.fp_per_hour),
        fp_per_hour_lower=fwd.fp_per_hour_lower,
        fp_per_hour_upper=fwd.fp_per_hour_upper,
        rev_sample_sensitivity_lower=rev["sample_sensitivity_lower"],
        rev_sample_sensitivity_upper=rev["sample_sensitivity_upper"],
        rev_sample_specificity_lower=rev["sample_specificity_lower"],
        rev_sample_specificity_upper=rev["sample_specificity_upper"],
    )


def run_validation(
    settings=VALIDATION_SETTINGS, *, seed: int = 1
) -> list[BridgeValidationResult]:
    """Run every setting; returns one :class:`BridgeValidationResult` each."""
    return [validate_setting(seed=seed, **cfg) for cfg in settings]


def results_to_frame(results: list[BridgeValidationResult]):
    """Tidy DataFrame: empirical vs bands + per-direction PASS flags."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "s": r.s,
                "specificity": r.spec,
                "prevalence": round(r.prevalence, 4),
                "K": r.K,
                "emp_alarm_sens": round(r.emp_alarm_sensitivity, 3),
                "alarm_sens_band": f"[{r.alarm_sensitivity_lower:.2f}, "
                f"{r.alarm_sensitivity_upper:.2f}]",
                "sens_PASS": r.alarm_sensitivity_in_band,
                "emp_fp_per_hour": round(r.emp_fp_per_hour, 3),
                "fp_per_hour_band": f"[{r.fp_per_hour_lower:.2f}, "
                f"{r.fp_per_hour_upper:.2f}]",
                "fph_PASS": r.fp_per_hour_in_band,
                "rev_sens_PASS": r.reverse_sensitivity_recovers,
                "rev_spec_PASS": r.reverse_specificity_recovers,
            }
            for r in results
        ]
    )


def make_figure(results: list[BridgeValidationResult], *, save_path=None):
    """Empirical values against analytic bands + reverse-recovery.

    Three panels: (1) alarm-sensitivity empirical vs band, (2) FP/hr
    empirical vs band, (3) reverse alarm->sample recovery of the true
    per-window ``s`` + specificity inside the returned ranges. Empirical /
    true markers landing inside every band is the post-fix PASS evidence.
    """
    import matplotlib.pyplot as plt

    n = len(results)
    x = np.arange(n)
    labels = [
        f"s={r.s:g}\nspec={r.spec:g}\nπ={r.prevalence:.0%}\nK={r.K}" for r in results
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    # Panel 1 — alarm sensitivity: band + empirical.
    ax = axes[0]
    lo = np.array([r.alarm_sensitivity_lower for r in results])
    up = np.array([r.alarm_sensitivity_upper for r in results])
    emp = np.array([r.emp_alarm_sensitivity for r in results])
    ax.vlines(x, lo, up, color="C0", linewidth=8, alpha=0.30, label="Analytic band")
    ax.scatter(x, lo, marker="_", s=260, color="C0")
    ax.scatter(x, up, marker="_", s=260, color="C0")
    ax.scatter(x, emp, color="C3", zorder=3, s=55, label="Empirical (Monte Carlo)")
    ax.set_ylabel("Alarm sensitivity")
    ax.set_title("Alarm sensitivity vs analytic band")
    ax.set_ylim(0, 1.05)

    # Panel 2 — FP per hour: band + empirical.
    ax = axes[1]
    lo = np.array([r.fp_per_hour_lower for r in results])
    up = np.array([r.fp_per_hour_upper for r in results])
    emp = np.array([r.emp_fp_per_hour for r in results])
    ax.vlines(x, lo, up, color="C0", linewidth=8, alpha=0.30, label="Analytic band")
    ax.scatter(x, lo, marker="_", s=260, color="C0")
    ax.scatter(x, up, marker="_", s=260, color="C0")
    ax.scatter(x, emp, color="C3", zorder=3, s=55, label="Empirical (Monte Carlo)")
    ax.set_ylabel("False positives per hour")
    ax.set_title("FP/hr vs analytic band")
    ax.set_ylim(bottom=0)

    # Panel 3 — reverse recovery: returned ranges + true value.
    ax = axes[2]
    s_lo = np.array([r.rev_sample_sensitivity_lower for r in results])
    s_up = np.array([r.rev_sample_sensitivity_upper for r in results])
    sp_lo = np.array([r.rev_sample_specificity_lower for r in results])
    sp_up = np.array([r.rev_sample_specificity_upper for r in results])
    true_s = np.array([r.s for r in results])
    true_sp = np.array([r.spec for r in results])
    off = 0.12
    ax.vlines(
        x - off,
        s_lo,
        s_up,
        color="C0",
        linewidth=8,
        alpha=0.30,
        label="Recovered s range",
    )
    ax.scatter(x - off, true_s, color="C0", marker="D", zorder=3, s=45, label="True s")
    ax.vlines(
        x + off,
        sp_lo,
        sp_up,
        color="C2",
        linewidth=8,
        alpha=0.30,
        label="Recovered specificity range",
    )
    ax.scatter(
        x + off,
        true_sp,
        color="C2",
        marker="D",
        zorder=3,
        s=45,
        label="True specificity",
    )
    ax.set_ylabel("Per-window metric")
    ax.set_title("Reverse alarm->sample recovery")
    ax.set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Setting")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.25)

    all_pass = all(r.all_pass for r in results)
    fig.suptitle(
        "Empirical validation of the sample<->alarm bridge — "
        f"{'all bounds hold (PASS)' if all_pass else 'BOUND VIOLATION'}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    if save_path is not None:
        import os

        root, _ = os.path.splitext(str(save_path))
        Path(root).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(root + ".pdf", bbox_inches="tight")
    return fig, axes


@stx.session
def main(CONFIG=stx.INJECTED, logger=stx.INJECTED) -> int:
    """Run every validation setting; write the table + README figure.

    The table is written under the session output directory
    (``CONFIG.SDIR_RUN`` -> ``06_bridge_validation_out/FINISHED_SUCCESS/
    <session_id>/``); the figure goes to ``docs/bridge_validation.{png,pdf}``
    because the README embeds it.
    """
    import matplotlib

    matplotlib.use("Agg")

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(CONFIG.SDIR_RUN)  # session-managed; FINISHED_SUCCESS on return
    fig_base = repo / "docs" / "bridge_validation"  # README asset

    results = run_validation()
    df = results_to_frame(results)
    logger.info("Bridge-validation results:\n" + df.to_string(index=False))
    all_pass = all(r.all_pass for r in results)
    logger.info(f"All settings PASS (both directions): {all_pass}")

    csv_path = out_dir / "bridge_validation.csv"
    df.to_csv(csv_path, index=False)
    make_figure(results, save_path=fig_base)
    logger.info(f"figure -> {fig_base}.png / .pdf")
    logger.info(f"table  -> {csv_path}")

    # Fail loud: a violated analytic bound must mark the session FAILED, not
    # quietly commit a green run.
    failed = [r for r in results if not r.all_pass]
    if failed:
        raise AssertionError(f"{len(failed)} setting(s) violated a bridge bound")
    return 0


if __name__ == "__main__":
    main()
