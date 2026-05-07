# Quick start

```bash
pip install epileval
```

## Detection (per-window classification)

```python
from epileval import detection

rep = detection.evaluate(y_true, y_proba, threshold=0.5, fs=1)
print(rep.roc_auc, rep.pr_auc, rep.brier, rep.mcc)
```

## Forecasting (continuous stream with explicit AlarmPolicy)

```python
from epileval import AlarmPolicy, forecasting

policy = AlarmPolicy(
    sph_seconds=300,
    sop_seconds=600,
    cadence_seconds=60,
    refractory_seconds=600,
    alarm_threshold=0.5,
    fp_denominator="interictal",
)

rep = forecasting.evaluate_stream(
    proba, times, seizures, policy,
    total_recording_time=24 * 3600,
)
print(rep.sensitivity, rep.fp_per_hour, rep.ioc, rep.time_in_warning_frac)
```

## Cross-paper bridge

```python
from epileval import bridge

bnd = bridge.sample_to_alarm(
    sample_sensitivity=0.79,
    sample_specificity=0.85,
    sop_seconds=600,
    cadence_seconds=60,
    refractory_seconds=600,
    prevalence=0.1,
)
print(bnd.alarm_sensitivity_upper, bnd.fp_per_hour_upper)
```
