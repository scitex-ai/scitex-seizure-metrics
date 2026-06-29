# `examples/` — runnable workflows

End-to-end, executable examples. The notebooks are smoke-tested in CI
(`tests/examples/`), so every one runs against the current API.

| File | What it shows |
| ---- | ------------- |
| `01_detection_quick_start.ipynb` | Sample-based evaluation: `detection.evaluate` → AUROC / AUPRC / Brier / MCC. |
| `02_forecasting_quick_start.ipynb` | Alarm-based evaluation: build an `AlarmPolicy`, run `forecasting.evaluate_stream`, read sensitivity / FP-per-hour / IoC / time-in-warning and (v0.2.0) the confusion-matrix scores + observed lead time. |
| `03_bridge_sample_to_alarm.ipynb` | Cross-paper bounds: `bridge.sample_to_alarm` / `alarm_to_sample`. |
| `04_calibration.ipynb` | Brier decomposition + reliability diagram. |
| `05_papers_andrade2024.ipynb` | The Andrade 2024 sample-vs-alarm panel via the paper-replica shim. |
| `06_bridge_validation.py` | Monte-Carlo validation of the analytic bridge; writes `docs/bridge_validation.{png,pdf}` and a results table to `06_bridge_validation_out/`. |

## Run

```bash
pip install -e ".[all]"
jupyter lab examples/        # notebooks
python examples/06_bridge_validation.py
```

For idiomatic guidance beyond these snippets, see the packaged skill at
`src/scitex_seizure_metrics/_skills/scitex-seizure-metrics/` (especially
`04_forecasting-classification.md`).
