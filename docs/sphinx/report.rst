report (MetricsReport)
======================

``MetricsReport`` is the single flat dataclass that carries a full
evaluation result for either regime. The same object exposes the
detection-style ranking metrics (``roc_auc`` / ``pr_auc`` / ``brier`` /
``mcc``) and the forecasting-style alarm metrics (``sensitivity`` /
``fp_per_hour`` / ``ioc`` / ``time_in_warning_frac``), plus — since
v0.2.0 — the forecasting confusion-matrix scores (``specificity`` /
``ppv`` / ``npv`` / ``forecasting_f1`` with the raw ``n_tn`` /
``n_opportunities`` denominators) and the observed lead time
(``lead_time_mean`` / ``lead_time_median``; the per-seizure array lives in
``extras["lead_times_seconds"]``). See
:doc:`math/alarm_confusion_matrix` for the definitions.

It serialises three ways: ``to_dict()`` (JSON-friendly), ``to_frame()``
(a one-row :class:`pandas.DataFrame` that stacks cleanly across
patients/folds), and ``to_json()``.

.. automodule:: scitex_seizure_metrics.report
   :members:
   :undoc-members:
   :show-inheritance:
