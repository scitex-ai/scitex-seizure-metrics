classification (forecasting confusion matrix)
=============================================

Forecasting-regime confusion-matrix primitives (v0.2.0). These turn the
alarm-vs-seizure match counts into the standard binary-classifier scores
(specificity, PPV, NPV, F1) and the observed lead time, under the
SOP-length-opportunity true-negative convention documented in
:doc:`math/alarm_confusion_matrix` and ADR-0001.

This is an internal module (``_classification``); its results are surfaced
to users through :doc:`forecasting` (``evaluate`` / ``evaluate_stream``)
and the :doc:`report` fields ``specificity`` / ``ppv`` / ``npv`` /
``forecasting_f1`` / ``n_tn`` / ``n_opportunities`` / ``lead_time_mean`` /
``lead_time_median``. It is documented here so the exact TN convention,
the NaN-on-empty (fail-loud) behaviour, and the lead-time definition are
discoverable in the API reference.

.. automodule:: scitex_seizure_metrics._classification
   :members:
   :undoc-members:
   :show-inheritance:
