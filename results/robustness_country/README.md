# Country-based Model C robustness check

This is an exploratory, post hoc test within MMRF CoMMpass, not external validation.

Model C (XGBoost) was trained on US patients, with a separate US validation group used for calibration and threshold selection. It was then evaluated on 116 patients from Spain and Canada. Patients were disjoint across these groups.

On Spain and Canada, binary improvement AUPRC was 0.451 (95% patient-bootstrap CI 0.401–0.507; 1,588 eligible visit pairs). Six-class response macro-F1 was 0.644 (95% CI 0.598–0.676; 1,694 visit pairs). US validation scores were 0.525 AUPRC and 0.647 macro-F1; that group informed model choices, so these comparisons are descriptive.

Italy was excluded from this comparison because its single site's visit records had a substantially different and sparse assessment pattern. A one-row treatment-flag correction produced identical held-out predictions in a separate sensitivity rerun.

The script is in `modeling/us_to_spain_canada_model_c.py`. The JSON files in this folder contain full metrics and confidence intervals. No independent cohort was used.
