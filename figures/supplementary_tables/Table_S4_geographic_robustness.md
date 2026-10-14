# Supplementary Table S4. Internal geographic hold-out within MMRF CoMMpass (Figure S2)

XGBoost information Layer C refit on US patients only (new random patient-level 82/18 split: 632 training, 139 validation); preprocessing and fitting on US training, calibration method and threshold on US validation; evaluated once on patients enrolled in Spain and Canada. Not independent external validation: hyperparameters were reused from the main analysis, whose tuning population included 80 of the 116 evaluated patients. No statistical comparison between cohorts was performed. Italy (138 patients) was excluded post hoc.

| Cohort | Patients | Visit pairs (binary / six-class) | Improvement events (rate) | AUPRC (95% CI) | AUROC (95% CI) | Macro-F1 (95% CI) | Patients in original tuning split |
|---|---|---|---|---|---|---|---|
| Spain + Canada (pooled) | 116 | 1,588 / 1,694 | 283 (17.8%) | 0.451 (0.401 to 0.507) | 0.770 (0.729 to 0.808) | 0.644 (0.598 to 0.676) | 80 |
| Spain | 80 | 949 / 1,034 | 185 (19.5%) | 0.463 (0.407 to 0.532) | 0.761 (0.718 to 0.809) | 0.639 (0.589 to 0.674) | 53 |
| Canada | 36 | 639 / 660 | 98 (15.3%) | 0.443 (0.369 to 0.541) | 0.773 (0.700 to 0.847) | 0.634 (0.492 to 0.686) | 27 |
| US validation (reference) | 139 | 1,911 / 2,076 | 351 (18.4%) | 0.525 (no CI) | 0.805 (no CI) | 0.647 (no CI) | — |

The event rate is the AUPRC of a non-informative model in each cohort. Source: `figures/data/FigS_country/`, cross-checked against `origin/modeling-country-robustness:results/robustness_country/binary_c_xgboost_us_to_spain_canada_metrics.json`.