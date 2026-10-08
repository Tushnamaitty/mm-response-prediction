# Supplementary Table S2. Calibration of binary next-visit improvement predictions, all test pairs and RNA-available test pairs

XGBoost information Layer C and Layer D on the held-out test set; uncalibrated probabilities (selected on the validation set for both layers). Layer D − Layer C differences are paired on identical visit pairs with 95% paired patient-level bootstrap CIs (1,000 resamples; calibrators fixed). For RNA-available pairs, per-layer values are point estimates recomputed from stored test probabilities with the locked formulas (their D − C equals the locked paired difference); per-layer CIs are not available. Intervals are unadjusted. Descriptive.

## All test pairs (1,930 pairs; 155 patients)

| Metric | Layer C | Layer D | D − C | 95% CI (D − C) | CI includes 0 |
|---|---|---|---|---|---|
| Calibration slope (ideal 1) | 1.087 (0.952 to 1.232) | 1.083 (0.953 to 1.225) | −0.004 | −0.029 to 0.024 | yes |
| Calibration intercept (ideal 0) | −0.066 (−0.199 to 0.075) | −0.075 (−0.210 to 0.069) | −0.009 | −0.024 to 0.005 | yes |
| Brier score (lower better) | 0.11749 (0.10748 to 0.12855) | 0.11759 (0.10768 to 0.12856) | +0.00010 | −0.00103 to 0.00105 | yes |
| Scaled Brier (higher better) | 0.214 (0.170 to 0.255) | 0.213 (0.170 to 0.254) | −0.001 | −0.007 to 0.007 | yes |
| Log loss (lower better) | 0.37821 (0.35086 to 0.40798) | 0.37873 (0.35133 to 0.40841) | +0.00052 | −0.00229 to 0.00297 | yes |
| Mean predicted probability | 0.1910 (0.1769 to 0.2061) | 0.1921 (0.1779 to 0.2076) | +0.0011 | −0.0006 to 0.0030 | yes |
| AUROC (higher better) | 0.813 (0.784 to 0.839) | 0.810 (0.781 to 0.837) | −0.003 | −0.007 to 0.001 | yes |
| AUPRC (higher better) | 0.493 (0.443 to 0.551) | 0.501 (0.454 to 0.554) | +0.008 | −0.009 to 0.023 | yes |

## RNA-available test pairs (1,323 pairs; 107 patients)

| Metric | Layer C | Layer D | D − C | 95% CI (D − C) | CI includes 0 |
|---|---|---|---|---|---|
| Calibration slope (ideal 1) | 1.023 | 1.033 | +0.010 | −0.022 to 0.043 | yes |
| Calibration intercept (ideal 0) | −0.115 | −0.141 | −0.026 | −0.045 to −0.007 | **no** |
| Brier score (lower better) | 0.12203 | 0.12188 | −0.00015 | −0.00162 to 0.00117 | yes |
| Scaled Brier (higher better) | 0.194 | 0.195 | +0.001 | −0.008 to 0.010 | yes |
| Log loss (lower better) | 0.39110 | 0.39107 | −0.00003 | −0.00390 to 0.00325 | yes |
| Mean predicted probability | 0.2003 | 0.2037 | +0.0034 | 0.0010 to 0.0058 | **no** |
| AUROC (higher better) | 0.801 | 0.798 | −0.004 | −0.009 to 0.001 | yes |
| AUPRC (higher better) | 0.474 | 0.487 | +0.013 | −0.010 to 0.029 | yes |

On RNA-available pairs, the intercept difference (−0.026) and mean-predicted difference (+0.003) have CIs excluding 0; these are small, unadjusted and descriptive. Observed improvement rate: 0.183 (all pairs), 0.186 (RNA-available pairs). Source: `figures/data/FigS_calibration/`, `artifacts/results_clean_rerun/calibration/probabilities/`.