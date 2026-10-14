# Supplementary Table S6. Held-out test performance of all 24 next-visit models (Figure 2)

Locked clean-rerun models; held-out test partition (155 patients), evaluated once. Information layers A–D as in Figure 1. Binary improvement (1,930 visit pairs; 353 improvement events, 18.3%) and six-class IMWG response (2,079 visit pairs) use different visit-pair populations and metrics and are not directly comparable. AUPRC, AUROC and macro-F1: higher is better; Brier score: lower is better (point estimate only). 95% patient-level bootstrap CIs (1,000 resamples of test patients; locked `artifacts/results_clean_rerun/uncertainty/main_model_cis.csv`), unadjusted. Validation-set values (point estimates) are shown for transparency; hyperparameters were tuned by cross-validation within training patients. No algorithm was selected as best; XGBoost is the reporting reference algorithm. Paired layer-to-layer differences (Figure 2C) are not reported here pending verification of their bootstrap provenance.

## Binary improvement

| Information layer | Algorithm | Test AUPRC (95% CI) | Test AUROC (95% CI) | Test Brier score | Validation AUPRC |
|---|---|---|---|---|---|
| A: current clinical state | Logistic Regression | 0.477 (0.432–0.530) | 0.781 (0.750–0.811) | 0.1219 | 0.419 |
| A: current clinical state | LightGBM | 0.477 (0.428–0.535) | 0.796 (0.768–0.824) | 0.1203 | 0.455 |
| A: current clinical state | XGBoost | 0.473 (0.427–0.532) | 0.797 (0.769–0.824) | 0.1211 | 0.441 |
| B: A + longitudinal history | Logistic Regression | 0.475 (0.429–0.528) | 0.784 (0.752–0.816) | 0.1219 | 0.431 |
| B: A + longitudinal history | LightGBM | 0.492 (0.444–0.551) | 0.813 (0.786–0.838) | 0.1168 | 0.465 |
| B: A + longitudinal history | XGBoost | 0.494 (0.446–0.552) | 0.810 (0.782–0.835) | 0.1180 | 0.462 |
| C: B + treatment context | Logistic Regression | 0.480 (0.433–0.533) | 0.790 (0.760–0.818) | 0.1211 | 0.435 |
| C: B + treatment context | LightGBM | 0.515 (0.460–0.574) | 0.815 (0.789–0.840) | 0.1154 | 0.474 |
| C: B + treatment context | XGBoost | 0.493 (0.443–0.551) | 0.813 (0.784–0.839) | 0.1175 | 0.474 |
| D: C + RNA pathway scores | Logistic Regression | 0.486 (0.439–0.538) | 0.789 (0.761–0.817) | 0.1204 | 0.432 |
| D: C + RNA pathway scores | LightGBM | 0.513 (0.463–0.567) | 0.817 (0.789–0.841) | 0.1159 | 0.475 |
| D: C + RNA pathway scores | XGBoost | 0.501 (0.454–0.554) | 0.810 (0.781–0.837) | 0.1176 | 0.479 |

## Six-class IMWG response

| Information layer | Algorithm | Test macro-F1 (95% CI) | Validation macro-F1 |
|---|---|---|---|
| A: current clinical state | Logistic Regression | 0.440 (0.395–0.480) | 0.477 |
| A: current clinical state | LightGBM | 0.640 (0.607–0.663) | 0.660 |
| A: current clinical state | XGBoost | 0.659 (0.628–0.682) | 0.669 |
| B: A + longitudinal history | Logistic Regression | 0.448 (0.403–0.488) | 0.491 |
| B: A + longitudinal history | LightGBM | 0.652 (0.620–0.675) | 0.667 |
| B: A + longitudinal history | XGBoost | 0.645 (0.612–0.669) | 0.658 |
| C: B + treatment context | Logistic Regression | 0.432 (0.393–0.464) | 0.468 |
| C: B + treatment context | LightGBM | 0.652 (0.620–0.675) | 0.663 |
| C: B + treatment context | XGBoost | 0.655 (0.624–0.678) | 0.657 |
| D: C + RNA pathway scores | Logistic Regression | 0.437 (0.388–0.479) | 0.448 |
| D: C + RNA pathway scores | LightGBM | 0.647 (0.615–0.671) | 0.658 |
| D: C + RNA pathway scores | XGBoost | 0.651 (0.619–0.676) | 0.653 |
