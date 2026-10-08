# Supplementary Table S5. Survival discrimination and overall prediction error with and without pre-treatment RNA (Figure 4)

Survival cohort: patients with one RNA sample collected on or before the start of first treatment (CoMMpass index date); official MMRF PFS and OS. Ridge-penalized Cox models; clinical baseline = age at index, gender, ISS, exact first-line regimen; expanded baseline adds LDH, β2-microglobulin, albumin, creatinine, haemoglobin, calcium and bone-marrow plasma-cell %; + RNA = 50 Hallmark pathway scores. Harrell's C-index (higher is better) on pooled out-of-fold predictions per repeat of 5×5 patient-level cross-validation, averaged over repeats. ΔC 95% CIs: patient bootstrap (2,000 resamples) of fixed out-of-fold predictions. Integrated Brier score (IBS; IPCW, 365–1,095 days; lower is better) point estimates only. Exploratory internal cross-validation, not external validation; the cohort includes 100 of the 155 Figure 2 test patients.

| Endpoint | Analysis | Patients / events | C-index: baseline → + RNA | ΔC (95% CI) | ΔC range over 5 repeats | Permutation p (one-sided) | IBS: baseline → + RNA |
|---|---|---|---|---|---|---|---|
| PFS | Primary | 674 / 459 | 0.601 → 0.642 | +0.040 (0.016 to 0.064) | 0.031–0.050 | 0.005 (0/200) | 0.205 → 0.185 (Δ −0.020; no CI) |
| PFS | Expanded clinical baseline | 674 / 459 | 0.610 → 0.645 | +0.035 (0.014 to 0.056) | 0.025–0.047 | not performed | not computed |
| PFS | RNA day −30 to 0 only | 624 / 423 | 0.619 → 0.646 | +0.028 (0.003 to 0.053) | 0.021–0.039 | not performed | 0.204 → 0.183 (Δ −0.020; no CI) |
| OS | Primary | 674 / 247 | 0.658 → 0.702 | +0.043 (0.012 to 0.074) | 0.039–0.051 | not performed | 0.100 → 0.089 (Δ −0.011; no CI) |
| OS | Expanded clinical baseline | 674 / 247 | 0.656 → 0.704 | +0.049 (0.022 to 0.076) | 0.034–0.060 | not performed | not computed |
| OS | RNA day −30 to 0 only | — | not evaluated | — | — | not performed | — |

No CIs exist for absolute C-index or IBS values. The OS RNA-timing analysis was specified for PFS only and was not run. Rows differ in cohort or comparator and were not compared statistically. Source: `figures/data/Fig4/`, cross-checked against `artifacts/results_clean_rerun/survival_incremental/` and `survival_richer_baseline_sensitivity/final_summary.json`.