# Supplementary Tables S1–S5: methods and provenance

Built by `build_supplementary_tables.py` (run from the repository root). All values are read from verified figure data
(`figures/data/`) and cross-checked against locked results (`artifacts/results_clean_rerun/`; country results from
`origin/modeling-country-robustness` via `git show`). No model was trained or refit, and no patient identifiers or
row-level data are written. 358 automatic checks pass (`supplementary_tables_verification.csv`); input SHA-256 hashes are in
`supplementary_tables_provenance.json`.

| Table | Content | Linked figure | Evaluation set | Uncertainty |
|---|---|---|---|---|
| S1 | RNA contribution (XGBoost Layer D − C) by treatment class and response trajectory; all strata, complements, reference and secondary interaction contrasts; AUPRC, AUROC, log loss, RPS | Figure 3C | 5×5 patient-grouped CV, training + validation patients | 95% patient-cluster bootstrap (2,000); Holm (primary), BH (exploratory/secondary), complements unadjusted |
| S2 | Binary calibration and discrimination, Layer C and D, all test pairs and RNA-available test pairs; D − C paired differences | Figure S1 | Held-out test set | 95% paired patient bootstrap (1,000; calibrators fixed); unadjusted |
| S3 | SHAP shares: net family magnitude (Figure S3A) and total source-feature importance; primary, patient-equal-weighted and RNA-available sensitivities; six-class per class | Figure S3 | Held-out test set | 95% patient bootstrap (1,000; model fixed) |
| S4 | Geographic hold-out: AUPRC, AUROC, macro-F1, denominators, event rates, tuning overlap; US validation as reference | Figure S2 | Spain + Canada evaluated once | 95% patient bootstrap (1,000); none for US validation |
| S5 | Survival C-index, ΔC, repeat ranges, permutation result and integrated Brier score; expanded-baseline and RNA-timing sensitivities | Figure 4 | 5×5 patient-level CV in the survival cohort | ΔC: patient bootstrap (2,000) on fixed predictions; no CIs for C-index or IBS |

Notes
- **S2, RNA-available pairs:** per-layer values are point estimates recomputed from stored test probabilities with the
  locked formulas; each recomputed D − C equals the locked paired difference (checked). Per-layer CIs are not available.
- **S3:** the two measures are not interchangeable; S3A plots net family magnitude only. Shares use largest-remainder
  rounding so each column sums to 100.0%, as in Figure S3A.
- **S1:** besides the primary-metric findings, secondary metrics gave significant results after BH adjustment of two
  distinct kinds. Subgroup effects (within-stratum D − C): two rows, both RNA worsening (anti-CD38 six-class log loss;
  improving-trajectory binary AUROC). Interaction contrasts (difference in D − C between strata): three rows (IMiD vs
  not, binary AUPRC; corticosteroid vs not, binary log loss; improving − stable, six-class log loss), which show that
  the RNA effect differed between strata, not that RNA helped or harmed within one. All are secondary and exploratory;
  BH is applied within each secondary family, not across families.
- S2 and S3 are descriptive analyses of the held-out test set; S1, S4 and S5 are post hoc and exploratory. S4 is not
  external validation; S5 is internal cross-validation and its cohort includes 100 of the 155 Figure 2 test patients.
