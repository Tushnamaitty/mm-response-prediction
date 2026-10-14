# Supplementary Tables S1–S5: methods and provenance

**Train-only redesign (current).** Tables S1, S4 and S5 were rebuilt after the validation-leakage revision (validation and test patients take no part in tuning, selection, fitting or preprocessing; held-out patients are scored once). The current files are `Table_S1_rna_subgroups_trainonly.md/.csv` (`build_table_s1_trainonly.py`, 6 checks), `Table_S4_geographic_robustness_origtrain.md` (`build_table_s4_origtrain.py`) and `Table_S5_survival_trainonly.md/.csv` (`build_table_s5_trainonly.py`, 13 checks). The earlier `Table_S1_fig3C_subgroups.md`, `Table_S4_geographic_robustness*.md` (other than origtrain) and `Table_S5_survival.md` are superseded and kept for provenance only; the rows of the table below that describe pooled 5×5 CV (S1, S5) refer to those superseded versions. See `../SUPERSEDED_ANALYSES.md`. Missing legacy inputs: `../LEGACY_MISSING_INPUTS.md`. Figure numbers in this folder follow the final scheme (Scheme A: Figure 1 overview, 2 design, 3 performance, 4 RNA contribution, 5 survival, 6 pathway coefficients; S-figures unchanged): `../FIGURE_NUMBERING.md` (mapping and residual earlier working labels; the pre-application audit is `../FIGURE_NUMBERING_AUDIT.md`). File names such as `Table_S6_full_figure2_performance.md` keep their earlier working names.

Built by `build_supplementary_tables.py` (run from the repository root). All values are read from verified figure data
(`figures/data/`) and cross-checked against locked results (`artifacts/results_clean_rerun/`; country results from
`origin/modeling-country-robustness` via `git show`). No model was trained or refit, and no patient identifiers or
row-level data are written. 358 automatic checks pass (`supplementary_tables_verification.csv`); input SHA-256 hashes are in
`supplementary_tables_provenance.json`.

| Table | Content | Linked figure | Evaluation set | Uncertainty |
|---|---|---|---|---|
| S1 | RNA contribution (XGBoost Layer D − C) by treatment class and response trajectory; all strata, complements, reference and secondary interaction contrasts; AUPRC, AUROC, log loss, RPS. **Current (train-only):** `Table_S1_rna_subgroups_trainonly.md`; locked train-only models scored once on 213 validation + test patients with RNA, plus a development-only 5×5 CV reference in 716 train patients. (Superseded: pooled train + validation 5×5 CV.) | Figure 4C | held-out validation + test patients with RNA | 95% patient-cluster bootstrap (2,000); Holm (primary), BH (exploratory/secondary), complements unadjusted |
| S2 | Binary calibration and discrimination, Layer C and D, all test pairs and RNA-available test pairs; D − C paired differences | Figure S1 | Held-out test set | 95% paired patient bootstrap (1,000; calibrators fixed); unadjusted |
| S3 | SHAP shares: net family magnitude (Figure S3A) and total source-feature importance; primary, patient-equal-weighted and RNA-available sensitivities; six-class per class | Figure S3 | Held-out test set | 95% patient bootstrap (1,000; model fixed) |
| S4 | Geographic hold-out (original-train-only revision): AUPRC, AUROC, macro-F1, denominators, event rates; US calibration set as reference. File: `Table_S4_geographic_robustness_origtrain.md` | Figure S2 | Spain + Canada evaluated once | 95% patient bootstrap (1,000); none for the US calibration set |
| S5 | Survival C-index, ΔC, permutation result and integrated Brier score; expanded-baseline, RNA-timing and partition-level analyses; development-only CV and earlier pooled estimates as references. **Current (train-only):** `Table_S5_survival_trainonly.md`. (Superseded: pooled 5×5 CV over all 674 patients.) | Figure 5 | frozen models fitted on 473 train patients, scored once on 201 held-out validation + test patients | ΔC: patient bootstrap (2,000) on held-out predictions; permutation (B = 10,000, PFS only); no CIs for C-index or IBS |

Notes
- **S2, RNA-available pairs:** per-layer values are point estimates recomputed from stored test probabilities with the
  locked formulas; each recomputed D − C equals the locked paired difference (checked). Per-layer CIs are not available.
- **S3:** the two measures are not interchangeable; S3A plots net family magnitude only. Shares use largest-remainder
  rounding so each column sums to 100.0%, as in Figure S3A.
- **S1 (train-only, current):** no within-stratum test on the primary metric is significant after adjustment; two six-class trajectory interaction contrasts (RPS) and three secondary-metric contrasts have BH p < 0.05 (exploratory, descriptive). Treatment strata receiving inference were fixed from train-patient counts before the held-out data were read. The superseded-S1 note that follows describes the earlier pooled-CV table.
- **S1 (superseded pooled-CV table):** besides the primary-metric findings, secondary metrics gave significant results after BH adjustment of two
  distinct kinds. Subgroup effects (within-stratum D − C): two rows, both RNA worsening (anti-CD38 six-class log loss;
  improving-trajectory binary AUROC). Interaction contrasts (difference in D − C between strata): three rows (IMiD vs
  not, binary AUPRC; corticosteroid vs not, binary log loss; improving − stable, six-class log loss), which show that
  the RNA effect differed between strata, not that RNA helped or harmed within one. All are secondary and exploratory;
  BH is applied within each secondary family, not across families.
- **S4 (revised).** Built by `build_table_s4_origtrain.py` (13 checks; run after `figures/figS_country_origtrain_extract_data.py`),
  not by `build_supplementary_tables.py`. Tuning, preprocessing, fitting, calibration and threshold used only US patients of the
  original training partition (439 fit, 96 calibration); 236 US validation/test-partition patients and Italy were excluded.
  The earlier S4 (`Table_S4_geographic_robustness.md`; hyperparameters reused from the main analysis, fitted on all US-train)
  and the intermediate `Table_S4_geographic_robustness_ustuned.md` are superseded and kept for provenance only; the S4
  function in `build_supplementary_tables.py` still reads the earlier results from `origin/modeling-country-robustness` and
  would regenerate the superseded table (do not use it for S4).
- S2 and S3 are descriptive analyses of the held-out test set; S1, S4 and S5 are post hoc and exploratory. S4 is not
  external validation; the superseded S5 was internal cross-validation whose cohort included 100 of the 155 Figure 3 test patients.
- **S5 (train-only, current):** held-out ΔC for PFS is +0.028 (95% CI −0.019 to +0.074; permutation p = 0.0001); the pre-specified criterion (CI lower bound > 0) was not met. The permutation null is centred below 0 because permuted RNA adds noise to a frozen model, so the permutation p tests association rather than ΔC > 0. The OS RNA-timing analysis was not run. The 201 held-out patients are the 101 validation and 100 test patients of the survival cohort; S5 is post hoc, exploratory and not independent or external validation. Integrated Brier scores are point estimates without CIs.
- **Legacy build status:** `build_supplementary_tables.py` (S1–S5 in one run) cannot run on this machine (its `figures/data/Fig3|Fig4` inputs are absent; its Table S3 function runs on the restored `figures/data/FigS_shap` data and reproduces `Table_S3_shap_attribution.md` byte for byte); the "358 automatic checks" and `supplementary_tables_provenance.json` above describe that earlier build and the superseded S1, S4 and S5. Its S2 and `build_table1_and_S6.py` S6 note strings were edited in place only to match the current wording (raw probabilities; validation values descriptive); no number changed and neither builder was re-run for those tables.
- **S5 (train-only):** the pooled-CV integrated Brier score and repeat ranges are transcribed (marked †) from the superseded `Table_S5_survival.md` because their source files are unavailable; the other pooled values come from the verified extraction. 13 checks.
- **S2 and S6:** no decision threshold or recalibration is used; probabilities are raw and validation values are descriptive only (the earlier exploratory validation comparison is disclosed in the Figure S1 caption and `../SUPERSEDED_ANALYSES.md`).
