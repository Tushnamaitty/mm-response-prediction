# Superseded analyses and their replacements

Four analyses were redesigned after Professor Song's feedback that validation and test patients must not take part in model
development (tuning, pathway selection, fitting, preprocessing or other learned decisions): the geographic analysis, the RNA
pathway selection (Figure 4D), the RNA-contribution analyses (Figures 4A–C, Table S1, low-variance add-on) and the survival
analyses (Figures 5–6, Table S5). The earlier files are kept unchanged for provenance and must not be used for the manuscript. All
redesigns are **post hoc and exploratory**: the earlier results were seen first, so none is a blind or independent validation, and
none is external validation. Each redesign followed a frozen, SHA-256-hashed protocol (development stage with held-out outcomes
unread, a hashed frozen artifact, then one held-out scoring run).

| Item | Superseded (kept, unchanged) | Replacement |
|---|---|---|
| Geographic analysis (Figure S2, Table S4) | `origin/modeling-country-robustness:modeling/us_to_spain_canada_model_c.py` and results (hyperparameters reused from the main analysis, tuned on a split holding 80 of the 116 evaluated patients; fitted on all US-train); intermediate `modeling/us_to_spain_canada_model_c_ustuned.py`, `artifacts/results_clean_rerun/geographic_us_tuned/`, `figures/publication_redesign/FigS_country_ustuned/`, `Table_S4_geographic_robustness_ustuned.md`; earlier `figures/figS_country_extract_data.py`, `figures/publication_redesign/FigS_country/`, `Table_S4_geographic_robustness.md` | `modeling/us_to_spain_canada_model_c_origtrain.py` → `artifacts/results_clean_rerun/geographic_origtrain/`; `figures/figS_country_origtrain_extract_data.py`; `figures/publication_redesign/FigS_country_origtrain/` (README holds the change log); `figures/supplementary_tables/build_table_s4_origtrain.py` → `Table_S4_geographic_robustness_origtrain.md` |
| Pathway selection and Figure 4D | `explainability/direction2_proliferation_replication.py`, `artifacts/results_clean_rerun/direction2_replication/proliferation_axis/` (family chosen from pooled data incl. test patients; train+validation discovery; test-only evaluation); panel D of `figures/fig3_extract_data.py`; `figures/publication_redesign/Fig3/figure3D.png` and `figure3D_plot.py` | `explainability/PATHWAY_TRAINONLY_PROTOCOL.md`; `explainability/direction2_pathway_selection_trainonly.py` (Stage 1); `explainability/direction2_proliferation_replication_trainonly.py` (Stage 2) → `artifacts/results_clean_rerun/direction2_replication_trainonly/`; `figures/fig3D_trainonly_extract_data.py`; `figures/publication_redesign/Fig3/figure3D_trainonly_plot.py` / `figure3D_trainonly.png` |
| RNA contribution: Figures 4A–C, Table S1, low-variance add-on | `figures/fig3_extract_data.py` (panels A–C); `figures/publication_redesign/Fig3/figure3A.png`, `figure3B.png`, `figure3C.png` and their `*_plot.py`; `figures/supplementary_tables/Table_S1_fig3C_subgroups.md` (pooled train + validation 5×5 CV in which validation patients trained the fold models); the pooled-CV analysis folders under `artifacts/results_clean_rerun/` | `explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md` (+ `.sha256`); `explainability/rna_contribution_trainonly_development.py` (Stage D) and `rna_contribution_trainonly_heldout.py` (Stage H) → `artifacts/results_clean_rerun/rna_contribution_trainonly/`; `figures/fig3ABC_trainonly_extract_data.py` (32 checks) → `figures/data/Fig3ABC_trainonly/`; `figures/publication_redesign/Fig3/figure3A_trainonly_plot.py`, `figure3B_trainonly_plot.py`, `figure3C_trainonly_plot.py` / `*_trainonly.png`; `figures/supplementary_tables/build_table_s1_trainonly.py` → `Table_S1_rna_subgroups_trainonly.md/.csv` |
| Survival: Figures 5–6, Table S5 | pooled 5×5 CV over all 674 patients: `figures/fig4_extract_data.py`, `figures/fig5_extract_data.py` (family column edited in place, see below), `figures/data/Fig4/`, `figures/data/Fig5/`, `figures/publication_redesign/Fig4/figure4A.png`, `figure4B.png`, `Fig5/figure5A.png`, `figure5B.png` and their `*_plot.py`; `Table_S5_survival.md` | `explainability/SURVIVAL_TRAINONLY_PROTOCOL.md` (+ `.sha256`); `explainability/survival_trainonly_development.py` and `survival_trainonly_heldout.py` → `artifacts/results_clean_rerun/survival_trainonly/`; `figures/fig45_trainonly_extract_data.py` (60 checks) → `figures/data/Fig45_trainonly/`; `figures/publication_redesign/Fig4/figure4A_trainonly_plot.py`, `figure4B_trainonly_plot.py`, `Fig5/figure5A_trainonly_plot.py`, `figure5B_trainonly_plot.py` / `*_trainonly.png`; `figures/supplementary_tables/build_table_s5_trainonly.py` → `Table_S5_survival_trainonly.md/.csv` |
| Figure 6 family column (old extractor) | `proliferation_myc_family` column and 7-name `LABELLED` list in `figures/fig5_extract_data.py` | edited in place: `train_selected_pathway_family` (read from the Stage-1 selection artifact), `LABELLED` = E2F, G2-M, Myc V1, Notch; **the old extractor is not re-run** (inputs missing, see below); the train-only extractor above supersedes it |
| Figure 1 (study overview) | first draft (amber rows; "80 of 116"; "redesign pending"; CV rows for RNA contribution and survival) | edited in place: `figures/publication_redesign/fig1_new/` (panel E rows now state the train-only development and held-out counts, verified against the result JSONs) |
| Captions | earlier Figure 4, 5, 6 and Figure S2 text | edited in place: `figures/FIGURE_CAPTIONS_DRAFT.md`; numbers checked by `figures/check_trainonly_captions.py` |

## Key differences in results

* **Pathway family:** E2F Targets, G2-M Checkpoint, Myc Targets V1, Mitotic Spindle and **Spermatogenesis** (earlier: Myc Targets V2 in
  place of Spermatogenesis). The composite is called the "train-selected pathway composite", not a proliferation/MYC axis.
* **Held-out evaluation (validation + test, 160 patients):** odds ratio 0.76 per SD (95% CI 0.67–0.87), meeting the rule set
  for this analysis (95% CI entirely below 0). The earlier test-only estimate was 0.82 (0.69–0.97; 84 patients).
* **Geographic:** pooled AUPRC 0.426 (0.374–0.484) and macro-F1 0.629 (0.585–0.658); earlier 0.451 and 0.644.
* **RNA contribution (held-out, 213 patients with RNA):** binary ΔAUPRC +0.0109 (−0.0026 to +0.0248); six-class ΔRPS −0.00006
  (−0.00114 to +0.00104); the earlier pooled estimates (+0.0048; +0.00075, a significant worsening) are superseded. Add-on ΔRPS
  −0.00018 (−0.00063 to +0.00028), permutation p = 0.0109: the success rule (CI upper bound < 0) is **not met** (earlier pooled: −0.00034,
  CI excluding 0, p = 0.005). No stratum is significant on its primary metric after adjustment; the earlier "RNA significantly worsened RPS in PI,
  IMiD, steroid and stable-trajectory pairs" is not reproduced.
* **Survival (held-out, 201 patients):** PFS C 0.630 → 0.657, ΔC +0.028 (−0.019 to +0.074), permutation p = 0.0001; **pre-specified
  criterion not met** (CI includes 0). OS ΔC +0.010 (−0.045 to +0.065). Earlier pooled PFS ΔC +0.040 (+0.016 to +0.064) and OS +0.043
  (+0.012 to +0.074) are superseded. Figure 6 refit within the 473 train patients: 12 concordant pathways (9 higher / 3 lower hazard;
  earlier 19 = 13 / 6; r = 0.81, earlier 0.83); of the Figure 4D family, E2F, G2-M, Myc V1 and Mitotic Spindle are concordant and
  **Spermatogenesis is not** (the earlier statement that all five were concordant is removed).

## Main-model threshold and calibration: validation-based decisions removed from the reported analysis

**What the code does (legacy, unchanged).** `modeling/train_models.py` tunes hyperparameters and fits preprocessing on train patients only, with no early stopping or
validation evaluation set. For the binary models it then selects the decision threshold that maximises F1 on the validation patients (0.25–0.35; `select_threshold`) and
writes threshold-dependent metrics (sensitivity, specificity, precision, recall, F1) to the per-model metrics JSONs. `modeling/calibration_analysis_clean_rerun.py`
(branch `subgroup-analysis`) compared uncalibrated / Platt / isotonic (binary) and uncalibrated / temperature (six-class) by 5-fold patient-grouped cross-fitting within the
validation patients and refit the candidates on all validation rows. Under the stated requirement (validation patients take no part in thresholds or calibration) these steps
are not acceptable as development decisions.

**Why no reported number changes.** Every reported main-model metric is threshold-free or argmax-based (AUPRC, AUROC, Brier score, six-class macro-F1; `main_model_cis.csv`
contains only AUPRC, AUROC and macro-F1). The binary comparison selected *uncalibrated* for both Layer C and Layer D, so Figure S1 and Table S2 show raw test probabilities
with no validation-fitted transformation. The six-class Layer D selection (temperature scaling) cannot change the argmax class and six-class calibration is not shown. The
threshold-dependent metrics appear in no figure, table or caption.

**What was changed (documentation and figure text only; no model, result file or protocol was touched).** The captions (Figure 1 D, Figure 2, Figure 3, Supplementary
Figure S1), the Figure 1 panel D validation box and footnote, the Figure S1A subtitle (image regenerated; only the subtitle strip differs from the earlier PNG), Tables S2 and S6
notes and the S1 extractor labels now state that probabilities are shown raw, that no decision threshold or recalibration enters any reported metric and that validation patients are
descriptive only. The geographic analysis (Figure S2, Table S4) already selects its threshold (0.20) and calibration on US train-partition patients only and its metrics are
threshold-free, so it is unchanged.

**Honest disclosure.** The decision to show raw probabilities is *not blind* to validation data: the exploratory validation-set comparison that favoured "uncalibrated" preceded
it. The comparison is retained only in the legacy `calibration_summary.json` (branch `subgroup-analysis`) and in the legacy per-model metrics JSONs, and is not a reported result.
`rerun_freshness_predictions.py` also selects a threshold on validation; its outputs (`uncertainty/rna_freshness_*`) appear in no caption or table.

**If an operating-point threshold or recalibrator is ever reported:** derive it from patient-grouped *train out-of-fold* predictions with the locked hyperparameters (not done; it
would require recomputing out-of-fold fits and your approval).

## Figure numbering and algorithm wording (applied 2026-10-14)

Scheme A: new study overview = Figure 1; the earlier Figure 1 (study design) = Figure 2; the earlier Figures 2–5 = Figures 3–6 (performance 3; RNA contribution and
pathway composite 4; survival 5; pathway coefficients 6); Supplementary Figures S1–S3 are unchanged. Manuscript-facing text (captions, tables, READMEs, documentation) and the images that drew a
figure label (Figures 1 [overview], 2, 3A–C, 6A–B) use the new numbers; this file names the superseded earlier versions of a figure by its new number. File names, folder names, extractor
and protocol text and the hash-chained artifacts keep the earlier working labels (for example `figure3A_trainonly.png` is Figure 4A; the frozen protocols still say "Figures 3A-C").
The full mapping is in `figures/FIGURE_NUMBERING.md`.

Algorithm wording: no algorithm was prespecified as primary, so none is called best. The phrase "reporting reference algorithm" is removed from captions, Table S6, the S1 data
definitions and the footnote drawn in `figureS3A.png`. Legacy statements outside the manuscript files that call a model "best" (for example the descriptions in the gitignored
`artifacts/results_clean_rerun/xai/publication_figures/publication_figures_manifest.csv` and the earlier synthesis text `explainability/step12.txt`, which is not in the repository) selected
models on test-set results and must not be used.

## Choice of XGBoost: search for an a priori justification (2026-10-14)

Searched: every text file (`.md`, `.txt`, `.py`) on all local and remote branches (`git grep`), the full commit history (`git log -S"xgboost"`), the pre-training design commit
`8413b3d` (Steps 9–12), `opus_plan.md`, the frozen protocols and the repository README.

* No commit before the first training results mentions any algorithm: `8413b3d` (2026-09-13 00:05) contains none; the first committed mention of XGBoost is `597037b` (2026-09-13 21:38,
  "Step 13 preliminary training and RNA sensitivity analyses"), the same commit that holds the preliminary results.
* `explainability/step12.txt` (branch `subgroup-analysis`) calls C-LightGBM (binary) and A-XGBoost (six-class) "the best-performing models" on the test set: a post hoc, test-based statement, not a
  prespecification.
* `PRIMARY_ALGORITHM = "xgboost"` (`explainability/subgroup_analysis_common.py`, branch `subgroup-analysis`, 2026-09-19) is described as "the same pair as the whole-cohort
  `rna_subset_paired_analysis_clean_rerun.py` D − C rows" and was written after the clean-rerun results (2026-09-14). It records a convention, not a reason.
* The country-robustness README says only "Model C (XGBoost)"; the calibration and SHAP scripts use XGBoost without comment.

**Conclusion:** no a priori justification for using XGBoost alone is recorded. **What was done:** the captions of Figures 3, 4, S1, S2 and S3 and Tables S1–S4 and S6 now state that the
analyses are XGBoost-specific, that no rationale is recorded, and that they do not show XGBoost to be superior to Logistic Regression or LightGBM (no algorithm is called best). **Still open:** if a
justification exists outside the repository (pre-analysis plan, correspondence), add it to the Methods; otherwise the Methods should say that XGBoost was used without a prespecified rationale.

## Figure 3C paired intervals: independent recomputation (2026-10-14)

`figures/verify_fig3c_paired_cis.py` recomputes the 24 paired differences and intervals of Figure 3C (earlier Figure 2C; `figures/data/Fig2/fig2_paired_deltas_full_test.csv`) from the 24 locked
test-prediction files, using the repository's patient-bootstrap function (`modeling/regenerate_main_model_cis_clean_rerun.py`: seed 42, 1,000 resamples, all rows of a sampled patient, percentile
2.5/97.5) with the same resample for both layers. Result: all 24 rows reproduce the plotted differences, the locked point-estimate differences and **both interval limits to 10⁻¹⁶**; the
"interval includes zero" status agrees in 24 of 24 rows. Three other seeds (7, 123, 2024) move the limits by at most 0.0053; all D − C intervals include zero under every seed, and the status
changes only for three borderline non-D − C rows (XGBoost binary B − A [lower limit 0.0011] and D − A [0.0000], XGBoost six-class C − B). **Limit of the verification:** the original extraction script
is still not in the repository (on no branch), so what is verified is that the plotted values are exactly what the repository's own procedure produces from the locked predictions, not the
identity of the original script. The recomputation reads predictions only; no model was trained and no result file was changed (its aggregate output
`figures/data/Fig2/fig2_paired_deltas_independent_check.csv` is gitignored and regenerable in about five minutes).

## Supplementary Figure S3A: source data restored and hash-verified; footnote replaced (2026-10-15)

On 2026-10-14 (about 23:35) a copy of the Supplementary Figure S3 data folder appeared in the working tree as `figures/data/FigS_Shap/` (the origin of the copy is not recorded; it is assumed to be a restoration from the original machine).
The scripts, the extractor and every provenance record spell the folder `FigS_shap`, which only resolved on a case-insensitive file system, so on 2026-10-15 the folder was renamed to `FigS_shap`
(contents untouched: all 8 file hashes identical before and after; still excluded by `figures/data/*` in `.gitignore`). Before that, the three S3A files had been searched for in the working tree, on all
local and remote branches and their history, in the reflog and stash, and on the local drive, and were nowhere. **Verification of the restored copy (read-only, nothing recomputed):**

* `figS3A_ext_component_shares.csv` (`9eee3d11…`), `figS3A_ext_component_shares_per_class.csv` (`3334d7de…`) and `figS3A_family_shares.csv` (`3dea4658…`) match, byte for byte, the SHA-256 hashes
  recorded earlier in `figures/supplementary_tables/supplementary_tables_provenance.json`. `figS3B_top15_source_features.csv` (`a1e7a33f…`), `figS3_ext_additivity.csv` and the binary source-level SHAP file
  match the hashes in the extension provenance `figS3_ext_provenance.json`, whose 15 recorded inputs (including `figS_shap_extend.py`, both locked XGBoost Layer D models and their predictions) also match the
  current repository files. The extension script uses native XGBoost `pred_contribs` (the SHAP package is not used), as the caption says; additivity error is at most 2.9 × 10⁻⁶ for all seven outputs.
* The data checks of `figureS3A_plot.py` pass on the restored file, and `build_supplementary_tables.py`'s own `table_s3()` (run in memory with the write step disabled; nothing was written)
  reproduces `Table_S3_shap_attribution.md` byte for byte with its 72 builder checks passing; separately, all 72 share/CI cells and 36 per-class cells of the table equal the CSV values.
* Independent recomputation of the shares (not from the restored CSVs): six-class from the older branch artifact `xai/shap_artifacts/multiclass_d_xgboost_*` (exact TreeSHAP of the locked model on the same
  2,079 test pairs; additivity 2.7 × 10⁻⁶) for all three analyses, all total-importance shares and the per-class table, with bootstrap CIs matching (seed 42); binary shares **and bootstrap CIs** from the restored source-level SHAP
  file (net family magnitude and total importance, all test pairs, patients weighted equally and RNA-available pairs; the RNA-available bootstrap resamples all 155 patients and keeps their RNA rows; all 36 share-and-CI triples
  (6 components x 2 measures x 3 analyses) per task match the CSV to 10⁻¹⁴ percentage points for binary and to 3 × 10⁻⁵ for six-class). The Figure S3B
  whiskers (1,000-resample patient bootstrap, seed 42, the repository's bootstrap function) were recomputed from the same two sources and match the plotted values (binary to 10⁻¹⁶, six-class to 10⁻⁶, the precision of the older
  float32 artifact); `python figures/verify_figS3_inputs.py` repeats the hash checks, the Table S3 cell comparison, the share and CI recomputation (net and total, three analyses, both tasks) and the S3B whisker recomputation (156 checks; read-only;
  it needs the patient-level restored file, which stays local). Only the per-class six-class CIs of Table S3 (in the CSV) were not independently recomputed. No discrepancy was found, so `figureS3A.png` was not changed beyond its footnote.
* **Not recoverable:** 13 of 16 protected earlier files and 8 of 11 extension outputs listed in `figS3_ext_provenance.json` are absent (among them `figS_shap_compute.py`, `figS3_ext_verification_checks.csv`
  and the six-class per-class SHAP files; list in `figures/LEGACY_MISSING_INPUTS.md`, section C). The provenance record's claim of 256 passed checks therefore cannot be confirmed here.

**What was changed:** the drawn footnote of `figureS3A.png` ("reporting reference algorithm, not a validation-selected best model") was replaced in `figureS3A_plot.py` by "Letters = information layer. Locked
XGBoost layer D, used as a single fixed model; no algorithm is claimed to perform better." and the image was regenerated from the verified CSV (the earlier PNG is byte-identical to the committed one and was
kept for comparison outside the repository). A pixel comparison with the earlier image shows that only the footnote band differs (28,399 pixels, all within rows 1581–1627 of 1,680; the data, bars and
labels are unchanged). The older SHAP artifacts on branches (LightGBM binary C/D and XGBoost six-class A/D) were used only for the independent six-class comparison above, never as a substitute for the S3A data.
No SHAP value was recomputed and no model was run. The data folder is excluded by `figures/data/*` in `.gitignore` and includes a patient-level source-SHAP file (`figS3_source_shap_binary_test.csv.gz`),
so it must stay out of Git.

## Reproducing the redesigned analyses

The pathway analyses need `statsmodels==0.15.0` (now pinned in `requirements.txt` together with its dependencies `formulaic`,
`interface_meta`, `patsy` and `wrapt`; verified to import and fit an ordinal GEE alongside the repository's other pins, and to
reproduce the earlier Direction 2 results). The geographic and RNA-contribution analyses additionally need `xgboost` (the analysis used 3.4.1) and
scikit-learn 1.9.0; **xgboost and lightgbm are not in `requirements.txt`** (a gap that predates this revision). The survival
analyses use the project environment (numpy, pandas, scipy, scikit-learn); no new package is needed.

```
python -m explainability.direction2_pathway_selection_trainonly              # Stage 1
python -m explainability.direction2_proliferation_replication_trainonly      # Stage 2 (stops before held-out outcomes if a gate fails)
python figures/fig3D_trainonly_extract_data.py
python figures/publication_redesign/Fig3/figure3D_trainonly_plot.py
python modeling/us_to_spain_canada_model_c_origtrain.py
python figures/figS_country_origtrain_extract_data.py
python figures/publication_redesign/FigS_country_origtrain/figureS2A_plot.py   # and figureS2B_plot.py
python figures/supplementary_tables/build_table_s4_origtrain.py
# RNA contribution and survival: Stage D, then Stage H (each verifies protocol and artifact hashes first; they refuse to overwrite)
python -m explainability.rna_contribution_trainonly_development ; python -m explainability.rna_contribution_trainonly_heldout
python -m explainability.survival_trainonly_development ; python -m explainability.survival_trainonly_heldout
python figures/fig3ABC_trainonly_extract_data.py
python figures/publication_redesign/Fig3/figure3A_trainonly_plot.py            # and figure3B / figure3C
python figures/supplementary_tables/build_table_s1_trainonly.py
python figures/fig45_trainonly_extract_data.py
python figures/publication_redesign/Fig4/figure4A_trainonly_plot.py            # and figure4B
python figures/publication_redesign/Fig5/figure5A_trainonly_plot.py            # and figure5B
python figures/supplementary_tables/build_table_s5_trainonly.py
python figures/publication_redesign/fig1_new/figure1_new_plot.py
python figures/check_trainonly_captions.py
python figures/figS_calibration_extract_data.py                                   # Figure S1 data (reads the local probabilities + branch via git show)
python figures/publication_redesign/FigS_calibration/figureS1A_plot.py            # and figureS1B_plot.py
python figures/audit_legacy_inputs.py                                             # writes figures/LEGACY_MISSING_INPUTS.md (read-only audit)
python figures/verify_fig3c_paired_cis.py                                          # Figure 3C paired-interval recomputation (about 5 minutes; reads locked predictions only)
python figures/verify_figS3_inputs.py                                             # Figure S3 / Table S3 hashes, table cells, share and S3B whisker recomputation (read-only; needs the local FigS_shap folder)
python figures/check_figure_numbering.py                                         # checks the final figure numbering in the manuscript-facing files
```

The analysis scripts refuse to overwrite existing outputs. The Stage D / Stage H scripts need the gitignored patient-level inputs under `data/`.

## Known blockers

* The old extractors cannot run on this machine and were not used for the rebuild. `figures/fig3_extract_data.py` (panels A–C and the old
  panel D) needs `direction1_sensitivity/entropy_slope/entropy_slope_primary_and_secondary.csv`,
  `rna_low_variance_incremental/permutation_null.csv` and the old `discovery_composite_loadings.csv`. `figures/fig5_extract_data.py` needs
  `survival_pathway_interpretation/{pathway_stability_both_endpoints,pfs_pathway_stability,os_pathway_stability,pfs_fold_coefficients,
  os_fold_coefficients}.csv`, and `figures/fig4_extract_data.py` needs the old `survival_incremental` out-of-fold files. These are CSV
  files excluded by the `*.csv` rule in `.gitignore`; they were searched for locally and on every branch (`git ls-tree`) and are on none.
  Consequently the old extracted data folders `figures/data/Fig3/`, `Fig4/` and `Fig5/` are also absent (not tracked), and the old
  `figure3A/3B/3C`, `figure4A/4B` and `figure5A/5B` plot scripts cannot be re-run; only their existing PNGs remain.
  The hashes of the Figure 6 inputs are in the folder's `output_manifest.json`, so copies from the original machine could still be
  verified. Nothing was fabricated or substituted; the superseded figures and tables were not regenerated.
* `figures/supplementary_tables/build_supplementary_tables.py` (S1–S5 in one run) cannot run for the same reason (it needs `figures/data/Fig3`,
  `Fig4`; its Table S3 function does run on the restored `FigS_shap` data); its S4 function would also regenerate the superseded S4. The train-only S1, S4 and S5 have their
  own builders.
* **Privacy caveat:** `artifacts/results_clean_rerun/survival_trainonly/frozen_models.json` (not matched by the `*.csv` rule) stores the
  Breslow baseline-hazard and Kaplan–Meier censoring step functions, i.e. the distinct day-level event/censoring times of the 473 train
  patients (no identifiers). It is excluded **locally** via `.git/info/exclude` (not versioned; `.gitignore` unchanged; `git add -f` would still override it, and a fresh clone
  does not carry the rule), the file itself and its hash (`3b4b79a6…`, recorded in `heldout_report.json`) are untouched, and the per-patient out-of-fold and held-out prediction CSVs are
  gitignored. Publish the hash only; if a public model card is needed, build a separate file without the step functions.
* **Table S5 legacy references:** the pooled-CV integrated Brier score and repeat ranges exist only in the superseded `Table_S5_survival.md`; the new Table S5 transcribes them
  verbatim, marked †, after checking that the legacy row agrees with the verified pooled C-index, ΔC and CI (13 checks). They cannot be re-derived here.
* **Main-model threshold and calibration:** resolved by removal from reporting, not by rerunning anything (next section).
* **Figure numbering:** Scheme A is applied (section below; `figures/FIGURE_NUMBERING.md`). Residual earlier working labels are listed there.
* **Missing legacy inputs and outputs:** itemised, with recorded SHA-256 hashes, in `figures/LEGACY_MISSING_INPUTS.md` (generated by `figures/audit_legacy_inputs.py`).
* **Choice of XGBoost:** no a priori rationale exists (section "Choice of XGBoost"); the captions and Tables S1–S4 describe the XGBoost-only analyses as XGBoost-specific.
* **Supplementary Figure S3A image:** source data restored and hash-verified; footnote replaced and image regenerated (section "Supplementary Figure S3A"). Not recoverable: 13 protected earlier files and 8 extension outputs, including the verification-checks file.
* **Figure 3C (earlier Figure 2C) intervals:** resolved by independent recomputation (section "Figure 3C paired intervals"); the original extraction script remains absent.
* **Imprecise held-out strata:** 42–207 patients per Figure 4C stratum (binary 42–206) and 100–101 per partition in Figure 5; intervals are wide.
