# Supplementary Figure S2 and Table S4, US-tuned revision (geographic analysis)

Response to Professor Song's feedback that evaluation patients must not influence model development. These files sit
beside, and do not replace, the earlier version (`../FigS_country/`, `../../supplementary_tables/Table_S4_geographic_robustness.md`,
`../../FIGURE_CAPTIONS_DRAFT.md`), which is unchanged. Adopting this revision means swapping those files and updating the
caption; that has not been done.

## What changed in the analysis

| | Earlier version | This revision |
|---|---|---|
| Hyperparameters | Reused from the main analysis, which tuned on the original training split (contained 80 of the 116 Spain/Canada patients) | Re-tuned (same 3-configuration grid, 5-fold patient-grouped CV, AUPRC / macro-F1) on **US patients in the original training partition and in US-train: 438 patients (binary), 439 (six-class)** |
| Patients in tuning that were evaluated | 80 of 116 | **0 of 116** (also 0 Italy, 0 US-validation, 0 original-validation/test patients) |
| Binary result | AUPRC 0.451 (0.401–0.507) | Identical (re-tuning selected the same parameters: 100 trees, depth 3, lr 0.05) |
| Six-class parameters | 200 trees, depth 4, lr 0.05 | 100 trees, depth 3, lr 0.05 |
| Six-class macro-F1, pooled | 0.644 (0.598–0.676) | 0.644 (0.596–0.675) (0.6441 → 0.6435) |
| Spain / Canada macro-F1 | 0.639 / 0.634 | 0.638 / 0.632 |
| US validation macro-F1 | 0.647 | 0.649 |
| Six-class calibration choice (US validation) | uncalibrated | temperature (T = 0.920) under the rule as implemented; see "Calibration" |

Unchanged by design: the US 82/18 split (632/139, seed 42), preprocessing fitted on US-train only, the final model fitted
on all of US-train, calibration method and binary threshold (0.25) chosen on US validation only, a single evaluation on
Spain + Canada (116 patients; 1,588 binary and 1,694 six-class pairs), Italy excluded, 1,000 patient-level bootstrap
resamples (seed 42).

## Calibration source (documentation correction)

The earlier script's docstring said hyperparameters and calibration were "taken from" the main results and
`calibration_summary.json`. The script never opens that file (it exists only on `origin/subgroup-analysis`). It re-implements
the main calibration design's candidate list and binary selection rule, and makes every selection by cross-fitting on US
validation only. The docstring of `modeling/us_to_spain_canada_model_c_ustuned.py` now says so.

**Rule discrepancy, now material for six-class.** The main analysis prefers the simpler method when cross-fitted Brier is
within 0.001; the country script's six-class rule uses cross-fitted log loss only. With the re-tuned model the two rules
disagree (log loss only: temperature, 0.9140 vs 0.9165; with Brier tie-break: uncalibrated, Brier 0.4417 vs 0.4418). The
applied rule (unchanged, to match the earlier script) selected temperature; the diagnostic is stored in the audit JSON.
This does not change any reported metric: temperature scaling rescales log-probabilities monotonically within each row, so
argmax and macro-F1 are identical (checked on the evaluation rows). It matters only if six-class probabilities are
reported. **Decision needed:** keep the log-loss-only rule, or align with the main rule (uncalibrated).

## Revised Supplementary Figure S2 caption (draft)

XGBoost information Layer C (143 source input features) was refit on US patients only, using a new random patient-level
82/18 split of US patients (632 training, 139 validation). Hyperparameters were re-tuned by 5-fold patient-grouped
cross-validation over the main analysis's three-configuration grid, using only US patients who were in both the original
training partition and the US training set (438 patients for the binary task, 439 for six-class); no Spain, Canada, Italy,
US-validation, original-validation or original-test patient entered tuning. Preprocessing and model fitting used US
training data; the calibration method and binary decision threshold (0.25; uncalibrated) were chosen on US validation. The
model was then evaluated once on all patients enrolled in Spain (80) and Canada (36). Patient sets were disjoint.

**(A)** Binary improvement AUPRC (higher is better). Spain and Canada pooled: 0.451 (95% CI 0.401–0.507; 116 patients,
1,588 pairs, 283 events, 17.8%); Spain 0.463 (0.407–0.532; 949 pairs, 19.5%); Canada 0.443 (0.369–0.541; 639 pairs, 15.3%).
Ticks mark each cohort's event rate, the AUPRC of a non-informative model. US validation, 0.525 (1,911 pairs, 18.4%), is a
descriptive reference only: it informed calibration and threshold choices and no CI was computed.

**(B)** Six-class IMWG response macro-F1 (higher is better): pooled 0.644 (0.596–0.675; 1,694 pairs); Spain 0.638
(0.588–0.675; 1,034 pairs); Canada 0.632 (0.491–0.683; 660 pairs); US validation 0.649 (2,076 pairs; no CI).

Intervals are 95% patient-level bootstrap percentile CIs (1,000 resamples). Selected hyperparameters were binary: 100 trees,
depth 3, learning rate 0.05; six-class: 100, 3, 0.05. This is a post hoc, exploratory internal geographic robustness check,
not independent external validation: an earlier version of this experiment, which reused hyperparameters tuned in the main
analysis on a split containing 80 of the 116 evaluated patients, had been seen before this redesign. Countries were not
compared statistically; Canada estimates are imprecise (36 patients). 57 Spain/Canada binary pairs had treatment-regimen
categories not seen in US training. Italy (138 patients) was excluded post hoc, after inspecting its outcome distribution
(single site, non-standard visit schedule, and an implausible 78.7% improvement rate); these patients remain in the main
analyses. AUROC (Supplementary Table S4): pooled 0.770 (0.729–0.808); Spain 0.761 (0.718–0.809); Canada 0.773
(0.700–0.847); US validation 0.805 (no CI).

## Files and how to regenerate (from the repository root, with `.venv`)

```
python modeling/us_to_spain_canada_model_c_ustuned.py                       # primary run (~80 s)
python modeling/us_to_spain_canada_model_c_ustuned.py --control-previous-params   # reproduction control
python figures/figS_country_ustuned_extract_data.py                          # 83 checks
python figures/publication_redesign/FigS_country_ustuned/figureS2A_plot.py
python figures/publication_redesign/FigS_country_ustuned/figureS2B_plot.py
python figures/supplementary_tables/build_table_s4_ustuned.py                # 13 checks
```

Outputs: `artifacts/results_clean_rerun/geographic_us_tuned/` (metrics and audit JSON; predictions CSV and models are
gitignored, as they hold patient identifiers), `.../geographic_us_tuned_control_previous_params/`,
`figures/data/FigS_country_ustuned/` (gitignored with the rest of `figures/data/`), the two PNGs here, and
`figures/supplementary_tables/Table_S4_geographic_robustness_ustuned.{md,csv}`.

## Verification summary

- **Reproduction control:** same code with the earlier hyperparameters reproduced the earlier results exactly (maximum
  absolute difference 0.0 in AUPRC/macro-F1, all CIs, thresholds, calibration choices). The pipeline and library versions
  (xgboost 3.4.1, scikit-learn 1.9.0) therefore do not explain any difference in the primary run.
- **Independent recount** (separate script, rebuilt from the split, subject and pair files): tuning-pool membership digests,
  patient and pair counts match the run; tuning ∩ {Spain/Canada, Italy, US-validation, original validation/test} = ∅;
  saved preprocessors equal an independent US-train-only refit and differ from one fit with evaluation patients.
- **Counts:** 632 / 139 / 116 patients (80 Spain, 36 Canada); 1,588 binary and 1,694 six-class evaluation pairs.

## Limits and open decisions

1. The final model is still fitted on all of US-train, which includes 88 US patients from the original validation split and
   105 from the original test split (193 of 632). Only tuning was restricted to the original training partition. If
   Professor Song's "validation patients" means the original validation split must never contribute to any model fitted in
   a geographic analysis, the fit population must also be restricted (a further rerun).
2. The tuning pool is restricted to US-train (not all 535 US patients in the original training partition) so that no
   US-validation patient, which sets calibration and threshold, enters tuning.
3. The six-class calibration rule discrepancy above.
4. The earlier country results were seen first; this is not a blind validation.
