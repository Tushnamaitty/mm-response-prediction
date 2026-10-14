# Supplementary Figure S2 and Table S4, original-train-only revision (geographic analysis)

Second revision for Professor Song's feedback that patients outside the training partition must not take part in any
model-development step. The first revision (`../FigS_country_ustuned/`) restricted only hyperparameter tuning; its final
model, preprocessing, calibration and threshold still used 193 + 43 US patients from the original validation and test
partitions. Here **every learned quantity** uses only US patients from the original training partition.
All earlier versions (`../FigS_country/`, `../FigS_country_ustuned/`, `Table_S4_geographic_robustness*.md`,
`../../FIGURE_CAPTIONS_DRAFT.md`) are unchanged; adopting this version means swapping files and updating the caption,
which has not been done.

## Design

| Step | Population |
|---|---|
| Hyperparameter tuning (5-fold patient-grouped CV, main-analysis 3-configuration grid, AUPRC / macro-F1) | **Fit set: 439 US patients** (438 for binary: one has no binary-eligible pair) in the original training partition |
| Preprocessing (imputation, one-hot encoding, column typing) | Fit set |
| Final model | Fit set |
| Calibration method and binary threshold (cross-fitted, then frozen) | **Calibration set: 96 US patients** in the original training partition |
| Excluded from every step | 236 US patients in the original validation (110) and test (126) partitions; 138 Italian patients |
| Evaluation (once) | 116 Spain (80) + Canada (36) patients; 1,588 binary / 1,694 six-class pairs (unchanged) |

Fit and calibration sets are the earlier random 82/18 US split (seed 42; 632/139) restricted to original-train
patients, so no new random draw was introduced. Of the 771 US patients, 535 are in the original training partition
(439 + 96). Spain/Canada patients are in no development set irrespective of their original partition.

## Results and changes

| Metric (95% patient-bootstrap CI) | Earlier experiment (hyperparameters reused) | US-tuned revision (fit on all US-train) | **This revision** |
|---|---|---|---|
| Binary AUPRC, pooled | 0.451 (0.401–0.507) | identical | **0.426 (0.374–0.484)** |
| Binary AUPRC, Spain / Canada | 0.463 / 0.443 | identical | **0.432 (0.375–0.514) / 0.430 (0.362–0.523)** |
| Binary AUROC, pooled | 0.770 (0.729–0.808) | identical | 0.769 (0.731–0.805) |
| Binary threshold / calibration | 0.25 / uncalibrated | same | **0.20** / uncalibrated |
| Binary hyperparameters | 100 trees, depth 3, lr 0.05 | same | same (re-selected) |
| Six-class macro-F1, pooled | 0.644 (0.598–0.676) | 0.644 (0.596–0.675) | **0.629 (0.585–0.658)** |
| Six-class macro-F1, Spain / Canada | 0.639 / 0.634 | 0.638 / 0.632 | **0.617 (0.568–0.655) / 0.634 (0.491–0.688)** |
| Six-class hyperparameters | 200 trees, depth 4, lr 0.05 | 100, 3, 0.05 | 100, 3, 0.05 |
| Six-class calibration (rule unchanged) | uncalibrated | temperature | temperature (T = 0.917) |
| US reference set, AUPRC / macro-F1 | 0.525 / 0.647 (139 patients) | 0.525 / 0.649 | 0.558 / 0.632 (96 patients) |

Training pairs fall from 8,630 to 6,133 (binary) and from 9,483 to 6,746 (six-class). Pooled AUPRC and macro-F1 are
about 0.025 and 0.015 lower than before; the intervals overlap heavily and no test of the differences was done (they
reflect both the smaller development set and a different threshold/preprocessing; the bootstrap covers
evaluation-set sampling only, not model-development variability). The calibration set has 96 patients, so the binary
threshold (0.20) rests on less data than before. Conclusions about generalization remain exploratory.

**Calibration rule (unchanged, flagged).** The six-class rule is cross-fitted log loss only, as in the earlier experiment.
With this model log loss selects temperature scaling (0.9237 vs 0.9273) while the main-analysis rule (simpler method if
cross-fitted Brier within 0.001) would select uncalibrated (Brier 0.4393 vs 0.4403). No reported metric is affected:
temperature scaling is a monotone within-row rescaling, so argmax and macro-F1 are identical (checked on the evaluation
rows). The disagreement is recorded in the audit JSON. Decision needed if six-class probabilities will be reported.

## Revised Supplementary Figure S2 caption (draft)

XGBoost information Layer C (143 source input features) was developed only on US patients in the original training
partition of the main analysis (535 of 771 US patients), using the earlier random 82/18 US split restricted to those
patients: 439 patients for hyperparameter tuning, preprocessing and model fitting, and 96 for the calibration method
(uncalibrated) and binary decision threshold (0.20) (binary pairs, 6,133 and 1,342; six-class pairs, 6,746 and 1,467).
Hyperparameters were selected by 5-fold patient-grouped cross-validation over the main analysis's three-configuration
grid. The 236 US patients in the original validation and test partitions took no part in any step. The model was
evaluated once on all patients enrolled in Spain (80) and Canada (36). Patient sets were disjoint.

**(A)** Binary improvement AUPRC (higher is better). Spain and Canada pooled: 0.426 (95% CI 0.374–0.484; 116 patients,
1,588 pairs, 283 events, 17.8%); Spain 0.432 (0.375–0.514; 949 pairs, 19.5%); Canada 0.430 (0.362–0.523; 639 pairs, 15.3%).
Ticks mark each cohort's event rate, the AUPRC of a non-informative model. The US calibration set, 0.558 (1,342 pairs,
19.0%), is a descriptive reference only: it informed calibration and threshold choices and no CI was computed.

**(B)** Six-class IMWG response macro-F1 (higher is better): pooled 0.629 (0.585–0.658; 1,694 pairs); Spain 0.617
(0.568–0.655; 1,034 pairs); Canada 0.634 (0.491–0.688; 660 pairs); US calibration set 0.632 (1,467 pairs; no CI).

Intervals are 95% patient-level bootstrap percentile CIs (1,000 resamples). Selected hyperparameters were binary: 100 trees,
depth 3, learning rate 0.05; six-class: 100, 3, 0.05. This is a post hoc, exploratory internal geographic robustness check,
not independent external validation: earlier versions of this experiment, which reused hyperparameters tuned in the main
analysis on a split containing 80 of the 116 evaluated patients, had been seen before this redesign. Countries were not
compared statistically; Canada estimates are imprecise (36 patients). 153 Spain/Canada binary pairs had a treatment-regimen
name, and 29 a regimen-class category, not seen in the development data (3 pairs an unseen heavy-chain type; the earlier
version, with a larger fit set, had 57 pairs with unseen regimen names). Italy (138
patients) was excluded post hoc, after inspecting its outcome distribution (single site, non-standard visit schedule, and
an implausible 78.7% improvement rate); these patients remain in the main analyses. AUROC (Supplementary Table S4): pooled
0.769 (0.731–0.805); Spain 0.755 (0.711–0.802); Canada 0.785 (0.722–0.849); US calibration set 0.817 (no CI).

## Files and regeneration (repository root, `.venv`)

```
python modeling/us_to_spain_canada_model_c_origtrain.py                    # ~75 s
python figures/figS_country_origtrain_extract_data.py                      # 90 checks
python figures/publication_redesign/FigS_country_origtrain/figureS2A_plot.py
python figures/publication_redesign/FigS_country_origtrain/figureS2B_plot.py
python figures/supplementary_tables/build_table_s4_origtrain.py            # 13 checks
```

Outputs: `artifacts/results_clean_rerun/geographic_origtrain/` (metrics and audit JSON; prediction CSVs and
`artifacts/models_clean_rerun/geographic_origtrain/` are gitignored), `figures/data/FigS_country_origtrain/`
(gitignored), the two PNGs here, `figures/supplementary_tables/Table_S4_geographic_robustness_origtrain.{md,csv}`.

## Verification

- Independent recount (separate script, rebuilt from the split, subject and pair files): 439 / 96 / 236 / 116 patient
  sets match the run by SHA-256 digest and are pairwise disjoint; tuning pool = fit set; all fit and calibration patients
  are US and original-train; saved preprocessors equal an independent refit on the fit set only and differ from a refit
  that also includes evaluation and excluded patients; six-class argmax is invariant to the temperature.
- Extractor: 90 checks; Table S4: 13 checks; plots check every plotted value against the result JSONs (1e-12).
- Pair counts: 1,588 binary / 1,694 six-class evaluation pairs (unchanged).

## Open issues

1. Unseen categories rose with the smaller fit set (regimen name: 153 evaluation pairs vs 57 before; regimen class 29;
   heavy-chain type 3; from `unseen_category_audit` in the binary metrics JSON). These are encoded as all zeros, and may
   contribute to the lower Spain/Canada performance; this was not analysed.
2. The US 82/18 assignment was kept (no re-split of the 535 patients). A fresh split or repeated splits would change the
   fit/calibration sets and could change results; not done.
3. Six-class calibration-rule disagreement (above).
4. Not blind: earlier results were seen first.
