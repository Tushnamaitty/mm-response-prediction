# Figure 1: Study overview

Overview figure requested after Professor Song's feedback (Figure 1 under the final numbering, Scheme A: the study-design figure becomes Figure 2 and the earlier Figures 2–5 become Figures 3–6; see `figures/FIGURE_NUMBERING.md`).
It contains no results. It shows data sources, cohorts, visit-pair construction, information layers A–D, both
tasks, partitions and modelling framework, and the validation status of each downstream analysis.
Updated after the geographic and pathway-selection redesigns (see "What changed in this update").

## Files

| File | Content |
|---|---|
| `figure1_new_plot.py` | Generation script (matplotlib; style block identical to `Fig1/figure1_plot.py`) |
| `figure1_new.png` | 600 dpi raster (7.2 × 10.08 in, double-column width) |
| `figure1_new.pdf`, `figure1_new.svg` | Vector versions (editable text) |

## Regenerate

From the repository root:

```
python figures/publication_redesign/fig1_new/figure1_new_plot.py                 # png + pdf + svg, 600 dpi
python figures/publication_redesign/fig1_new/figure1_new_plot.py --formats png --dpi 150
```

Requires `matplotlib`, `pandas`. The script stops if any number differs from the verified value.

## Provenance of every number

- Read from tracked aggregate files in `figures/data/Fig1/`: cohort and task counts, RNA-available counts, partition
  sizes, information-layer feature counts (78 / 115 / 143 / 194; segments +78 / +37 / +28 / +51), survival cohort
  (674 patients, 459 PFS events, 247 deaths), survival-cohort partition overlap (101 validation, 100 test),
  test improvement rate (353 of 1,930 pairs = 18.3%).
- Constants in the script, re-derived automatically when the private `data/` directory is present:
  - 1,143 subject records (`data/clinical/subject_deid.csv`);
  - Spain 80, Canada 36, 116 total (`data/clinical/subject_deid.csv`, `data/splits/patient_split_assignments.csv`).
- Geographic counts (439 fit / 96 calibration / 236 US patients excluded / 116 evaluated) and pathway counts (371 train
  patients in selection and discovery; 160 held-out patients = 76 validation + 84 test) come from the result JSONs and are
  checked against them whenever those files are present:
  `artifacts/results_clean_rerun/geographic_origtrain/full_run_audit_and_summary.json`,
  `artifacts/results_clean_rerun/direction2_replication_trainonly/{discovery,heldout}_report.json`. Both were independently
  recounted from the split, subject and pair files during verification.
- RNA-contribution counts (716 train patients in development; 213 held-out validation + test patients with RNA) and survival counts
  (473 train patients in development; 201 held-out patients = 101 validation + 100 test) come from
  `artifacts/results_clean_rerun/rna_contribution_trainonly/{heldout_report,frozen_development}.json` and
  `artifacts/results_clean_rerun/survival_trainonly/{heldout_report,devcv_pfs_summary}.json`, and are checked against them whenever those
  files are present (the plot script stops on a mismatch).
- Modelling facts come from `modeling/train_models.py`: 5-fold patient-grouped CV on train patients only
  (AUPRC for binary, macro-F1 for six-class), three algorithms, 24 models. Validation patients are descriptive only: no decision threshold
  or recalibration enters any reported metric (probabilities are shown raw; the earlier validation-based threshold and exploratory calibration
  comparison are legacy and disclosed in `figures/SUPERSEDED_ANALYSES.md`; Figure S1 caption in `figures/FIGURE_CAPTIONS_DRAFT.md`).

## Design choices

- No row is amber any more: the two analyses flagged in the first draft (pathway selection; geographic hold-out) were
  redesigned. The footnote states what each analysis used for tuning, selection, fitting, preprocessing, thresholds and
  calibration. Wording avoids "validated" and "independent" except to say results are not independent or externally validated.
- No figure number is drawn as a figure title; panel E cites one downstream figure by number ("Fig. 6", the survival development cross-validation), which follows the final numbering.
- Panel letters A–E coincide with layer letters A–D (as in Figure 2); layers are always written "Layer X".
- Layer-D bar is cumulative source features; the derived RNA-availability indicator is noted, not drawn.

## What changed in this update

| Row of panel E | Before | Now |
|---|---|---|
| Pathway analysis | "Proliferation / MYC association"; discovery train + validation; test 84 patients; family chosen from pooled data incl. test; amber | "Train-selected pathway composite (association)"; selection and discovery 371 train patients; evaluation 160 validation + test patients; post hoc, exploratory; not amber |
| Geographic hold-out | fit 632 / calibration 139 US patients; "80 of 116 evaluated patients were in the tuning split"; amber | tuning and fit 439 / threshold and calibration 96 US train-partition patients; 236 US validation/test-partition patients and Italy excluded; 116 Spain/Canada scored once |
| RNA contribution, uncertainty and subgroup analyses | "5×5 grouped CV on train + validation patients"; "CV cohort overlaps the tuning patients" | development: 716 train patients (CV reference); evaluation: 213 validation + test patients with RNA; locked models, scored once; post hoc, exploratory, earlier pooled results seen first |
| Survival and pathway coefficients | "674 patients, 5×5 CV; includes 101 validation and 100 test patients" | development: 473 train patients (CV; Figure 6); evaluation: 201 validation + test patients (101 + 100); frozen models, scored once; post hoc, exploratory, not independent |
| Panel D, Validation box | "Binary decision threshold; calibration comparison" | "Descriptive only: no tuning, threshold or calibration" |
| Footnote | "known validation concerns under revision" (later: "the cross-validated RNA and survival analyses fit within their own cohorts") | states that every redesigned analysis (RNA contribution, pathway composite, survival, geographic) uses train-partition patients only for tuning, selection, fitting and preprocessing and scores held-out patients once, and where validation patients still set binary thresholds and calibration (main models) |

## Unresolved issues

1. **Scope of "validation patients" (resolved by removal).** The validation-based binary threshold and calibration-method choice
   were removed from the reported analysis: every reported main-model metric is threshold-free or argmax-based and probabilities are
   shown raw, so no number changed. Panel D's validation box now reads "Descriptive only: no tuning, threshold or calibration" and the footnote says so.
   The decision to show raw probabilities is not blind to an earlier exploratory validation comparison (disclosed in the Figure S1 caption and
   `figures/SUPERSEDED_ANALYSES.md`). Professor Song should confirm that this is acceptable; if an operating-point threshold is wanted, it would be
   derived from train out-of-fold predictions (not done).
2. **Cross-validated analyses (resolved).** The RNA-contribution and survival analyses were redesigned (frozen protocols; train-only
   development, held-out scoring once; see `figures/SUPERSEDED_ANALYSES.md`); panel E states the development and held-out counts. They remain
   post hoc and exploratory because the earlier pooled results were seen first.
3. **1,143 → 1,025 patients.** 118 subject records have no visit pair; the reason is not shown or verified here.
4. **PRELIMINARY label.** Main-model test metrics are still labelled PRELIMINARY in the JSON files. The figure calls
   the test set an "internal held-out test split" and avoids "locked".
5. **Renumbering (resolved).** Scheme A (this overview = Figure 1; study design = Figure 2; earlier Figures 2–5 = Figures 3–6) is
   applied in the captions, tables, documentation and in the image text; this folder keeps its working name `fig1_new`.
6. **Survival row** (473 development and 201 held-out patients) is verified against the result JSONs; the "RNA by day 0" definition and the panel A
   survival-cohort numbers (674 patients, 459 events, 247 deaths) follow the Figure 5 caption and `figures/data/Fig1/`.
7. **Not independently reviewed.** No co-author or domain check of the wording has been done.
