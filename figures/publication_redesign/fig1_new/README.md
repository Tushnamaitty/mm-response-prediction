# New Figure 1: Study overview

Draft overview figure requested after Professor Song's feedback (new Figure 1; the current Figure 1 becomes Figure 2).
It contains no results. It shows data sources, cohorts, visit-pair construction, information layers A–D, both
tasks, partitions and modelling framework, and the validation status of each downstream analysis.

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

Requires `matplotlib`, `pandas` (versions pinned in `requirements.txt`, which is UTF-16). The script stops if any
number differs from the verified value.

## Provenance of every number

- Read from tracked aggregate files in `figures/data/Fig1/`: cohort and task counts, RNA-available counts, partition
  sizes, information-layer feature counts (78 / 115 / 143 / 194; segments +78 / +37 / +28 / +51), survival cohort
  (674 patients, 459 PFS events, 247 deaths), survival-cohort partition overlap (101 validation, 100 test),
  test improvement rate (353 of 1,930 pairs = 18.3%).
- Constants in the script, re-derived automatically when the private `data/` directory is present:
  - 1,143 subject records (`data/clinical/subject_deid.csv`);
  - Spain 80, Canada 36, 116 total; 80 of 116 in the original training partition
    (`data/splits/patient_split_assignments.csv`).
- Constants not re-derivable from this branch (taken from the `origin/modeling-country-robustness` audit,
  `results/robustness_country/full_run_audit_and_summary.json`): 632 US-train and 139 US-validation patients.
  I reproduced the 632 / 139 split locally (`train_test_split`, 82/18, seed 42, sorted US patients); it matches.
- Modelling facts come from `modeling/train_models.py`: 5-fold patient-grouped CV on train patients only
  (AUPRC for binary, macro-F1 for six-class), validation patients used for the binary threshold, three algorithms,
  24 models. Calibration comparison on validation: `figures/FIGURE_CAPTIONS_DRAFT.md` (Figure S1).
- 84 test patients in the proliferation/MYC test stage: `figures/fig3_extract_data.py`.

## Design choices

- Amber marks the two analyses with concerns raised by Professor Song (pathway selection, geographic hold-out). It is
  used for no other purpose. Wording avoids "validated" and "independent" except to say they are *not* independent.
- No figure numbers appear inside the figure, because the renumbering of Figures 2–5 is not settled.
- Panel letters A–E coincide with layer letters A–D (as in the current Figure 1); layers are always written "Layer X".
- Layer-D bar is cumulative source features; the derived RNA-availability indicator is noted, not drawn.

## Unresolved issues

1. **Scope of "validation patients".** The figure shows validation patients setting the binary threshold and choosing
   calibration, which is a development step under a strict reading of Song's note. The footnote says the scope is being
   clarified; update once she answers.
2. **Redesign pending.** Panel E must be updated when the pathway-selection and geographic analyses are redone
   (counts, wording, amber flags). The geographic row's "80 of 116" disappears if tuning is rerun on US patients only.
3. **1,143 → 1,025 patients.** 118 subject records have no visit pair; the reason is not shown or verified here.
4. **PRELIMINARY label.** Main-model test metrics are still labelled PRELIMINARY in the JSON files. The figure calls
   the test set an "internal held-out test split" and avoids "locked".
5. **Renumbering.** Whether Figures 2–5 are renumbered (Song only specified Figure 1 → 2) is an assumption to confirm.
6. **Survival row** (101 validation and 100 test patients in the CV cohort) and the "RNA by day 0" definition follow
   the Figure 4 caption; not re-derived here.
7. **Not independently reviewed.** No co-author or domain check of the wording has been done.
