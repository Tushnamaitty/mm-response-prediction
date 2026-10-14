# Protocol: train-only development and held-out evaluation of pre-treatment RNA pathway scores for PFS and OS

Version 1.0, written 2026-10-14, **before any survival model of this redesign has been fitted or any held-out survival
outcome has been read**. Its SHA-256 (computed on line-feed-normalised bytes) is recorded in
`explainability/SURVIVAL_TRAINONLY_PROTOCOL.sha256` and will be written into every output of the scripts named below.
No part of this protocol may be changed after the first Stage D run; a change would require a new, separately hashed version
and a statement of why.

Status of the evidence: **post hoc and exploratory.** The earlier pooled analysis (all 674 patients in 5 x 5 cross-validation:
PFS delta C-index +0.040, patient-bootstrap 95% CI 0.016 to 0.064, permutation p = 0.005 with 200 permutations; OS +0.043,
0.012 to 0.074) was seen before this redesign, and the survival outcomes of the 201 held-out patients below were part of it.
The redesign is therefore not blind, not an independent validation and not external validation; "validated", "independent"
and "replicated" are not used.

## 1. Why the analysis is redesigned

Professor Song's feedback is that validation patients must not influence model development. The earlier survival analysis
pooled all 674 patients, including the 101 validation and 100 test patients of the main analysis. Those patients trained the
fold models (ridge-Cox coefficients, standardisation, ridge penalty) and were also evaluated. Under the strict reading the
validation (and test) patients must be held out of every development step. The earlier outputs are kept unchanged and are
relabelled "pooled cross-validation; held-out patients took part in model fitting".

## 2. Question, cohort and populations

**Question (unchanged):** does pre-treatment RNA pathway information add prognostic discrimination for PFS beyond baseline
clinical characteristics and first-line treatment?

**Cohort (unchanged, built by `survival_incremental_analysis.build_cohort`):** one row per patient with an RNA sample at or
before day 0 (first treatment); official MMRF PFS (primary endpoint) and OS (secondary); 674 patients, 459 PFS events, 247
deaths. The cohort definition uses neither partition membership nor any model output.

**Partitions (the main analysis's patient-level partitions, `data/splits/patient_split_assignments.csv`; nothing re-randomised):**

| Group | Partitions | Patients | PFS events | OS deaths |
|---|---|---|---|---|
| **Development** | train | 473 | 310 | 165 |
| **Held-out** | validation + test | 201 (101 + 100) | 149 (71 + 78) | 82 (45 + 37) |
| Total | | 674 | 459 | 247 |

Sensitivity cohort (RNA day in [-30, 0], unchanged definition): development 438 patients / 286 PFS events / 154 deaths;
held-out 186 / 137 / 75 (624 patients in total).

Validation and test patients are pooled as one held-out set because neither may enter development; no other held-out
definition (test only, validation only) is a criterion or a primary result.

## 3. Models and preserved analysis rules

All rules below are those of `explainability/survival_incremental_analysis.py` (module spec SHA-256 `fe0792db...db6e4`), which
is **imported unchanged**. Preserved without modification:

* comparator **C0**: age at index, gender, ISS (not_reported = missing category), exact first-line regimen; RNA model
  **C0 + RNA**: C0 plus the same fixed 50 Hallmark pathway scores; no pathway selection;
* ridge-penalised Cox model, Breslow ties, the existing L-BFGS-B solver; lambda grid {0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100};
* preprocessing fitted on the training rows only: median age imputation plus missing indicator, one-hot gender / ISS /
  first-line (a category unseen in the training rows maps to all zeros), every column standardised with the training mean and
  SD (ddof = 0); **in this protocol "training rows" are development patients only**;
* lambda chosen by mean validation Harrell C-index in 5-fold inner cross-validation, ties to the largest lambda;
* Harrell C-index (tied risks count 0.5); patient bootstrap with 2,000 resamples, seed 42 for PFS and 1,042 for OS, percentile
  95% CI; descriptive IPCW integrated Brier score over 365 to 1,095 days (50-point grid, KM censoring distribution estimated in
  the training rows), no CI for IBS;
* whole-RNA-vector permutation as the null (clinical variables, outcomes and timing fixed).

## 4. Necessary changes from the earlier design (each justified)

| # | Change | Justification |
|---|---|---|
| N1 | Models are fitted on the 473 development patients once and **frozen**; they are scored once on the 201 held-out patients (instead of 5 x 5 outer cross-validation over 674). | A held-out evaluation requires a model that never saw the held-out patients. |
| N2 | The ridge penalty of each model is chosen in the development patients by 5-fold inner CV **repeated 5 times** (seeds `SEED + 10000 + 10*r`, r = 0..4); the 25 validation C-indices per lambda are averaged and the earlier tie rule applies. | The earlier design chose a lambda per outer fold from one 5-fold split. With a single final model one noisy split would determine the whole result, so the same inner-CV rule is averaged over five splits. |
| N3 | The permutation null permutes the RNA vectors **among the held-out patients** and re-scores the frozen RNA model (no refitting); B = 10,000; one-sided p = (1 + k) / (B + 1), k = number of null delta C >= observed. | The earlier null refitted models inside every outer fold (hence B = 200). A frozen model needs no refit, so a conditional null on the held-out set is exact and cheap, and B can be larger. |
| N4 | A pre-specified success rule is added (section 6). | The earlier analysis reported the CI and the permutation p separately without a pass/fail rule. With a single held-out test the rule must be fixed in advance. |
| N5 | The delta C-index is the single held-out difference C(C0 + RNA) - C(C0) (no averaging over repeats, because there are no repeats). | Follows from N1. |

Nothing else changes: no new pathways, covariates, penalty grid, endpoints, cut-offs or algorithms are introduced.

## 5. Staged procedure and blinding

Scripts (to be written after this protocol is approved and hashed): `explainability/survival_trainonly_development.py`
(Stage D) and `explainability/survival_trainonly_heldout.py` (Stage H). Outputs only under
`artifacts/results_clean_rerun/survival_trainonly/`; existing outputs are never overwritten and the earlier survival
directories are only read.

**Stage D (development; held-out outcomes blanked).** The frame handed to Stage D has the survival times and event indicators
of all held-out patients set to missing (other held-out variables are not used by any learned step).

1. Gates. **Any failure stops the run before any held-out outcome is read.**
   * G1: module spec SHA = `fe0792db...db6e4`; cohort 674 / 459 / 247 asserted; partition counts 473 / 101 / 100; PFS events by
     partition 310 / 71 / 78 and deaths 165 / 45 / 37 asserted; sensitivity cohort counts (section 2) asserted;
   * G2 (reproduction): the imported locked machinery, run on all 674 patients in the earlier pooled design for PFS, reproduces
     `survival_incremental/primary_pfs_summary.json` (`c_C0` 0.6012025, `c_C0_RNA` 0.6416971, `delta_c` 0.0404946) to within 1e-9;
   * G3: development and held-out patient sets are disjoint; preprocessing parameters (age median, categories, means, SDs,
     Breslow baseline, KM censoring) are computed from development rows only (verified by recomputation);
   * G4: every ridge-Cox fit converges.
2. Development fits, per analysis in section 7: choose lambda (N2), fit C0 and C0 + RNA on all development patients, and store
   coefficients, preprocessing parameters, Breslow baseline hazard and the censoring KM in `frozen_models.json`.
3. Development-only cross-validation (descriptive, reference for optimism): the earlier 5 x 5 outer design with nested lambda,
   run inside the 473 development patients. Its per-fold lambdas are also used by the Figure 5 refit (section 7, A6).
4. Hash `frozen_models.json` (SHA-256 of line-feed-normalised bytes) **before** Stage H. Stage H reads only that file plus the
   held-out patients' predictors and, after the hash check, their outcomes.

**Stage H (held-out evaluation; run once).** Verifies the hashes and file order, attaches held-out outcomes, computes the
analyses of section 7 and writes the report. No re-run, re-tuning or alternative model follows a Stage H result.

## 6. Primary analysis and success criteria

**Primary analysis (A1):** PFS, primary cohort, held-out n = 201 (149 events). Estimand: delta C = C(C0 + RNA) - C(C0) on the
held-out patients, using the frozen models. One primary hypothesis, one test; direction: RNA improves discrimination
(delta C > 0).

**The criterion is met only if all three hold:**
1. observed delta C > 0;
2. the 95% patient-bootstrap percentile CI lower bound is > 0 (2,000 resamples, seed 42);
3. the one-sided permutation p (N3) is <= 0.025 (the same one-sided level as the 95% CI).

Otherwise it is **not met**. Not meeting the criterion is not evidence of no RNA effect: the interpretation is "inconclusive"
unless the CI upper bound is below 0, in which case the interpretation is "RNA did not improve discrimination in the held-out
patients". No other outcome (OS, sensitivities, partition-level results, development cross-validation) can rescue or replace
the primary result.

**Planning arithmetic (not an experiment; approximate).** The pooled bootstrap CI half-width of 0.024 at 459 events implies a
standard error of about 0.012; with 149 held-out events it scales to about 0.021. For a true delta C of 0.030 / 0.040 / 0.050
the power of the one-sided 2.5% criterion is about 29% / 46% / 64%. The earlier +0.040 was estimated in-sample (including the
held-out patients) and the development set is smaller than the earlier training folds, so the true held-out effect may be
smaller. An inconclusive result is therefore a likely and acceptable outcome of this design and is reported as such.

## 7. Planned analyses

| ID | Analysis | Frozen on | Evaluated on | Reported |
|---|---|---|---|---|
| **A1** | **PFS primary** (section 6) | 473 development | 201 held-out | C(C0), C(C0 + RNA), delta C, CI, permutation p, criterion verdict |
| A2 | OS secondary | 473 | 201 (82 deaths) | delta C and CI only; descriptive (seed 1,042) |
| A3 | IBS (PFS and OS) | A1, A2 models | 201 | IBS of both models and the difference; no CI, as before |
| S1 | PFS, RNA day in [-30, 0] | 438 | 186 (137 events) | delta C and CI; no permutation, as before |
| S2 | Expanded baseline, PFS and OS (adds LDH, B2M, albumin, creatinine, haemoglobin, calcium, bone-marrow plasma-cell %; earlier prespecified definitions and handling, fold-local median imputation + missing indicator, now development-only) | 473 | 201 | delta C and CI; no permutation, as before |
| S3 | Development-only 5 x 5 CV (section 5 step 3) | 473 (within folds) | 473 (out of fold) | delta C and CI; descriptive reference for optimism; not an alternative primary |
| S4 | Partition-level description of A1 (validation 101 patients / 71 events; test 100 / 78) | A1 models | each partition | delta C with bootstrap CI; descriptive heterogeneity check only, no test |
| A6 | Figure 5 refit: the 50 pathway coefficients of C0 + RNA in each of the 25 development-CV fits (lambda = that fold's nested-CV value) and in the frozen A1 / A2 models; direction rule (same sign in all 25 fits), concordance rule and summaries exactly as before | development only | none (descriptive) | coefficient tables; no hypothesis tests |
| P | The earlier pooled 5 x 5 CV (674 patients) | not rerun | not rerun | kept and labelled "pooled; held-out patients took part in fitting"; not leakage-free |

S2 requires the pinned pipeline modules `build_clinical_features` and `build_flow_cytometry_features` (SHA-256 checks in
`survival_richer_baseline_sensitivity.py`); if either cannot be obtained with the pinned hash, S2 is reported as not run and
nothing is substituted. No sensitivity, subgroup, cut-off, model or pathway analysis beyond this table is run.

## 8. Multiplicity and reporting rules

* One primary test (A1). A2, A3, S1-S4 and A6 are descriptive or sensitivity analyses with no multiplicity adjustment and no
  pass/fail role.
* Report all of A1-A3, S1-S4 and A6 whether or not they favour RNA, with the number of patients and events.
* The held-out result is described as "evaluated in held-out validation and test patients; post hoc, exploratory; the earlier
  pooled results had been seen". RNA is described as adding (or not adding) prognostic discrimination in this model, never as
  having or lacking biological value.
* Figures 4 and 5 and Supplementary Table S5 are rebuilt from the new outputs; the earlier pooled figures are relabelled.

## 9. Stopping rules

* A failed Stage D gate stops the run; no held-out outcome is read and nothing is retried with different settings.
* If Stage H runs, its single result is reported. If the A1 criterion is not met, no alternative rule, lambda grid, covariate
  set, cohort, endpoint or test is tried to obtain a different verdict.
* The protocol, Stage D artifact and Stage H report hashes are recorded in the output manifest; a mismatch invalidates the
  run.
