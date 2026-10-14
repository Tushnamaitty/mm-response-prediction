# Protocol: train-only development and held-out evaluation of the RNA contribution analyses (Figures 3A-C, Table S1, low-variance add-on)

Version 1.0, written 2026-10-14, **before any model of this redesign has been fitted and before any held-out outcome has been
read**. Its SHA-256 (computed on line-feed-normalised bytes) is recorded in `explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.sha256`
and will be written into every output of the scripts named below. It may not be changed after the first Stage D run; a change would
need a new, separately hashed version and a statement of why.

Status of the evidence: **post hoc and exploratory.** The earlier results (pooled train+validation cross-validation) were seen
before this redesign and the validation patients' outcomes were part of them; the test patients' outcomes were seen in the main
analysis. The redesign is not blind, not an independent validation and not external validation. The words "validated",
"independent" and "replicated" are not used, and no claim is made that RNA has or lacks biological value.

## 1. Audit: where validation patients still influence these analyses

All of the following were run in 5 x 5 patient-grouped cross-validation over train + validation patients
(`split != "test"`: 716 + 154 = 870 patients). Validation patients therefore trained the fold models and were evaluated.

| Analysis (code) | Output | What validation patients influenced | Leak under the strict reading |
|---|---|---|---|
| Direct D - C, Layer D vs Layer C XGBoost (`direction3_predictive`, group `all`; `repeated_cv_c_vs_d_clean_rerun`) | Fig 3A, Table S1 | training and evaluation rows of every fold model | yes |
| Low-variance RNA ridge add-on (`rna_low_variance_incremental`) | Fig 3A | inner Model-C cross-fits, ridge penalty, ridge coefficients, standardisation | yes (this is the only positive next-visit RNA result: delta RPS -0.00034, permutation p = 0.005) |
| Uncertainty quartiles (`direction1_uncertainty`, `direction1_q4_permutation_null`, `direction1_entropy_slope`) | Fig 3B | fold models, and the quartile cut-points (ranks within the train+validation cohort) | yes |
| Treatment-class strata (`direction3_predictive`, `direction3_common.support_table`) | Fig 3C, Table S1 | fold models; which treatments get inference (support counts include validation patients) | yes |
| Trajectory strata (`direction4_predictive`, reusing the D3 out-of-fold predictions) | Fig 3C, Table S1 | fold models; stratum support | yes |
| Pooled molecular associations (`direction2_association` S3/S5, D3/D4 Part B) | not in Fig 3A-C / Table S1 | none: association tests develop no model | no (relabel as pooled descriptive only) |

Not leaking and unchanged: the main 24 models (tuned, preprocessed and fitted on train patients only); their locked hyperparameters,
which were selected in train patients only.

## 2. Recommended design (and why)

Same principle as the pathway, geographic and survival redesigns: **develop on train patients only, evaluate once on validation +
test patients with models that never saw them.**

* The locked Layer C and Layer D models (`artifacts/models_clean_rerun/{binary,multiclass}_{c,d}_xgboost.joblib`, fitted on train patients,
  hyperparameters tuned in train patients) are already train-only. Applying them to the held-out RNA-available rows is
  **inference only**: no refitting and no tuning, so D - C, the uncertainty groups and the strata can be evaluated exactly as before
  on patients that played no role in development (Section 5).
* Anything that must be *learned* or *decided* (Model-C out-of-fold offsets and the ridge RNA shift of the add-on, uncertainty
  cut-points, which treatments have enough support) is learned or decided from train patients only (Section 4), then frozen.
* Train-only 5 x 5 cross-validation of the earlier design is kept as a **development-only reference** (it is the only way to get
  usable power for the small strata); it never replaces the held-out evaluation and cannot rescue it.

Why not simply rerun the earlier cross-validation on train patients? It would still reuse the same patients for fitting and
evaluation and for hyperparameters tuned on them. Why not only the held-out set? Subgroups there are small, so the strata are reported
with their intervals but the development-only reference is provided alongside.

## 3. Populations and fixed inputs

**Partitions** (main analysis, `data/splits/patient_split_assignments.csv`; nothing re-randomised): development = train (716 patients);
held-out = validation + test (154 + 155 = 309 patients; validation and test are pooled and no other held-out definition is a criterion).

**RNA-available visit pairs** (`days_since_rna_sample` notna; bookkeeping, not outcomes):

| Task | Development | Held-out |
|---|---|---|
| Six-class | 6,484 rows / 494 patients | 2,689 rows / 213 patients |
| Binary (eligible_for_binary) | 5,857 rows / 490 patients | 2,405 rows / 213 patients |

The held-out rows are about 35% of the earlier train+validation RNA-available rows (six-class 7,724; binary 6,939), so held-out
intervals will be roughly 1.7 times wider than before.

**Fixed (not re-estimated):** the locked XGBoost hyperparameters (`artifacts/results_clean_rerun/{task}_{c,d}_xgboost_metrics.json`,
`best_params`, SHA-256 recorded in the artifact); the Model C / Model D feature sets; the 50 Hallmark pathway scores; the metrics (binary
AUPRC primary, AUROC, log loss; six-class RPS primary, log loss; macro-F1 descriptive); the patient-cluster bootstrap (2,000 draws, seed 42,
one resample shared across groups and metrics); Holm and BH families exactly as in the earlier code.

## 4. Stage D: development (train patients only; held-out outcomes never read)

Script: `explainability/rna_contribution_trainonly_development.py`. Outputs only under
`artifacts/results_clean_rerun/rna_contribution_trainonly/`; existing outputs are never overwritten and the earlier directories are only read.
The master table is reduced to train patients immediately after loading; no held-out frame and no file containing held-out outcomes
(e.g. the val/test probability CSVs) is read by Stage D.

**Gates (any failure stops the run):**
* G1: protocol hash; locked-parameter file hashes; partition counts 716 / 154 / 155 patients; development / held-out RNA-available counts of
  the table above.
* G2 (machinery reproduction): the imported locked `oof_task` code, run on the earlier train+validation universe for the multiclass task,
  repeat 0, reproduces the committed repeat-0 `all`-group RPS of Model C and Model D in
  `direction3/partA_predictive/partA_repeat_level_metrics.csv` to within 1e-9. (This gate necessarily uses validation patients inside the
  earlier pooled design; only an aggregate pass/fail and the difference are kept.)
* G3: the add-on's self-checks (gradient check, planted-effect recovery, beta = 0 identity) pass; the probability floor changes mean RPS by
  < 1e-5 on the development out-of-fold probabilities.
* G4: the objective support rule applied to development counts reproduces the declared roles (PI, IMiD, steroid primary; CD38, chemo
  exploratory; SLAMF7, BCMA unsupported). Checked on development counts before this protocol was written: reproduced (see Section 8).
* G5: development and held-out patient sets are disjoint; serialised frozen objects reproduce their in-memory values exactly.

**D-A. Development-only 5 x 5 CV (reference).** The locked `oof_task` with locked parameters, folds seeded `42 + repeat` on train patients only,
for Model C and Model D, binary and six-class. From the saved out-of-fold predictions: overall D - C; treatment strata (support from train counts,
D-B); trajectory states; uncertainty groups; all with the earlier bootstrap, Holm/BH families and metrics. Reported as "development-only
cross-validation reference; hyperparameters were tuned in the same patients, so absolute performance is optimistic".

**D-B. Support and strata (decided in development).** `support_table` rule (Tier 1: >= 150 patients, >= 150 events, >= 150 non-events;
Tier 2: >= 80 / 75 / 75; multiclass additionally >= 20 rows per class) is applied to **train** RNA-available counts. The treatments with inference
per task are frozen: binary PI, IMiD, steroid, CD38, chemo; six-class PI, IMiD, steroid, CD38 (chemo has < 20 rows in one class). Trajectory states
(improving, stable, worsening; no_history descriptive; the six audited continuity rows excluded) are the earlier definitions, computed from
visit-response fields only (`direction4_support.build_trajectory`); primary states stable and worsening, secondary improving and no_history.

**D-C. Uncertainty cut-points (decided in development).** Normalised Model-C entropy (binary: Bernoulli; six-class: Shannon / log 6), averaged over
the 5 development repeats per pair. The frozen cut-points are the 0.25, 0.50, 0.75, 2/3 and 0.80 quantiles of this development distribution
(`numpy.quantile`, linear). Held-out pairs are assigned to groups by comparing their entropy with those fixed values:
Q1 <= q25 < Q2 <= q50 < Q3 <= q75 < Q4; "top 33%" > q(2/3); "top 20%" > q80. Primary subgroup Q4; comparator Q1-Q3.

**D-D. Low-variance ridge add-on, frozen per repeat.** For each repeat r = 0..4: cross-fit Model C (multiclass, locked parameters) within the
development patients by 5-fold patient-grouped CV (folds `make_repeat_folds(train_patients, 5, seed = 42 + 10000 + 10 r)`, the earlier inner
cross-fit scheme with fold = 0); lambda and the ridge coefficients are then chosen and fitted by the **unchanged** `choose_lambda_and_fit`
(lambda grid {1, 10, ..., 1e5, inf}, nested patient CV on the cross-fit folds, tie to the largest lambda, standardisation by distinct RNA samples of
the training patients, common shift across cut-points, minimally floored Model-C probabilities). Frozen: (lambda_r, beta_r, mean_r, sd_r) for r = 0..4.

**D-E. Freeze and hash.** `frozen_development.json` (support sets, state roles, entropy cut-points, add-on fits, locked-parameter hashes,
patient-id-set hashes, code and protocol hashes) and the development reference tables; SHA-256 of line-feed-normalised bytes is written to
`frozen_development.sha256` **before** Stage H.

## 5. Stage H: held-out evaluation (run once)

Script: `explainability/rna_contribution_trainonly_heldout.py`. It first verifies the protocol hash, the frozen-artifact hash and the Stage D gate
report; only then does it read the held-out probability files (`artifacts/results_clean_rerun/calibration/probabilities/{task}_{c,d}_xgboost_{val,test}.csv`,
inference output of the locked train-only models, restricted to RNA-available rows) and, through them, the held-out outcomes.
Gate H: the held-out probabilities equal a fresh inference of the locked models (joblib) on the same rows to within 1e-6, and the held-out patient set
equals the frozen one (hash).

Analyses (all held-out, locked C / D models, single repeat, earlier bootstrap scheme):

| ID | Analysis | Rule (unchanged unless listed in Section 6) |
|---|---|---|
| **P1** | Fig 3A direct D - C, overall: binary delta AUPRC, six-class delta RPS (primary metrics), plus AUROC / log loss; validation-only and test-only descriptive rows | CI only, no pass/fail (as before) |
| **P2** | Fig 3B uncertainty: D - C by quartile; primary interaction (D - C in Q4) - (D - C in Q1-Q3); top-20% / top-33% sensitivities | Holm across the two primary interactions (binary AUPRC, six-class RPS) |
| **P3** | Low-variance add-on: delta RPS of the frozen RNA-shifted model vs floored Model C, mean over the 5 frozen repeats | **Success only if** delta < 0 AND patient-bootstrap 95% CI upper < 0 AND one-sided permutation p <= 0.05; otherwise "suggestive or inconclusive" / "no incremental value detected" exactly as before |
| **P4** | Fig 3C / Table S1 treatment strata (D-B sets): on / not-on D - C, interactions | Holm over PI, IMiD, steroid per task on the primary metric; BH for exploratory on-treatment tests and for interaction contrasts; complements and secondary metrics descriptive |
| **P5** | Table S1 trajectory states: per-state D - C, interactions (worsening - stable, improving - stable) | Holm over stable and worsening per task (primary metric); BH for secondary states and interactions |
| **P6** | Negative control: RNA vectors permuted among held-out distinct (patient, sample) units, locked Model D re-scored; all P1-P5 groups | descriptive, no pass/fail |
| **P7** | Continuous uncertainty: weighted slope of the per-pair log-loss benefit (binary) on the entropy percentile (empirical CDF of the development entropy) with patient-bootstrap CI | descriptive (as before, outside the Holm family) |
| **R** | Reference: the development-only CV results of D-A and the earlier pooled results (read from their existing files) | not part of any test |

"Significant after adjustment" means adjusted p < 0.05 within the stated family, and only ever "post hoc, exploratory".
A metric that cannot be computed in a held-out stratum (a class or outcome missing) is reported as "not evaluable"; a stratum is never dropped
or substituted on the basis of held-out outcomes.

## 6. Necessary changes from the earlier designs (each justified)

| # | Change | Justification |
|---|---|---|
| N1 | Held-out evaluation with the single locked train-only models replaces 5 x 5 CV over train+validation (one prediction per pair, bootstrap over patients). | Evaluation patients must not train the evaluated models. |
| N2 | Uncertainty groups use cut-points fixed from the development entropy distribution instead of ranks inside the evaluated cohort. | Cut-points are a learned quantity and must come from development. |
| N3 | Treatment support tiers and the inference sets are decided from train counts. Held-out strata are reported whatever their size, with intervals. | Decisions must precede held-out data; the earlier thresholds (150 patients) cannot be met in the 213-patient held-out set, so small strata will be imprecise, not hidden. |
| N4 | Add-on: five per-repeat frozen fits evaluated on the held-out rows and averaged (the earlier estimand: mean over repeats); the permutation null re-scores the frozen fits with RNA vectors permuted among held-out units, B = 10,000 (was 200 refits). | Frozen models need no refit; the exact null is cheap. Success rule unchanged. |
| N5 | Negative control for D - C (P6): permutation among held-out units with the locked Model D re-scored, B = 1,000 (was 20 retrainings of Model D and 200 for the binary Q4 interaction). | Same reason; no retraining possible or needed. |
| N6 | Not re-run (superseded by P1-P5 or outside Fig 3A-C / Table S1): D4 gap <= 180 d and exclude-Vt-PD sensitivities, D1 spline / Spearman / raw-entropy variants, scrambled-RNA Q4 retraining null, `repeated_cv_c_vs_d`, Part B pooled analyses, freshness variants. | They cannot be defined without CV refits or are not part of the named outputs; they remain available as labelled pooled results. |

## 7. Planning arithmetic (not an experiment; approximate)

Held-out intervals scale with the inverse square root of the number of patients: about 1.7 times wider than the pooled CV for the overall D - C (213 vs
about 600 patients) and wider for Q4 (about 25% of rows) and treatment strata. The add-on's pooled standard error of about 0.00012 RPS scales to
about 0.0002 RPS in the held-out rows, so for a true delta of -0.00034 the power of the three-part success rule is roughly 40%. An inconclusive
result is therefore likely and acceptable for the add-on and for most strata; it is reported as such.

## 8. Facts established in development before this protocol was written

Development-only counts (train patients, no held-out outcome): support rule reproduces the declared roles (PI 341 / IMiD 339 / steroid 381 patients, tier 1;
CD38 106 and chemo 149 patients, tier 2; SLAMF7 33 and BCMA 6 patients, tier 3); six-class minimum class counts 118 / 291 / 134 / 27 / 1 (chemo below 20).
RNA-available counts as in Section 3. The locked model files and the held-out probability files exist locally.

## 9. Reporting and stopping rules

* Report every analysis of Section 5 with n patients / rows, effect, CI and adjusted p where defined, whether or not favourable; Figures 3A-C and
  Table S1 are rebuilt only afterwards (not part of this protocol).
* RNA is described as adding (or not adding) predictive information in these models, never as having or lacking biological value.
* A failed Stage D gate stops the run; no held-out outcome is read and nothing is retried with other settings. After Stage H no analysis is rerun, re-tuned
  or replaced to change a verdict. The protocol, Stage D artifact and Stage H report hashes are recorded in the output manifest; a mismatch invalidates the run.
