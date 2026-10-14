# Figure captions (draft for scientific review)

Status: draft. Figures are numbered by the final scheme (Scheme A, applied 2026-10-14): Figure 1 study overview, Figure 2 study
design and information layers, Figure 3 next-visit performance, Figure 4 RNA contribution and train-selected pathway composite,
Figure 5 survival, Figure 6 pathway coefficients; supplementary figures S1–S3 keep their numbers. Image files, folders and scripts
keep their earlier working names (for example `figure3A_trainonly.png` is Figure 4A): see `figures/FIGURE_NUMBERING.md`. Every
number below was checked against the extracted figure data in `figures/data/` and the locked results in
`artifacts/results_clean_rerun/`; supplementary tables are in `figures/supplementary_tables/`. Open items are listed under
Unresolved items at the end. Day 0 is the start of first treatment (CoMMpass index date) throughout.

Revision note: the Figure 1, Figure 4 (A–D), Figure 5, Figure 6 and Supplementary Figure S2 captions below describe the
redesigned analyses, in which validation and test patients take no part in tuning, selection, fitting or preprocessing
(pathway selection and discovery, RNA-contribution models, survival models and the geographic model all use
original-training-partition patients only; held-out patients are scored once). They replace the earlier captions (pooled
cross-validation including validation patients; pooled pathway selection; hyperparameters reused from the main analysis),
whose figure files are superseded: see `figures/SUPERSEDED_ANALYSES.md`. The redesigned analyses are post hoc and
exploratory. Main-model
threshold and calibration: no decision threshold or recalibration enters any reported metric (probabilities are shown raw and
all headline metrics are threshold-free or argmax-based); the validation-based binary threshold and the earlier exploratory
calibration comparison remain only in legacy files and are disclosed in `figures/SUPERSEDED_ANALYSES.md`.

---

## Main figures

### Figure 1. Study overview: data, prediction tasks, partitions and validation strategy.

**(A)** Data sources and cohorts. MMRF CoMMpass interim analysis 24 contributes 1,143 subject records: clinical data
(diagnosis, laboratory, treatment, response, adverse events, symptoms) and bulk RNA sequencing (TPM converted to 50 MSigDB
Hallmark ssGSEA pathway scores per RNA sample). Consecutive visit pairs V(t) → V(t+1) define 13,451 pairs from 1,025 patients
(six-class IMWG response task) and 12,298 pairs from 1,018 patients (binary improvement task, which excludes 1,153 pairs with
sCR at V(t)). An RNA sample at or before V(t) is available for 9,173 pairs (707 patients). The survival cohort comprises 674
patients with pre-treatment RNA (459 progression-free-survival events, 247 deaths). **(B)** Visit-pair construction: inputs
are information available at or before V(t) (the latest RNA sample is never later than V(t)); the target is the IMWG category
at V(t+1); every pair of a patient stays in one data partition. **(C)** Nested information layers (source input features): A
current clinical state, 78; B plus longitudinal history, 115; C plus treatment context, 143; D plus 50 pathway scores and RNA
age, 194 (a derived RNA-availability indicator is also supplied to Layer D). **(D)** Patient-level partitions (train 716,
validation 154, test 155 patients; binary 709 / 154 / 155; visit pairs shown on the figure) and the modelling framework
(three algorithms × four layers × two tasks = 24 models; binary AUPRC, six-class macro-F1). In the main models,
hyperparameters are tuned and preprocessing is fitted on train patients only, and test patients are scored once; no decision
threshold or recalibration enters any reported metric (probabilities are shown raw), and validation patients are reported
descriptively only. **(E)** Downstream analyses and their validation status. RNA contribution (Figures 4A–C): locked models
tuned and fitted on the 716 train patients, scored once on 213 validation and test patients with RNA. The train-selected
pathway composite is selected and discovered in 371 train patients and evaluated in 160 validation and test patients with RNA
(76 and 84). Survival (Figures 5–6): models fitted on 473 train patients (25 development-CV fits for the pathway
coefficients) and scored once on 201 validation and test patients (101 and 100). The geographic hold-out tunes and fits on
439 US patients of the original training partition, sets the threshold and calibration on 96 such patients, excludes 236 US
patients of the original validation and test partitions and the Italian patients, and scores 116 Spain and Canada patients
once. The RNA-contribution, pathway, survival and geographic analyses are post hoc and exploratory (the earlier pooled
results were seen first); all evaluations are internal to CoMMpass and none is externally validated.

---

### Figure 2. Study design, nested information layers and RNA-sequencing timing in MMRF CoMMpass.

**(A)** Next-visit prediction framework and cohort. Each observation is a consecutive visit pair: input features available at
or before the current visit, V(t), are used to predict the International Myeloma Working Group (IMWG) response at the next
visit, V(t+1). The six-class task (progressive disease [PD], stable disease [SD], partial response [PR], very good partial
response [VGPR], complete response [CR], stringent complete response [sCR]) comprised 1,025 patients and 13,451 visit pairs.
The binary task (improvement to a better IMWG category at V(t+1)) comprised 1,018 patients and 12,298 pairs after excluding
1,153 pairs with sCR at V(t), which cannot improve. An RNA sample collected on or before V(t) was available for 9,173
six-class pairs (707 patients) and 8,262 binary pairs (703 patients). Patients were assigned once to patient-grouped
training, validation and test partitions (716, 154 and 155 patients; binary task 709, 154 and 155), shared by every
information layer and algorithm, so that no patient contributes to more than one partition. Hyperparameters were tuned by
patient-grouped cross-validation within training patients; no decision threshold or recalibration enters any reported metric,
validation patients are reported descriptively only, and test patients were scored once. The survival cohort (Figures 5 and
6) comprised the 674 patients with an RNA sample collected on or before day 0 (459 progression-free survival [PFS] events;
247 deaths).

**(B)** Nested information layers A–D are input-feature sets, not algorithms; each layer contains all features of the
previous layer. Layer A, current clinical state (78 source input features); B, A + longitudinal history (115); C, B +
treatment context (143); D, C + 50 Hallmark RNA pathway scores (single-sample gene set enrichment analysis, ssGSEA) and
RNA age (194, plus one derived RNA-availability indicator). Source input features are counted before encoding; after
one-hot encoding of categorical features and addition of missing-value indicators, the fitted models received 148, 185,
302 and 354 model-input columns, respectively. Each layer was fitted with Logistic Regression, LightGBM and XGBoost.
Cohort characteristics by partition, RNA availability and survival cohort are given in Table 1.

**(C)** RNA timing. Left: collection day of the 751 distinct RNA samples relative to day 0, shown separately for the
first sample of each of 707 patients and for 44 additional (later) samples from 38 patients; 636 of 751 samples (84.7%),
all first samples, were collected within ±30 days of day 0. Middle: RNA age at V(t), the number of days from the latest
RNA sample collected on or before V(t) to V(t), across the 9,173 six-class RNA-available pairs (median 866 days;
interquartile range 380–1,465). Right: alignment rule; only the latest sample collected on or before V(t) is used, and a
later sample is never used for an earlier visit. Pairs without such a sample are RNA-unavailable.

---

### Figure 3. Next-visit response prediction across nested information layers A–D.

Performance of Logistic Regression, LightGBM and XGBoost for each information layer (Figure 2B) on the held-out test
partition (155 patients), evaluated once. No algorithm was prespecified as primary or selected as best, and the figure does not rank them. XGBoost
alone is used in Figures 4A–C and Supplementary Figures S1–S3 as a single fixed model for those analyses; this is not a statement
that it performed better than Logistic Regression or LightGBM, and the repository records no a priori rationale for choosing it
(see Unresolved items). All plotted metrics are threshold-free (AUPRC) or use the highest-probability class (macro-F1);
no decision threshold or recalibration is applied.

**(A)** Area under the precision–recall curve (AUPRC; higher is better) for binary improvement at V(t+1), 1,930 visit
pairs including 353 improvement events. The dotted line marks the test improvement prevalence (18.3%), the AUPRC expected
from a non-informative model.

**(B)** Macro-averaged F1 (unweighted mean of the six class-specific F1 scores; higher is better) for six-class IMWG
response, 2,079 visit pairs. Logistic Regression performed substantially below the tree-based models for this task.

**(C)** Paired change in performance when a layer is added: longitudinal history (B − A), treatment context (C − B) and
RNA pathway scores (D − C); positive values favour the larger layer. For XGBoost, adding longitudinal history increased
binary AUPRC by +0.022 but changed six-class macro-F1 by −0.014; adding RNA changed binary AUPRC by +0.008 and six-class
macro-F1 by −0.004. All D − C intervals included zero for all three algorithms and both tasks, so there was no clear evidence that RNA improved next-visit prediction on the
held-out test set.

Points are test-set estimates; whiskers are 95% patient-level bootstrap confidence intervals (CIs), unadjusted for multiple
comparisons. The panel A–B intervals are the locked 1,000-resample patient bootstrap in
`artifacts/results_clean_rerun/uncertainty/main_model_cis.csv`; the plotted values in `figures/data/Fig2/` match that file
exactly (24 of 24 rows). The panel C paired differences and intervals (1,000 resamples; `fig2_paired_deltas_full_test.csv`) were
independently recomputed from the locked test predictions (`figures/verify_fig3c_paired_cis.py`) with the repository's patient-bootstrap
procedure (seed 42, all rows of each sampled patient, the same resample for both layers, 2.5th and 97.5th percentiles): all 24 rows
(3 algorithms × 2 tasks × B − A, C − B, D − C and D − A) reproduce the plotted differences and both interval limits to 10⁻¹⁶, and the
differences equal the differences of the locked point estimates. The original extraction script is not in the repository, so it is the
procedure, not that script, that is verified. With three other seeds the interval limits moved by at most 0.0053; all D − C intervals
included zero under every seed, whereas whether an interval includes zero changed for three borderline rows that the text does not use
(XGBoost binary B − A and D − A, XGBoost six-class C − B). Exact test-set values and CIs for all 24 models (from the locked
`main_model_cis.csv`, with validation-set values) are given in Supplementary Table S6; the panel C paired differences and intervals are in
the tracked file `figures/data/Fig2/fig2_paired_deltas_full_test.csv` and are not repeated in Table S6. The two tasks use different visit-pair populations and metrics and are not directly
comparable. These are internal held-out evaluations within CoMMpass, not external validation.

---

### Figure 4. Exploratory analyses of the incremental contribution of RNA to next-visit prediction and of a train-selected pathway composite.

Panels A–C evaluate locked XGBoost Layer C and Layer D models, whose hyperparameters were tuned and which were fitted on the
716 train patients only, once on the 213 validation and test patients with RNA (binary 2,405 visit pairs; six-class 2,689
pairs), comparing the two layers on identical RNA-available pairs. These analyses are specific to XGBoost: it was used alone, no a priori
rationale for choosing it is recorded, and the results neither show that XGBoost is superior to Logistic Regression or LightGBM nor describe
how RNA would contribute to those algorithms. A development-only 5×5 patient-grouped cross-validation
within the train patients is shown as a reference for optimism and is not an evaluation. Changes are Layer D − Layer C.
AUPRC: higher is better. RPS, ranked probability score over the six ordered IMWG categories: lower is better, so ΔRPS > 0
means RNA worsened predictions. Intervals are 95% patient-cluster bootstrap CIs (2,000 resamples). All analyses are post hoc
and exploratory: the earlier pooled results (5×5 cross-validation over train + validation patients, in which validation
patients trained the fold models) had been seen before this redesign, so the held-out evaluation is neither independent nor
external validation, and the earlier estimates are superseded.

**(A)** Overall RNA contribution, held-out. Direct XGBoost Layer D − Layer C: binary ΔAUPRC +0.0109 (95% CI −0.0026 to
+0.0248; Layer C AUPRC 0.474); six-class ΔRPS −0.00006 (−0.00114 to +0.00104; −0.08% of Layer C RPS). Neither interval
excludes zero. The development-only reference gave binary ΔAUPRC +0.0007 (−0.0058 to +0.0069) and six-class ΔRPS +0.0011
(+0.0005 to +0.0018), a small worsening. Separately, a strongly regularized RNA add-on (ridge shift of the Layer C
cumulative-logit offsets using the 50 pathway scores; six-class only; frozen penalty λ = 1,000; five frozen per-repeat fits)
gave ΔRPS −0.00018 (−0.00063 to +0.00028; −0.22% of Layer C RPS), with a one-sided permutation p of 0.0109 (RNA vectors
permuted among held-out patients; 10,000 permutations). The pre-specified success rule (ΔRPS < 0, CI upper bound < 0 and
permutation p ≤ 0.05) was not met because the CI includes zero, so the add-on result is inconclusive. The permutation null is
centred away from zero because scrambled RNA adds noise to a frozen model, so it tests association between RNA and outcome,
not ΔRPS < 0. The direct and add-on approaches are different models and their estimates are not interchangeable.

**(B)** RNA contribution by quartile of Layer C prediction entropy (uncertainty), computed without outcomes or RNA, with cut
points taken from the train patients' entropy (so held-out groups differ in size). The prespecified test was the interaction
(D − C in the highest-uncertainty quartile Q4) − (D − C in Q1–Q3), with Holm adjustment across the two primary interactions.
Binary: Q4 ΔAUPRC +0.0140 (−0.0115 to +0.0385); interaction +0.0101 (−0.0173 to +0.0379; Holm p = 0.94). Six-class: Q4 ΔRPS
+0.00002 (−0.00224 to +0.00237); interaction +0.00012 (−0.00235 to +0.00261; Holm p = 0.94). Neither interaction was
significant. A descriptive continuous-uncertainty analysis (binary log-loss benefit at the 90th minus 10th percentile of
train-patient entropy) gave +0.0058 (−0.0027 to +0.0176). Quartile-level estimates are descriptive.

**(C)** RNA contribution by drug class active at V(t) (strata overlap) and by response trajectory (IMWG category at V(t)
compared with the previous visit), for binary ΔAUPRC and six-class ΔRPS, held-out. Which treatments received inference was
fixed from train-patient counts before the held-out data were read (proteasome inhibitor, immunomodulatory drug and
corticosteroid primary; anti-CD38 and chemotherapy exploratory; six-class chemotherapy not evaluable because of too few
CR/sCR outcomes). No stratum changed significantly on its primary metric after multiplicity adjustment (Holm across the three
primary treatments and, separately, across stable and worsening trajectories, per task; Benjamini–Hochberg [BH] for
exploratory treatments and secondary states). Strata contained 42–207 patients (binary 42–206; six-class 42–207), and estimates are imprecise. The earlier
pooled-CV finding that RNA significantly worsened six-class RPS in proteasome-inhibitor, immunomodulatory-drug,
corticosteroid and stable-trajectory pairs was not reproduced in the held-out evaluation. Two secondary six-class trajectory
interaction contrasts had BH p < 0.05 (worsening − stable ΔRPS −0.0030, −0.0059 to −0.0003, BH p = 0.032; improving − stable
−0.0034, −0.0063 to −0.0007, BH p = 0.022), as did three secondary-metric contrasts (binary AUROC, proteasome inhibitor;
six-class log loss, improving − stable and corticosteroid); these are exploratory, unreplicated and descriptive. Full
per-stratum results, including complement strata, secondary metrics, interaction contrasts and the development-only
reference: Supplementary Table S1.

**(D)** Train-selected pathway composite: the first principal component (68.1% of variance; loadings 0.31–0.53, all positive)
of the five pathways with the smallest p-values in a covariate-adjusted ordinal regression run in train patients only (E2F
Targets, G2-M Checkpoint, Myc Targets V1, Mitotic Spindle and Spermatogenesis), and the ordered next IMWG category
(cumulative-logit ordinal GEE, patient-clustered robust SEs; odds ratio < 1 means a higher score is associated with a worse next
response). Stage 1, selection evidence: single-pathway regressions in train patients only, with pathway scores z-scored using
train RNA samples only (5,100 pairs, 371 patients; odds ratios 0.79–0.85 per SD; Benjamini–Hochberg q ≤ 1.2 × 10⁻⁴ across 50
pathways). Stage 2, discovery in the same train patients: odds ratio 0.77 per SD (95% Wald CI 0.72–0.84); this estimate is
optimistic because the same patients selected the pathways. Stage 3, held-out evaluation in 160 validation and test patients
(76 and 84; 2,179 pairs) with the composite and a clinical linear predictor frozen before they were scored: odds ratio 0.76
(0.67–0.87; one-sided p = 1.1 × 10⁻⁵), meeting the rule set for this analysis (95% CI entirely below 0). Validation and test
patients took no part in pathway selection, composite construction or discovery. The one-sided direction was fixed after earlier
pooled analyses and was not prespecified. An earlier version of this analysis, which chose the family from pooled data that
included validation and test patients (it contained Myc Targets V2 in place of Spermatogenesis) and evaluated it in 84 test
patients, had been seen before this redesign; the present analysis is therefore post hoc and exploratory, and stage 3 is neither
independent nor external validation. Spermatogenesis is not a proliferation or MYC pathway, so the composite is not
interpreted as a proliferation axis. The stages use different patients and adjustment models, so their estimates are not
directly comparable. The prespecified primary analysis, a matched-pair global test across all 50 pathways pooled over all
patients, was null (six-class p = 0.149; binary p = 0.441). This is an association, not evidence of predictive gain or a causal
mechanism.

---

### Figure 5. Exploratory incremental prognostic value of pre-treatment RNA pathway scores for progression-free and overall survival.

Survival cohort: 674 patients with one RNA sample collected on or before the start of first treatment (CoMMpass index date);
official MMRF PFS (459 events) and overall survival (OS; 247 deaths). The existing patient-level partitions define the
development set (473 train patients; 310 PFS events, 165 deaths) and the held-out set (201 validation + test patients, 101 +
100; 149 PFS events, 82 deaths). Clinical baseline: ridge-penalized Cox model with age at index, gender, International
Staging System (ISS) stage and exact first-line regimen. Clinical baseline + RNA: the same model plus the 50 Hallmark pathway
scores. Penalties were chosen by 5-fold inner cross-validation repeated five times within the train patients and then frozen;
models were fitted on train patients only and scored once on the held-out patients. Harrell's C-index (higher is better); ΔC
= C(clinical baseline + RNA) − C(clinical baseline). Intervals are 95% patient-bootstrap CIs (2,000 resamples) of the
held-out predictions; they reflect patient sampling, not model-refitting variability. No CIs are available for absolute
C-index values. The pre-specified criterion for the PFS analysis was ΔC > 0, CI lower bound > 0 and one-sided permutation p ≤
0.025. The earlier pooled cross-validation results had been seen before this redesign, so the analysis is post hoc and
exploratory, and neither independent nor external validation.

**(A)** Held-out C-index without and with RNA. PFS: 0.630 → 0.657 (ΔC +0.028, 95% CI −0.019 to +0.074). OS: 0.696 → 0.706 (ΔC
+0.010, −0.045 to +0.065). Both intervals include zero, so the PFS criterion was not met, and the data do not establish or
exclude an effect. A one-sided whole-RNA-vector permutation test (RNA vectors permuted among the held-out patients, frozen
models re-scored, 10,000 permutations) was performed for PFS only: 0 of 10,000 permuted ΔC reached the observed value (p =
0.0001, the smallest attainable). The permutation null is centred below zero because scrambled RNA adds noise to the frozen
model, so this p-value tests the association between RNA and outcome, not ΔC > 0. No permutation test was performed for OS.
Small hollow points show validation-only (0.646 → 0.702; ΔC +0.056) and test-only (0.605 → 0.604; ΔC −0.001) patients; these
descriptive partitions are imprecise and were not tested.

**(B)** ΔC across the held-out primary and sensitivity analyses and the reference estimates. Expanded clinical baseline
(clinical baseline + baseline LDH, β2-microglobulin, albumin, creatinine, haemoglobin, calcium and bone-marrow plasma-cell
percentage): PFS ΔC +0.010 (−0.029 to +0.049; C 0.648 → 0.658) and OS ΔC −0.002 (−0.049 to +0.045; C 0.704 → 0.702). RNA
collected within 30 days before first treatment (day −30 to 0; 186 held-out patients, 137 PFS events): PFS ΔC +0.014 (−0.036
to +0.061; C 0.648 → 0.662); this timing analysis was specified for PFS only and was not evaluated for OS. Reference
estimates: development-only 5×5 cross-validation within the 473 train patients (PFS ΔC +0.022, −0.010 to +0.054; OS +0.048,
+0.009 to +0.088; fitted and evaluated in the same patients, so not an evaluation) and the earlier pooled 5×5
cross-validation over all 674 patients (grey; PFS +0.040, +0.016 to +0.064; OS +0.043, +0.012 to +0.074), in which validation
and test patients also trained the fold models; the pooled estimates are superseded and not comparable with the held-out
rows. Rows differ in cohort or comparator and were not compared statistically. R-ISS was excluded before fitting (overlap
with ISS and LDH; about 57% missing), and cytogenetic (FISH) adjustment was not possible because no such fields were
available.

Held-out integrated Brier score (inverse-probability-weighted, 365–1,095 days; lower is better; point estimates only, no CI):
PFS 0.233 → 0.208; OS 0.138 → 0.122. The 201 held-out patients are 101 of the 154 validation and 100 of the 155 test patients
of the next-visit partitions, so the survival cohort overlaps those partitions and these results are not independent of
Figures 3–4. All values, including repeat-level ranges of ΔC for the development reference, are given in Supplementary Table
S5. All results are post hoc and exploratory.

---

### Figure 6. Exploratory RNA pathway coefficient patterns across progression-free and overall survival models.

Coefficients come from the Figure 5 clinical baseline + RNA ridge Cox models, refitted in each of 25 overlapping development
cross-validation fits (5 repeats × 5 folds) per endpoint within the 473 train patients, with the penalty chosen by nested
cross-validation inside each fit; validation and test patients were not used. β is the log-hazard per training-fold SD of a
pathway score (hazard ratio [HR] = e^β); β > 0 means a higher score is associated with higher modelled hazard. A pathway is
consistently directed for an endpoint if all 25 fits share the same sign (PFS: 12 positive, 6 negative, 32 mixed; OS: 16, 10,
24) and concordant if consistently directed with the same sign for both endpoints.

**(A)** Mean PFS versus mean OS coefficient for all 50 Hallmark pathways, with zero lines and the identity line (dashed).
Twelve pathways were concordant: 9 associated with higher hazard and 3 (IL-6/JAK/STAT3 Signaling, Inflammatory Response,
Notch Signaling) with lower hazard in both endpoints; none was consistently directed in opposite directions. The five
pathways of the Figure 4D family are ringed: E2F Targets, G2-M Checkpoint, Myc Targets V1 and Mitotic Spindle are concordant
(higher hazard), whereas Spermatogenesis is not concordant. This is descriptive context, not corroboration. Pearson r = 0.81
across the 50 mean coefficients is descriptive: PFS and OS share the same patients, RNA and covariates, and deaths are also
PFS events, so their agreement is not independent corroboration.

**(B)** The 12 concordant pathways, grouped by direction: mean PFS and OS coefficients with the minimum–maximum range across
the 25 fits. Ranges are not confidence intervals; the fits share about 80% of their training patients, so consistency across
fits is not independent replication. Effects are small (HR per SD of the mean coefficients 0.94–1.07 for PFS and 0.88–1.11
for OS).

Correlated pathways (for example, the five train-selected pathway scores of Figure 4D share 68.1% of their variance in their
first principal component) have interdependent ridge coefficients that are not individually identifiable. No pathway-level
CIs, p-values or causal effects were estimated. The development patients are the patients that selected the Figure 4D family
and overlap those of Figures 4 and 5, so agreement with the Figure 4D result is not independent evidence. Patterns are
exploratory model associations, not protective mechanisms, biological drivers or validated biomarkers. The earlier version of
this figure (50 pathways fitted on all 674 patients; 19 concordant pathways; r = 0.83) is superseded.

---

## Supplementary figures

### Supplementary Figure S1. Calibration of binary next-visit improvement predictions with and without RNA.

XGBoost information Layer C (clinical + longitudinal history + treatment context) and Layer D (Layer C + RNA) on the held-out
test set (155 patients; 1,930 visit pairs; improvement rate 18.3%). The assessment is specific to XGBoost (no a priori rationale for using
it alone is recorded; the calibration of the other algorithms was not assessed). Raw (uncalibrated) predicted probabilities are shown: no
recalibration, decision threshold or other validation-fitted transformation was applied to any reported number. An earlier
exploratory comparison of uncalibrated, Platt and isotonic recalibration (5-fold patient-grouped cross-fitting within the
validation patients; lowest log loss, preferring the simpler method when Brier scores differed by < 0.001) had favoured the
uncalibrated probabilities for both layers, so the decision to show raw probabilities is not blind to validation data; that
comparison is not a reported result (see `figures/SUPERSEDED_ANALYSES.md`).

**(A)** Reliability curves from 10 equal-count quantile bins of predicted probability (193 pairs each): bin-mean predicted
probability against observed improvement rate; the dashed line marks perfect calibration. The strip shows the
distribution of predicted probabilities (24 Layer C and 23 Layer D predictions above 0.7 are not drawn). Calibration
slope (ideal 1) and intercept (calibration-in-the-large; ideal 0), from logistic recalibration on logit(p): Layer C 1.087
(95% CI 0.952–1.232) and −0.066 (−0.199 to 0.075); Layer D 1.083 (0.953–1.224) and −0.075 (−0.210 to 0.069). Mean
predicted probability 0.191 (C) and 0.192 (D) against an observed rate of 0.183.

**(B)** Layer D − Layer C differences on identical test pairs, each metric on its own scale: calibration slope −0.004
(−0.029 to 0.024), intercept −0.009 (−0.024 to 0.005), Brier score +0.0001 (−0.0010 to 0.0010) and log loss +0.0005
(−0.0023 to 0.0030). All intervals included zero, indicating no clear overall calibration difference; equivalence was not
tested.

Intervals are 95% paired patient-level bootstrap CIs (1,000 resamples of test patients, calibrators fixed). No bin-level
intervals or expected calibration error were computed, and six-class calibration is not shown. Results for RNA-available
test pairs only (1,323 pairs; 107 patients), all D − C intervals included zero except calibration intercept (−0.026,
−0.045 to −0.007) and mean predicted probability (+0.003, +0.001 to +0.006), which are small, unadjusted and
descriptive; per-layer values for this subset are point estimates without CIs (Supplementary Table S2). Internal
held-out evaluation within CoMMpass.

---

### Supplementary Figure S2. Internal geographic hold-out of clinical, longitudinal and treatment-based prediction within MMRF CoMMpass (original-training-partition revision).

XGBoost information Layer C (143 source input features; the check is specific to this algorithm and layer, no a priori rationale for choosing
them is recorded, and the other algorithms and layers were not evaluated) was developed only on US patients in the original training partition of
the main analysis (535 of 771 US patients), using the earlier random 82/18 split of US patients (seed 42) restricted to those
patients: 439 patients (438 for the binary task, because one has no binary-eligible pair) for hyperparameter tuning, preprocessing
and model fitting, and 96 for the calibration method and the binary decision threshold (0.20) (binary pairs, 6,133 and 1,342; six-class
pairs, 6,746 and 1,467). The calibration method was uncalibrated for the binary task; for the six-class task temperature scaling was
selected, which cannot change the predicted class or macro-F1 (no six-class calibrated probability is reported).
Hyperparameters were selected by 5-fold patient-grouped cross-validation over the main analysis's three-configuration grid. The
236 US patients in the original validation and test partitions took no part in any step. The model was then evaluated once on all
patients enrolled in Spain (80) and Canada (36). Patient sets were disjoint.

**(A)** Binary improvement AUPRC (higher is better). Spain and Canada pooled: 0.426 (95% CI 0.374–0.484; 116 patients, 1,588
pairs, 283 events, 17.8%); Spain 0.432 (0.375–0.514; 949 pairs, 19.5%); Canada 0.430 (0.362–0.523; 639 pairs, 15.3%). Ticks
mark each cohort's event rate, the AUPRC of a non-informative model. The US calibration set, 0.558 (1,342 pairs, 19.0%), is a
descriptive reference only: it informed calibration and threshold choices and no CI was computed.

**(B)** Six-class IMWG response macro-F1 (higher is better): pooled 0.629 (0.585–0.658; 1,694 pairs); Spain 0.617
(0.568–0.655; 1,034 pairs); Canada 0.634 (0.491–0.688; 660 pairs); US calibration set 0.632 (1,467 pairs; no CI).

Intervals are 95% patient-level bootstrap percentile CIs (1,000 resamples). Selected hyperparameters were binary: 100 trees,
depth 3, learning rate 0.05; six-class: 100, 3, 0.05. This is a post hoc, exploratory internal geographic robustness check, not
independent external validation: earlier versions of this experiment, which reused hyperparameters tuned in the main analysis on
a split containing 80 of the 116 evaluated patients and then fitted on US patients of all partitions, had been seen before this
redesign. Countries were not compared statistically; Canada estimates are imprecise (36 patients). With the smaller development
set, 153 Spain/Canada binary pairs had a treatment-regimen name, and 29 a regimen-class category, not seen in the development
data (3 pairs an unseen heavy-chain type); these are encoded as all zeros. Italy (138 patients) was excluded post hoc, after
inspecting its outcome distribution (single site, non-standard visit schedule and an implausible 78.7% improvement rate); these
patients remain in the main analyses. AUROC (Supplementary Table S4): pooled 0.769 (0.731–0.805); Spain 0.755 (0.711–0.802);
Canada 0.785 (0.722–0.849); US calibration set 0.817 (no CI).

---

### Supplementary Figure S3. SHAP attribution for XGBoost information Layer D models.

Exact TreeSHAP values were computed with native XGBoost for the locked Layer D models on the held-out test set (155
patients; 1,930 binary and 2,079 six-class visit pairs); contributions plus the bias reproduced each row's raw model
output within 3 × 10⁻⁶. Binary contributions are in log-odds of improvement; six-class contributions are class-specific raw
scores (softmax inputs), averaged equally over the six IMWG classes, and are not probability attributions. Signed
contributions of all fitted model-input columns of a source feature (one-hot levels, missing-value indicators) were summed
before taking absolute values. XGBoost Layer D is used as a single fixed model for attribution (no a priori rationale is recorded); this does not imply that it
performed better than the other algorithms (no algorithm was selected as best), and the attributions describe these XGBoost models, not
Logistic Regression or LightGBM.

**(A)** Net family attribution share across six mutually exclusive groups: current IMWG response; other current clinical
state (rest of Layer A); longitudinal history (B); treatment context (C); 50 Hallmark RNA pathway scores (D); and RNA
timing/availability (RNA age and the derived availability indicator; D). For each group, signed source contributions were
summed within each visit pair, the absolute value averaged over pairs (and classes), and the six values normalized to
100% (labels rounded to sum to 100.0%). Binary: 58.1% (95% CI 56.3–59.8), 10.4% (9.2–11.6), 18.0% (17.0–18.9), 8.7%
(8.4–9.1), 3.3% (2.9–3.9) and 1.5% (1.4–1.7). Six-class: 53.2% (52.0–54.4), 16.8% (16.0–17.8), 14.7% (14.2–15.2), 6.8%
(6.5–7.0), 7.3% (6.5–8.1) and 1.2% (1.1–1.3). With patients weighted equally, the current-response share was 47.5%
(binary) and 49.0% (six-class). These are shares of model attribution, not accuracy, variance explained or causation.

**(B)** Top 15 source input features per task ranked by mean absolute source-feature SHAP, with 95% CIs; this is a
different measure from panel A. Current IMWG response ranked first in both tasks; in the six-class test set, the current
response equalled the next response in 69.8% of pairs, so its dominance partly reflects persistence. RNA features in the
top 15 were RNA age (#11), TNF-α signalling via NF-κB (#12) and TGF-β signalling (#15) for binary, and RNA age (#15) for
six-class. Summing feature importances within a family gives larger RNA pathway shares (6.7% binary, 12.9% six-class)
than panel A because pathway contributions partly cancel within visit pairs; the two sets of shares are not
interchangeable.

Intervals are 95% patient-level bootstrap CIs (1,000 resamples of test patients; model and SHAP values fixed). Visit
pairs from the same patient are not independent. Correlated features share attribution, so individual rankings are
unstable. RNA attributions on RNA-unavailable pairs reflect imputed values. Both measures for all three analyses and
six-class shares by class: Supplementary Table S3.

---

## Unresolved items

1. **Figure 3 provenance [resolved by recomputation; original script absent]:** (working name Fig2) the panel A–B intervals match the locked
   `main_model_cis.csv` exactly (24 of 24 rows). The panel C differences and intervals were recomputed from the locked test predictions with the
   repository's bootstrap procedure and match 24 of 24 rows to 10⁻¹⁶ (`figures/verify_fig3c_paired_cis.py`); the original extraction script
   for `figures/data/Fig2/` is still absent, so the verification is of the procedure, and borderline intervals depend on the seed (see Figure 3).
2. **Main-model threshold and calibration [resolved by removal; disclosed]:** no decision threshold or recalibration enters
   any reported metric (Figures 3 and S1, Tables S2 and S6 use threshold-free or argmax metrics and raw probabilities); no
   reported number changed. The validation-selected binary threshold (0.25–0.35) and the exploratory calibration comparison
   remain only in the legacy metrics files and in `calibration_summary.json` on the `subgroup-analysis` branch; the choice to
   show raw probabilities is not blind to that comparison. If an operating-point threshold or a recalibrator is ever
   reported, derive it from train out-of-fold predictions (not done).
3. **Figures 4A–C, 5 and 6 and Supplementary Tables S1 and S5 [rebuilt, provisional]:** the train-only versions
   (`*_trainonly.png`, `Table_S1_rna_subgroups_trainonly.*`, `Table_S5_survival_trainonly.*`) replace the earlier pooled-CV
   versions, which are kept unchanged and superseded (`figures/SUPERSEDED_ANALYSES.md`).
4. **Missing legacy inputs and outputs:** the old extractors for the superseded earlier versions of Figures 4A–D, 5 and 6
   (working labels 3A–D, 4 and 5) cannot be re-run; 47
   files recorded in legacy manifests are absent and the old `figures/data/Fig3|Fig4|Fig5` folders are not on any branch. See
   `figures/LEGACY_MISSING_INPUTS.md` (recorded hashes allow verification of recovered copies). Legacy values quoted in Table
   S5 are marked as transcriptions.
5. **Held-out strata are imprecise:** the 213 held-out patients with RNA leave 42–207 patients per Figure 4C stratum (binary 42–206) and
   100–101 patients per partition in Figure 5; the confidence intervals are wide and the analyses cannot exclude effects of
   the size seen in the earlier pooled results.
6. **Figure numbering [resolved; residual working labels]:** Scheme A is applied in the captions, tables, READMEs, documentation
   and the five images that drew figure labels (mapping and residual items: `figures/FIGURE_NUMBERING.md`). File and folder
   names, extractor and protocol text, hash-chained artifacts and the superseded figures keep their earlier working labels.
7. **Choice of XGBoost for later analyses [searched; no a priori rationale exists; handled by wording]:** all branches and the full history
   were searched (`figures/SUPERSEDED_ANALYSES.md`, section "Choice of XGBoost"): the first mention of XGBoost in any committed file is in the
   same commit as the preliminary training results, the commits before it contain no algorithm choice, and the only designation (a "primary
   algorithm" constant in the subgroup scripts, written after the clean-rerun results) gives no rationale. The captions of Figures 3, 4 and
   S1–S3 therefore describe those analyses as XGBoost-specific and make no superiority claim. If a justification was written down outside the
   repository (for example a pre-analysis plan or correspondence), add it to the Methods; otherwise state in the Methods that XGBoost was used
   for these analyses without a prespecified rationale.
8. **Supplementary Figure S3 source data [restored and hash-verified; S3A footnote replaced]:** the three S3A source files are present again in
   `figures/data/FigS_shap/` and match the SHA-256 hashes recorded earlier (`figures/LEGACY_MISSING_INPUTS.md`, section D;
   `SUPERSEDED_ANALYSES.md`, section "Supplementary Figure S3A"). `figureS3A.png` was regenerated from them with the footnote changed to "used as a single
   fixed model; no algorithm is claimed to perform better" (only the footnote band differs from the earlier image). The shares and CIs of Table S3 equal
   the restored CSVs; six-class shares and CIs were also recomputed independently from the older branch artifact, and binary shares and CIs from the restored source-level
   SHAP file (net and total, all three analyses; match to 10⁻¹⁴ percentage points); the Figure S3B whiskers (1,000-resample patient bootstrap, seed 42) were recomputed from the same sources and match the plotted values
   (binary to 10⁻¹⁶; six-class to 10⁻⁶, the precision of the older float32 artifact), reproducible with `figures/verify_figS3_inputs.py`.
   **Still open:** the per-class six-class CIs of Table S3 were not independently recomputed; 13 protected earlier files
   and 8 extension outputs recorded in the extension provenance (including its verification-checks file and `figS_shap_compute.py`) are absent, so its
   claim of 256 passed checks cannot be confirmed here; the origin of the restored copy is not recorded.
9. **Table 1 small-cell disclosure [decision needed; not resolved]:** Table 1 shows 8 cells with 1–4 patients or pairs and 7 with 5–9 (mostly missing-data counts and one
   country-by-partition cell; all other supplementary tables have none with 1–4). No MMRF data-use terms are stored in the repository, and a public search on 2026-10-15 found no
   MMRF small-cell or minimum-count rule; the only public publication terms found (the IA19 Researcher Gateway dataset listing, not IA24) require acknowledgement of the MMRF and at
   least 30 days' notice before publication, and the full terms sit behind the Researcher Gateway or Virtual Lab agreement. Nothing was suppressed or merged. Decision for the authors:
   check the IA24 data-use agreement or ask the MMRF data-access team, then either keep the table as is or report such cells as "<5" or merge categories (the Table 1 builder prints the
   smallest displayed count, so a change is a one-line edit there).

Notes: Table 1 and Supplementary Tables S1–S6 are in `figures/supplementary_tables/`; Supplementary Figures and Supplementary
Tables are numbered independently. "Days since index at V(t)" (Figure S3B) is the number of days from the start of first
treatment (day 0) to the current visit. Hallmark pathway names keep their MSigDB spelling (Myc Targets V1/V2). The Figure 4D
composite is called the "train-selected pathway composite"; it is not called a proliferation/MYC axis because Spermatogenesis
is one of its five pathways. Held-out always means the validation + test patients (not used for tuning, selection, fitting or
preprocessing in the redesigned analyses); no result is described as validated or independent, and no algorithm is called
best.
