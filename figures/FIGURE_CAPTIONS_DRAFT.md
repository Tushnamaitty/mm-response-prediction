# Figure captions (draft for scientific review)

Status: draft. Every number below was checked against the extracted figure data in `figures/data/` and the locked
results in `artifacts/results_clean_rerun/`; supplementary tables are in `figures/supplementary_tables/`. The only item
marked **[pending verification]** is the Figure 2 bootstrap provenance. Day 0 is the start of first treatment (CoMMpass
index date) throughout.

---

## Main figures

### Figure 1. Study design, nested information layers and RNA-sequencing timing in MMRF CoMMpass.

**(A)** Next-visit prediction framework and cohort. Each observation is a consecutive visit pair: input features
available at or before the current visit, V(t), are used to predict the International Myeloma Working Group (IMWG)
response at the next visit, V(t+1). The six-class task (progressive disease [PD], stable disease [SD], partial response
[PR], very good partial response [VGPR], complete response [CR], stringent complete response [sCR]) comprised 1,025
patients and 13,451 visit pairs. The binary task (improvement to a better IMWG category at V(t+1)) comprised 1,018
patients and 12,298 pairs after excluding 1,153 pairs with sCR at V(t), which cannot improve. An RNA sample collected on
or before V(t) was available for 9,173 six-class pairs (707 patients) and 8,262 binary pairs (703 patients). Patients
were assigned once to patient-grouped training, validation and test partitions (716, 154 and 155 patients; binary task
709, 154 and 155), shared by every information layer and algorithm, so that no patient contributes to more than one
partition. Hyperparameters were tuned by patient-grouped cross-validation within training patients; validation patients
set the binary decision threshold; test patients were scored once. The survival cohort (Figures 4 and 5) comprised the
674 patients with an RNA sample collected on or before day 0 (459 progression-free survival [PFS] events; 247 deaths).

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

### Figure 2. Next-visit response prediction across nested information layers A–D.

Performance of Logistic Regression, LightGBM and XGBoost for each information layer (Figure 1B) on the held-out test
partition (155 patients), evaluated once. No algorithm was selected as best; XGBoost serves as the reporting reference
algorithm in later figures.

**(A)** Area under the precision–recall curve (AUPRC; higher is better) for binary improvement at V(t+1), 1,930 visit
pairs including 353 improvement events. The dotted line marks the test improvement prevalence (18.3%), the AUPRC expected
from a non-informative model.

**(B)** Macro-averaged F1 (unweighted mean of the six class-specific F1 scores; higher is better) for six-class IMWG
response, 2,079 visit pairs. Logistic Regression performed substantially below the tree-based models for this task.

**(C)** Paired change in performance when a layer is added: longitudinal history (B − A), treatment context (C − B) and
RNA pathway scores (D − C); positive values favour the larger layer. For XGBoost, adding longitudinal history increased
binary AUPRC by +0.022 but changed six-class macro-F1 by −0.014; adding RNA changed binary AUPRC by +0.008 and six-class
macro-F1 by −0.004. In the plotted intervals (pending verification; see below), all D − C intervals included zero for
all three algorithms and both tasks, so there was no clear evidence that RNA improved next-visit prediction on the
held-out test set.

Points are test-set estimates; whiskers are 95% patient-level bootstrap confidence intervals (CIs), unadjusted for
multiple comparisons. The panel A–B intervals correspond to the locked 1,000-resample patient bootstrap in
`artifacts/results_clean_rerun/uncertainty/main_model_cis.csv`. **[pending verification]** The figure was drawn from
extraction files (`figures/data/Fig2/`) that are not yet in the repository; the exact plotted intervals, and in
particular the paired-difference intervals in panel C, have not been independently verified against a reproducible
extraction. Exact test-set values and CIs for all 24 models (from the locked `main_model_cis.csv`, with validation-set
values) are given in Supplementary Table S6; the panel C paired differences are not tabulated pending verification. The
two tasks use different visit-pair populations and metrics and are not directly comparable. These are internal held-out
evaluations within CoMMpass, not external validation.

---

### Figure 3. Exploratory analyses of the incremental contribution of RNA to next-visit prediction and of a proliferation/MYC association.

Panels A–C use out-of-fold predictions from 5 repeats of 5-fold patient-grouped cross-validation on training and
validation patients only (not the held-out test set of Figure 2): XGBoost Layer C and Layer D were refitted in each fold
with the locked hyperparameters and compared on identical RNA-available visit pairs (binary 6,939 pairs / 596 patients;
six-class 7,724 / 600). Changes are Layer D − Layer C. AUPRC: higher is better. RPS, ranked probability score over the six
ordered IMWG categories: lower is better, so ΔRPS > 0 means RNA worsened predictions. Unless stated otherwise, intervals
are 95% patient-cluster bootstrap CIs (2,000 resamples). The locked hyperparameters were originally tuned by
cross-validation within the training patients, who form most of this cross-validation cohort, so absolute performance
may be optimistic; the same hyperparameters were used for Layers C and D. All analyses are post hoc and exploratory.

**(A)** Overall RNA contribution. Direct XGBoost Layer D − Layer C: binary ΔAUPRC +0.0048 (95% CI −0.0017 to +0.0109;
no detectable difference); six-class ΔRPS +0.00075 (+0.00023 to +0.00133; +0.97% of Layer C RPS), a small worsening. A
separate, strongly regularized RNA add-on (ridge-penalized correction of Layer C six-class probabilities using the 50
pathway scores) gave ΔRPS −0.00034 (−0.00058 to −0.00011; −0.44% of Layer C RPS), with a one-sided whole-RNA-vector
permutation p = 0.005 (0 of 200 permutations as extreme; smallest attainable p = 1/201). The add-on was evaluated for the
six-class task only. The two approaches are different models and their estimates are not interchangeable.

**(B)** RNA contribution by quartile of Layer C prediction entropy (uncertainty), computed without outcomes or RNA. The
prespecified test was the interaction (D − C in the highest-uncertainty quartile Q4) − (D − C in Q1–Q3), with Holm
adjustment across the two primary interactions. Binary: Q4 ΔAUPRC +0.0100 (−0.0007 to +0.0202); interaction +0.0139
(+0.0014 to +0.0267; unadjusted p = 0.030; Holm p = 0.060). Six-class: Q4 ΔRPS +0.0016 (+0.0002 to +0.0031), a worsening;
interaction Holm p = 0.142. Neither interaction was significant after adjustment. Post-hoc checks outside the Holm family:
scrambled-RNA permutation p = 0.020 for the binary interaction; continuous-uncertainty slope p = 0.121. Quartile-level
estimates are descriptive.

**(C)** RNA contribution by drug class active at V(t) (strata overlap) and by response trajectory (IMWG category at V(t)
compared with the previous visit), for binary ΔAUPRC and six-class ΔRPS. No stratum improved significantly with RNA.
Six-class RPS was significantly worse after multiplicity adjustment for proteasome inhibitor (Holm p = 0.045), immuno-
modulatory drug (0.042), corticosteroid (0.042) and stable-trajectory pairs (0.002), and for the exploratory anti-CD38
stratum (Benjamini–Hochberg [BH] p = 0.013); these increases were small (ΔRPS +0.95 to +2.86 × 10⁻³; 1.3–3.5% of Layer C
RPS). Holm adjustment was applied across the three primary treatments and, separately, across the two primary trajectory
states, per task on its primary metric; BH adjustment was used for exploratory treatments and secondary states. The
six-class chemotherapy stratum was not evaluated (too few CR/sCR outcomes). Among secondary metrics, two further
subgroup (within-stratum D − C) results were significant after BH adjustment, both RNA worsening: six-class log loss in
the anti-CD38 stratum (+0.021, 95% CI +0.002 to +0.043; BH p = 0.026) and binary AUROC in improving-trajectory pairs
(−0.0075, −0.0133 to −0.0013; BH p = 0.036). Separately, three secondary interaction contrasts (differences in D − C
between strata) were significant after BH adjustment (IMiD and corticosteroid binary contrasts; improving − stable
six-class log loss); these are descriptive. Full per-stratum results, including complement strata, secondary metrics and
interaction contrasts: Supplementary Table S1.

**(D)** Proliferation/MYC composite (first principal component of E2F Targets, G2-M Checkpoint, Mitotic Spindle, Myc
Targets V1 and Myc Targets V2; 71.5% of their variance) and the ordered next IMWG category (cumulative-logit ordinal GEE,
patient-clustered robust SEs; odds ratio < 1 means a higher score is associated with a worse next response). Stage 1,
pathway-selection evidence: pooled single-pathway sensitivity regressions that included the test patients motivated the
post-hoc choice of the five-pathway family (odds ratios 0.78–0.86 per SD). Stage 2, discovery in training and validation
patients: odds ratio 0.76 per SD (95% Wald CI 0.71–0.81; 6,212 pairs, 448 patients). Stage 3, post-hoc internal
split-sample test in held-out test patients with the composite frozen: odds ratio 0.82 (0.69–0.97; one-sided p = 0.011;
1,221 pairs, 84 patients). The one-sided direction was fixed after the pooled stage-1 results and before stages 2–3; it
was not prespecified. Because the family was selected using data that included the test patients, stage 3 is not
independent replication. The stages use different patients and adjustment models, so their estimates are not directly
comparable. The prespecified primary analysis, a matched-pair global test across all 50 pathways, was null (six-class
p = 0.149; binary p = 0.441). This is an association, not evidence of predictive gain or a causal mechanism.

---

### Figure 4. Exploratory incremental prognostic value of pre-treatment RNA pathway scores for progression-free and overall survival.

Survival cohort: 674 patients with one RNA sample collected on or before the start of first treatment (CoMMpass index
date); official MMRF PFS (459 events) and overall survival (OS; 247 deaths). Clinical baseline: ridge-penalized Cox model
with age at index, gender, International Staging System (ISS) stage and exact first-line regimen. Clinical baseline +
RNA: the same model plus the 50 Hallmark pathway scores. Harrell's C-index (higher is better) was computed on pooled
out-of-fold predictions within each of 5 repeats of patient-level 5-fold cross-validation (ridge penalty chosen by inner
5-fold cross-validation) and averaged over repeats. ΔC = C(clinical baseline + RNA) − C(clinical baseline). Intervals are
95% patient-bootstrap CIs (2,000 resamples) on fixed out-of-fold predictions; they reflect patient sampling, not
model-refitting variability. No CIs are available for absolute C-index values.

**(A)** C-index without and with RNA. PFS: 0.601 → 0.642 (ΔC +0.040, 95% CI +0.016 to +0.064). OS: 0.658 → 0.702 (ΔC
+0.043, +0.012 to +0.074). Small hollow points show the five cross-validation repeats and are not confidence intervals. A
one-sided whole-RNA-vector permutation test (200 permutations, penalties frozen) was performed for PFS only: 0 of 200
permuted ΔC reached the observed value (p = 0.005, the smallest attainable). No permutation test was performed for OS.

**(B)** ΔC across primary and sensitivity analyses. Expanded clinical baseline (clinical baseline + baseline LDH,
β2-microglobulin, albumin, creatinine, haemoglobin, calcium and bone-marrow plasma-cell percentage): PFS ΔC +0.035
(+0.014 to +0.056; C 0.610 → 0.645) and OS ΔC +0.049 (+0.022 to +0.076; C 0.656 → 0.704). RNA collected within 30 days
before first treatment (day −30 to 0; 624 patients, 423 PFS events): PFS ΔC +0.028 (+0.003 to +0.053; C 0.619 → 0.646);
this timing analysis was specified for PFS only and was not evaluated for OS. All displayed intervals excluded zero, but
rows differ in cohort or comparator and were not compared statistically. R-ISS was excluded before fitting (overlap with
ISS and LDH; about 57% missing), and cytogenetic (FISH) adjustment was not possible because no such fields were available.

Integrated Brier score (inverse-probability-weighted, 365–1,095 days; lower is better; point estimates only): PFS 0.205 →
0.185; OS 0.100 → 0.089; PFS day −30 to 0, 0.204 → 0.183. The survival cohort overlaps the next-visit partitions (473
training, 101 validation and 100 of the 155 test patients), so these results are not independent of Figures 2–3. All
values, including repeat-level ranges of ΔC for every analysis, are given in Supplementary Table S5. All results are post
hoc, exploratory internal cross-validation, not external validation.

---

### Figure 5. Exploratory RNA pathway coefficient patterns across progression-free and overall survival models.

Coefficients come from the Figure 4 clinical baseline + RNA ridge Cox models, refitted in each of 25 overlapping outer
cross-validation fits (5 repeats × 5 folds) per endpoint with the locked penalty (λ = 1.0 in every fit); refitting
reproduced the locked out-of-fold risks within 10⁻⁹. β is the log-hazard per training-fold SD of a pathway score (hazard
ratio [HR] = e^β); β > 0 means a higher score is associated with higher modelled hazard. A pathway is consistently
directed for an endpoint if all 25 fits share the same sign (PFS: 16 positive, 9 negative, 25 mixed; OS: 21, 10, 19) and
concordant if consistently directed with the same sign for both endpoints.

**(A)** Mean PFS versus mean OS coefficient for all 50 Hallmark pathways, with zero lines and the identity line (dashed).
Nineteen pathways were concordant: 13 associated with higher hazard and 6 with lower hazard in both endpoints; none was
consistently directed in opposite directions. E2F Targets, G2-M Checkpoint, Myc Targets V1 and Notch Signaling are
labelled. Pearson r = 0.83 across the 50 mean coefficients is descriptive: PFS and OS share the same 674 patients, RNA
and covariates, and deaths are also PFS events, so their agreement is not independent corroboration.

**(B)** The 19 concordant pathways, grouped by direction: mean PFS and OS coefficients with the minimum–maximum range across
the 25 fits. Ranges are not confidence intervals; the fits share about 80% of their training patients, so consistency
across fits is not independent replication. Effects are small (concordant HR per SD 0.93–1.07 for PFS and 0.90–1.09 for
OS).

Correlated pathways (for example, the five proliferation/MYC scores share 71.5% of their variance) have interdependent
ridge coefficients that are not individually identifiable. No pathway-level CIs, p-values or causal effects were
estimated. The cohort overlaps Figures 3 and 4, so agreement with the proliferation/MYC result in Figure 3D is not
independent evidence. Patterns are exploratory model associations, not protective mechanisms, biological drivers or
validated biomarkers.

---

## Supplementary figures

### Supplementary Figure S1. Calibration of binary next-visit improvement predictions with and without RNA.

XGBoost information Layer C (clinical + longitudinal history + treatment context) and Layer D (Layer C + RNA) on the
held-out test set (155 patients; 1,930 visit pairs; improvement rate 18.3%). Uncalibrated, Platt and isotonic
recalibration were compared by 5-fold patient-grouped cross-fitting on the validation set (lowest log loss, preferring
the simpler method when Brier scores differed by < 0.001); uncalibrated was selected for both layers, so raw model
probabilities are shown and no validation-fitted transformation was applied.

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

### Supplementary Figure S2. Internal geographic hold-out of clinical, longitudinal and treatment-based prediction within MMRF CoMMpass.

XGBoost information Layer C (143 source input features) was refit on US patients only, using a new random patient-level
82/18 split of US patients (632 training, 139 validation). Preprocessing and model fitting used US training data;
calibration method (uncalibrated selected) and binary decision threshold (0.25) were chosen on US validation. The model
was then evaluated once on all patients enrolled in Spain (80) and Canada (36). Patient sets were disjoint.

**(A)** Binary improvement AUPRC (higher is better). Spain and Canada pooled: 0.451 (95% CI 0.401–0.507; 116 patients,
1,588 pairs, 283 events, 17.8%); Spain 0.463 (0.407–0.532; 949 pairs, 19.5%); Canada 0.443 (0.369–0.541; 639 pairs,
15.3%). Ticks mark each cohort's event rate, the AUPRC of a non-informative model. US validation, 0.525 (1,911 pairs,
18.4%), is a descriptive reference only: it informed calibration and threshold choices and no CI was computed.

**(B)** Six-class IMWG response macro-F1 (higher is better): pooled 0.644 (0.598–0.676; 1,694 pairs); Spain 0.639
(0.589–0.674; 1,034 pairs); Canada 0.634 (0.492–0.686; 660 pairs); US validation 0.647 (2,076 pairs; no CI).

Intervals are 95% patient-level bootstrap percentile CIs (1,000 resamples). Hyperparameters (binary: 100 trees, depth 3,
learning rate 0.05; six-class: 200, 4, 0.05) were reused from the main analysis, where they were selected among three
configurations by patient-grouped cross-validation on a training split that included 80 of the 116 evaluated patients
(Spain 53, Canada 27); no Spain or Canada data were used for fitting, preprocessing, calibration or threshold selection in
this experiment. This is an internal geographic robustness check, not independent external validation. Countries were
not compared statistically; Canada estimates are imprecise (36 patients; e.g. 21 sCR and 27 SD six-class pairs). 57
Spain/Canada binary pairs had treatment-regimen categories not seen in US training. Italy (138 patients) was excluded post
hoc (single site, non-standard visit schedule with a mean visit gap of about 350 days, and an implausible 78.7%
improvement rate); these patients remain in the main analyses. AUROC (Supplementary Table S4): pooled 0.770
(0.729–0.808); Spain 0.761 (0.718–0.809); Canada 0.773 (0.700–0.847); US validation 0.805 (no CI).

---

### Supplementary Figure S3. SHAP attribution for XGBoost information Layer D models.

Exact TreeSHAP values were computed with native XGBoost for the locked Layer D models on the held-out test set (155
patients; 1,930 binary and 2,079 six-class visit pairs); contributions plus the bias reproduced each row's raw model
output within 3 × 10⁻⁶. Binary contributions are in log-odds of improvement; six-class contributions are class-specific raw
scores (softmax inputs), averaged equally over the six IMWG classes, and are not probability attributions. Signed
contributions of all fitted model-input columns of a source feature (one-hot levels, missing-value indicators) were summed
before taking absolute values. XGBoost Layer D is the reporting reference model, not a validation-selected best model.

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

1. **Figure 2 provenance [pending]:** `figures/data/Fig2/` and its extraction script are not in the repository; the panel C
   paired-difference CIs cannot be traced to a locked file.

Notes: Table 1 and Supplementary Tables S1–S6 are in `figures/supplementary_tables/`; Supplementary Figures and
Supplementary Tables are numbered independently. "Days since index at V(t)" (Figure S3B) is the number of days from the
start of first treatment (day 0) to the current visit. Hallmark pathway names keep their MSigDB spelling (Myc
Targets V1/V2); "MYC" is used for the proliferation/MYC programme.
