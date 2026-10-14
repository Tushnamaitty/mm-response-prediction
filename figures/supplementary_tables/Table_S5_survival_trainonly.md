# Supplementary Table S5. Survival discrimination and overall prediction error with and without pre-treatment RNA: held-out evaluation (Figure 5), train-only redesign

Survival cohort: patients with one RNA sample collected on or before the start of first treatment (CoMMpass index date); official MMRF PFS and OS. Ridge-penalized Cox models (Breslow); clinical baseline = age at index, gender, ISS, exact first-line regimen; expanded baseline adds LDH, β2-microglobulin, albumin, creatinine, haemoglobin, calcium and bone-marrow plasma-cell %; + RNA = 50 Hallmark pathway scores. **Models were fitted on the 473 train patients only** (penalties chosen by 5-fold inner CV repeated 5 times within those patients, then frozen) and **scored once on the 201 held-out validation + test patients** (101 + 100; PFS 149 events, OS 82 deaths). Harrell's C-index (higher is better); ΔC 95% CIs: patient bootstrap (2,000 resamples). Integrated Brier score (IBS; IPCW, 365–1,095 days; lower is better): point estimates only. Pre-specified PFS criterion: ΔC > 0 AND CI lower bound > 0 AND one-sided permutation p ≤ 0.025 (RNA vectors permuted among held-out patients, B = 10,000). Post hoc and exploratory: the earlier pooled results had been seen before the redesign, so this is not independent or external validation.

## Held-out evaluation (validation + test patients)

| Endpoint | Analysis | Patients / events | C-index: baseline → + RNA | ΔC (95% CI) | ΔC range over 5 CV repeats | Permutation p (one-sided) | IBS: baseline → + RNA | Frozen λ: baseline / + RNA |
|---|---|---|---|---|---|---|---|---|
| PFS | Primary (held-out) | 201 / 149 | 0.630 → 0.657 | +0.028 (−0.019 to 0.074) | — | 0.0001 | 0.233 → 0.208 (Δ −0.025; no CI) | 100 / 1 |
| PFS | Expanded clinical baseline | 201 / 149 | 0.648 → 0.658 | +0.010 (−0.029 to 0.049) | — | not performed | not computed | 1 / 1 |
| PFS | RNA day −30 to 0 only | 186 / 137 | 0.648 → 0.662 | +0.014 (−0.036 to 0.061) | — | not performed | not computed | 100 / 1 |
| PFS | Validation patients only (descriptive) | 101 / 71 | 0.646 → 0.702 | +0.056 (0.003 to 0.109) | — | not performed | not computed | 100 / 1 |
| PFS | Test patients only (descriptive) | 100 / 78 | 0.605 → 0.604 | −0.001 (−0.082 to 0.076) | — | not performed | not computed | 100 / 1 |
| OS | Primary (held-out) | 201 / 82 | 0.696 → 0.706 | +0.010 (−0.045 to 0.065) | — | not performed | 0.138 → 0.122 (Δ −0.016; no CI) | 100 / 1 |
| OS | Expanded clinical baseline | 201 / 82 | 0.704 → 0.702 | −0.002 (−0.049 to 0.045) | — | not performed | not computed | 10 / 1 |
| OS | RNA day −30 to 0 only | — | not evaluated | — | — | not performed | — | — |

**The pre-specified PFS criterion was not met**: ΔC = +0.028 with a 95% CI that includes 0 (−0.019 to 0.074), although the permutation p was 0.0001 (0 of 10,000 permuted statistics reached the observed ΔC). The permutation null is centred below 0 because permuted RNA adds noise to the frozen model, so the permutation result tests association between RNA and outcome, not ΔC > 0. The OS point estimate is also small with a CI that includes 0. Neither result is evidence of no effect. The validation-only and test-only rows are descriptive partitions of the 201 patients and are imprecise; they were not tested against each other. The OS RNA-timing analysis was specified for PFS only and was not run.

## Development-only cross-validation reference (train patients; not an evaluation)

5 × 5 patient-level CV within the 473 train patients (pooled out-of-fold C per repeat, averaged over repeats), shown for context and to gauge optimism. It cannot replace the held-out evaluation.

| Endpoint | Analysis | Patients / events | C-index: baseline → + RNA | ΔC (95% CI) | ΔC range over 5 CV repeats | Permutation p (one-sided) | IBS: baseline → + RNA | Frozen λ: baseline / + RNA |
|---|---|---|---|---|---|---|---|---|
| PFS | Development-only 5×5 CV (train patients) | 473 / 310 | 0.595 → 0.617 | +0.022 (−0.010 to 0.054) | 0.005 to 0.040 | not performed | 0.195 → 0.179 (Δ −0.016; no CI) | — |
| OS | Development-only 5×5 CV (train patients) | 473 / 165 | 0.638 → 0.685 | +0.048 (0.009 to 0.088) | 0.022 to 0.066 | not performed | 0.085 → 0.076 (Δ −0.009; no CI) | — |

## Earlier pooled estimate (superseded; context only)

5 × 5 CV over all 674 patients, in which the validation and test patients also trained the fold models. These values are not leakage-free and are not comparable with the held-out rows.

| Endpoint | Analysis | Patients / events | C-index: baseline → + RNA | ΔC (95% CI) | ΔC range over 5 CV repeats | Permutation p (one-sided) | IBS: baseline → + RNA | Frozen λ: baseline / + RNA |
|---|---|---|---|---|---|---|---|---|
| PFS | Earlier pooled 5×5 CV (all 674 patients) | 674 / 459 | 0.601 → 0.642 | +0.040 (0.016 to 0.064) | 0.031–0.050 † | 0.0050 | 0.205 → 0.185 (Δ −0.020; no CI) † | — |
| OS | Earlier pooled 5×5 CV (all 674 patients) | 674 / 247 | 0.658 → 0.702 | +0.043 (0.012 to 0.074) | 0.039–0.051 † | not performed | 0.100 → 0.089 (Δ −0.011; no CI) † | — |

† Integrated Brier score and repeat range transcribed verbatim from the superseded Table S5 (`Table_S5_survival.md`); the pooled out-of-fold prediction files it was built from are not available on this machine (see `figures/LEGACY_MISSING_INPUTS.md`), so these two values cannot be re-derived or independently checked here. The pooled C-index, ΔC, CI and permutation p come from the verified extraction.

No CIs exist for absolute C-index or IBS values. Rows differ in cohort or comparator and were not compared statistically. Sources: `figures/data/Fig45_trainonly/` (built by `figures/fig45_trainonly_extract_data.py`), cross-checked against `artifacts/results_clean_rerun/survival_trainonly/` and the frozen protocol `explainability/SURVIVAL_TRAINONLY_PROTOCOL.md`. The earlier Table S5 (`Table_S5_survival.md`; pooled 5×5 CV) is superseded.