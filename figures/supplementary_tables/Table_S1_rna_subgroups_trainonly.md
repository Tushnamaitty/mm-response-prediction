# Supplementary Table S1. RNA contribution (Layer D − Layer C) by treatment class and response trajectory: held-out evaluation (Figure 4C), train-only redesign

XGBoost Layer C and Layer D, fitted and tuned on train patients only (locked models; no refitting), were evaluated once on the 213 held-out validation + test patients with RNA (2,405 binary visit pairs; 2,689 six-class pairs). This analysis is specific to XGBoost, which was used alone without a recorded a priori rationale; it does not show that XGBoost is superior to Logistic Regression or LightGBM. Treatment strata are the class active at V(t) and overlap; trajectory strata compare the IMWG category at V(t) with the previous visit (six audited continuity rows excluded). Which treatments receive inference was fixed from train-patient counts before the held-out data were read (PI, IMiD, steroid primary; CD38, chemotherapy exploratory; six-class chemotherapy not evaluable: too few CR/sCR). Δ = Layer D − Layer C; for AUPRC and AUROC positive = RNA better; for RPS and log loss negative = RNA better. 95% patient-cluster bootstrap CIs (2,000 resamples). Adjustment: Holm over the three primary treatments and, separately, over stable + worsening, per task on the primary metric; BH for exploratory treatments, secondary trajectory states and interaction contrasts; secondary metrics and complements are descriptive. Post hoc and exploratory; not independent validation.

| Stratum | Role | Patients (binary / six-class) | Binary ΔAUPRC (95% CI) | Binary ΔAUROC (95% CI) | Binary Δlog loss (95% CI) | Six-class ΔRPS ×10⁻³ (95% CI) | Six-class Δlog loss (95% CI) |
|---|---|---|---|---|---|---|---|
| All RNA-available visit pairs (reference, unadjusted) | reference | 213 / 213 | +0.0109 (−0.0026 to 0.0248) | −0.0032 (−0.0073 to 0.0014) | −0.0009 (−0.0039 to 0.0017) | −0.06 (−1.14 to 1.04) | +0.0002 (−0.0116 to 0.0120) |
| Proteasome inhibitor | primary | 149 / 149 | +0.0018 (−0.0253 to 0.0303); Holm p = 1.00 | +0.0047 (−0.0024 to 0.0119) | −0.0036 (−0.0086 to 0.0013) | −0.55 (−2.06 to 1.06); Holm p = 0.97 | −0.0086 (−0.0272 to 0.0105) |
| Immunomodulatory drug | primary | 150 / 151 | +0.0010 (−0.0197 to 0.0176); Holm p = 1.00 | −0.0054 (−0.0122 to 0.0015) | +0.0009 (−0.0022 to 0.0037) | +0.04 (−1.30 to 1.41); Holm p = 0.97 | −0.0056 (−0.0212 to 0.0088) |
| Corticosteroid | primary | 172 / 172 | +0.0064 (−0.0125 to 0.0237); Holm p = 1.00 | +0.0005 (−0.0045 to 0.0057) | −0.0015 (−0.0053 to 0.0021) | −1.03 (−2.48 to 0.38); Holm p = 0.51 | −0.0157 (−0.0340 to 0.0021) |
| Anti-CD38 antibody | exploratory | 42 / 42 | −0.0009 (−0.0497 to 0.0341); BH p = 0.94 | −0.0077 (−0.0221 to 0.0086); BH p = 0.49 | +0.0058 (−0.0010 to 0.0138); BH p = 0.20 | +2.08 (−1.45 to 5.75); BH p = 0.27 | +0.0020 (−0.0415 to 0.0516); BH p = 0.95 |
| Chemotherapy | exploratory | 72 / — | −0.0038 (−0.0524 to 0.0320); BH p = 0.94 | +0.0049 (−0.0083 to 0.0164); BH p = 0.49 | −0.0049 (−0.0142 to 0.0045); BH p = 0.27 | not evaluable (frozen support rule) | not evaluable |
| Stable trajectory | primary | 172 / 172 | −0.0001 (−0.0212 to 0.0214); Holm p = 1.00 | −0.0102 (−0.0190 to −0.0019) | +0.0019 (−0.0004 to 0.0040) | +0.81 (−0.31 to 1.98); Holm p = 0.20 | +0.0071 (−0.0054 to 0.0218) |
| Worsening trajectory | primary | 132 / 132 | +0.0050 (−0.0280 to 0.0372); Holm p = 1.00 | +0.0085 (−0.0111 to 0.0273) | −0.0075 (−0.0177 to 0.0022) | −2.21 (−4.80 to 0.32); Holm p = 0.20 | −0.0086 (−0.0376 to 0.0207) |
| Improving trajectory | secondary | 146 / 153 | −0.0009 (−0.0210 to 0.0203); BH p = 0.91 | −0.0001 (−0.0183 to 0.0176); BH p = 0.98 | +0.0000 (−0.0055 to 0.0049); BH p = 1.00 | −2.61 (−5.35 to −0.03); BH p = 0.10 | −0.0284 (−0.0562 to −0.0010); BH p = 0.07 |
| No previous visit | secondary | 206 / 207 | +0.0318 (−0.0101 to 0.0726); BH p = 0.33 | +0.0060 (−0.0070 to 0.0201); BH p = 0.70 | −0.0094 (−0.0222 to 0.0019); BH p = 0.21 | +0.05 (−3.74 to 3.80); BH p = 0.96 | +0.0146 (−0.0198 to 0.0465); BH p = 0.40 |

No within-stratum test on the primary metric is significant after adjustment. The patients column counts held-out patients contributing rows to the stratum.

## Interaction contrasts (held-out)

| Task | Contrast (on − not on / state − stable) | Metric | Interaction (95% CI; RPS ×10⁻³) | Unadjusted p | BH p | Family |
|---|---|---|---|---|---|---|
| binary | improving minus stable | auprc | −0.0008 (−0.0309 to 0.0287) | 0.971 | 0.971 | binary|auprc|state interaction contrasts |
| binary | improving minus stable | auroc | +0.0101 (−0.0109 to 0.0305) | 0.369 | 0.369 | binary|auroc|state interaction contrasts |
| binary | improving minus stable | logloss | −0.0019 (−0.0076 to 0.0038) | 0.507 | 0.507 | binary|logloss|state interaction contrasts |
| binary | worsening minus stable | auprc | +0.0052 (−0.0344 to 0.0432) | 0.789 | 0.971 | binary|auprc|state interaction contrasts |
| binary | worsening minus stable | auroc | +0.0187 (−0.0022 to 0.0402) | 0.081 | 0.162 | binary|auroc|state interaction contrasts |
| binary | worsening minus stable | logloss | −0.0094 (−0.0192 to 0.0001) | 0.055 | 0.110 | binary|logloss|state interaction contrasts |
| binary | on minus not cd38 | auprc | −0.0125 (−0.0599 to 0.0271) | 0.458 | 0.816 | binary|auprc|treatment interaction contrasts |
| binary | on minus not cd38 | auroc | −0.0048 (−0.0202 to 0.0119) | 0.560 | 0.560 | binary|auroc|treatment interaction contrasts |
| binary | on minus not cd38 | logloss | +0.0072 (0.0000 to 0.0159) | 0.051 | 0.255 | binary|logloss|treatment interaction contrasts |
| binary | on minus not chemo | auprc | −0.0147 (−0.0667 to 0.0248) | 0.495 | 0.816 | binary|auprc|treatment interaction contrasts |
| binary | on minus not chemo | auroc | +0.0091 (−0.0047 to 0.0206) | 0.186 | 0.310 | binary|auroc|treatment interaction contrasts |
| binary | on minus not chemo | logloss | −0.0042 (−0.0137 to 0.0058) | 0.368 | 0.460 | binary|logloss|treatment interaction contrasts |
| binary | on minus not imid | auprc | −0.0172 (−0.0443 to 0.0071) | 0.180 | 0.816 | binary|auprc|treatment interaction contrasts |
| binary | on minus not imid | auroc | −0.0042 (−0.0118 to 0.0039) | 0.293 | 0.366 | binary|auroc|treatment interaction contrasts |
| binary | on minus not imid | logloss | +0.0034 (−0.0012 to 0.0083) | 0.165 | 0.325 | binary|logloss|treatment interaction contrasts |
| binary | on minus not pi | auprc | −0.0057 (−0.0346 to 0.0278) | 0.816 | 0.816 | binary|auprc|treatment interaction contrasts |
| binary | on minus not pi | auroc | +0.0123 (0.0037 to 0.0210) | 0.009 | 0.045 | binary|auroc|treatment interaction contrasts |
| binary | on minus not pi | logloss | −0.0037 (−0.0094 to 0.0021) | 0.195 | 0.325 | binary|logloss|treatment interaction contrasts |
| binary | on minus not steroid | auprc | −0.0041 (−0.0299 to 0.0232) | 0.800 | 0.816 | binary|auprc|treatment interaction contrasts |
| binary | on minus not steroid | auroc | +0.0091 (−0.0001 to 0.0181) | 0.055 | 0.137 | binary|auroc|treatment interaction contrasts |
| binary | on minus not steroid | logloss | −0.0010 (−0.0064 to 0.0044) | 0.725 | 0.725 | binary|logloss|treatment interaction contrasts |
| multiclass | improving minus stable | logloss | −0.0354 (−0.0647 to −0.0074) | 0.013 | 0.026 | multiclass|logloss|state interaction contrasts |
| multiclass | improving minus stable | rps | −3.42 (−6.26 to −0.73) | 0.011 | 0.022 | multiclass|rps|state interaction contrasts |
| multiclass | worsening minus stable | logloss | −0.0157 (−0.0477 to 0.0164) | 0.326 | 0.326 | multiclass|logloss|state interaction contrasts |
| multiclass | worsening minus stable | rps | −3.02 (−5.88 to −0.34) | 0.032 | 0.032 | multiclass|rps|state interaction contrasts |
| multiclass | on minus not cd38 | logloss | +0.0020 (−0.0418 to 0.0475) | 0.946 | 0.946 | multiclass|logloss|treatment interaction contrasts |
| multiclass | on minus not cd38 | rps | +2.29 (−1.42 to 6.08) | 0.233 | 0.466 | multiclass|rps|treatment interaction contrasts |
| multiclass | on minus not imid | logloss | −0.0109 (−0.0343 to 0.0097) | 0.327 | 0.436 | multiclass|logloss|treatment interaction contrasts |
| multiclass | on minus not imid | rps | +0.20 (−1.77 to 2.11) | 0.837 | 0.837 | multiclass|rps|treatment interaction contrasts |
| multiclass | on minus not pi | logloss | −0.0117 (−0.0320 to 0.0082) | 0.272 | 0.436 | multiclass|logloss|treatment interaction contrasts |
| multiclass | on minus not pi | rps | −0.64 (−2.58 to 1.19) | 0.525 | 0.700 | multiclass|rps|treatment interaction contrasts |
| multiclass | on minus not steroid | logloss | −0.0255 (−0.0467 to −0.0054) | 0.007 | 0.028 | multiclass|logloss|treatment interaction contrasts |
| multiclass | on minus not steroid | rps | −1.55 (−3.40 to 0.30) | 0.103 | 0.412 | multiclass|rps|treatment interaction contrasts |

Two six-class trajectory contrasts (worsening − stable and improving − stable, ΔRPS) have BH p < 0.05. They are secondary, exploratory and unreplicated; no primary test is significant.

## Complement strata (held-out; descriptive)

| Task | Complement stratum | Metric | Patients | Δ (95% CI; RPS ×10⁻³); descriptive, unadjusted |
|---|---|---|---|---|
| binary | not on cd38 | auprc | 213 | +0.0115 (−0.0030 to 0.0260) |
| binary | not on cd38 | auroc | 213 | −0.0029 (−0.0075 to 0.0022) |
| binary | not on cd38 | logloss | 213 | −0.0014 (−0.0046 to 0.0013) |
| binary | not on chemo | auprc | 209 | +0.0110 (−0.0047 to 0.0262) |
| binary | not on chemo | auroc | 209 | −0.0042 (−0.0086 to 0.0008) |
| binary | not on chemo | logloss | 209 | −0.0006 (−0.0036 to 0.0019) |
| binary | not on imid | auprc | 185 | +0.0182 (−0.0014 to 0.0357) |
| binary | not on imid | auroc | 185 | −0.0012 (−0.0063 to 0.0041) |
| binary | not on imid | logloss | 185 | −0.0025 (−0.0068 to 0.0016) |
| binary | not on pi | auprc | 197 | +0.0075 (−0.0101 to 0.0247) |
| binary | not on pi | auroc | 197 | −0.0076 (−0.0133 to −0.0016) |
| binary | not on pi | logloss | 197 | +0.0001 (−0.0033 to 0.0032) |
| binary | not on steroid | auprc | 179 | +0.0105 (−0.0090 to 0.0295) |
| binary | not on steroid | auroc | 179 | −0.0086 (−0.0167 to −0.0009) |
| binary | not on steroid | logloss | 179 | −0.0005 (−0.0047 to 0.0031) |
| multiclass | not on cd38 | logloss | 213 | +0.0001 (−0.0116 to 0.0110) |
| multiclass | not on cd38 | rps | 213 | −0.21 (−1.35 to 0.92) |
| multiclass | not on imid | logloss | 186 | +0.0053 (−0.0114 to 0.0241) |
| multiclass | not on imid | rps | 186 | −0.16 (−1.70 to 1.51) |
| multiclass | not on pi | logloss | 197 | +0.0031 (−0.0094 to 0.0162) |
| multiclass | not on pi | rps | 197 | +0.10 (−1.21 to 1.40) |
| multiclass | not on steroid | logloss | 179 | +0.0098 (−0.0034 to 0.0235) |
| multiclass | not on steroid | rps | 179 | +0.52 (−0.85 to 1.92) |

## Development-only cross-validation reference (train patients; not an evaluation)

5 × 5 patient-grouped CV within the 716 train patients (locked hyperparameters, tuned in the same patients, so absolute performance is optimistic), with the same strata, families and bootstrap. Shown for context only; it cannot replace the held-out evaluation.

| Stratum | Role | Patients (binary) | Binary ΔAUPRC (95% CI) | Six-class ΔRPS ×10⁻³ (95% CI) |
|---|---|---|---|---|
| All RNA-available visit pairs (reference, unadjusted) | reference | — | not evaluable | not evaluable |
| Proteasome inhibitor | primary | 341 | −0.0082 (−0.0174 to 0.0013) | +0.79 (−0.18 to 1.85) |
| Immunomodulatory drug | primary | 339 | −0.0037 (−0.0117 to 0.0052) | +0.83 (0.06 to 1.66) |
| Corticosteroid | primary | 381 | −0.0094 (−0.0184 to 0.0000) | +1.51 (0.47 to 2.79); Holm p = 0.003 |
| Anti-CD38 antibody | exploratory | 106 | −0.0124 (−0.0302 to 0.0052) | +2.56 (−0.86 to 7.72) |
| Chemotherapy | exploratory | 149 | −0.0014 (−0.0163 to 0.0168) | not evaluable |
| Stable trajectory | primary | 392 | −0.0033 (−0.0104 to 0.0052) | +1.52 (0.69 to 2.39); Holm p = 0.002 |
| Worsening trajectory | primary | 328 | −0.0047 (−0.0203 to 0.0106) | +0.66 (−0.68 to 2.02) |
| Improving trajectory | secondary | 358 | −0.0114 (−0.0248 to 0.0053) | +0.66 (−0.70 to 2.00) |
| No previous visit | secondary | 476 | +0.0171 (−0.0015 to 0.0313) | −0.48 (−3.01 to 1.67) |

Sources: `figures/data/Fig3ABC_trainonly/` (built by `figures/fig3ABC_trainonly_extract_data.py`), cross-checked against `artifacts/results_clean_rerun/rna_contribution_trainonly/` and the frozen protocol `explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md`. The earlier Table S1 (pooled train+validation CV, in which validation patients trained the fold models) is superseded.