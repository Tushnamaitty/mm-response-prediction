# Supplementary Table S1. RNA contribution by treatment class and response trajectory (Figure 3C)

XGBoost information Layer D − Layer C, out-of-fold predictions from 5 repeats × 5-fold patient-grouped cross-validation on training and validation patients (not the held-out test set). AUPRC: higher is better. RPS (ranked probability score): lower is better, so ΔRPS > 0 means RNA worsened predictions. 95% patient-cluster bootstrap CIs (2,000 resamples). Multiplicity: Holm across the three primary treatments (PI, IMiD, corticosteroid) and, separately, the two primary trajectory states (stable, worsening), per task on its primary metric; Benjamini–Hochberg (BH) for exploratory treatments and secondary states; complement rows and the reference row are unadjusted. Treatment strata overlap. Exploratory, post hoc.

## Binary improvement

| Analysis | Stratum | Role | Metric | Visit pairs / patients | D − C | 95% CI | Raw p | Adjusted p | Significant after adjustment |
|---|---|---|---|---|---|---|---|---|---|
| Treatment class active at V(t) | All RNA-available visit pairs | reference | ΔAUPRC | 6,939 / 596 | +0.0048 | −0.0017 to 0.0109 | 0.154 | unadjusted | n/a |
| Treatment class active at V(t) | On Proteasome inhibitor (PI) at V(t) | primary | ΔAUPRC | 1,902 / 412 | −0.0050 | −0.0130 to 0.0027 | 0.234 | 0.468 (Holm) | no |
| Treatment class active at V(t) | Not on Proteasome inhibitor (PI) at V(t) | complement (descriptive) | ΔAUPRC | 5,037 / 549 | +0.0078 | −0.0004 to 0.0158 | 0.068 | unadjusted | n/a |
| Treatment class active at V(t) | On Immunomodulatory drug (IMiD) at V(t) | primary | ΔAUPRC | 3,312 / 410 | −0.0064 | −0.0132 to 0.0007 | 0.078 | 0.234 (Holm) | no |
| Treatment class active at V(t) | Not on Immunomodulatory drug (IMiD) at V(t) | complement (descriptive) | ΔAUPRC | 3,627 / 550 | +0.0100 | 0.0009 to 0.0188 | 0.032 | unadjusted | n/a |
| Treatment class active at V(t) | On Corticosteroid at V(t) | primary | ΔAUPRC | 2,666 / 464 | −0.0038 | −0.0120 to 0.0044 | 0.388 | 0.468 (Holm) | no |
| Treatment class active at V(t) | Not on Corticosteroid at V(t) | complement (descriptive) | ΔAUPRC | 4,273 / 524 | +0.0092 | −0.0001 to 0.0174 | 0.053 | unadjusted | n/a |
| Treatment class active at V(t) | On Anti-CD38 antibody at V(t) | exploratory | ΔAUPRC | 532 / 129 | −0.0078 | −0.0297 to 0.0161 | 0.554 | 0.554 (BH) | no |
| Treatment class active at V(t) | Not on Anti-CD38 antibody at V(t) | complement (descriptive) | ΔAUPRC | 6,407 / 596 | +0.0055 | −0.0009 to 0.0119 | 0.096 | unadjusted | n/a |
| Treatment class active at V(t) | On Chemotherapy at V(t) | exploratory | ΔAUPRC | 400 / 184 | +0.0045 | −0.0097 to 0.0181 | 0.552 | 0.554 (BH) | no |
| Treatment class active at V(t) | Not on Chemotherapy at V(t) | complement (descriptive) | ΔAUPRC | 6,539 / 584 | +0.0049 | −0.0021 to 0.0116 | 0.186 | unadjusted | n/a |
| Response trajectory at V(t) | Improving (higher category than previous visit) | secondary | ΔAUPRC | 1,028 / 423 | +0.0003 | −0.0106 to 0.0121 | 0.947 | 0.947 (BH) | no |
| Response trajectory at V(t) | Stable (same IMWG category as previous visit) | primary | ΔAUPRC | 4,403 / 473 | +0.0012 | −0.0070 to 0.0100 | 0.775 | 1.000 (Holm) | no |
| Response trajectory at V(t) | Worsening (lower category than previous visit) | primary | ΔAUPRC | 928 / 385 | −0.0013 | −0.0118 to 0.0099 | 0.814 | 1.000 (Holm) | no |
| Response trajectory at V(t) | No previous visit (first visit pair) | secondary | ΔAUPRC | 579 / 579 | +0.0139 | −0.0043 to 0.0299 | 0.161 | 0.322 (BH) | no |

## Six-class IMWG response

| Analysis | Stratum | Role | Metric | Visit pairs / patients | D − C | 95% CI | Raw p | Adjusted p | Significant after adjustment |
|---|---|---|---|---|---|---|---|---|---|
| Treatment class active at V(t) | All RNA-available visit pairs | reference | ΔRPS | 7,724 / 600 | +0.00075 | 0.00023 to 0.00133 | 0.004 | unadjusted | n/a |
| Treatment class active at V(t) | On Proteasome inhibitor (PI) at V(t) | primary | ΔRPS | 2,021 / 414 | +0.00104 | 0.00003 to 0.00212 | 0.045 | 0.045 (Holm) | **yes** |
| Treatment class active at V(t) | Not on Proteasome inhibitor (PI) at V(t) | complement (descriptive) | ΔRPS | 5,703 / 554 | +0.00065 | 0.00003 to 0.00129 | 0.041 | unadjusted | n/a |
| Treatment class active at V(t) | On Immunomodulatory drug (IMiD) at V(t) | primary | ΔRPS | 3,676 / 413 | +0.00095 | 0.00020 to 0.00177 | 0.014 | 0.042 (Holm) | **yes** |
| Treatment class active at V(t) | Not on Immunomodulatory drug (IMiD) at V(t) | complement (descriptive) | ΔRPS | 4,048 / 559 | +0.00057 | −0.00016 to 0.00141 | 0.122 | unadjusted | n/a |
| Treatment class active at V(t) | On Corticosteroid at V(t) | primary | ΔRPS | 2,793 / 464 | +0.00113 | 0.00029 to 0.00206 | 0.014 | 0.042 (Holm) | **yes** |
| Treatment class active at V(t) | Not on Corticosteroid at V(t) | complement (descriptive) | ΔRPS | 4,931 / 529 | +0.00054 | −0.00015 to 0.00122 | 0.149 | unadjusted | n/a |
| Treatment class active at V(t) | On Anti-CD38 antibody at V(t) | exploratory | ΔRPS | 559 / 129 | +0.00286 | 0.00057 to 0.00602 | 0.013 | 0.013 (BH) | **yes** |
| Treatment class active at V(t) | Not on Anti-CD38 antibody at V(t) | complement (descriptive) | ΔRPS | 7,165 / 600 | +0.00059 | 0.00004 to 0.00117 | 0.038 | unadjusted | n/a |
| Response trajectory at V(t) | Improving (higher category than previous visit) | secondary | ΔRPS | 1,192 / 448 | +0.00006 | −0.00124 to 0.00140 | 0.957 | 0.957 (BH) | no |
| Response trajectory at V(t) | Stable (same IMWG category as previous visit) | primary | ΔRPS | 5,013 / 478 | +0.00114 | 0.00049 to 0.00183 | < 0.001 | 0.002 (Holm) | **yes** |
| Response trajectory at V(t) | Worsening (lower category than previous visit) | primary | ΔRPS | 928 / 385 | +0.00043 | −0.00090 to 0.00177 | 0.500 | 0.500 (Holm) | no |
| Response trajectory at V(t) | No previous visit (first visit pair) | secondary | ΔRPS | 587 / 587 | −0.00042 | −0.00231 to 0.00149 | 0.709 | 0.957 (BH) | no |

## Secondary interaction contrasts (difference in D − C between strata)

| Analysis | Contrast | Task | Metric | Interaction | 95% CI | Raw p | Adjusted p | Adjustment |
|---|---|---|---|---|---|---|---|---|
| Direction 3 | (D - C on Proteasome inhibitor (PI)) - (D - C not on) | Binary improvement | auprc | −0.0128 | −0.0238 to −0.0014 | 0.028 | 0.063 | BH (binary|auprc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Proteasome inhibitor (PI)) - (D - C not on) | Binary improvement | auroc | −0.0013 | −0.0054 to 0.0027 | 0.533 | 0.888 | BH (binary|auroc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Proteasome inhibitor (PI)) - (D - C not on) | Binary improvement | logloss | +0.0025 | 0.0003 to 0.0048 | 0.027 | 0.067 | BH (binary|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Immunomodulatory drug (IMiD)) - (D - C not on) | Binary improvement | auprc | −0.0164 | −0.0262 to −0.0051 | 0.007 | 0.035 | BH (binary|auprc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Immunomodulatory drug (IMiD)) - (D - C not on) | Binary improvement | auroc | −0.0014 | −0.0050 to 0.0024 | 0.477 | 0.888 | BH (binary|auroc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Immunomodulatory drug (IMiD)) - (D - C not on) | Binary improvement | logloss | +0.0020 | −0.0000 to 0.0040 | 0.052 | 0.087 | BH (binary|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Corticosteroid) - (D - C not on) | Binary improvement | auprc | −0.0130 | −0.0242 to −0.0009 | 0.038 | 0.063 | BH (binary|auprc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Corticosteroid) - (D - C not on) | Binary improvement | auroc | −0.0026 | −0.0064 to 0.0012 | 0.178 | 0.888 | BH (binary|auroc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Corticosteroid) - (D - C not on) | Binary improvement | logloss | +0.0029 | 0.0008 to 0.0050 | 0.007 | 0.035 | BH (binary|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Anti-CD38 antibody) - (D - C not on) | Binary improvement | auprc | −0.0133 | −0.0351 to 0.0110 | 0.296 | 0.370 | BH (binary|auprc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Anti-CD38 antibody) - (D - C not on) | Binary improvement | auroc | −0.0005 | −0.0062 to 0.0054 | 0.877 | 0.894 | BH (binary|auroc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Anti-CD38 antibody) - (D - C not on) | Binary improvement | logloss | +0.0017 | −0.0019 to 0.0052 | 0.382 | 0.477 | BH (binary|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Chemotherapy) - (D - C not on) | Binary improvement | auprc | −0.0004 | −0.0165 to 0.0143 | 0.927 | 0.927 | BH (binary|auprc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Chemotherapy) - (D - C not on) | Binary improvement | auroc | +0.0006 | −0.0054 to 0.0059 | 0.894 | 0.894 | BH (binary|auroc|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Chemotherapy) - (D - C not on) | Binary improvement | logloss | +0.0004 | −0.0039 to 0.0048 | 0.817 | 0.817 | BH (binary|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Proteasome inhibitor (PI)) - (D - C not on) | Six-class IMWG response | rps | +0.00038 | −0.00088 to 0.00160 | 0.515 | 0.515 | BH (multiclass|rps|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Proteasome inhibitor (PI)) - (D - C not on) | Six-class IMWG response | logloss | +0.0009 | −0.0114 to 0.0135 | 0.885 | 0.885 | BH (multiclass|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Immunomodulatory drug (IMiD)) - (D - C not on) | Six-class IMWG response | rps | +0.00037 | −0.00078 to 0.00150 | 0.508 | 0.515 | BH (multiclass|rps|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Immunomodulatory drug (IMiD)) - (D - C not on) | Six-class IMWG response | logloss | +0.0066 | −0.0034 to 0.0166 | 0.206 | 0.412 | BH (multiclass|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Corticosteroid) - (D - C not on) | Six-class IMWG response | rps | +0.00059 | −0.00051 to 0.00182 | 0.311 | 0.515 | BH (multiclass|rps|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Corticosteroid) - (D - C not on) | Six-class IMWG response | logloss | +0.0052 | −0.0049 to 0.0158 | 0.313 | 0.417 | BH (multiclass|logloss|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Anti-CD38 antibody) - (D - C not on) | Six-class IMWG response | rps | +0.00227 | −0.00016 to 0.00551 | 0.072 | 0.288 | BH (multiclass|rps|interaction contrasts across treatments (secondary)) |
| Direction 3 | (D - C on Anti-CD38 antibody) - (D - C not on) | Six-class IMWG response | logloss | +0.0152 | −0.0049 to 0.0375 | 0.129 | 0.412 | BH (multiclass|logloss|interaction contrasts across treatments (secondary)) |
| Direction 4 | Worsening - stable (D - C differences) | Binary improvement | auprc | −0.0024 | −0.0172 to 0.0119 | 0.738 | 0.862 | BH (binary|auprc|state interaction contrasts (secondary)) |
| Direction 4 | Worsening - stable (D - C differences) | Binary improvement | auroc | +0.0012 | −0.0060 to 0.0088 | 0.753 | 0.753 | BH (binary|auroc|state interaction contrasts (secondary)) |
| Direction 4 | Worsening - stable (D - C differences) | Binary improvement | logloss | −0.0043 | −0.0082 to −0.0001 | 0.042 | 0.084 | BH (binary|logloss|state interaction contrasts (secondary)) |
| Direction 4 | Improving - stable (D - C differences) | Binary improvement | auprc | −0.0008 | −0.0155 to 0.0134 | 0.862 | 0.862 | BH (binary|auprc|state interaction contrasts (secondary)) |
| Direction 4 | Improving - stable (D - C differences) | Binary improvement | auroc | −0.0044 | −0.0112 to 0.0028 | 0.221 | 0.442 | BH (binary|auroc|state interaction contrasts (secondary)) |
| Direction 4 | Improving - stable (D - C differences) | Binary improvement | logloss | +0.0002 | −0.0023 to 0.0025 | 0.886 | 0.886 | BH (binary|logloss|state interaction contrasts (secondary)) |
| Direction 4 | Worsening - stable (D - C differences) | Six-class IMWG response | rps | −0.00071 | −0.00217 to 0.00078 | 0.333 | 0.333 | BH (multiclass|rps|state interaction contrasts (secondary)) |
| Direction 4 | Worsening - stable (D - C differences) | Six-class IMWG response | logloss | −0.0123 | −0.0273 to 0.0040 | 0.138 | 0.138 | BH (multiclass|logloss|state interaction contrasts (secondary)) |
| Direction 4 | Improving - stable (D - C differences) | Six-class IMWG response | rps | −0.00108 | −0.00252 to 0.00031 | 0.131 | 0.262 | BH (multiclass|rps|state interaction contrasts (secondary)) |
| Direction 4 | Improving - stable (D - C differences) | Six-class IMWG response | logloss | −0.0181 | −0.0306 to −0.0060 | 0.005 | 0.010 | BH (multiclass|logloss|state interaction contrasts (secondary)) |

Six-class chemotherapy stratum not evaluated (stratum support rule: too few CR/sCR outcomes). AUROC and log-loss rows (secondary metrics) are in the CSV. After BH adjustment, secondary metrics gave two kinds of significant result, which answer different questions:
- **Subgroup effects (within-stratum Layer D − Layer C; two rows)**, both in the direction of RNA worsening: On Anti-CD38 antibody at V(t) (Six-class IMWG response) Δlog loss +0.0211 [0.0021 to 0.0431], BH p = 0.026; Improving (higher category than previous visit) (Binary improvement) ΔAUROC −0.0075 [−0.0133 to −0.0013], BH p = 0.036.
- **Interaction contrasts (difference in D − C between strata; three rows)**: IMiD vs not (Binary improvement) AUPRC −0.0164 [−0.0262 to −0.0051], BH p = 0.035; corticosteroid vs not (Binary improvement) log loss +0.0029 [0.0008 to 0.0050], BH p = 0.035; improving − stable (Six-class IMWG response) log loss −0.0181 [−0.0306 to −0.0060], BH p = 0.010. A significant contrast means the RNA effect differed between strata; it does not by itself show that RNA helped or harmed within either stratum.

All of these are secondary, post hoc and exploratory; BH controls the false discovery rate within each secondary family only, not across the many families in this table. Source: `figures/data/Fig3/fig3C_*.csv`, cross-checked against `artifacts/results_clean_rerun/direction3|direction4/partA_predictive/`.