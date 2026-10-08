# Supplementary Table S3. SHAP attribution shares for XGBoost information Layer D

Held-out test set (155 patients; 1,930 binary and 2,079 six-class visit pairs). Two different measures, not interchangeable: **net family magnitude** (Figure S3A) = mean over pairs of |sum of signed source SHAP within the family|; **total source-feature importance** = sum over the family's features of mean |source SHAP|. Total importance is larger for groups whose features push in opposite directions (notably RNA pathways). Six-class values are raw class scores averaged equally over classes (class average) or shown per class. Shares are of model attribution, not accuracy, variance explained or causation. Shares use largest-remainder rounding (each column sums to 100.0%, as in Figure S3A); CIs are from the exact shares. 95% patient-level bootstrap CIs (1,000 resamples).

## Net family magnitude (Figure S3A) — all test pairs
Binary 1,930 pairs / 155 patients; six-class 2,079 / 155.

| Component | Layer | Binary share (95% CI) | Six-class share (95% CI) |
|---|---|---|---|
| Current IMWG response | A | 58.1% (56.3–59.8) | 53.2% (52.0–54.4) |
| Other current clinical state | A | 10.4% (9.2–11.6) | 16.8% (16.0–17.8) |
| Longitudinal history | B | 18.0% (17.0–18.9) | 14.7% (14.2–15.2) |
| Treatment context | C | 8.7% (8.4–9.1) | 6.8% (6.5–7.0) |
| RNA pathway scores | D | 3.3% (2.9–3.9) | 7.3% (6.5–8.1) |
| RNA timing / availability | D | 1.5% (1.4–1.7) | 1.2% (1.1–1.3) |

## Net family magnitude (Figure S3A) — patients weighted equally
Binary 1,930 pairs / 155 patients; six-class 2,079 / 155.

| Component | Layer | Binary share (95% CI) | Six-class share (95% CI) |
|---|---|---|---|
| Current IMWG response | A | 47.5% (42.8–51.7) | 49.0% (47.1–50.6) |
| Other current clinical state | A | 20.9% (16.3–25.9) | 20.6% (19.0–22.4) |
| Longitudinal history | B | 18.7% (17.6–19.7) | 14.9% (14.6–15.4) |
| Treatment context | C | 8.0% (7.4–8.7) | 7.1% (6.8–7.4) |
| RNA pathway scores | D | 3.4% (2.9–3.9) | 7.2% (6.6–7.9) |
| RNA timing / availability | D | 1.5% (1.4–1.8) | 1.2% (1.1–1.3) |

## Net family magnitude (Figure S3A) — RNA-available pairs only
Binary 1,323 pairs / 107 patients; six-class 1,449 / 107.

| Component | Layer | Binary share (95% CI) | Six-class share (95% CI) |
|---|---|---|---|
| Current IMWG response | A | 57.9% (55.6–59.9) | 51.7% (50.4–53.0) |
| Other current clinical state | A | 10.3% (9.0–11.8) | 16.9% (15.8–18.0) |
| Longitudinal history | B | 17.6% (16.5–18.8) | 14.4% (13.9–15.1) |
| Treatment context | C | 8.6% (8.2–9.0) | 6.6% (6.3–6.9) |
| RNA pathway scores | D | 3.9% (3.3–4.6) | 8.9% (8.0–9.8) |
| RNA timing / availability | D | 1.7% (1.6–1.9) | 1.5% (1.4–1.6) |

## Total source-feature importance (not Figure S3A) — all test pairs
Binary 1,930 pairs / 155 patients; six-class 2,079 / 155.

| Component | Layer | Binary share (95% CI) | Six-class share (95% CI) |
|---|---|---|---|
| Current IMWG response | A | 41.9% (40.5–43.3) | 33.7% (32.8–34.6) |
| Other current clinical state | A | 18.2% (17.5–19.0) | 24.3% (23.7–24.9) |
| Longitudinal history | B | 22.9% (22.1–23.8) | 21.3% (20.9–21.8) |
| Treatment context | C | 9.2% (8.9–9.6) | 7.0% (6.8–7.3) |
| RNA pathway scores | D | 6.7% (5.9–7.5) | 12.9% (11.9–13.9) |
| RNA timing / availability | D | 1.1% (1.0–1.2) | 0.8% (0.7–0.8) |

## Total source-feature importance (not Figure S3A) — patients weighted equally
Binary 1,930 pairs / 155 patients; six-class 2,079 / 155.

| Component | Layer | Binary share (95% CI) | Six-class share (95% CI) |
|---|---|---|---|
| Current IMWG response | A | 35.1% (31.9–37.9) | 31.1% (30.0–32.2) |
| Other current clinical state | A | 25.0% (21.8–28.7) | 27.0% (25.8–28.3) |
| Longitudinal history | B | 23.0% (22.1–24.0) | 21.3% (20.7–21.9) |
| Treatment context | C | 9.0% (8.5–9.5) | 7.3% (7.1–7.6) |
| RNA pathway scores | D | 6.7% (6.0–7.4) | 12.5% (11.7–13.4) |
| RNA timing / availability | D | 1.2% (1.0–1.3) | 0.8% (0.7–0.8) |

## Total source-feature importance (not Figure S3A) — RNA-available pairs only
Binary 1,323 pairs / 107 patients; six-class 1,449 / 107.

| Component | Layer | Binary share (95% CI) | Six-class share (95% CI) |
|---|---|---|---|
| Current IMWG response | A | 41.3% (39.6–42.9) | 32.3% (31.4–33.2) |
| Other current clinical state | A | 18.0% (17.3–19.0) | 23.9% (23.2–24.7) |
| Longitudinal history | B | 22.4% (21.5–23.4) | 20.9% (20.4–21.3) |
| Treatment context | C | 9.1% (8.6–9.5) | 6.8% (6.6–7.0) |
| RNA pathway scores | D | 8.0% (7.1–9.1) | 15.2% (14.3–16.3) |
| RNA timing / availability | D | 1.2% (1.1–1.4) | 0.9% (0.8–1.0) |

## Six-class net family magnitude share by class (all test pairs)

| Component | PD | SD | PR | VGPR | CR | sCR |
|---|---|---|---|---|---|---|
| Current IMWG response | 14.9% | 61.1% | 66.6% | 64.1% | 48.8% | 55.7% |
| Other current clinical state | 19.1% | 10.3% | 8.3% | 14.3% | 31.5% | 14.0% |
| Longitudinal history | 38.8% | 7.5% | 13.1% | 9.5% | 9.9% | 15.8% |
| Treatment context | 14.5% | 11.4% | 5.2% | 3.3% | 3.5% | 5.9% |
| RNA pathway scores | 11.9% | 9.0% | 6.0% | 5.9% | 5.9% | 6.8% |
| RNA timing / availability | 0.8% | 0.7% | 0.8% | 2.9% | 0.4% | 1.8% |

Per-class CIs and the total-importance measure per class are in the CSV. Source: `figures/data/FigS_shap/figS3A_ext_component_shares*.csv` (cross-checked against `figS3A_family_shares.csv`).