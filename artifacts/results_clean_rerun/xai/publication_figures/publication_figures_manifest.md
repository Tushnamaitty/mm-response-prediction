# XAI publication figures manifest

## 01_binary_shap_summary.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/beeswarm_binary_c_lightgbm.png` (Step 3)
- **Question answered:** What drives improvement prediction? (best binary model, C-LightGBM)

## 02_multiclass_shap_summary.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/global_importance_multiclass_a_xgboost.png` (Step 3)
- **Question answered:** What drives exact IMWG response prediction? (best multiclass model, A-XGBoost)

## 03_feature_family_comparison.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/feature_family_comparison.png` (Step 4)
- **Question answered:** Clinical vs temporal vs treatment vs RNA contribution, across all 4 selected models

## 04_binary_vs_multiclass_family_D_vs_D.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/family_comparison_D_vs_D.png` (Step 6)
- **Question answered:** Fair (identical feature space) comparison of family reliance between binary and multiclass tasks

## 05_class_specific_progressive_disease.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/class_importance_multiclass_d_xgboost_progressive_disease.png` (Step 7)
- **Question answered:** What drives prediction of progressive disease specifically (Model D)
- **Caveat:** PD's top driver, m_protein_change, is not part of the flagged m_protein_current/previous correlation pair (Step 10, Check 3) - see manuscript note on this distinction.

## 06_class_specific_stringent_complete_response.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/class_importance_multiclass_d_xgboost_stringent_complete_response.png` (Step 7)
- **Question answered:** What drives prediction of stringent complete response, as a contrast to progressive disease

## 07_rna_pathways_binary.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/rna_pathways_binary_d_lightgbm.png` (Step 8)
- **Question answered:** Which Hallmark pathways rank highest in the binary RNA-integrated model
- **Caveat:** Individual pathway rankings should be read with Step 10's correlation finding in mind: 155/1225 pathway pairs exceed |r|=0.7, so a single pathway's rank may reflect a broader correlated biological program rather than a unique, isolated contribution.

## 08_rna_pathways_multiclass.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/rna_pathways_multiclass_d_xgboost.png` (Step 8)
- **Question answered:** Which Hallmark pathways rank highest in the multiclass RNA-integrated model
- **Caveat:** Same correlation caveat as 07 applies.

## 09_dependence_top_feature_binary.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/dependence_binary_c_lightgbm_m_protein_current.png` (Step 5)
- **Question answered:** Direction of effect for the top continuous biomarker in the best binary model

## 10_dependence_top_feature_multiclass_pd.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/dependence_multiclass_d_xgboost_m_protein_change.png` (Step 5)
- **Question answered:** Direction of effect for m_protein_change - the feature driving the PD-specific finding

## 11_local_case_correct_improvement.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/local_correct_improvement.png` (Step 9)
- **Question answered:** Patient-level example: correctly predicted improvement

## 12_local_case_informative_error.png
- **Source:** `artifacts/results_clean_rerun/xai/figures/local_informative_error.png` (Step 9)
- **Question answered:** Patient-level example: a confidently wrong prediction, illustrating the vt_disease_response anchoring failure mode discussed in the Step 9 review

