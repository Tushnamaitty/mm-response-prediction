"""
Step 11 (XAI): Publication figure consolidation.

Does NO new computation - purely selects, copies, and documents
existing figures from Steps 3-9 into a compact, publication-oriented
set (per the original plan's Section 11 table), rather than leaving a
large scattered collection of per-model plots.

Fails loudly if any expected source figure is missing - never silently
skips or substitutes an alternative, since a manuscript figure with an
undocumented substitution is worse than a clear error here.

Outputs (under artifacts/results_clean_rerun/xai/publication_figures/):
- 01_binary_shap_summary.png
- 02_multiclass_shap_summary.png
- 03_feature_family_comparison.png
- 04_binary_vs_multiclass_family_D_vs_D.png
- 05_class_specific_progressive_disease.png
- 06_class_specific_stringent_complete_response.png
- 07_rna_pathways_binary.png
- 08_rna_pathways_multiclass.png
- 09_dependence_top_feature_binary.png
- 10_dependence_top_feature_multiclass_pd.png
- 11_local_case_correct_improvement.png
- 12_local_case_informative_error.png
- publication_figures_manifest.csv / .md
"""

import os, shutil
import pandas as pd

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
FIG_DIR = f"{OUT_DIR}/figures"
PUB_DIR = f"{OUT_DIR}/publication_figures"

# (publication_filename, source_path, source_step, question_answered, caveat)
FIGURE_MANIFEST = [
    ("01_binary_shap_summary.png",
     f"{FIG_DIR}/beeswarm_binary_c_lightgbm.png", "Step 3",
     "What drives improvement prediction? (best binary model, C-LightGBM)", ""),

    ("02_multiclass_shap_summary.png",
     f"{FIG_DIR}/global_importance_multiclass_a_xgboost.png", "Step 3",
     "What drives exact IMWG response prediction? (best multiclass model, A-XGBoost)", ""),

    ("03_feature_family_comparison.png",
     f"{FIG_DIR}/feature_family_comparison.png", "Step 4",
     "Clinical vs temporal vs treatment vs RNA contribution, across all 4 selected models", ""),

    ("04_binary_vs_multiclass_family_D_vs_D.png",
     f"{FIG_DIR}/family_comparison_D_vs_D.png", "Step 6",
     "Fair (identical feature space) comparison of family reliance between binary and multiclass tasks", ""),

    ("05_class_specific_progressive_disease.png",
     f"{FIG_DIR}/class_importance_multiclass_d_xgboost_progressive_disease.png", "Step 7",
     "What drives prediction of progressive disease specifically (Model D)",
     "PD's top driver, m_protein_change, is not part of the flagged m_protein_current/previous "
     "correlation pair (Step 10, Check 3) - see manuscript note on this distinction."),

    ("06_class_specific_stringent_complete_response.png",
     f"{FIG_DIR}/class_importance_multiclass_d_xgboost_stringent_complete_response.png", "Step 7",
     "What drives prediction of stringent complete response, as a contrast to progressive disease", ""),

    ("07_rna_pathways_binary.png",
     f"{FIG_DIR}/rna_pathways_binary_d_lightgbm.png", "Step 8",
     "Which Hallmark pathways rank highest in the binary RNA-integrated model",
     "Individual pathway rankings should be read with Step 10's correlation finding in mind: "
     "155/1225 pathway pairs exceed |r|=0.7, so a single pathway's rank may reflect a broader "
     "correlated biological program rather than a unique, isolated contribution."),

    ("08_rna_pathways_multiclass.png",
     f"{FIG_DIR}/rna_pathways_multiclass_d_xgboost.png", "Step 8",
     "Which Hallmark pathways rank highest in the multiclass RNA-integrated model",
     "Same correlation caveat as 07 applies."),

    ("09_dependence_top_feature_binary.png",
     f"{FIG_DIR}/dependence_binary_c_lightgbm_m_protein_current.png", "Step 5",
     "Direction of effect for the top continuous biomarker in the best binary model", ""),

    ("10_dependence_top_feature_multiclass_pd.png",
     f"{FIG_DIR}/dependence_multiclass_d_xgboost_m_protein_change.png", "Step 5",
     "Direction of effect for m_protein_change - the feature driving the PD-specific finding", ""),

    ("11_local_case_correct_improvement.png",
     f"{FIG_DIR}/local_correct_improvement.png", "Step 9",
     "Patient-level example: correctly predicted improvement", ""),

    ("12_local_case_informative_error.png",
     f"{FIG_DIR}/local_informative_error.png", "Step 9",
     "Patient-level example: a confidently wrong prediction, illustrating the vt_disease_response "
     "anchoring failure mode discussed in the Step 9 review", ""),
]


if __name__ == "__main__":
    os.makedirs(PUB_DIR, exist_ok=True)

    missing = [(pub_name, src) for pub_name, src, *_ in FIGURE_MANIFEST if not os.path.exists(src)]
    if missing:
        print("MISSING SOURCE FIGURES - cannot consolidate until these exist:")
        for pub_name, src in missing:
            print(f"  {pub_name} <- {src}")
        raise FileNotFoundError(
            f"{len(missing)} source figure(s) missing. Check the exact filenames Steps 5/7/9 "
            f"actually produced (e.g. via 'dir artifacts\\results_clean_rerun\\xai\\figures') "
            f"and tell me if any manifest path above needs correcting."
        )

    manifest_rows = []
    for pub_name, src, step, question, caveat in FIGURE_MANIFEST:
        dst = f"{PUB_DIR}/{pub_name}"
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
        manifest_rows.append({
            "publication_filename": pub_name, "source_file": src, "generating_step": step,
            "question_answered": question, "caveat": caveat,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(f"{PUB_DIR}/publication_figures_manifest.csv", index=False)

    with open(f"{PUB_DIR}/publication_figures_manifest.md", "w", encoding="utf-8") as f:
        f.write("# XAI publication figures manifest\n\n")
        for row in manifest_rows:
            f.write(f"## {row['publication_filename']}\n")
            f.write(f"- **Source:** `{row['source_file']}` ({row['generating_step']})\n")
            f.write(f"- **Question answered:** {row['question_answered']}\n")
            if row["caveat"]:
                f.write(f"- **Caveat:** {row['caveat']}\n")
            f.write("\n")

    print(f"\n{'='*70}\nSTEP 11 COMPLETE\n{'='*70}")
    print(f"{len(manifest_rows)} figures consolidated to: {PUB_DIR}/")
    print(f"Manifest: {PUB_DIR}/publication_figures_manifest.csv and .md")
    