"""
Step 8 (XAI): RNA-focused interpretability analysis.

Reads ONLY existing Step 3/4/7 outputs - no new SHAP computation.
Isolates the RNA family's individual members (50 Hallmark pathway
scores, days_since_rna_sample, rna_available) from Step 3's full
rankings for the two D models, and checks Step 7's per-class rankings
for any RNA feature appearing in a class-specific top-15 even though
none made the whole-dataset top 10 (Step 3).

Directly answers the plan's four RNA questions:
1. Do pathway scores contribute meaningfully? -> RNA family total (from
   Step 4, cross-checked here) and individual pathway ranks.
2. Which pathways rank highest? -> ranked pathway-only table.
3. Is timing more influential than pathway content? -> explicit rank
   comparison between days_since_rna_sample and every individual
   pathway.
4. Does availability itself matter? -> rna_available's rank/importance
   (already established as exactly 0 in both D models from Step 4's
   discussion - reconfirmed here).

Outputs (under artifacts/results_clean_rerun/xai/):
- rna_focused_shap_{key}.csv          - RNA family members only, ranked
- rna_per_class_appearances.csv       - any RNA feature in a class's top 15 (Step 7)
- figures/rna_pathways_{key}.png      - top-15 pathway bar chart
- step8_rna_focused_summary.json
"""

import json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
FIG_DIR = f"{OUT_DIR}/figures"
FEATURE_SETS_PATH = "data_pipeline/model_feature_sets.json"

RNA_INDICATOR_COL = "rna_available"
STALENESS_COL = "days_since_rna_sample"

D_MODELS = [
    {"task": "binary", "stage": "d", "algorithm": "lightgbm"},
    {"task": "multiclass", "stage": "d", "algorithm": "xgboost"},
]
MULTICLASS_CLASSES_FOR_STEP7 = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]
PER_CLASS_TOP_N = 15
TOP_N_PATHWAY_PLOT = 15


def get_rna_columns(fs):
    model_c = set(fs["model_c"])
    model_d = set(fs["model_d"])
    return (model_d - model_c) | {RNA_INDICATOR_COL}


def process_model(task, stage, algorithm, rna_cols, family_total_from_step4):
    key = f"{task}_{stage}_{algorithm}"
    print(f"\n{'='*70}\n{key}\n{'='*70}")

    importance_path = f"{OUT_DIR}/global_importance_{key}.csv"
    assert os.path.exists(importance_path), f"Missing {importance_path} - run Step 3 first."
    full_df = pd.read_csv(importance_path)

    rna_df = full_df[full_df["original_column"].isin(rna_cols)].copy()
    rna_df = rna_df.sort_values("importance", ascending=False).reset_index(drop=True)
    rna_df.insert(0, "rna_rank", range(1, len(rna_df) + 1))
    rna_df["whole_dataset_rank"] = rna_df["rank"]
    rna_df.drop(columns=["rank"], inplace=True)
    rna_df.to_csv(f"{OUT_DIR}/rna_focused_shap_{key}.csv", index=False)

    # --- Sanity check against Step 4's already-reported RNA family total ---
    step8_rna_sum = rna_df["importance"].sum()
    diff = abs(step8_rna_sum - family_total_from_step4)
    rel_diff = diff / family_total_from_step4 if family_total_from_step4 != 0 else diff
    print(f"RNA family sum (this step): {step8_rna_sum:.6f}   "
          f"(Step 4 reported: {family_total_from_step4:.6f}, relative diff: {rel_diff:.2e})")
    assert rel_diff < 1e-5, (
        f"{key}: RNA sum mismatch vs Step 4 (relative diff {rel_diff:.2e}) - "
        f"family membership may have diverged between steps."
    )
    print("Cross-check vs Step 4 PASSED.")

    # --- Q1/Q2: pathway-only ranking ---
    pathway_df = rna_df[~rna_df["original_column"].isin([RNA_INDICATOR_COL, STALENESS_COL])]
    print(f"\nTop 10 pathways by importance:")
    print(pathway_df.head(10)[["rna_rank", "whole_dataset_rank", "original_column", "importance"]].to_string(index=False))

    # --- Q3: timing vs pathway content ---
    staleness_row = rna_df[rna_df["original_column"] == STALENESS_COL]
    staleness_rank = int(staleness_row["whole_dataset_rank"].iloc[0]) if len(staleness_row) else None
    n_pathways_beaten = int((pathway_df["whole_dataset_rank"] > staleness_rank).sum()) if staleness_rank else None
    print(f"\n{STALENESS_COL}: whole-dataset rank {staleness_rank} "
          f"(individually outranks {n_pathways_beaten} of {len(pathway_df)} pathways)")

    # --- Q4: availability indicator ---
    indicator_row = rna_df[rna_df["original_column"] == RNA_INDICATOR_COL]
    indicator_importance = float(indicator_row["importance"].iloc[0]) if len(indicator_row) else None
    print(f"{RNA_INDICATOR_COL}: importance = {indicator_importance} "
          f"(whole-dataset rank {int(indicator_row['whole_dataset_rank'].iloc[0]) if len(indicator_row) else 'N/A'})")

    # --- Plot: top pathways only ---
    top = pathway_df.head(TOP_N_PATHWAY_PLOT).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["original_column"], top["importance"])
    plt.xlabel("Mean |SHAP| (rolled)")
    plt.title(f"{key} - top {TOP_N_PATHWAY_PLOT} Hallmark pathways")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rna_pathways_{key}.png", dpi=150)
    plt.close()

    return {
        "key": key, "rna_family_sum": step8_rna_sum, "step4_cross_check_relative_diff": rel_diff,
        "top_5_pathways": pathway_df.head(5)[["original_column", "importance", "whole_dataset_rank"]].to_dict("records"),
        "days_since_rna_sample_whole_dataset_rank": staleness_rank,
        "days_since_rna_sample_outranks_n_pathways": n_pathways_beaten,
        "n_pathways_total": len(pathway_df),
        "rna_available_importance": indicator_importance,
    }


def check_per_class_appearances(rna_cols):
    """New check: does any RNA feature crack a class-SPECIFIC top-15,
    even though none made the whole-dataset top 10 (Step 3)?"""
    key = "multiclass_d_xgboost"
    path = f"{OUT_DIR}/class_specific_importance_{key}.csv"
    assert os.path.exists(path), f"Missing {path} - run Step 7 first."
    df = pd.read_csv(path)

    appearances = []
    for class_name in MULTICLASS_CLASSES_FOR_STEP7:
        class_df = df[df["class"] == class_name].sort_values("rank").head(PER_CLASS_TOP_N)
        rna_hits = class_df[class_df["original_column"].isin(rna_cols)]
        for _, row in rna_hits.iterrows():
            appearances.append({
                "class": class_name, "original_column": row["original_column"],
                "rank_within_class": int(row["rank"]), "importance": float(row["importance"]),
            })

    print(f"\n{'='*70}\nPer-class RNA appearances (top-{PER_CLASS_TOP_N} per class, multiclass_d_xgboost)\n{'='*70}")
    if appearances:
        appearances_df = pd.DataFrame(appearances)
        print(appearances_df.to_string(index=False))
    else:
        appearances_df = pd.DataFrame(columns=["class", "original_column", "rank_within_class", "importance"])
        print(f"No RNA feature appears in any class's top-{PER_CLASS_TOP_N}.")

    appearances_df.to_csv(f"{OUT_DIR}/rna_per_class_appearances.csv", index=False)
    return appearances


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(FEATURE_SETS_PATH) as f:
        fs = json.load(f)
    rna_cols = get_rna_columns(fs)
    print(f"RNA family: {len(rna_cols)} columns (50 pathways + rna_available"
          f"{' + ' + STALENESS_COL if STALENESS_COL in rna_cols else ''})")

    step4_family_totals = {}
    for spec in D_MODELS:
        key = f"{spec['task']}_{spec['stage']}_{spec['algorithm']}"
        fam_df = pd.read_csv(f"{OUT_DIR}/feature_family_importance_{key}.csv")
        step4_family_totals[key] = float(fam_df.loc[fam_df["family"] == "rna", "importance"].iloc[0])

    all_results = []
    for spec in D_MODELS:
        key = f"{spec['task']}_{spec['stage']}_{spec['algorithm']}"
        result = process_model(spec["task"], spec["stage"], spec["algorithm"], rna_cols, step4_family_totals[key])
        all_results.append(result)

    per_class_appearances = check_per_class_appearances(rna_cols)

    summary = {"per_model": all_results, "per_class_rna_appearances": per_class_appearances}
    with open(f"{OUT_DIR}/step8_rna_focused_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}\nSTEP 8 COMPLETE\n{'='*70}")
    print(f"RNA-only rankings: {OUT_DIR}/rna_focused_shap_{{key}}.csv")
    print(f"Per-class RNA appearances: {OUT_DIR}/rna_per_class_appearances.csv")
    print(f"Figures: {FIG_DIR}/rna_pathways_*.png")
    print(f"Summary: {OUT_DIR}/step8_rna_focused_summary.json")