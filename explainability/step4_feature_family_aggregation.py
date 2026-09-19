"""
Step 4 (XAI): Feature-family aggregation (Clinical / Temporal /
Treatment / RNA), using the existing model_feature_sets.json buckets
as the source of truth - not a redefinition of families.

Family definitions (mutually exclusive):
- Clinical  = Model A's full column set
- Temporal  = Model B's columns minus Model A's (B's additions)
- Treatment = Model C's columns minus Model B's (C's additions)
- RNA       = Model D's columns minus Model C's (D's additions),
              plus the engineered 'rna_available' indicator (added at
              preprocessing time, not present in model_feature_sets.json)

Reads Step 3's rolled-up global_importance_{key}.csv files - does not
recompute SHAP or re-roll up categoricals. Every rolled-up original
column for a given model MUST map to exactly one family; an unmapped
column raises immediately rather than being silently dropped from the
family totals.

Outputs (under artifacts/results_clean_rerun/xai/):
- feature_family_importance_{key}.csv  - family totals + % of model total
- feature_family_comparison.csv        - all 4 models side by side
- figures/feature_family_{key}.png     - per-model bar chart
- figures/feature_family_comparison.png - grouped bar chart, all 4 models
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
FIG_DIR = f"{OUT_DIR}/figures"
FEATURE_SETS_PATH = "data_pipeline/model_feature_sets.json"

RNA_INDICATOR_COL = "rna_available"

SELECTED_MODELS = [
    {"task": "binary", "stage": "c", "algorithm": "lightgbm"},
    {"task": "binary", "stage": "d", "algorithm": "lightgbm"},
    {"task": "multiclass", "stage": "a", "algorithm": "xgboost"},
    {"task": "multiclass", "stage": "d", "algorithm": "xgboost"},
]

FAMILY_ORDER = ["clinical", "temporal", "treatment", "rna"]


def build_family_buckets(fs):
    model_a = set(fs["model_a"])
    model_b = set(fs["model_b"])
    model_c = set(fs["model_c"])
    model_d = set(fs["model_d"])

    clinical = model_a
    temporal = model_b - model_a
    treatment = model_c - model_b
    rna = (model_d - model_c) | {RNA_INDICATOR_COL}

    # --- Sanity check: buckets are mutually exclusive by construction ---
    all_pairs = [("clinical", clinical), ("temporal", temporal),
                 ("treatment", treatment), ("rna", rna)]
    for i in range(len(all_pairs)):
        for j in range(i + 1, len(all_pairs)):
            overlap = all_pairs[i][1] & all_pairs[j][1]
            assert not overlap, (
                f"Family buckets '{all_pairs[i][0]}' and '{all_pairs[j][0]}' overlap: {overlap} - "
                f"buckets must be mutually exclusive."
            )

    print(f"Family bucket sizes: clinical={len(clinical)}, temporal={len(temporal)}, "
          f"treatment={len(treatment)}, rna={len(rna)}")
    return {"clinical": clinical, "temporal": temporal, "treatment": treatment, "rna": rna}


def family_map_for_stage(stage, buckets):
    """Which families apply to a given model stage, and the combined
    column->family map for just those families."""
    stage_families = {
        "a": ["clinical"],
        "b": ["clinical", "temporal"],
        "c": ["clinical", "temporal", "treatment"],
        "d": ["clinical", "temporal", "treatment", "rna"],
    }[stage]
    col_to_family = {}
    for fam in stage_families:
        for col in buckets[fam]:
            col_to_family[col] = fam
    return col_to_family


def process_model(task, stage, algorithm, buckets):
    key = f"{task}_{stage}_{algorithm}"
    print(f"\n{'='*70}\n{key}\n{'='*70}")

    importance_path = f"{OUT_DIR}/global_importance_{key}.csv"
    assert os.path.exists(importance_path), (
        f"Missing {importance_path} - run Step 3 first for this model."
    )
    imp_df = pd.read_csv(importance_path)

    col_to_family = family_map_for_stage(stage, buckets)

    unmapped = [c for c in imp_df["original_column"] if c not in col_to_family]
    if unmapped:
        raise AssertionError(
            f"{key}: {len(unmapped)} original column(s) from Step 3 do not map to any family "
            f"bucket for stage '{stage}': {unmapped[:10]}{'...' if len(unmapped) > 10 else ''} - "
            f"model_feature_sets.json may not match this model's actual column set. "
            f"Refusing to silently drop these from the family totals."
        )

    imp_df = imp_df.copy()
    imp_df["family"] = imp_df["original_column"].map(col_to_family)

    total = imp_df["importance"].sum()
    family_totals = (imp_df.groupby("family")["importance"].sum()
                      .reindex([f for f in FAMILY_ORDER if f in col_to_family.values()])
                      .reset_index())
    family_totals["pct_of_total"] = 100 * family_totals["importance"] / total

    # --- Sanity check: family totals sum to 100% of model total ---
    pct_sum = family_totals["pct_of_total"].sum()
    print(f"Family totals sum to {pct_sum:.4f}% of model total (should be ~100%)")
    assert abs(pct_sum - 100.0) < 1e-3, f"{key}: family percentages sum to {pct_sum:.4f}%, not ~100%"

    family_totals.to_csv(f"{OUT_DIR}/feature_family_importance_{key}.csv", index=False)
    print(family_totals.to_string(index=False))

    plt.figure(figsize=(5, 4))
    plt.bar(family_totals["family"], family_totals["pct_of_total"])
    plt.ylabel("% of total SHAP importance")
    plt.title(f"{key} - feature family contribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_family_{key}.png", dpi=150)
    plt.close()

    family_totals["key"] = key
    return family_totals


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(FEATURE_SETS_PATH) as f:
        fs = json.load(f)
    buckets = build_family_buckets(fs)

    all_family_rows = []
    for spec in SELECTED_MODELS:
        result = process_model(spec["task"], spec["stage"], spec["algorithm"], buckets)
        all_family_rows.append(result)

    combined = pd.concat(all_family_rows, ignore_index=True)
    combined.to_csv(f"{OUT_DIR}/feature_family_comparison.csv", index=False)

    # --- Grouped bar chart across all 4 models ---
    pivot = combined.pivot(index="family", columns="key", values="pct_of_total").reindex(FAMILY_ORDER)
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("% of total SHAP importance")
    plt.title("Feature family contribution across models")
    plt.xticks(rotation=0)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_family_comparison.png", dpi=150)
    plt.close()

    print(f"\n{'='*70}\nSTEP 4 COMPLETE\n{'='*70}")
    print(f"Per-model family tables: {OUT_DIR}/feature_family_importance_{{key}}.csv")
    print(f"Combined comparison: {OUT_DIR}/feature_family_comparison.csv")
    print(f"Figures: {FIG_DIR}/")
    print("\nCombined table:")
    print(combined.pivot(index="family", columns="key", values="pct_of_total").reindex(FAMILY_ORDER).round(2))