"""
Step 6 (XAI): Binary vs. multiclass interpretability comparison.

Reads ONLY Step 3's global_importance_{key}.csv and Step 4's
feature_family_importance_{key}.csv files - no new SHAP or importance
computation. Two deliberately separate comparisons:

1. D vs D (binary_d_lightgbm vs multiclass_d_xgboost): apples-to-apples,
   since both models see the identical four-family feature space
   (clinical/temporal/treatment/RNA). This is the fair comparison of
   what each TASK relies on given equal access to everything.

2. Best-performing models (binary_c_lightgbm vs multiclass_a_xgboost):
   NOT apples-to-apples - Model A only ever sees clinical features by
   construction, so it will trivially show 100% clinical family share.
   Kept because these are the actual headline/reported models, but
   labeled explicitly so it is never confused with comparison 1.

Also documents the vt_disease_response ceiling-effect finding from
Step 5 (identical structural pattern in every one of the 4 models:
patients already in deep response, e.g. VGPR/CR/sCR, have less room to
register as "improved" on a bounded response scale, so the model
correctly - not incorrectly - pushes them toward "not improved"/lower
classes). Recorded once here since it is a binary-vs-multiclass-wide
structural fact, not specific to one model.

Outputs (under artifacts/results_clean_rerun/xai/):
- binary_vs_multiclass_family_comparison_D_vs_D.csv
- binary_vs_multiclass_family_comparison_best_vs_best.csv
- top_feature_overlap.csv
- figures/family_comparison_D_vs_D.png
- figures/family_comparison_best_vs_best.png
- step6_summary.json (includes the ceiling-effect note as structured text)
"""

import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
FIG_DIR = f"{OUT_DIR}/figures"

D_VS_D = {"binary": "binary_d_lightgbm", "multiclass": "multiclass_d_xgboost"}
BEST_VS_BEST = {"binary": "binary_c_lightgbm", "multiclass": "multiclass_a_xgboost"}

TOP_N_OVERLAP = 10
FAMILY_ORDER = ["clinical", "temporal", "treatment", "rna"]

CEILING_EFFECT_NOTE = (
    "vt_disease_response is the rank-1 feature in all 4 selected models. In every "
    "model, its category medians are monotonic with the IMWG response hierarchy in "
    "the SAME direction: patients already in a deeper response at Vt (e.g. VGPR, CR, "
    "sCR) show LOWER predicted-improvement SHAP contributions than patients with "
    "stable or progressive disease at Vt. This reflects a real structural ceiling "
    "effect in the target definition - 'improved' measures upward movement on a "
    "bounded response scale, and a patient starting near the top of that scale has "
    "mechanically less room to register as 'improved' regardless of true clinical "
    "trajectory. This is consistent with, and the same underlying phenomenon as, the "
    "project's existing decision to exclude Vt=sCR pairs from the binary task for an "
    "identical ceiling reason (sCR cannot improve further by definition) - this "
    "pattern is that same effect showing up one or two rungs lower on the ladder "
    "(VGPR/CR), not severe enough on its own to warrant exclusion, but present."
)


def family_comparison(pair_dict, label):
    rows = []
    for task, key in pair_dict.items():
        path = f"{OUT_DIR}/feature_family_importance_{key}.csv"
        assert os.path.exists(path), f"Missing {path} - run Step 4 first for {key}."
        df = pd.read_csv(path)
        df["task"] = task
        df["key"] = key
        rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(f"{OUT_DIR}/binary_vs_multiclass_family_comparison_{label}.csv", index=False)

    pivot = combined.pivot(index="family", columns="task", values="pct_of_total").reindex(FAMILY_ORDER)
    pivot.plot(kind="bar", figsize=(7, 5))
    plt.ylabel("% of total SHAP importance")
    plt.title(f"Feature family: binary vs multiclass ({label.replace('_', ' ')})")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/family_comparison_{label}.png", dpi=150)
    plt.close()

    print(f"\n--- Family comparison ({label}) ---")
    print(pivot.round(2))
    return combined


def top_feature_overlap(pair_dict, label, n=TOP_N_OVERLAP):
    top_sets = {}
    top_lists = {}
    for task, key in pair_dict.items():
        path = f"{OUT_DIR}/global_importance_{key}.csv"
        assert os.path.exists(path), f"Missing {path} - run Step 3 first for {key}."
        df = pd.read_csv(path).head(n)
        top_sets[task] = set(df["original_column"])
        top_lists[task] = df["original_column"].tolist()

    overlap = top_sets["binary"] & top_sets["multiclass"]
    union = top_sets["binary"] | top_sets["multiclass"]
    jaccard = len(overlap) / len(union) if union else 0.0

    print(f"\n--- Top-{n} feature overlap ({label}) ---")
    print(f"Binary top-{n}:     {top_lists['binary']}")
    print(f"Multiclass top-{n}: {top_lists['multiclass']}")
    print(f"Overlap ({len(overlap)}/{n}): {sorted(overlap)}")
    print(f"Jaccard index: {jaccard:.3f}")

    return {
        "label": label, "n": n,
        "binary_top": top_lists["binary"], "multiclass_top": top_lists["multiclass"],
        "overlap": sorted(overlap), "n_overlap": len(overlap), "jaccard_index": jaccard,
    }


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("COMPARISON 1: D vs D (apples-to-apples, identical feature space)")
    print("=" * 70)
    family_d_vs_d = family_comparison(D_VS_D, "D_vs_D")
    overlap_d_vs_d = top_feature_overlap(D_VS_D, "D_vs_D")

    print("\n" + "=" * 70)
    print("COMPARISON 2: best-performing models (NOT apples-to-apples - "
          "Model A is clinical-only by construction)")
    print("=" * 70)
    family_best_vs_best = family_comparison(BEST_VS_BEST, "best_vs_best")
    overlap_best_vs_best = top_feature_overlap(BEST_VS_BEST, "best_vs_best")

    all_overlaps = pd.DataFrame([overlap_d_vs_d, overlap_best_vs_best])
    all_overlaps.to_csv(f"{OUT_DIR}/top_feature_overlap.csv", index=False)

    summary = {
        "d_vs_d_models": D_VS_D, "best_vs_best_models": BEST_VS_BEST,
        "top_feature_overlap": [overlap_d_vs_d, overlap_best_vs_best],
        "ceiling_effect_note": CEILING_EFFECT_NOTE,
    }
    with open(f"{OUT_DIR}/step6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nSTEP 6 COMPLETE\n{'='*70}")
    print(f"Family comparisons: {OUT_DIR}/binary_vs_multiclass_family_comparison_*.csv")
    print(f"Feature overlap: {OUT_DIR}/top_feature_overlap.csv")
    print(f"Figures: {FIG_DIR}/family_comparison_*.png")
    print(f"Summary (incl. ceiling-effect note): {OUT_DIR}/step6_summary.json")