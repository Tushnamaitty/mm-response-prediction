"""
Step 7 (XAI): Class-specific multiclass SHAP analysis for the two
multiclass models, with particular attention to Progressive Disease.

Different question from Step 5: Step 5 asked, per row, whether a
feature pushed toward whatever outcome ACTUALLY happened. Step 7 asks,
per CLASS, which features drive the model TOWARD PREDICTING that class
in general - using each class's own SHAP array (shape n_rows x
n_features) across ALL test rows, independent of what actually
happened for any given row.

Reuses Step 2's saved shap_values (n_classes, n_rows, n_features) and
Step 3's rollup mapping (output_feature -> original_column) - no new
SHAP computation, no rebuilt mapping.

All 6 classes get a full ranked importance CSV. Plots and the
direction-of-effect breakdown are limited to Progressive Disease (the
class flagged in the project's plan as hardest to predict and most
clinically important to understand) and Stringent Complete Response
(the opposite clinical extreme, as a contrast).

Outputs (under artifacts/results_clean_rerun/xai/):
- class_specific_importance_{key}.csv       - all 6 classes, ranked, rolled up
- figures/class_importance_{key}_{class}.png - PD and sCR only
- figures/class_dependence_{key}_{class}_{feature}.png - top-5 PD/sCR features
- step7_summary.json
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "modeling")
from modeling.train_models import load_master_data, MULTICLASS_CLASSES

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
SHAP_DIR = f"{OUT_DIR}/shap_artifacts"
FIG_DIR = f"{OUT_DIR}/figures"

MULTICLASS_MODELS = [
    {"task": "multiclass", "stage": "a", "algorithm": "xgboost"},
    {"task": "multiclass", "stage": "d", "algorithm": "xgboost"},
]

FOCUS_CLASSES = ["progressive_disease", "stringent_complete_response"]
TOP_N_PLOT = 15
TOP_N_DIRECTION = 5
NUMERIC_COERCION_THRESHOLD = 0.95
CATEGORICAL_MAX_UNIQUE = 15


def is_categorical_column(raw_v):
    coerced = pd.to_numeric(pd.Series(raw_v), errors="coerce")
    frac_numeric = coerced.notna().mean()
    fails_numeric_coercion = frac_numeric < NUMERIC_COERCION_THRESHOLD
    n_unique = pd.Series(raw_v).nunique()
    is_low_cardinality = n_unique <= CATEGORICAL_MAX_UNIQUE
    return (fails_numeric_coercion or is_low_cardinality), coerced


def direction_summary(original_column, raw_values, effect):
    valid = ~pd.isna(raw_values)
    raw_v, eff_v = raw_values[valid], effect[valid]
    if len(raw_v) < 5:
        return None
    is_cat, coerced = is_categorical_column(raw_v)
    if is_cat:
        medians = pd.DataFrame({"category": raw_v, "effect": eff_v}).groupby("category")["effect"].median()
        return {"type": "categorical", "n_categories": int(pd.Series(raw_v).nunique()),
                "category_medians": medians.sort_values().to_dict()}
    else:
        raw_f = coerced[coerced.notna()]
        eff_f = eff_v[coerced.notna().to_numpy()]
        sp = pd.Series(raw_f).corr(pd.Series(eff_f), method="spearman")
        return {"type": "continuous", "spearman_corr": float(sp) if not pd.isna(sp) else None}


def process_model(task, stage, algorithm):
    key = f"{task}_{stage}_{algorithm}"
    print(f"\n{'='*70}\n{key}\n{'='*70}")

    shap_values = np.load(f"{SHAP_DIR}/{key}_shap_values.npy")  # (n_classes, n_rows, n_features)
    with open(f"{SHAP_DIR}/{key}_feature_names.json") as f:
        feature_names = json.load(f)
    row_ids = pd.read_csv(f"{SHAP_DIR}/{key}_row_ids.csv")
    rollup_df = pd.read_csv(f"{OUT_DIR}/rollup_mapping_{key}.csv")
    fn_to_original = dict(zip(rollup_df["output_feature"], rollup_df["original_column"]))

    assert shap_values.shape[0] == len(MULTICLASS_CLASSES)
    assert shap_values.shape[2] == len(feature_names)

    master = load_master_data().set_index("pair_id")
    raw_lookup = master.reindex(row_ids["pair_id"])

    all_class_rows = []
    class_summaries = {}

    for c_idx, class_name in enumerate(MULTICLASS_CLASSES):
        class_shap = shap_values[c_idx]  # (n_rows, n_features)
        per_output_importance = np.mean(np.abs(class_shap), axis=0)

        df = pd.DataFrame({
            "output_feature": feature_names,
            "original_column": [fn_to_original[fn] for fn in feature_names],
            "importance": per_output_importance,
        })
        rolled = (df.groupby("original_column")["importance"].sum()
                  .reset_index().sort_values("importance", ascending=False).reset_index(drop=True))
        rolled.insert(0, "rank", range(1, len(rolled) + 1))
        rolled.insert(0, "class", class_name)
        all_class_rows.append(rolled)

        print(f"  {class_name}: top 5 -> "
              f"{rolled.head(5)['original_column'].tolist()}")

        if class_name in FOCUS_CLASSES:
            top = rolled.head(TOP_N_PLOT).iloc[::-1]
            plt.figure(figsize=(8, 6))
            plt.barh(top["original_column"], top["importance"])
            plt.xlabel("Mean |SHAP| (this class's array)")
            plt.title(f"{key} - {class_name} - top {TOP_N_PLOT}")
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/class_importance_{key}_{class_name}.png", dpi=150)
            plt.close()

            direction_rows = []
            for _, row in rolled.head(TOP_N_DIRECTION).iterrows():
                original_column = row["original_column"]
                if original_column not in raw_lookup.columns:
                    continue
                output_cols = rollup_df.loc[rollup_df["original_column"] == original_column, "output_feature"].tolist()
                idxs = [feature_names.index(c) for c in output_cols]
                effect = class_shap[:, idxs].sum(axis=1)
                raw_values = raw_lookup[original_column].to_numpy()
                summary = direction_summary(original_column, raw_values, effect)
                if summary is not None:
                    summary.update({"original_column": original_column, "rank": int(row["rank"])})
                    direction_rows.append(summary)
            class_summaries[class_name] = direction_rows

    combined = pd.concat(all_class_rows, ignore_index=True)
    combined.to_csv(f"{OUT_DIR}/class_specific_importance_{key}.csv", index=False)

    return {"key": key, "focus_class_direction": class_summaries}


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    all_results = []
    for spec in MULTICLASS_MODELS:
        result = process_model(spec["task"], spec["stage"], spec["algorithm"])
        all_results.append(result)

    with open(f"{OUT_DIR}/step7_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}\nSTEP 7 COMPLETE\n{'='*70}")
    print(f"Per-class importance (all 6 classes): {OUT_DIR}/class_specific_importance_{{key}}.csv")
    print(f"Focus-class plots (PD, sCR): {FIG_DIR}/class_importance_*.png")
    print(f"Summary (incl. PD/sCR direction): {OUT_DIR}/step7_summary.json")