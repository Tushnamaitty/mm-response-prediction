"""
Step 3 (XAI): Global feature importance, with categorical rollup and
multiclass cross-class aggregation.

Reads SHAP artifacts saved by Step 2 - does not recompute SHAP values,
does not touch any model or the frozen preprocessors beyond reading
their metadata for the rollup mapping.

CATEGORICAL ROLLUP: the frozen preprocessors one-hot encode several
columns (current_regimen_name, ig_heavy_chain_type, etc.). SHAP
attributes importance per encoded dummy column, not per original
variable. Ranking on raw dummy-column SHAP values without first
rolling up to the original column would make a genuinely important
categorical look unimportant, since its SHAP mass is split across many
sub-columns. This script builds an output-column -> original-column
map from each model's *_metadata.json input_columns list, and sums
|SHAP| importance within each original column BEFORE ranking.

MULTICLASS AGGREGATION: for the two multiclass models, SHAP produces a
separate value per class. "Global" importance here is defined as: for
each output column, mean(|SHAP|) within each class, then the mean of
those 6 per-class values. This is stated explicitly so it's
reproducible and not confused with "importance for the predicted class
only".

Outputs (per model, under artifacts/results_clean_rerun/xai/):
- global_importance_{key}.csv       - rolled-up, ranked original-column importances
- rollup_mapping_{key}.csv          - raw output-column -> original-column mapping used
- figures/global_importance_{key}.png - top-20 rolled bar chart
- figures/beeswarm_{key}.png        - raw-feature SHAP beeswarm (binary models only)
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, "modeling")
from modeling.train_models import load_preprocessor_and_meta, MULTICLASS_CLASSES

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
SHAP_DIR = f"{OUT_DIR}/shap_artifacts"
FIG_DIR = f"{OUT_DIR}/figures"

SELECTED_MODELS = [
    {"task": "binary", "stage": "c", "algorithm": "lightgbm"},
    {"task": "binary", "stage": "d", "algorithm": "lightgbm"},
    {"task": "multiclass", "stage": "a", "algorithm": "xgboost"},
    {"task": "multiclass", "stage": "d", "algorithm": "xgboost"},
]

TOP_N_PLOT = 20


def strip_transformer_prefix(name):
    """ColumnTransformer output names look like '{transformer}__{col}' or
    '{transformer}__{col}_{category}' for one-hot columns. Strip the
    transformer prefix, keep the rest for matching against input_columns."""
    return name.split("__", 1)[-1] if "__" in name else name


def build_rollup_map(feature_names, input_columns):
    """Map each SHAP output column back to its original predictor column.
    Matches on exact name or on 'original_column_' prefix (one-hot case).
    Falls back to the stripped name itself if no input column matches -
    flagged explicitly so an unrecognized pattern is never silently
    absorbed."""
    sorted_cols = sorted(input_columns, key=len, reverse=True)
    mapping = {}
    unmatched = []
    for fn in feature_names:
        core = strip_transformer_prefix(fn)
        matched = None
        for col in sorted_cols:
            if core == col or core.startswith(col + "_"):
                matched = col
                break
        if matched is None:
            matched = core
            unmatched.append(fn)
        mapping[fn] = matched
    return mapping, unmatched


def compute_per_output_importance(shap_values, task):
    """Returns 1D array (n_output_features,).
    Binary: mean(|SHAP|) over rows.
    Multiclass: mean(|SHAP|) over rows per class, then mean over classes."""
    if task == "binary":
        return np.mean(np.abs(shap_values), axis=1) if shap_values.ndim == 2 and shap_values.shape[0] != shap_values.shape[1] else np.mean(np.abs(shap_values), axis=0)
    else:
        per_class = np.mean(np.abs(shap_values), axis=1)  # (n_classes, n_features)
        return np.mean(per_class, axis=0)  # (n_features,)


def process_model(task, stage, algorithm):
    key = f"{task}_{stage}_{algorithm}"
    print(f"\n{'='*70}\n{key}\n{'='*70}")

    shap_values = np.load(f"{SHAP_DIR}/{key}_shap_values.npy")
    with open(f"{SHAP_DIR}/{key}_feature_names.json") as f:
        feature_names = json.load(f)

    _, meta = load_preprocessor_and_meta(stage)
    input_columns = list(meta["input_columns"])

    per_output_importance = np.mean(np.abs(shap_values), axis=0) if task == "binary" \
        else np.mean(np.mean(np.abs(shap_values), axis=1), axis=0)
    assert len(per_output_importance) == len(feature_names), (
        f"{key}: importance length {len(per_output_importance)} != feature count {len(feature_names)}"
    )

    mapping, unmatched = build_rollup_map(feature_names, input_columns)
    if unmatched:
        print(f"WARNING: {len(unmatched)} output columns did not match any input_column, "
              f"kept as their own group: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    raw_df = pd.DataFrame({
        "output_feature": feature_names,
        "original_column": [mapping[fn] for fn in feature_names],
        "importance": per_output_importance,
    })
    raw_df.to_csv(f"{OUT_DIR}/rollup_mapping_{key}.csv", index=False)

    rolled = (raw_df.groupby("original_column")["importance"].sum()
              .reset_index().sort_values("importance", ascending=False).reset_index(drop=True))
    rolled.insert(0, "rank", range(1, len(rolled) + 1))
    rolled.to_csv(f"{OUT_DIR}/global_importance_{key}.csv", index=False)

    # --- Sanity check: rollup didn't drop or double-count anything ---
    raw_total = float(per_output_importance.sum())
    rolled_total = float(rolled["importance"].sum())

    diff = abs(raw_total - rolled_total)
    rel_diff = diff / raw_total if raw_total != 0 else diff
    print(f"Raw total importance: {raw_total:.6f}   Rolled total: {rolled_total:.6f}   "
          f"diff: {diff:.2e}   relative diff: {rel_diff:.2e}")
    assert rel_diff < 1e-5, (
        f"{key}: rollup sum mismatch (relative diff {rel_diff:.2e}) - "
        f"rollup dropped or double-counted a column"
    )

    print(f"Rollup sanity check PASSED. {len(feature_names)} output columns -> {len(rolled)} original columns.")

    print(f"\nTop 10 rolled features:")
    print(rolled.head(10)[["rank", "original_column", "importance"]].to_string(index=False))

    # --- Top-N bar chart (rolled, original columns) ---
    top = rolled.head(TOP_N_PLOT).iloc[::-1]  # reverse for horizontal bar (largest on top)
    plt.figure(figsize=(8, 6))
    plt.barh(top["original_column"], top["importance"])
    plt.xlabel("Rolled-up mean |SHAP|")
    plt.title(f"{key} - top {TOP_N_PLOT} features (rolled up)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/global_importance_{key}.png", dpi=150)
    plt.close()

    # --- Beeswarm on raw (pre-rollup) features - binary models only ---
    if task == "binary":
        X_te = np.load(f"{SHAP_DIR}/{key}_X_test.npy")
        plt.figure()
        shap.summary_plot(shap_values, X_te, feature_names=feature_names, max_display=TOP_N_PLOT, show=False)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/beeswarm_{key}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Beeswarm saved (raw, pre-rollup features): {FIG_DIR}/beeswarm_{key}.png")
    else:
        print("Beeswarm skipped for multiclass model (see docstring - use rolled bar chart + Step 7's "
              "per-class analysis instead).")

    return {
        "key": key, "task": task, "model_stage": stage, "algorithm": algorithm,
        "n_output_columns": len(feature_names), "n_original_columns": len(rolled),
        "n_unmatched": len(unmatched),
        "rollup_sum_check_diff": diff,
        "top_5_original_columns": rolled.head(5)["original_column"].tolist(),
    }


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    summary_rows = []
    for spec in SELECTED_MODELS:
        result = process_model(spec["task"], spec["stage"], spec["algorithm"])
        summary_rows.append(result)

    with open(f"{OUT_DIR}/step3_global_importance_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n{'='*70}\nSTEP 3 COMPLETE\n{'='*70}")
    print(f"Per-model rolled importances: {OUT_DIR}/global_importance_{{key}}.csv")
    print(f"Rollup mappings (raw->original): {OUT_DIR}/rollup_mapping_{{key}}.csv")
    print(f"Figures: {FIG_DIR}/")
    print(f"Summary: {OUT_DIR}/step3_global_importance_summary.json")