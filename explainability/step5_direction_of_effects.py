"""
Step 5 (XAI): Direction of effects for the top rolled-up features per
model.

Binary models (C, D): SHAP values are already for the positive class
("improved") - a positive contribution means "pushes toward
improvement."

Multiclass models (A, D): direction is ambiguous with 6 classes and no
single improvement axis. This uses, for each row, the SHAP value for
that row's ACTUAL OBSERVED class (not the predicted class, not one
fixed reference class) - i.e. "did this feature push toward the
response that actually happened?" This is one defensible framing, not
the only one; Step 7 does the full per-class breakdown.

TYPE DETECTION: a column is treated as categorical if EITHER:
(a) fewer than NUMERIC_COERCION_THRESHOLD of its non-null values
    coerce to a number (e.g. string regimen names) - catches the case
    where pandas represents string data as 'object' or as StringDtype
    ('string', 'string[pyarrow]') depending on version/backend, which
    a naive dtype check can silently misclassify as continuous; OR
(b) it has at most CATEGORICAL_MAX_UNIQUE unique values, even if it
    coerces cleanly to numbers (e.g. an integer-coded
    n_prior_transplants with values 0/1/2, or a binary 0/1 flag) -
    low-cardinality integers behave as ordinal/categorical, not
    continuous, and a Spearman correlation or qcut-binned trend line
    on 2-3 distinct values is misleading.
Both conditions are checked; either one triggers categorical handling.

HIGH-CARDINALITY CATEGORICALS: if a categorical has more than
CATEGORICAL_TOP_K_DISPLAY unique values, the PLOT shows only the top-K
most frequent categories plus an 'other' bucket (for readability), but
the SAVED SUMMARY JSON always includes medians for every category,
regardless of plot truncation - nothing is dropped from the record.

For each of the top N_TOP_FEATURES per model (by Step 3's rolled
ranking), reconstructs each row's RAW (pre-preprocessing) feature
value from the frozen master data, paired with that row's summed SHAP
contribution across all output columns that rolled up to this original
column (from Step 3's rollup_mapping_{key}.csv).

Does NOT retrain or modify any model. Writes ONLY under
artifacts/results_clean_rerun/xai/.
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "modeling")
from modeling.train_models import (
    load_master_data, load_preprocessor_and_meta, get_multiclass_xyz,
)

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

N_TOP_FEATURES = 8
NUMERIC_COERCION_THRESHOLD = 0.95   # fraction of non-null values that must coerce to number
CATEGORICAL_MAX_UNIQUE = 15         # columns with this many or fewer unique values are treated as categorical
CATEGORICAL_TOP_K_DISPLAY = 12      # max categories shown individually in the plot


def per_row_effect(shap_values, task, y_te=None):
    """Return (n_rows, n_output_features) signed SHAP matrix.
    Binary: already (n_rows, n_features), return as-is.
    Multiclass: select each row's SHAP toward its ACTUAL observed class."""
    if task == "binary":
        return shap_values
    n_rows = shap_values.shape[1]
    assert y_te is not None and len(y_te) == n_rows
    return shap_values[y_te, np.arange(n_rows), :]


def summed_effect_for_original_column(per_row_shap, rollup_df, original_column, feature_names):
    output_cols = rollup_df.loc[rollup_df["original_column"] == original_column, "output_feature"].tolist()
    idxs = [feature_names.index(c) for c in output_cols]
    return per_row_shap[:, idxs].sum(axis=1)


def is_categorical_column(raw_v):
    """A column is categorical if it fails numeric coercion OR has low
    cardinality, regardless of stored dtype. See module docstring."""
    coerced = pd.to_numeric(pd.Series(raw_v), errors="coerce")
    frac_numeric = coerced.notna().mean()
    fails_numeric_coercion = frac_numeric < NUMERIC_COERCION_THRESHOLD
    n_unique = pd.Series(raw_v).nunique()
    is_low_cardinality = n_unique <= CATEGORICAL_MAX_UNIQUE
    return (fails_numeric_coercion or is_low_cardinality), coerced


def plot_feature_direction(key, original_column, raw_values, effect, out_path):
    valid = ~pd.isna(raw_values)
    raw_v = raw_values[valid]
    eff_v = effect[valid]
    if len(raw_v) < 5:
        print(f"  SKIP {original_column}: fewer than 5 valid non-null rows")
        return None

    is_categorical, coerced_numeric = is_categorical_column(raw_v)
    n_unique = pd.Series(raw_v).nunique()

    plt.figure(figsize=(6, 4))

    if is_categorical:
        df_plot = pd.DataFrame({"category": raw_v, "effect": eff_v})
        # Full medians for EVERY category go in the summary, regardless of display truncation
        full_medians = df_plot.groupby("category")["effect"].median().sort_values()

        if n_unique > CATEGORICAL_TOP_K_DISPLAY:
            top_cats = df_plot["category"].value_counts().head(CATEGORICAL_TOP_K_DISPLAY).index
            df_plot["category_display"] = df_plot["category"].where(
                df_plot["category"].isin(top_cats), other="other")
            title_suffix = f" (top {CATEGORICAL_TOP_K_DISPLAY} of {n_unique}, rest grouped as 'other')"
        else:
            df_plot["category_display"] = df_plot["category"]
            title_suffix = ""

        order = df_plot.groupby("category_display")["effect"].median().sort_values().index
        df_plot["category_display"] = pd.Categorical(df_plot["category_display"], categories=order, ordered=True)
        df_plot.boxplot(column="effect", by="category_display", ax=plt.gca(), rot=45)
        plt.suptitle("")
        plt.title(f"{original_column} (categorical){title_suffix}")
        plt.ylabel("SHAP contribution")
        plt.xlabel("")
        summary = {"type": "categorical", "n_categories": int(n_unique),
                   "displayed_top_k": int(min(n_unique, CATEGORICAL_TOP_K_DISPLAY)),
                   "category_medians": full_medians.to_dict()}
    else:
        raw_v_f = coerced_numeric[coerced_numeric.notna()]
        eff_v_f = eff_v[coerced_numeric.notna().to_numpy()]
        plt.scatter(raw_v_f, eff_v_f, s=8, alpha=0.35)
        n_bins = min(20, max(5, len(raw_v_f) // 50))
        bins = pd.qcut(raw_v_f, q=n_bins, duplicates="drop")
        trend = pd.DataFrame({"x": raw_v_f, "y": eff_v_f, "bin": bins}).groupby("bin", observed=True).agg(
            x_mid=("x", "mean"), y_mean=("y", "mean"))
        plt.plot(trend["x_mid"], trend["y_mean"], color="black", linewidth=1.5, label="binned mean")
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.xlabel(original_column)
        plt.ylabel("SHAP contribution")
        plt.title(f"{original_column} (continuous)")
        plt.legend()
        spearman = pd.Series(raw_v_f).corr(pd.Series(eff_v_f), method="spearman")
        summary = {"type": "continuous", "spearman_corr": float(spearman) if not pd.isna(spearman) else None}

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return summary


def process_model(task, stage, algorithm):
    key = f"{task}_{stage}_{algorithm}"
    print(f"\n{'='*70}\n{key}\n{'='*70}")

    shap_values = np.load(f"{SHAP_DIR}/{key}_shap_values.npy")
    with open(f"{SHAP_DIR}/{key}_feature_names.json") as f:
        feature_names = json.load(f)
    row_ids = pd.read_csv(f"{SHAP_DIR}/{key}_row_ids.csv")
    rollup_df = pd.read_csv(f"{OUT_DIR}/rollup_mapping_{key}.csv")
    top_features = pd.read_csv(f"{OUT_DIR}/global_importance_{key}.csv").head(N_TOP_FEATURES)

    y_te = None
    if task == "multiclass":
        df = load_master_data()
        preprocessor, meta = load_preprocessor_and_meta(stage)
        _, y_te_arr, _, pid_te, _ = get_multiclass_xyz(df, stage, preprocessor, meta, "test")
        assert list(pid_te) == list(row_ids["pair_id"]), (
            f"{key}: pair_id order mismatch between freshly-computed y_te and saved row_ids.csv"
        )
        y_te = y_te_arr
        print(f"Using each row's ACTUAL OBSERVED class for direction (not predicted class).")

    per_row_shap = per_row_effect(shap_values, task, y_te)

    master = load_master_data().set_index("pair_id")
    raw_lookup = master.reindex(row_ids["pair_id"])

    summaries = []
    for _, row in top_features.iterrows():
        original_column = row["original_column"]
        if original_column not in raw_lookup.columns:
            print(f"  SKIP {original_column}: not a direct column in master data "
                  f"(likely the engineered rna_available indicator)")
            continue
        effect = summed_effect_for_original_column(per_row_shap, rollup_df, original_column, feature_names)
        raw_values = raw_lookup[original_column].to_numpy()
        out_path = f"{FIG_DIR}/dependence_{key}_{original_column}.png"
        summary = plot_feature_direction(key, original_column, raw_values, effect, out_path)
        if summary is not None:
            summary.update({"key": key, "original_column": original_column, "rank": int(row["rank"])})
            summaries.append(summary)
            if summary["type"] == "continuous":
                desc = f"spearman={summary['spearman_corr']:.3f}" if summary["spearman_corr"] is not None else "spearman=NA"
            else:
                desc = f"{summary['n_categories']} categories"
            print(f"  rank {row['rank']:>2}: {original_column} ({summary['type']}, {desc}) -> {out_path}")

    return summaries


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    all_summaries = []
    for spec in SELECTED_MODELS:
        all_summaries.extend(process_model(spec["task"], spec["stage"], spec["algorithm"]))

    with open(f"{OUT_DIR}/step5_direction_of_effects_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    print(f"\n{'='*70}\nSTEP 5 COMPLETE\n{'='*70}")
    print(f"Figures: {FIG_DIR}/dependence_*.png")
    print(f"Summary: {OUT_DIR}/step5_direction_of_effects_summary.json")