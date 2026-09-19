"""
Step 9 (XAI): Local, patient-level SHAP explanations for 5
representative test-set cases.

Case selection is automatic and criteria-based:
1. Correctly predicted improvement       - binary_d_lightgbm, highest-confidence true positive
2. Correctly predicted non-improvement   - binary_d_lightgbm, highest-confidence true negative
3. Correctly predicted progressive disease - multiclass_d_xgboost, highest-confidence correct PD
4. Correctly predicted stringent CR      - multiclass_d_xgboost, highest-confidence correct sCR
   (the deep-response class tracked throughout Steps 7-8)
5. An informative error                  - multiclass_d_xgboost, largest margin gap between the
   true class and the (wrong) predicted class - i.e. the most confidently wrong row.

Reuses Step 2's saved SHAP arrays and Step 3's rollup mapping - no new
SHAP computation. Waterfall plots use shap.plots.waterfall on the RAW
(pre-rollup) output columns, since that is the library's native,
already-validated visualization; the printed top-8 table is rolled up
to original columns for readability.

Outputs (under artifacts/results_clean_rerun/xai/):
- figures/local_{case_name}.png
- step9_local_cases_summary.json
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, "modeling")
from modeling.train_models import load_master_data, MULTICLASS_CLASSES

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
SHAP_DIR = f"{OUT_DIR}/shap_artifacts"
FIG_DIR = f"{OUT_DIR}/figures"

BINARY_KEY = "binary_d_lightgbm"
MULTICLASS_KEY = "multiclass_d_xgboost"
TOP_N_TABLE = 8


def load_model_artifacts(key):
    shap_values = np.load(f"{SHAP_DIR}/{key}_shap_values.npy")
    X_te = np.load(f"{SHAP_DIR}/{key}_X_test.npy")
    with open(f"{SHAP_DIR}/{key}_feature_names.json") as f:
        feature_names = json.load(f)
    with open(f"{SHAP_DIR}/{key}_expected_value.json") as f:
        expected_value = json.load(f)["expected_value"]
    row_ids = pd.read_csv(f"{SHAP_DIR}/{key}_row_ids.csv")
    rollup_df = pd.read_csv(f"{OUT_DIR}/rollup_mapping_{key}.csv")
    preds_df = pd.read_csv(f"{CLEAN_RESULTS_DIR}/{key}_predictions.csv")
    return shap_values, X_te, feature_names, expected_value, row_ids, rollup_df, preds_df


def rolled_top_table(shap_row, feature_names, rollup_df, X_row, n=TOP_N_TABLE):
    df = pd.DataFrame({
        "output_feature": feature_names,
        "original_column": [rollup_df.set_index("output_feature").loc[f, "original_column"] for f in feature_names],
        "shap_value": shap_row,
        "raw_output_value": X_row,
    })
    rolled = df.groupby("original_column").agg(
        shap_value=("shap_value", "sum"),
        raw_output_value=("raw_output_value", "first"),  # representative, may be dummy-encoded
    ).reset_index()
    rolled["abs_shap"] = rolled["shap_value"].abs()
    return rolled.sort_values("abs_shap", ascending=False).head(n).drop(columns="abs_shap")


def make_waterfall(shap_row, expected_value, feature_names, X_row, title, out_path):
    explanation = shap.Explanation(
        values=shap_row, base_values=expected_value,
        data=X_row, feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=TOP_N_TABLE, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def select_binary_cases(shap_values, X_te, feature_names, expected_value, row_ids, rollup_df, preds_df):
    cases = {}
    correct_pos = preds_df[(preds_df["y_true"] == 1) & (preds_df["pred"] == 1)]
    correct_neg = preds_df[(preds_df["y_true"] == 0) & (preds_df["pred"] == 0)]

    tp_idx = correct_pos.loc[correct_pos["proba"].idxmax()].name
    tn_idx = correct_neg.loc[correct_neg["proba"].idxmin()].name

    cases["correct_improvement"] = {"row_idx": int(tp_idx), "label": "Correctly predicted: improvement"}
    cases["correct_non_improvement"] = {"row_idx": int(tn_idx), "label": "Correctly predicted: non-improvement"}
    return cases


def select_multiclass_cases(shap_values, X_te, feature_names, expected_value, row_ids, rollup_df, preds_df):
    cases = {}
    class_to_idx = {c: i for i, c in enumerate(MULTICLASS_CLASSES)}
    correct = preds_df[preds_df["y_true"] == preds_df["pred"]]

    for target_class, case_name, label in [
        ("progressive_disease", "correct_pd", "Correctly predicted: progressive disease"),
        ("stringent_complete_response", "correct_scr", "Correctly predicted: stringent complete response"),
    ]:
        subset = correct[correct["pred"] == target_class]
        c_idx = class_to_idx[target_class]
        margins = shap_values[c_idx, subset.index, :].sum(axis=1) + expected_value[c_idx]
        best_row = subset.index[np.argmax(margins)]
        cases[case_name] = {"row_idx": int(best_row), "label": label, "explain_class": target_class}

    wrong = preds_df[preds_df["y_true"] != preds_df["pred"]]
    gaps = []
    for idx in wrong.index:
        true_c, pred_c = class_to_idx[wrong.loc[idx, "y_true"]], class_to_idx[wrong.loc[idx, "pred"]]
        margin_pred = shap_values[pred_c, idx, :].sum() + expected_value[pred_c]
        margin_true = shap_values[true_c, idx, :].sum() + expected_value[true_c]
        gaps.append((idx, margin_pred - margin_true))
    worst_idx, worst_gap = max(gaps, key=lambda t: t[1])
    cases["informative_error"] = {
        "row_idx": int(worst_idx),
        "label": f"Informative error: true={wrong.loc[worst_idx, 'y_true']}, predicted={wrong.loc[worst_idx, 'pred']} "
                 f"(margin gap {worst_gap:.3f})",
        "explain_class": wrong.loc[worst_idx, "pred"],  # explain the model's (wrong) choice
    }
    return cases


def process_binary_cases():
    key = BINARY_KEY
    shap_values, X_te, feature_names, expected_value, row_ids, rollup_df, preds_df = load_model_artifacts(key)
    cases = select_binary_cases(shap_values, X_te, feature_names, expected_value, row_ids, rollup_df, preds_df)

    results = []
    for case_name, info in cases.items():
        idx = info["row_idx"]
        pair_id = row_ids.iloc[idx]["pair_id"]
        print(f"\n{case_name} ({key}): pair_id={pair_id}, {info['label']}")
        table = rolled_top_table(shap_values[idx], feature_names, rollup_df, X_te[idx])
        print(table.to_string(index=False))

        out_path = f"{FIG_DIR}/local_{case_name}.png"
        make_waterfall(shap_values[idx], expected_value, feature_names, X_te[idx],
                        f"{key} | {info['label']} | pair_id={pair_id}", out_path)

        results.append({"case_name": case_name, "key": key, "pair_id": str(pair_id),
                         "label": info["label"], "top_features": table.to_dict("records")})
    return results


def process_multiclass_cases():
    key = MULTICLASS_KEY
    shap_values, X_te, feature_names, expected_value, row_ids, rollup_df, preds_df = load_model_artifacts(key)
    cases = select_multiclass_cases(shap_values, X_te, feature_names, expected_value, row_ids, rollup_df, preds_df)

    results = []
    for case_name, info in cases.items():
        idx = info["row_idx"]
        pair_id = row_ids.iloc[idx]["pair_id"]
        c_idx = MULTICLASS_CLASSES.index(info["explain_class"])
        print(f"\n{case_name} ({key}): pair_id={pair_id}, {info['label']}, "
              f"explaining class '{info['explain_class']}'")
        row_shap = shap_values[c_idx, idx, :]
        table = rolled_top_table(row_shap, feature_names, rollup_df, X_te[idx])
        print(table.to_string(index=False))

        out_path = f"{FIG_DIR}/local_{case_name}.png"
        make_waterfall(row_shap, expected_value[c_idx], feature_names, X_te[idx],
                        f"{key} | {info['label']} | pair_id={pair_id}", out_path)

        results.append({"case_name": case_name, "key": key, "pair_id": str(pair_id),
                         "label": info["label"], "explained_class": info["explain_class"],
                         "top_features": table.to_dict("records")})
    return results


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("BINARY CASES")
    print("=" * 70)
    binary_results = process_binary_cases()

    print("\n" + "=" * 70)
    print("MULTICLASS CASES")
    print("=" * 70)
    multiclass_results = process_multiclass_cases()

    with open(f"{OUT_DIR}/step9_local_cases_summary.json", "w") as f:
        json.dump({"binary_cases": binary_results, "multiclass_cases": multiclass_results}, f, indent=2, default=str)

    print(f"\n{'='*70}\nSTEP 9 COMPLETE\n{'='*70}")
    print(f"Figures: {FIG_DIR}/local_*.png")
    print(f"Summary: {OUT_DIR}/step9_local_cases_summary.json")