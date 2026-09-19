"""
Step 1 (XAI): Canonical model reconfirmation + reproducibility gate.

Two checks, in order:
1. Reconfirm best-performing binary/multiclass model from
   artifacts/results_clean_rerun/summary_PRELIMINARY.csv - not assumed
   from any prior document.
2. For each of the 4 models selected for XAI, reload the model +
   frozen preprocessor + test data independently and confirm
   recomputed predictions match the already-saved locked predictions
   CSV. No SHAP computation proceeds until all 4 pass.

Does NOT retrain, retune, or modify any model, preprocessor, split, or
prediction file. Reads only from artifacts/models_clean_rerun/ and
artifacts/results_clean_rerun/. Writes ONLY to
artifacts/results_clean_rerun/xai/xai_model_verification.json.
"""

import json, os, sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "modeling")
from modeling.train_models import (
    load_master_data, load_preprocessor_and_meta,
    get_binary_xyz, get_multiclass_xyz, MULTICLASS_CLASSES,
)

CLEAN_MODELS_DIR = "artifacts/models_clean_rerun"
CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"

# The 4 models selected for XAI, per the plan discussed - reconfirmed,
# not re-derived, in step 1a below.
SELECTED_MODELS = [
    {"task": "binary", "stage": "c", "algorithm": "lightgbm", "role": "best-performing binary model"},
    {"task": "binary", "stage": "d", "algorithm": "lightgbm", "role": "RNA-integrated comparison"},
    {"task": "multiclass", "stage": "a", "algorithm": "xgboost", "role": "best-performing multiclass model"},
    {"task": "multiclass", "stage": "d", "algorithm": "xgboost", "role": "RNA-integrated comparison"},
]


def reconfirm_ranking(summary_df):
    """Step 1a: print the actual top model per task from the clean-rerun
    summary, so model selection is verified, not assumed."""
    print("=== Reconfirming best-performing model per task (clean rerun) ===")
    rankings = {}
    for task in ["binary", "multiclass"]:
        task_df = summary_df[summary_df["task"] == task].copy()
        task_df = task_df.sort_values("test_main_metric_PRELIMINARY", ascending=False)
        print(f"\n{task} (ranked by test_main_metric_PRELIMINARY):")
        print(task_df[["model_stage", "algorithm", "test_main_metric_PRELIMINARY"]].to_string(index=False))
        top = task_df.iloc[0]
        rankings[task] = {"model_stage": top["model_stage"], "algorithm": top["algorithm"],
                           "test_main_metric_PRELIMINARY": float(top["test_main_metric_PRELIMINARY"])}
    return rankings


def verify_one_model(task, stage, algorithm):
    """Step 1b: reload model + preprocessor + test data independently,
    recompute predictions, compare to the locked predictions CSV."""
    key = f"{task}_{stage}_{algorithm}"
    print(f"\n--- Verifying {key} ---")

    model_path = f"{CLEAN_MODELS_DIR}/{key}.joblib"
    preds_path = f"{CLEAN_RESULTS_DIR}/{key}_predictions.csv"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing clean-rerun model: {model_path}")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Missing clean-rerun predictions: {preds_path}")

    model = joblib.load(model_path)
    locked_preds = pd.read_csv(preds_path)

    df = load_master_data()
    preprocessor, meta = load_preprocessor_and_meta(stage)
    getter = get_binary_xyz if task == "binary" else get_multiclass_xyz
    X_te, y_te, g_te, pid_te, feat_names = getter(df, stage, preprocessor, meta, "test")

    pair_id_match = list(locked_preds["pair_id"]) == list(pid_te)
    if not pair_id_match:
        raise AssertionError(
            f"{key}: pair_id order mismatch between freshly-built test set and locked "
            f"predictions CSV - feature/eligibility set may have drifted since the clean rerun."
        )

    if task == "binary":
        recomputed_proba = model.predict_proba(X_te)[:, 1]
        matches = np.allclose(recomputed_proba, locked_preds["proba"].to_numpy(), atol=1e-8)
        max_abs_diff = float(np.max(np.abs(recomputed_proba - locked_preds["proba"].to_numpy())))
    else:
        recomputed_pred = model.predict(X_te)
        recomputed_labels = [MULTICLASS_CLASSES[i] for i in recomputed_pred]
        matches = recomputed_labels == list(locked_preds["pred"])
        max_abs_diff = None  # not applicable to label comparison

    status = "PASS" if matches else "FAIL"
    print(f"{key}: {status}" + (f" (max abs diff: {max_abs_diff:.2e})" if max_abs_diff is not None else ""))

    return {
        "key": key, "task": task, "model_stage": stage, "algorithm": algorithm,
        "pair_id_order_match": pair_id_match,
        "predictions_match": bool(matches),
        "max_abs_diff": max_abs_diff,
        "n_test_rows": int(len(y_te)),
        "n_test_patients": int(len(set(g_te))),
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_path = f"{CLEAN_RESULTS_DIR}/summary_PRELIMINARY.csv"
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing clean-rerun summary: {summary_path}")
    summary_df = pd.read_csv(summary_path)

    rankings = reconfirm_ranking(summary_df)

    expected_best = {
    "binary": {"model_stage": "c", "algorithm": "lightgbm"},
    "multiclass": {"model_stage": "a", "algorithm": "xgboost"},
    }

    for task, expected in expected_best.items():
     actual = rankings[task]

     if (
        str(actual["model_stage"]).lower() != expected["model_stage"]
        or str(actual["algorithm"]).lower() != expected["algorithm"]
     ):
        raise AssertionError(
            f"{task}: clean-rerun winner has changed. "
            f"Expected {expected['model_stage']}-{expected['algorithm']}, "
            f"but clean summary gives "
            f"{actual['model_stage']}-{actual['algorithm']}. "
            "STOP and review XAI model selection before proceeding."
        )

    print("\nClean-rerun winners confirmed:")
    print("  Binary: C-LightGBM")
    print("  Multiclass: A-XGBoost")

    print("\n=== Reproducibility gate: verifying the 4 selected XAI models ===")
    verification_results = []
    for spec in SELECTED_MODELS:
        result = verify_one_model(spec["task"], spec["stage"], spec["algorithm"])
        result["role"] = spec["role"]
        verification_results.append(result)

    all_passed = all(r["predictions_match"] and r["pair_id_order_match"] for r in verification_results)

    output = {
        "reconfirmed_best_per_task": rankings,
        "selected_models_verification": verification_results,
        "all_checks_passed": all_passed,
    }
    with open(f"{OUT_DIR}/xai_model_verification.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL CHECKS PASSED. Safe to proceed to Step 2 (SHAP computation).")
    else:
        failed = [r["key"] for r in verification_results if not (r["predictions_match"] and r["pair_id_order_match"])]
        print(f"CHECKS FAILED for: {failed}")
        print("DO NOT proceed to Step 2 until this is resolved.")
    print(f"Saved: {OUT_DIR}/xai_model_verification.json")