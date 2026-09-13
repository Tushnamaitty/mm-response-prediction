"""
Reruns ONLY the 6 RNA-freshness models (3 cutoffs x 2 tasks, XGBoost) to
save row-level test predictions, which the original freshness analysis
never saved. Same logic/seed as before - this is additive only, not a
methodology change. Does not touch train_models.py, the original
freshness JSON, or any main A/B/C/D results.
"""

import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "modeling")
from train_models import (
    load_master_data, load_preprocessor_and_meta, build_model,
    select_best_params, select_threshold, get_binary_xyz, get_multiclass_xyz,
    binary_metrics, multiclass_metrics, STALENESS_COL, MULTICLASS_CLASSES,
)

RESULTS_DIR = "artifacts/results"
PRED_DIR = "artifacts/results/rna_freshness_predictions"
FRESHNESS_CUTOFFS_DAYS = [365, 545, 730]


def run_freshness_cutoff(df, task, cutoff, algorithm="xgboost"):
    preprocessor, meta = load_preprocessor_and_meta("d")
    has_rna = df[STALENESS_COL].notna()
    fresh_mask = has_rna & (df[STALENESS_COL] <= cutoff)
    work_df = df.copy()
    rna_cols = meta["rna_columns_imputed"]
    stale_rows = has_rna & ~fresh_mask
    work_df.loc[stale_rows, rna_cols] = np.nan
    work_df.loc[stale_rows, STALENESS_COL] = np.nan

    getter = get_binary_xyz if task == "binary" else get_multiclass_xyz
    X_tr, y_tr, g_tr, pid_tr, _ = getter(work_df, "d", preprocessor, meta, "train")
    X_va, y_va, g_va, pid_va, _ = getter(work_df, "d", preprocessor, meta, "val")
    X_te, y_te, g_te, pid_te, _ = getter(work_df, "d", preprocessor, meta, "test")

    X_trv = np.vstack([X_tr, X_va]); y_trv = np.concatenate([y_tr, y_va]); g_trv = np.concatenate([g_tr, g_va])
    best_params, _ = select_best_params(X_trv, y_trv, g_trv, algorithm, task)
    model = build_model(algorithm, task, best_params)
    model.fit(X_tr, y_tr)

    if task == "binary":
        val_proba = model.predict_proba(X_va)[:, 1]
        threshold = select_threshold(y_va, val_proba)
        test_proba = model.predict_proba(X_te)[:, 1]
        metrics = binary_metrics(y_te, test_proba, threshold)
        preds_df = pd.DataFrame({"pair_id": pid_te, "public_id": g_te, "y_true": y_te,
                                  "proba": test_proba, "cutoff": cutoff, "task": task})
    else:
        test_pred = model.predict(X_te)
        metrics = multiclass_metrics(y_te, test_pred)
        preds_df = pd.DataFrame({"pair_id": pid_te, "public_id": g_te,
                                  "y_true": [MULTICLASS_CLASSES[i] for i in y_te],
                                  "pred": [MULTICLASS_CLASSES[i] for i in test_pred],
                                  "cutoff": cutoff, "task": task})

    return metrics, preds_df


if __name__ == "__main__":
    os.makedirs(PRED_DIR, exist_ok=True)
    df = load_master_data()

    freshness_results = {"binary": {"cutoff_results_PRELIMINARY": {}},
                          "multiclass": {"cutoff_results_PRELIMINARY": {}}}

    for task in ["binary", "multiclass"]:
        for cutoff in FRESHNESS_CUTOFFS_DAYS:
            print(f"{task} | cutoff_{cutoff}d")
            metrics, preds_df = run_freshness_cutoff(df, task, cutoff)
            preds_df.to_csv(f"{PRED_DIR}/{task}_cutoff_{cutoff}d_predictions.csv", index=False)
            freshness_results[task]["cutoff_results_PRELIMINARY"][f"cutoff_{cutoff}d"] = {"metrics": metrics}

    # Rerun summary saved to a NEW file - original freshness JSON is untouched
    with open(f"{RESULTS_DIR}/rna_freshness_predictions_rerun_PRELIMINARY.json", "w") as f:
        json.dump(freshness_results, f, indent=2)

    print("DONE. Predictions saved to artifacts/results/rna_freshness_predictions/")
    print("Rerun summary saved to artifacts/results/rna_freshness_predictions_rerun_PRELIMINARY.json")
    print("Original rna_freshness_sensitivity_PRELIMINARY.json was NOT touched.")