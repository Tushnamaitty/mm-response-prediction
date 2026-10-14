"""
RNA-indicator ablation: locked clean-rerun Model D (WITH rna_available)
vs. a single new Model D variant with ONLY rna_available removed
(Hallmark pathway scores and days_since_rna_sample are kept).

FIXED-HYPERPARAMETER ABLATION BY DESIGN: this reuses the locked clean
Model D XGBoost's already-selected best_params for the no-indicator
variant, with NO retuning. This is intentional, not an oversight -
the goal is to isolate the presence/absence of a single column as the
ONLY modeling difference. If hyperparameters were re-tuned for the
no-indicator variant, any observed difference could reflect a
different hyperparameter choice rather than the indicator itself.

Does NOT retrain the locked Model D - loads
artifacts/models_clean_rerun/{task}_d_xgboost.joblib directly and cross-
checks its predictions against the locked
artifacts/results_clean_rerun/{task}_d_xgboost_predictions.csv before
using it, so this script cannot silently drift from the locked result.

Does NOT touch or overwrite the old, untrusted
model_d_without_rna_indicator_PRELIMINARY.json. Writes ONLY under
artifacts/results_clean_rerun/rna_ablation/.
"""

import json, os, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, "modeling")
from train_models import (
    load_master_data, load_preprocessor_and_meta, build_model,
    get_binary_xyz, get_multiclass_xyz, binary_metrics, multiclass_metrics,
    RNA_INDICATOR_COL, MULTICLASS_CLASSES, RANDOM_SEED,
)

CLEAN_MODELS_DIR = "artifacts/models_clean_rerun"
CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/rna_ablation"

N_BOOTSTRAP = 1000


def patient_bootstrap_indices(public_ids, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    uniq = np.unique(public_ids)
    patient_to_rows = {p: np.where(public_ids == p)[0] for p in uniq}
    resamples = []
    for _ in range(n):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        row_idx = np.concatenate([patient_to_rows[p] for p in sampled])
        resamples.append(row_idx)
    return resamples


def percentile_ci(scores):
    scores = np.array([s for s in scores if s is not None])
    if len(scores) == 0:
        return None, None
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def find_indicator_index(feature_names):
    matches = [i for i, name in enumerate(feature_names) if name.endswith(RNA_INDICATOR_COL)]
    assert len(matches) == 1, (
        f"Expected exactly one '{RNA_INDICATOR_COL}' output column, found {len(matches)}: "
        f"{[feature_names[i] for i in matches]}"
    )
    return matches[0]


def run_task(df, task):
    preprocessor, meta = load_preprocessor_and_meta("d")
    getter = get_binary_xyz if task == "binary" else get_multiclass_xyz

    X_tr, y_tr, g_tr, pid_tr, feat_names = getter(df, "d", preprocessor, meta, "train")
    X_te, y_te, g_te, pid_te, _ = getter(df, "d", preprocessor, meta, "test")

    ind_idx = find_indicator_index(feat_names)
    print(f"[{task}] rna_available located at output column index {ind_idx} "
          f"of {len(feat_names)} ('{feat_names[ind_idx]}')")

    # --- Load the LOCKED full-D model (no retraining) ---
    locked_model_path = f"{CLEAN_MODELS_DIR}/{task}_d_xgboost.joblib"
    assert os.path.exists(locked_model_path), f"Missing locked model: {locked_model_path}"
    full_d_model = joblib.load(locked_model_path)

    # --- Cross-check against the locked predictions CSV (no drift) ---
    locked_preds_path = f"{CLEAN_RESULTS_DIR}/{task}_d_xgboost_predictions.csv"
    assert os.path.exists(locked_preds_path), f"Missing locked predictions: {locked_preds_path}"
    locked_preds = pd.read_csv(locked_preds_path)
    assert list(locked_preds["pair_id"]) == list(pid_te), (
        f"[{task}] pair_id order mismatch between freshly-built X_te and locked predictions CSV - "
        f"refusing to proceed, feature/eligibility set may have drifted."
    )

    if task == "binary":
        full_d_proba = full_d_model.predict_proba(X_te)[:, 1]
        assert np.allclose(full_d_proba, locked_preds["proba"].to_numpy(), atol=1e-8), (
            f"[{task}] Recomputed locked-D probabilities do not match the saved locked predictions CSV."
        )
    else:
        full_d_pred = full_d_model.predict(X_te)  # integer class indices, same encoding as y_te
        full_d_pred_labels = [MULTICLASS_CLASSES[i] for i in full_d_pred]
        assert full_d_pred_labels == list(locked_preds["pred"]), (
            f"[{task}] Recomputed locked-D predictions do not match the saved locked predictions CSV."
        )
    print(f"[{task}] Locked D cross-check PASSED - recomputed predictions match saved locked results.")

    # --- Build D-without-indicator matrices by dropping ind_idx post-hoc ---
    X_tr_noind = np.delete(X_tr, ind_idx, axis=1)
    X_te_noind = np.delete(X_te, ind_idx, axis=1)
    assert X_tr_noind.shape[1] == X_tr.shape[1] - 1
    assert X_te_noind.shape[1] == X_te.shape[1] - 1

    # --- FIXED-HYPERPARAMETER ABLATION: reuse locked D's best_params, no re-tuning ---
    with open(f"{CLEAN_RESULTS_DIR}/{task}_d_xgboost_metrics.json") as f:
        locked_metrics = json.load(f)
    best_params = locked_metrics["best_params"]

    noind_model = build_model("xgboost", task, best_params)
    noind_model.fit(X_tr_noind, y_tr)

    if task == "binary":
        noind_proba = noind_model.predict_proba(X_te_noind)[:, 1]
        full_m = binary_metrics(y_te, full_d_proba, threshold=0.5)  # threshold-free metrics (auprc/auroc) unaffected
        noind_m = binary_metrics(y_te, noind_proba, threshold=0.5)
        preds_out = pd.DataFrame({"pair_id": pid_te, "public_id": g_te, "y_true": y_te,
                                   "proba": noind_proba})
        metric_fns = {"auprc": average_precision_score, "auroc": roc_auc_score}
        full_vals = {"auprc": full_m["auprc"], "auroc": full_m["auroc"]}
        noind_vals = {"auprc": noind_m["auprc"], "auroc": noind_m["auroc"]}
        val_arrays = {"full": full_d_proba, "noind": noind_proba}
        y_for_metric = y_te
    else:
        # FIX: noind predictions and full predictions are both integer class
        # indices in the SAME encoding as y_te - metrics are computed against
        # the ACTUAL y_te for both models. (Previously this incorrectly used
        # indices derived from the no-indicator model's own predictions as
        # y_true, which trivially inflated its apparent performance.)
        noind_pred = noind_model.predict(X_te_noind)
        full_m = multiclass_metrics(y_te, full_d_pred)
        noind_m = multiclass_metrics(y_te, noind_pred)
        preds_out = pd.DataFrame({"pair_id": pid_te, "public_id": g_te,
                                   "y_true": [MULTICLASS_CLASSES[i] for i in y_te],
                                   "pred": [MULTICLASS_CLASSES[i] for i in noind_pred]})
        metric_fns = {"macro_f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)}
        full_vals = {"macro_f1": full_m["macro_f1"]}
        noind_vals = {"macro_f1": noind_m["macro_f1"]}
        val_arrays = {"full": full_d_pred, "noind": noind_pred}
        y_for_metric = y_te

    # --- Paired patient-level bootstrap: full D vs D-without-indicator ---
    resamples = patient_bootstrap_indices(g_te)
    paired_rows = []
    for metric_name, metric_fn in metric_fns.items():
        observed_diff = float(metric_fn(y_for_metric, val_arrays["full"])
                               - metric_fn(y_for_metric, val_arrays["noind"]))
        boot_diffs = []
        for idx in resamples:
            if len(idx) < 2:
                continue
            yt = y_for_metric[idx]
            if task == "binary" and len(np.unique(yt)) < 2:
                continue
            d = metric_fn(yt, val_arrays["full"][idx]) - metric_fn(yt, val_arrays["noind"][idx])
            boot_diffs.append(d)
        lo, hi = percentile_ci(boot_diffs)
        paired_rows.append({
            "task": task, "metric": metric_name,
            "full_D_value": full_vals[metric_name], "D_without_indicator_value": noind_vals[metric_name],
            "observed_diff_full_minus_noind": observed_diff,
            "ci_lower": lo, "ci_upper": hi,
            "n_rows": int(len(y_te)), "n_patients": int(len(set(g_te))),
        })

    preds_out.to_csv(f"{OUT_DIR}/{task}_d_without_indicator_predictions.csv", index=False)
    with open(f"{OUT_DIR}/{task}_d_without_indicator_metrics.json", "w") as f:
        json.dump({
            "full_D": full_vals, "D_without_indicator": noind_vals,
            "best_params_reused_no_retuning": best_params,
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        }, f, indent=2)

    return paired_rows


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_master_data()

    all_paired_rows = []
    for task in ["binary", "multiclass"]:
        print(f"\n=== {task} ===")
        all_paired_rows.extend(run_task(df, task))

    pd.DataFrame(all_paired_rows).to_csv(f"{OUT_DIR}/rna_indicator_ablation_paired_differences.csv", index=False)
    print(f"\nDONE. Saved: {OUT_DIR}/rna_indicator_ablation_paired_differences.csv")
    print("Locked models/results were read-only. Old model_d_without_rna_indicator_PRELIMINARY.json untouched.")