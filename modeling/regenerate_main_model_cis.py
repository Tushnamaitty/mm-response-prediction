"""
Standalone regeneration of main_model_cis.csv only.

Fixes the same point-estimate issue already corrected in the freshness
and RNA-subset scripts: point_estimate is now the metric computed once
on the full observed test set. Bootstrap resamples are used ONLY for
ci_lower/ci_upper.

No retraining, no changes to data/models/predictions. Does not touch
rna_subset_paired_differences.csv, rna_freshness_cis.csv, or
rna_freshness_paired_differences.csv - this script never references
those files or the code that produces them.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

RESULTS_DIR = "artifacts/results"
OUT_DIR = "artifacts/results/uncertainty"

N_BOOTSTRAP = 1000
SEED = 42
MODEL_STAGES = ["a", "b", "c", "d"]
ALGORITHMS = ["logistic", "lightgbm", "xgboost"]
MULTICLASS_CLASSES = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]


def patient_bootstrap_indices(public_ids, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    uniq = np.unique(public_ids)
    patient_to_rows = {p: np.where(public_ids == p)[0] for p in uniq}
    resamples = []
    for _ in range(n):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        row_idx = np.concatenate([patient_to_rows[p] for p in sampled])
        resamples.append(row_idx)
    return resamples


def ci_only(scores):
    scores = np.array([s for s in scores if s is not None])
    if len(scores) == 0:
        return None, None
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def load_preds(task, stage, algo):
    path = f"{RESULTS_DIR}/{task}_{stage}_{algo}_predictions.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main_rows = []

    for task in ["binary", "multiclass"]:
        for stage in MODEL_STAGES:
            for algo in ALGORITHMS:
                df = load_preds(task, stage, algo)
                if df is None:
                    print(f"MISSING predictions file: {task}_{stage}_{algo} - skipped")
                    continue

                pid = df["public_id"].to_numpy()
                resamples = patient_bootstrap_indices(pid)

                if task == "binary":
                    y = df["y_true"].to_numpy()
                    proba = df["proba"].to_numpy()

                    auprc_point = float(average_precision_score(y, proba))
                    auprc_boot = [average_precision_score(y[idx], proba[idx]) for idx in resamples
                                  if len(idx) >= 2 and len(np.unique(y[idx])) >= 2]
                    lo, hi = ci_only(auprc_boot)
                    main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                       "metric": "auprc", "point_estimate": auprc_point,
                                       "ci_lower": lo, "ci_upper": hi})

                    auroc_point = float(roc_auc_score(y, proba))
                    auroc_boot = [roc_auc_score(y[idx], proba[idx]) for idx in resamples
                                  if len(idx) >= 2 and len(np.unique(y[idx])) >= 2]
                    lo, hi = ci_only(auroc_boot)
                    main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                       "metric": "auroc", "point_estimate": auroc_point,
                                       "ci_lower": lo, "ci_upper": hi})
                else:
                    y = df["y_true"].to_numpy()
                    pred = df["pred"].to_numpy()

                    f1_point = float(f1_score(y, pred, labels=MULTICLASS_CLASSES,
                                               average="macro", zero_division=0))
                    f1_boot = [f1_score(y[idx], pred[idx], labels=MULTICLASS_CLASSES,
                                         average="macro", zero_division=0)
                               for idx in resamples if len(idx) >= 2]
                    lo, hi = ci_only(f1_boot)
                    main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                       "metric": "macro_f1", "point_estimate": f1_point,
                                       "ci_lower": lo, "ci_upper": hi})

                print(f"done: {task}_{stage}_{algo}")

    pd.DataFrame(main_rows).to_csv(f"{OUT_DIR}/main_model_cis.csv", index=False)
    print("DONE. main_model_cis.csv regenerated.")
    print("rna_subset_paired_differences.csv, rna_freshness_cis.csv, and "
          "rna_freshness_paired_differences.csv were NOT touched.")