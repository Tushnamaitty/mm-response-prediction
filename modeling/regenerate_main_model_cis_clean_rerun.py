"""
Publication-quality patient-level bootstrap CIs for the CLEAN RERUN's
24 main A/B/C/D x algorithm x task results.

Reads ONLY from artifacts/results_clean_rerun/*_predictions.csv - never
from the historical artifacts/results/. No retraining, no changes to
any prediction file. Writes ONLY to
artifacts/results_clean_rerun/uncertainty/main_model_cis.csv - does not
touch any historical uncertainty output.

Point estimates are computed ONCE on the full observed clean-rerun test
predictions (never the bootstrap mean). Bootstrap resamples are used
ONLY to build the 95% CI around that fixed point estimate.

Fails loudly (raises) if any of the 24 expected clean-rerun prediction
files is missing - never silently substitutes a historical file.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/uncertainty"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
MODEL_STAGES = ["a", "b", "c", "d"]
ALGORITHMS = ["logistic", "lightgbm", "xgboost"]
MULTICLASS_CLASSES = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]


def patient_bootstrap_indices(public_ids, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Patient-level bootstrap: sample patients with replacement, include
    ALL rows for each sampled patient, preserving multiplicity if a
    patient is drawn more than once."""
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


def load_clean_predictions(task, stage, algo):
    path = f"{CLEAN_RESULTS_DIR}/{task}_{stage}_{algo}_predictions.csv"
    if not os.path.exists(path):
        return None, path
    return pd.read_csv(path), path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Fail-fast check: all 24 clean-rerun prediction files must exist ---
    missing = []
    for task in ["binary", "multiclass"]:
        for stage in MODEL_STAGES:
            for algo in ALGORITHMS:
                _, path = load_clean_predictions(task, stage, algo)
                if not os.path.exists(path):
                    missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing clean-rerun prediction file(s) - cannot regenerate "
            "main_model_cis.csv from an incomplete clean rerun. "
            "Refusing to fall back to historical artifacts/results/. "
            f"Missing: {missing}"
        )

    main_rows = []
    for task in ["binary", "multiclass"]:
        for stage in MODEL_STAGES:
            for algo in ALGORITHMS:
                df, path = load_clean_predictions(task, stage, algo)
                pid = df["public_id"].to_numpy()
                resamples = patient_bootstrap_indices(pid)

                if task == "binary":
                    y = df["y_true"].to_numpy()
                    proba = df["proba"].to_numpy()
                    point_auprc = float(average_precision_score(y, proba))
                    point_auroc = float(roc_auc_score(y, proba))
                    auprc_scores, auroc_scores = [], []
                    for idx in resamples:
                        if len(idx) < 2 or len(np.unique(y[idx])) < 2:
                            continue
                        auprc_scores.append(average_precision_score(y[idx], proba[idx]))
                        auroc_scores.append(roc_auc_score(y[idx], proba[idx]))
                    auprc_lo, auprc_hi = percentile_ci(auprc_scores)
                    auroc_lo, auroc_hi = percentile_ci(auroc_scores)
                    main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                       "metric": "auprc", "point_estimate": point_auprc,
                                       "ci_lower": auprc_lo, "ci_upper": auprc_hi})
                    main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                       "metric": "auroc", "point_estimate": point_auroc,
                                       "ci_lower": auroc_lo, "ci_upper": auroc_hi})
                else:
                    y = df["y_true"].to_numpy()
                    pred = df["pred"].to_numpy()
                    point_f1 = float(f1_score(y, pred, labels=MULTICLASS_CLASSES,
                                               average="macro", zero_division=0))
                    f1_scores = []
                    for idx in resamples:
                        if len(idx) < 2:
                            continue
                        f1_scores.append(f1_score(y[idx], pred[idx], labels=MULTICLASS_CLASSES,
                                                   average="macro", zero_division=0))
                    lo, hi = percentile_ci(f1_scores)
                    main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                       "metric": "macro_f1", "point_estimate": point_f1,
                                       "ci_lower": lo, "ci_upper": hi})
                print(f"done: {task}_{stage}_{algo}")

    pd.DataFrame(main_rows).to_csv(f"{OUT_DIR}/main_model_cis.csv", index=False)
    print(f"\nDONE. Saved: {OUT_DIR}/main_model_cis.csv")
    print("Source: artifacts/results_clean_rerun/ ONLY. Historical artifacts/results/ untouched.")