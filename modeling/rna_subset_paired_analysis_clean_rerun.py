"""
Paired patient-level bootstrap comparison of D vs A and D vs C, using
the CLEAN RERUN's XGBoost predictions, on RNA-available test rows only.

Reads ONLY artifacts/results_clean_rerun/*_xgboost_predictions.csv and
the frozen data/clinical/visit_pairs_with_rna.csv (for the existing
RNA-availability definition: days_since_rna_sample notna()). No
retraining, no changes to predictions or task/eligibility/RNA-alignment
definitions.

Writes ONLY to artifacts/results_clean_rerun/uncertainty/
rna_subset_paired_differences.csv.

Positive observed_diff means Model D outperforms the comparator on that
metric. This script reports point estimates and 95% CIs only - it does
not claim statistical significance.

Fails loudly if any required clean-rerun prediction file is missing.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/uncertainty"
CLINICAL_DIR = "data/clinical"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
STALENESS_COL = "days_since_rna_sample"  # existing project RNA-availability definition
MULTICLASS_CLASSES = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]


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


def load_clean_preds(task, stage, algo="xgboost"):
    path = f"{CLEAN_RESULTS_DIR}/{task}_{stage}_{algo}_predictions.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing clean-rerun prediction file: {path} - cannot run the paired "
            f"RNA analysis on an incomplete clean rerun. Refusing to substitute a "
            f"historical file."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    rna_source_path = f"{CLINICAL_DIR}/visit_pairs_with_rna.csv"
    if not os.path.exists(rna_source_path):
        raise FileNotFoundError(f"Missing frozen input: {rna_source_path}")
    master = pd.read_csv(rna_source_path, low_memory=False)
    rna_available_pairs = set(master.loc[master[STALENESS_COL].notna(), "pair_id"])

    paired_rows = []

    for task in ["binary", "multiclass"]:
        preds = {stage: load_clean_preds(task, stage) for stage in ["a", "c", "d"]}

        merged = preds["a"][["pair_id", "public_id", "y_true"]].copy()
        merged = merged.merge(preds["c"][["pair_id", "public_id", "y_true"]],
                               on="pair_id", suffixes=("", "_c_check"))
        merged = merged.merge(preds["d"][["pair_id", "public_id", "y_true"]],
                               on="pair_id", suffixes=("", "_d_check"))

        # --- Hard assertions: identical pair_ids/public_ids/y_true across A/C/D ---
        assert set(preds["a"]["pair_id"]) == set(preds["c"]["pair_id"]) == set(preds["d"]["pair_id"]), \
            f"{task}: A/C/D prediction files do not cover identical pair_id sets."
        assert (merged["y_true"] == merged["y_true_c_check"]).all(), f"{task}: A/C y_true misalignment"
        assert (merged["y_true"] == merged["y_true_d_check"]).all(), f"{task}: A/D y_true misalignment"
        assert (merged["public_id"] == merged["public_id_c_check"]).all(), f"{task}: A/C public_id misalignment"
        assert (merged["public_id"] == merged["public_id_d_check"]).all(), f"{task}: A/D public_id misalignment"

        merged = merged[merged["pair_id"].isin(rna_available_pairs)].reset_index(drop=True)
        print(f"{task}: {len(merged)} RNA-available test rows for paired comparison "
              f"({merged['public_id'].nunique()} patients)")

        col = "proba" if task == "binary" else "pred"
        a_vals = preds["a"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        c_vals = preds["c"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        d_vals = preds["d"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        y_true = merged["y_true"].to_numpy()
        pid = merged["public_id"].to_numpy()

        # Same resample indices reused for both comparators -> a true paired bootstrap
        resamples = patient_bootstrap_indices(pid)
        n_patients = len(set(pid))

        if task == "binary":
            metrics = {"auprc": average_precision_score, "auroc": roc_auc_score}
        else:
            metrics = {"macro_f1": lambda yt, yp: f1_score(
                yt, yp, labels=MULTICLASS_CLASSES, average="macro", zero_division=0)}

        for metric_name, metric_fn in metrics.items():
            for comparator, comp_vals in [("A", a_vals), ("C", c_vals)]:
                observed_diff = float(metric_fn(y_true, d_vals) - metric_fn(y_true, comp_vals))
                boot_diffs = []
                for idx in resamples:
                    if len(idx) < 2:
                        continue
                    if task == "binary" and len(np.unique(y_true[idx])) < 2:
                        continue
                    boot_diffs.append(metric_fn(y_true[idx], d_vals[idx]) - metric_fn(y_true[idx], comp_vals[idx]))
                lo, hi = percentile_ci(boot_diffs)
                paired_rows.append({
                    "task": task, "comparison": f"D - {comparator}", "metric": metric_name,
                    "observed_diff": observed_diff, "ci_lower": lo, "ci_upper": hi,
                    "n_rows": len(merged), "n_patients": n_patients,
                })

    pd.DataFrame(paired_rows).to_csv(f"{OUT_DIR}/rna_subset_paired_differences.csv", index=False)
    print(f"\nDONE. Saved: {OUT_DIR}/rna_subset_paired_differences.csv")
    print("Source: artifacts/results_clean_rerun/ ONLY. Historical artifacts/results/ untouched.")