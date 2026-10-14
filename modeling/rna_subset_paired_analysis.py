"""
Standalone RNA-available-subset paired comparison (D vs A, D vs C).
Uses observed_diff (computed once on the full matched observed data),
with bootstrap resamples used ONLY for the CI - not the bootstrap mean.

This is a separate script specifically so it cannot touch the already
corrected freshness outputs (rna_freshness_cis.csv,
rna_freshness_paired_differences.csv). No retraining, no changes to
data/models/predictions.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

RESULTS_DIR = "artifacts/results"
OUT_DIR = "artifacts/results/uncertainty"
CLINICAL_DIR = "data/clinical"

N_BOOTSTRAP = 1000
SEED = 42
STALENESS_COL = "days_since_rna_sample"
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


def flag(lo, hi):
    if lo is None:
        return "inconclusive (insufficient data)"
    if lo > 0:
        return "favors D"
    if hi < 0:
        return "favors comparator"
    return "inconclusive (CI includes 0)"


def load_preds(task, stage, algo="xgboost"):
    path = f"{RESULTS_DIR}/{task}_{stage}_{algo}_predictions.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    master = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    rna_available_pairs = set(master.loc[master[STALENESS_COL].notna(), "pair_id"])

    paired_rows = []

    for task in ["binary", "multiclass"]:
        preds = {}
        for stage in ["a", "c", "d"]:
            df = load_preds(task, stage)
            if df is None:
                print(f"MISSING: {task}_{stage}_xgboost predictions - cannot do paired comparison for {task}")
                preds = None
                break
            preds[stage] = df

        if preds is None:
            continue

        merged = preds["a"][["pair_id", "public_id", "y_true"]].copy()
        merged = merged.merge(preds["c"][["pair_id", "public_id", "y_true"]], on="pair_id", suffixes=("", "_c_check"))
        merged = merged.merge(preds["d"][["pair_id", "public_id", "y_true"]], on="pair_id", suffixes=("", "_d_check"))
        assert (merged["y_true"] == merged["y_true_c_check"]).all(), "A/C y_true misalignment"
        assert (merged["y_true"] == merged["y_true_d_check"]).all(), "A/D y_true misalignment"
        assert (merged["public_id"] == merged["public_id_c_check"]).all(), "A/C public_id misalignment"
        assert (merged["public_id"] == merged["public_id_d_check"]).all(), "A/D public_id misalignment"

        merged = merged[merged["pair_id"].isin(rna_available_pairs)].reset_index(drop=True)
        print(f"{task}: {len(merged)} RNA-available test rows for paired comparison")

        col = "proba" if task == "binary" else "pred"
        a_vals = preds["a"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        c_vals = preds["c"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        d_vals = preds["d"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        y_true = merged["y_true"].to_numpy()
        pid = merged["public_id"].to_numpy()
        resamples = patient_bootstrap_indices(pid)
        n_patients = len(set(pid))

        if task == "binary":
            metrics = {"auprc": average_precision_score, "auroc": roc_auc_score}
        else:
            metrics = {"macro_f1": lambda yt, yp: f1_score(yt, yp, labels=MULTICLASS_CLASSES,
                                                            average="macro", zero_division=0)}

        for metric_name, metric_fn in metrics.items():
            for comparator, comp_vals in [("A", a_vals), ("C", c_vals)]:
                observed_diff = float(metric_fn(y_true, d_vals) - metric_fn(y_true, comp_vals))
                boot_diffs = []
                for idx in resamples:
                    if len(idx) < 2:
                        continue
                    if task == "binary" and len(np.unique(y_true[idx])) < 2:
                        continue  # skip replicate with only one class present
                    boot_diffs.append(metric_fn(y_true[idx], d_vals[idx]) - metric_fn(y_true[idx], comp_vals[idx]))
                lo, hi = ci_only(boot_diffs)
                paired_rows.append({
                    "task": task, "comparison": f"D - {comparator}", "metric": metric_name,
                    "observed_diff": observed_diff, "ci_lower": lo, "ci_upper": hi,
                    "interpretation": flag(lo, hi), "n_rows": len(merged), "n_patients": n_patients,
                })

    pd.DataFrame(paired_rows).to_csv(f"{OUT_DIR}/rna_subset_paired_differences.csv", index=False)
    print("DONE. See artifacts/results/uncertainty/rna_subset_paired_differences.csv")
    print("rna_freshness_cis.csv and rna_freshness_paired_differences.csv were NOT touched.")