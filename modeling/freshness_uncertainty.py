"""
Computes patient-level bootstrap CIs and paired cutoff differences for
the RNA freshness analysis, using the row-level predictions saved by
rerun_freshness_predictions.py. No retraining.

point_estimate / observed_diff = computed ONCE on the full observed test
data. Bootstrap resamples are used ONLY to build the 95% CI around that
fixed point estimate - the bootstrap mean is never reported as the
point estimate itself.
"""

import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

PRED_DIR = "artifacts/results/rna_freshness_predictions"
OUT_DIR = "artifacts/results/uncertainty"
N_BOOTSTRAP = 1000
SEED = 42
CUTOFFS = [365, 545, 730]
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
    """Returns (ci_lower, ci_upper) from bootstrap scores - never a point estimate."""
    scores = np.array([s for s in scores if s is not None])
    if len(scores) == 0:
        return None, None
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def flag(lo, hi):
    if lo is None:
        return "inconclusive (insufficient data)"
    if lo > 0:
        return "favors cutoff 1"
    if hi < 0:
        return "favors cutoff 2"
    return "inconclusive (CI includes 0)"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    ci_rows = []
    paired_rows = []

    for task in ["binary", "multiclass"]:
        preds = {}
        for cutoff in CUTOFFS:
            path = f"{PRED_DIR}/{task}_cutoff_{cutoff}d_predictions.csv"
            preds[cutoff] = pd.read_csv(path)

        # ---- per-cutoff: point estimate on full data, CI from bootstrap ----
        for cutoff, df in preds.items():
            pid = df["public_id"].to_numpy()
            resamples = patient_bootstrap_indices(pid)
            if task == "binary":
                y, proba = df["y_true"].to_numpy(), df["proba"].to_numpy()
                for metric_name, metric_fn in [("auprc", average_precision_score), ("auroc", roc_auc_score)]:
                    point_estimate = float(metric_fn(y, proba))  # observed, full data
                    boot_scores = [metric_fn(y[idx], proba[idx]) for idx in resamples
                                   if len(idx) >= 2 and len(np.unique(y[idx])) >= 2]
                    lo, hi = ci_only(boot_scores)
                    ci_rows.append({"task": task, "cutoff": f"cutoff_{cutoff}d", "metric": metric_name,
                                     "point_estimate": point_estimate, "ci_lower": lo, "ci_upper": hi})
            else:
                y, pred = df["y_true"].to_numpy(), df["pred"].to_numpy()
                point_estimate = float(f1_score(y, pred, labels=MULTICLASS_CLASSES, average="macro", zero_division=0))
                boot_scores = [f1_score(y[idx], pred[idx], labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                               for idx in resamples if len(idx) >= 2]
                lo, hi = ci_only(boot_scores)
                ci_rows.append({"task": task, "cutoff": f"cutoff_{cutoff}d", "metric": "macro_f1",
                                 "point_estimate": point_estimate, "ci_lower": lo, "ci_upper": hi})

        # ---- paired differences: observed_diff on full matched data, CI from bootstrap ----
        pairs = [(365, 545), (365, 730), (545, 730)]
        for c1, c2 in pairs:
            df1, df2 = preds[c1], preds[c2]
            common = set(df1["pair_id"]) & set(df2["pair_id"])
            d1 = df1[df1["pair_id"].isin(common)].sort_values("pair_id").reset_index(drop=True)
            d2 = df2[df2["pair_id"].isin(common)].sort_values("pair_id").reset_index(drop=True)
            assert (d1["public_id"].values == d2["public_id"].values).all(), "public_id mismatch on common rows"
            assert (d1["y_true"].values == d2["y_true"].values).all(), "y_true mismatch on common rows"

            pid = d1["public_id"].to_numpy()
            y = d1["y_true"].to_numpy()
            resamples = patient_bootstrap_indices(pid)

            if task == "binary":
                p1, p2 = d1["proba"].to_numpy(), d2["proba"].to_numpy()
                for metric_name, metric_fn in [("auprc", average_precision_score), ("auroc", roc_auc_score)]:
                    observed_diff = float(metric_fn(y, p1) - metric_fn(y, p2))  # on full matched data
                    boot_diffs = []
                    for idx in resamples:
                        if len(idx) < 2 or len(np.unique(y[idx])) < 2:
                            continue
                        boot_diffs.append(metric_fn(y[idx], p1[idx]) - metric_fn(y[idx], p2[idx]))
                    lo, hi = ci_only(boot_diffs)
                    paired_rows.append({"task": task, "comparison": f"{c1}d - {c2}d", "metric": metric_name,
                                         "observed_diff": observed_diff, "ci_lower": lo, "ci_upper": hi,
                                         "interpretation": flag(lo, hi),
                                         "n_rows": len(common), "n_patients": len(set(pid))})
            else:
                pr1, pr2 = d1["pred"].to_numpy(), d2["pred"].to_numpy()
                f1_1 = f1_score(y, pr1, labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                f1_2 = f1_score(y, pr2, labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                observed_diff = float(f1_1 - f1_2)  # on full matched data
                boot_diffs = []
                for idx in resamples:
                    if len(idx) < 2:
                        continue
                    fa = f1_score(y[idx], pr1[idx], labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                    fb = f1_score(y[idx], pr2[idx], labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                    boot_diffs.append(fa - fb)
                lo, hi = ci_only(boot_diffs)
                paired_rows.append({"task": task, "comparison": f"{c1}d - {c2}d", "metric": "macro_f1",
                                     "observed_diff": observed_diff, "ci_lower": lo, "ci_upper": hi,
                                     "interpretation": flag(lo, hi),
                                     "n_rows": len(common), "n_patients": len(set(pid))})

    pd.DataFrame(ci_rows).to_csv(f"{OUT_DIR}/rna_freshness_cis.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(f"{OUT_DIR}/rna_freshness_paired_differences.csv", index=False)

    with open(f"{OUT_DIR}/rna_freshness_uncertainty_summary.json", "w") as f:
        json.dump({"cis": ci_rows, "paired_differences": paired_rows}, f, indent=2)

    print("DONE. See artifacts/results/uncertainty/rna_freshness_cis.csv and rna_freshness_paired_differences.csv")