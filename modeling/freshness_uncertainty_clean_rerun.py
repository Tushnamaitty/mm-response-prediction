"""
Paired patient-level bootstrap uncertainty for the RNA freshness cutoffs
(365/545/730 days), reported alongside the clean-rerun uncertainty
outputs.

The 6 freshness models (3 cutoffs x 2 tasks, XGBoost) are NOT retrained
here - this reads the row-level predictions already saved by the
already-corrected rerun_freshness_predictions.py at
artifacts/results/rna_freshness_predictions/ (untouched, not part of
the 24-model clean rerun, since freshness models are a separate
analysis track).

Writes ONLY to artifacts/results_clean_rerun/uncertainty/ - does not
touch the historical artifacts/results/uncertainty/ outputs.

Point estimates are computed ONCE on the full observed test data (never
the bootstrap mean); bootstrap resamples are used ONLY for the 95% CI.
Fails loudly if any expected freshness prediction file is missing.
"""

import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

FRESHNESS_PRED_DIR = "artifacts/results/rna_freshness_predictions"  # existing, not retrained
OUT_DIR = "artifacts/results_clean_rerun/uncertainty"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
CUTOFFS = [365, 545, 730]
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


def load_freshness_preds(task, cutoff):
    path = f"{FRESHNESS_PRED_DIR}/{task}_cutoff_{cutoff}d_predictions.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing freshness prediction file: {path}. This script does not "
            f"retrain freshness models - run the already-corrected "
            f"rerun_freshness_predictions.py first if this file is genuinely absent."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    ci_rows = []
    paired_rows = []

    for task in ["binary", "multiclass"]:
        preds = {cutoff: load_freshness_preds(task, cutoff) for cutoff in CUTOFFS}

        # ---- per-cutoff: point estimate on full data, CI from bootstrap ----
        for cutoff, df in preds.items():
            pid = df["public_id"].to_numpy()
            resamples = patient_bootstrap_indices(pid)
            if task == "binary":
                y, proba = df["y_true"].to_numpy(), df["proba"].to_numpy()
                for metric_name, metric_fn in [("auprc", average_precision_score), ("auroc", roc_auc_score)]:
                    point_estimate = float(metric_fn(y, proba))
                    boot_scores = [metric_fn(y[idx], proba[idx]) for idx in resamples
                                   if len(idx) >= 2 and len(np.unique(y[idx])) >= 2]
                    lo, hi = percentile_ci(boot_scores)
                    ci_rows.append({"task": task, "cutoff": f"cutoff_{cutoff}d", "metric": metric_name,
                                     "point_estimate": point_estimate, "ci_lower": lo, "ci_upper": hi})
            else:
                y, pred = df["y_true"].to_numpy(), df["pred"].to_numpy()
                point_estimate = float(f1_score(y, pred, labels=MULTICLASS_CLASSES, average="macro", zero_division=0))
                boot_scores = [f1_score(y[idx], pred[idx], labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                               for idx in resamples if len(idx) >= 2]
                lo, hi = percentile_ci(boot_scores)
                ci_rows.append({"task": task, "cutoff": f"cutoff_{cutoff}d", "metric": "macro_f1",
                                 "point_estimate": point_estimate, "ci_lower": lo, "ci_upper": hi})

        # ---- paired differences between cutoff pairs: observed_diff on matched data ----
        pairs = [(365, 545), (365, 730), (545, 730)]
        for c1, c2 in pairs:
            df1, df2 = preds[c1], preds[c2]
            common = set(df1["pair_id"]) & set(df2["pair_id"])
            d1 = df1[df1["pair_id"].isin(common)].sort_values("pair_id").reset_index(drop=True)
            d2 = df2[df2["pair_id"].isin(common)].sort_values("pair_id").reset_index(drop=True)

            # --- Hard assertions: identical pair_ids/public_ids after filtering ---
            assert list(d1["pair_id"]) == list(d2["pair_id"]), \
                f"{task} {c1}d/{c2}d: pair_id mismatch after common-set filtering"
            assert (d1["public_id"].values == d2["public_id"].values).all(), \
                f"{task} {c1}d/{c2}d: public_id mismatch on common rows"
            assert (d1["y_true"].values == d2["y_true"].values).all(), \
                f"{task} {c1}d/{c2}d: y_true mismatch on common rows"

            pid = d1["public_id"].to_numpy()
            y = d1["y_true"].to_numpy()
            resamples = patient_bootstrap_indices(pid)

            if task == "binary":
                p1, p2 = d1["proba"].to_numpy(), d2["proba"].to_numpy()
                for metric_name, metric_fn in [("auprc", average_precision_score), ("auroc", roc_auc_score)]:
                    observed_diff = float(metric_fn(y, p1) - metric_fn(y, p2))
                    boot_diffs = []
                    for idx in resamples:
                        if len(idx) < 2 or len(np.unique(y[idx])) < 2:
                            continue
                        boot_diffs.append(metric_fn(y[idx], p1[idx]) - metric_fn(y[idx], p2[idx]))
                    lo, hi = percentile_ci(boot_diffs)
                    paired_rows.append({"task": task, "comparison": f"{c1}d - {c2}d", "metric": metric_name,
                                         "observed_diff": observed_diff, "ci_lower": lo, "ci_upper": hi,
                                         "n_rows": len(common), "n_patients": len(set(pid))})
            else:
                pr1, pr2 = d1["pred"].to_numpy(), d2["pred"].to_numpy()
                f1_1 = f1_score(y, pr1, labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                f1_2 = f1_score(y, pr2, labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                observed_diff = float(f1_1 - f1_2)
                boot_diffs = []
                for idx in resamples:
                    if len(idx) < 2:
                        continue
                    fa = f1_score(y[idx], pr1[idx], labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                    fb = f1_score(y[idx], pr2[idx], labels=MULTICLASS_CLASSES, average="macro", zero_division=0)
                    boot_diffs.append(fa - fb)
                lo, hi = percentile_ci(boot_diffs)
                paired_rows.append({"task": task, "comparison": f"{c1}d - {c2}d", "metric": "macro_f1",
                                     "observed_diff": observed_diff, "ci_lower": lo, "ci_upper": hi,
                                     "n_rows": len(common), "n_patients": len(set(pid))})

    pd.DataFrame(ci_rows).to_csv(f"{OUT_DIR}/rna_freshness_cis.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(f"{OUT_DIR}/rna_freshness_paired_differences.csv", index=False)

    with open(f"{OUT_DIR}/rna_freshness_uncertainty_summary.json", "w") as f:
        json.dump({"cis": ci_rows, "paired_differences": paired_rows}, f, indent=2)

    print(f"DONE. Saved: {OUT_DIR}/rna_freshness_cis.csv")
    print(f"      Saved: {OUT_DIR}/rna_freshness_paired_differences.csv")
    print(f"      Saved: {OUT_DIR}/rna_freshness_uncertainty_summary.json")
    print(f"Source predictions: {FRESHNESS_PRED_DIR}/ (not retrained). "
          f"Historical artifacts/results/uncertainty/ untouched.")