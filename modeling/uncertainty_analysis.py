"""
Uncertainty analysis on existing Step 13 results. No retraining, no
tuning - reads already-saved prediction files only.

Patient-level bootstrap throughout: a patient sampled twice contributes
all their rows twice in that replicate (preserves true multiplicity).
"""

import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

RESULTS_DIR = "artifacts/results"
OUT_DIR = "artifacts/results/uncertainty"
CLINICAL_DIR = "data/clinical"

N_BOOTSTRAP = 1000
SEED = 42
STALENESS_COL = "days_since_rna_sample"
MODEL_STAGES = ["a", "b", "c", "d"]
ALGORITHMS = ["logistic", "lightgbm", "xgboost"]
MULTICLASS_CLASSES = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]


def patient_bootstrap_indices(public_ids, n=N_BOOTSTRAP, seed=SEED):
    """Precompute resample row-indices once so they can be reused
    identically across paired models."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(public_ids)
    patient_to_rows = {p: np.where(public_ids == p)[0] for p in uniq}
    resamples = []
    for _ in range(n):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        row_idx = np.concatenate([patient_to_rows[p] for p in sampled])
        resamples.append(row_idx)
    return resamples


def ci_from_scores(scores):
    scores = np.array([s for s in scores if s is not None])
    if len(scores) == 0:
        return None, None, None
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def binary_ci(y_true, proba, public_ids):
    resamples = patient_bootstrap_indices(public_ids)
    auprc_scores, auroc_scores = [], []
    for idx in resamples:
        if len(idx) < 2 or len(np.unique(y_true[idx])) < 2:
            continue
        auprc_scores.append(average_precision_score(y_true[idx], proba[idx]))
        auroc_scores.append(roc_auc_score(y_true[idx], proba[idx]))
    return {"auprc": ci_from_scores(auprc_scores), "auroc": ci_from_scores(auroc_scores)}


def multiclass_ci(y_true, y_pred, public_ids):
    resamples = patient_bootstrap_indices(public_ids)
    f1_scores = []
    for idx in resamples:
        if len(idx) < 2:
            continue
        f1_scores.append(f1_score(y_true[idx], y_pred[idx], labels=MULTICLASS_CLASSES,
                                   average="macro", zero_division=0))
    return {"macro_f1": ci_from_scores(f1_scores)}


def flag(mean_diff, lo, hi):
    if lo is None:
        return "inconclusive (insufficient data)"
    if lo > 0:
        return "favors D"
    if hi < 0:
        return "favors comparator"
    return "inconclusive (CI includes 0)"


def paired_diff(y_true, proba_or_pred_1, proba_or_pred_2, public_ids, metric_fn, resamples):
    """model_1 - model_2, using the SAME resample indices for both."""
    diffs = []
    for idx in resamples:
        if len(idx) < 2:
            continue
        try:
            s1 = metric_fn(y_true[idx], proba_or_pred_1[idx])
            s2 = metric_fn(y_true[idx], proba_or_pred_2[idx])
            diffs.append(s1 - s2)
        except (ValueError, ZeroDivisionError):
            continue
    return ci_from_scores(diffs)


def load_preds(task, stage, algo):
    path = f"{RESULTS_DIR}/{task}_{stage}_{algo}_predictions.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    #  Part 1: main model CIs, all 24 existing combos 
    main_rows = []
    for task in ["binary", "multiclass"]:
        for stage in MODEL_STAGES:
            for algo in ALGORITHMS:
                df = load_preds(task, stage, algo)
                if df is None:
                    print(f"MISSING predictions file: {task}_{stage}_{algo} - skipped")
                    continue
                pid = df["public_id"].to_numpy()
                if task == "binary":
                    y = df["y_true"].to_numpy()
                    proba = df["proba"].to_numpy()
                    cis = binary_ci(y, proba, pid)
                    for metric, (mean, lo, hi) in cis.items():
                        main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                           "metric": metric, "point_estimate": mean,
                                           "ci_lower": lo, "ci_upper": hi})
                else:
                    y = df["y_true"].to_numpy()
                    pred = df["pred"].to_numpy()
                    cis = multiclass_ci(y, pred, pid)
                    for metric, (mean, lo, hi) in cis.items():
                        main_rows.append({"task": task, "model_stage": stage, "algorithm": algo,
                                           "metric": metric, "point_estimate": mean,
                                           "ci_lower": lo, "ci_upper": hi})
                print(f"done: {task}_{stage}_{algo}")

    pd.DataFrame(main_rows).to_csv(f"{OUT_DIR}/main_model_cis.csv", index=False)

    #  Part 2: paired RNA-subset comparison, XGBoost A/C/D 
    master = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    rna_available_pairs = set(master.loc[master[STALENESS_COL].notna(), "pair_id"])

    paired_rows = []
    summary_rows = []

    for task in ["binary", "multiclass"]:
        preds = {}
        for stage in ["a", "c", "d"]:
            df = load_preds(task, stage, "xgboost")
            if df is None:
                print(f"MISSING: {task}_{stage}_xgboost predictions - cannot do paired comparison for {task}")
                preds = None
                break
            preds[stage] = df

        if preds is None:
            continue

        merged = preds["a"][["pair_id", "public_id", "y_true"]].copy()
        merged = merged.merge(preds["c"][["pair_id", "y_true"]], on="pair_id", suffixes=("", "_c_check"))
        merged = merged.merge(preds["d"][["pair_id", "y_true"]], on="pair_id", suffixes=("", "_d_check"))
        assert (merged["y_true"] == merged["y_true_c_check"]).all(), "A/C row misalignment"
        assert (merged["y_true"] == merged["y_true_d_check"]).all(), "A/D row misalignment"

        merged = merged[merged["pair_id"].isin(rna_available_pairs)].reset_index(drop=True)
        print(f"{task}: {len(merged)} RNA-available test rows for paired comparison")

        col = "proba" if task == "binary" else "pred"
        a_vals = preds["a"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        c_vals = preds["c"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        d_vals = preds["d"].set_index("pair_id").loc[merged["pair_id"], col].to_numpy()
        y_true = merged["y_true"].to_numpy()
        pid = merged["public_id"].to_numpy()
        resamples = patient_bootstrap_indices(pid)

        if task == "binary":
            metrics = {"auprc": average_precision_score, "auroc": roc_auc_score}
        else:
            metrics = {"macro_f1": lambda yt, yp: f1_score(yt, yp, labels=MULTICLASS_CLASSES,
                                                            average="macro", zero_division=0)}

        for metric_name, metric_fn in metrics.items():
            for comparator, comp_vals in [("A", a_vals), ("C", c_vals)]:
                mean, lo, hi = paired_diff(y_true, d_vals, comp_vals, pid, metric_fn, resamples)
                row = {"task": task, "comparison": f"D - {comparator}", "metric": metric_name,
                       "mean_diff": mean, "ci_lower": lo, "ci_upper": hi,
                       "interpretation": flag(mean, lo, hi), "n_rows": len(merged),
                       "n_patients": len(set(pid))}
                paired_rows.append(row)
                summary_rows.append(row)

    pd.DataFrame(paired_rows).to_csv(f"{OUT_DIR}/rna_subset_paired_differences.csv", index=False)

    #  Part 3: freshness - BLOCKED, point estimates only 
    freshness_path = f"{RESULTS_DIR}/rna_freshness_sensitivity_PRELIMINARY.json"
    freshness_rows = []
    if os.path.exists(freshness_path):
        with open(freshness_path) as f:
            freshness_data = json.load(f)
        for task, task_data in freshness_data.items():
            for cutoff_key, cutoff_data in task_data.get("cutoff_results_PRELIMINARY", {}).items():
                m = cutoff_data["metrics"]
                if task == "binary":
                    vals = {"auprc": m["auprc"], "auroc": m["auroc"]}
                else:
                    vals = {"macro_f1": m["macro_f1"]}
                for metric, val in vals.items():
                    freshness_rows.append({
                        "task": task, "cutoff": cutoff_key, "metric": metric,
                        "point_estimate": val, "ci_lower": None, "ci_upper": None,
                        "ci_status": "UNAVAILABLE - row-level predictions were not saved "
                                     "for freshness-cutoff models; point estimate only",
                    })
    pd.DataFrame(freshness_rows).to_csv(f"{OUT_DIR}/rna_freshness_cis.csv", index=False)

    with open(f"{OUT_DIR}/rna_freshness_paired_differences.csv", "w") as f:
        f.write("status\nBLOCKED - row-level predictions not saved for freshness-cutoff models. "
                "Paired 365d-vs-730d and 365d-vs-545d bootstrap differences require a small rerun "
                "of rna_freshness_sensitivity() with a predictions-save step added.\n")

    #  Summary 
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump({
            "rna_subset_paired_differences": summary_rows,
            "note": "Freshness CIs and paired differences are blocked - see rna_freshness_cis.csv "
                    "and rna_freshness_paired_differences.csv for details.",
        }, f, indent=2)

    print("DONE. See artifacts/results/uncertainty/")