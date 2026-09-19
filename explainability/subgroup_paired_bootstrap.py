"""
Subgroup analysis - File 3 of 5 (PRIMARY analysis): paired patient-level
bootstrap D-vs-C comparison within each a priori subgroup and its
complement, plus subgroup-by-model interaction tests.

Uses ONLY the LOCKED clean-rerun whole-cohort test predictions of
C-XGBoost and D-XGBoost (D = C + RNA) for BOTH tasks - the same pair as
the whole-cohort "D - C" rows of rna_subset_paired_analysis_clean_rerun.py.
NO retraining. Analysis set: task-eligible test rows with RNA available
(days_since_rna_sample notna), matching that whole-cohort script.

Per group: rows are aligned on pair_id; one set of patient-level
bootstrap resamples (1000, seed RANDOM_SEED) is reused for C and D (true
paired bootstrap). Positive difference = Model D outperforms. Resamples
where a metric is undefined (e.g. single-class AUROC) are skipped and
counted, never silently dropped.

Interactions: (D-C)_a - (D-C)_b for deep vs non-deep and frontline vs
later-line. The same patient can contribute rows to both sides (response
depth and treatment line change over follow-up), so patients are
resampled JOINTLY from the union of both groups and both effects are
recomputed inside each resample - the two effects are never treated as
independent.

Point estimates and 95% percentile CIs only - no p-values, no claim of
statistical significance, no multiplicity adjustment (see summary JSON).

Does NOT modify or retrain any canonical model or prediction file.
Writes ONLY under artifacts/results_clean_rerun/subgroup_analysis/.
"""

import os

import numpy as np
import pandas as pd

from explainability.subgroup_analysis_common import (
    load_master_data, get_subgroup_mask, assert_no_leakage_in_subgroup_definitions,
    patient_bootstrap_indices, percentile_ci, outcome_summary,
    TASK_METRICS, TASK_SCORE_COL, SUBGROUPS, INTERACTIONS, TASKS, SUBGROUP_FIELDS,
    PRIMARY_ALGORITHM, COMPARATOR_STAGE, RNA_STAGE, BINARY_TARGET, MULTICLASS_TARGET,
    STALENESS_COL, RANDOM_SEED, N_BOOTSTRAP, SUBGROUP_OUT_DIR,
)
from rna_subset_paired_analysis_clean_rerun import load_clean_preds

DIFFERENCES_PATH = f"{SUBGROUP_OUT_DIR}/subgroup_paired_bootstrap_differences.csv"
INTERACTIONS_PATH = f"{SUBGROUP_OUT_DIR}/subgroup_interactions.csv"
COMPARISON_LABEL = f"D - C ({PRIMARY_ALGORITHM})"
ANALYSIS_SET_LABEL = "test split, task-eligible, RNA-available"


def load_aligned_analysis_frame(df, task):
    """Locked C and D test predictions aligned on pair_id, joined to the
    Vt-side membership fields, restricted to RNA-available rows. Returns
    (frame, alignment_report)."""
    score_col = TASK_SCORE_COL[task]
    c = load_clean_preds(task, COMPARATOR_STAGE, PRIMARY_ALGORITHM)
    d = load_clean_preds(task, RNA_STAGE, PRIMARY_ALGORITHM)
    assert c["pair_id"].is_unique and d["pair_id"].is_unique, f"{task}: duplicate pair_id in predictions"
    assert set(c["pair_id"]) == set(d["pair_id"]), f"{task}: C/D predictions cover different pair_id sets"

    merged = c[["pair_id", "public_id", "y_true", score_col]].merge(
        d[["pair_id", "public_id", "y_true", score_col]], on="pair_id",
        suffixes=("_c", "_d"), validate="one_to_one")
    assert (merged["public_id_c"] == merged["public_id_d"]).all(), f"{task}: C/D public_id mismatch"
    assert (merged["y_true_c"] == merged["y_true_d"]).all(), f"{task}: C/D y_true mismatch"

    target = BINARY_TARGET if task == "binary" else MULTICLASS_TARGET
    meta_cols = ["pair_id", "public_id", "split", "eligible_for_binary", target, STALENESS_COL,
                 *sorted(set(SUBGROUP_FIELDS.values()))]
    frame = merged.merge(df[meta_cols], on="pair_id", how="left", validate="one_to_one")
    assert frame["split"].notna().all(), f"{task}: prediction pair_id missing from master data"
    assert (frame["split"] == "test").all(), f"{task}: non-test row in locked test predictions"
    assert (frame["public_id"] == frame["public_id_c"]).all(), f"{task}: public_id differs from master data"
    if task == "binary":
        assert frame["eligible_for_binary"].astype(bool).all(), "binary: ineligible row in predictions"
    y_master = frame[target].astype(int) if task == "binary" else frame[target]
    assert (y_master == frame["y_true_c"]).all(), f"{task}: y_true differs from master {target}"

    frame = frame.rename(columns={"y_true_c": "y_true", f"{score_col}_c": "score_c", f"{score_col}_d": "score_d"})
    nan_rows = frame[["y_true", "score_c", "score_d"]].isna().any(axis=1)
    rna_available = frame[STALENESS_COL].notna()
    report = {"task": task, "n_test_rows_aligned": int(len(frame)),
              "n_rows_dropped_nan": int(nan_rows.sum()),
              "n_rows_dropped_rna_unavailable": int((~nan_rows & ~rna_available).sum())}
    frame = frame[~nan_rows & rna_available].reset_index(drop=True)
    report["n_rows_analysis_set"] = int(len(frame))
    report["n_patients_analysis_set"] = int(frame["public_id"].nunique())
    return frame, report


def paired_diff(metric_fn, y, c, d, idx):
    return metric_fn(y[idx], d[idx]) - metric_fn(y[idx], c[idx])


def evaluate_group(frame, mask, task, subgroup_name):
    """C, D, D-C and 95% patient-bootstrap CI for one group."""
    sub = frame[mask].reset_index(drop=True)
    y, c, d = sub["y_true"].to_numpy(), sub["score_c"].to_numpy(), sub["score_d"].to_numpy()
    all_idx = np.arange(len(sub))
    resamples = patient_bootstrap_indices(sub["public_id"].to_numpy(), n=N_BOOTSTRAP, seed=RANDOM_SEED)

    base = {"task": task, "subgroup": subgroup_name, "comparison": COMPARISON_LABEL,
            "analysis_set": ANALYSIS_SET_LABEL, "n_rows": int(len(sub)),
            "n_patients": int(sub["public_id"].nunique()), **outcome_summary(y, task)}
    rows = []
    for metric_name, metric_fn in TASK_METRICS[task].items():
        boot = [paired_diff(metric_fn, y, c, d, idx) for idx in resamples]
        lo, hi, n_valid = percentile_ci(boot)
        rows.append({**base, "metric": metric_name,
                     "C": metric_fn(y, c), "D": metric_fn(y, d),
                     "observed_diff": paired_diff(metric_fn, y, c, d, all_idx),
                     "ci_lower": lo, "ci_upper": hi,
                     "n_boot_valid": n_valid, "n_boot_skipped": N_BOOTSTRAP - n_valid})
    return rows


def evaluate_interaction(frame, group_a, group_b, task):
    """(D-C)_a - (D-C)_b with a JOINT patient bootstrap over the union of
    both groups: each resample draws patients once, then splits the
    drawn rows by group membership and recomputes both effects."""
    mask_a = get_subgroup_mask(frame, group_a).to_numpy()
    mask_b = get_subgroup_mask(frame, group_b).to_numpy()
    assert not (mask_a & mask_b).any(), f"{group_a}/{group_b} share rows - not complementary"
    union = frame[mask_a | mask_b].reset_index(drop=True)
    in_a = mask_a[mask_a | mask_b]
    y, c, d = union["y_true"].to_numpy(), union["score_c"].to_numpy(), union["score_d"].to_numpy()
    resamples = patient_bootstrap_indices(union["public_id"].to_numpy(), n=N_BOOTSTRAP, seed=RANDOM_SEED)

    idx_a_all, idx_b_all = np.where(in_a)[0], np.where(~in_a)[0]
    pids_a, pids_b = set(union.loc[in_a, "public_id"]), set(union.loc[~in_a, "public_id"])
    rows = []
    for metric_name, metric_fn in TASK_METRICS[task].items():
        diff_a = paired_diff(metric_fn, y, c, d, idx_a_all)
        diff_b = paired_diff(metric_fn, y, c, d, idx_b_all)
        boot = [paired_diff(metric_fn, y, c, d, idx[in_a[idx]]) - paired_diff(metric_fn, y, c, d, idx[~in_a[idx]])
                for idx in resamples]
        lo, hi, n_valid = percentile_ci(boot)
        rows.append({
            "task": task, "metric": metric_name, "group_a": group_a, "group_b": group_b,
            "contrast": f"({COMPARISON_LABEL})_{group_a} - ({COMPARISON_LABEL})_{group_b}",
            "diff_a": diff_a, "diff_b": diff_b, "observed_interaction": diff_a - diff_b,
            "ci_lower": lo, "ci_upper": hi,
            "n_boot_valid": n_valid, "n_boot_skipped": N_BOOTSTRAP - n_valid,
            "n_rows_a": int(len(idx_a_all)), "n_rows_b": int(len(idx_b_all)),
            "n_patients_a": len(pids_a), "n_patients_b": len(pids_b),
            "n_patients_in_both": len(pids_a & pids_b),
            "n_patients_resampled": int(union["public_id"].nunique()),
            "analysis_set": ANALYSIS_SET_LABEL,
        })
    return rows


def run_primary_analysis(df):
    """Returns (differences_df, interactions_df, alignment_reports)."""
    assert_no_leakage_in_subgroup_definitions(df)
    diff_rows, inter_rows, reports = [], [], []
    for task in TASKS:
        frame, report = load_aligned_analysis_frame(df, task)
        reports.append(report)
        print(f"\n[{task}] analysis set: {report['n_rows_analysis_set']} rows / "
              f"{report['n_patients_analysis_set']} patients "
              f"(dropped: {report['n_rows_dropped_nan']} NaN, "
              f"{report['n_rows_dropped_rna_unavailable']} RNA-unavailable)")

        diff_rows.extend(evaluate_group(frame, pd.Series(True, index=frame.index), task, "all"))
        for name in SUBGROUPS:
            diff_rows.extend(evaluate_group(frame, get_subgroup_mask(frame, name), task, name))
        for group_a, group_b in INTERACTIONS:
            inter_rows.extend(evaluate_interaction(frame, group_a, group_b, task))

    return pd.DataFrame(diff_rows), pd.DataFrame(inter_rows), reports


def _fmt(v):
    return "NA" if v is None or not np.isfinite(v) else f"{v:+.4f}"


def print_results(diff_df, inter_df):
    for _, r in diff_df.iterrows():
        print(f"  {r['task']:10s} {r['subgroup']:20s} {r['metric']:8s} n={r['n_rows']:4d}/{r['n_patients']:3d}pts "
              f"C={r['C']:.4f} D={r['D']:.4f} D-C={_fmt(r['observed_diff'])} "
              f"[{_fmt(r['ci_lower'])}, {_fmt(r['ci_upper'])}] skipped={r['n_boot_skipped']}")
    for _, r in inter_df.iterrows():
        print(f"  {r['task']:10s} {r['group_a']} vs {r['group_b']} {r['metric']:8s} "
              f"interaction={_fmt(r['observed_interaction'])} [{_fmt(r['ci_lower'])}, {_fmt(r['ci_upper'])}] "
              f"patients in both={r['n_patients_in_both']} skipped={r['n_boot_skipped']}")


if __name__ == "__main__":
    os.makedirs(SUBGROUP_OUT_DIR, exist_ok=True)
    diff_df, inter_df, _ = run_primary_analysis(load_master_data())
    print_results(diff_df, inter_df)
    diff_df.to_csv(DIFFERENCES_PATH, index=False)
    inter_df.to_csv(INTERACTIONS_PATH, index=False)
    print(f"\nSaved: {DIFFERENCES_PATH}\nSaved: {INTERACTIONS_PATH}")
