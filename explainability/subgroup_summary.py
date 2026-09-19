"""
Subgroup analysis - File 5 of 5: regenerates the counts report and the
PRIMARY locked-model paired bootstrap + interaction outputs, reads the
SECONDARY repeated-CV outputs written by File 4, runs cross-file
consistency checks, and writes subgroup_analysis_summary.json.

Run order (repo root):
    python -m explainability.subgroup_repeated_cv   # File 4, slow (retrains XGBoost)
    python -m explainability.subgroup_summary       # this file

Does NOT modify or retrain any canonical model. Fails loudly if a
required input is missing. Writes ONLY under
artifacts/results_clean_rerun/subgroup_analysis/.

Outputs:
- subgroup_counts_report.csv
- subgroup_paired_bootstrap_differences.csv
- subgroup_interactions.csv
- subgroup_analysis_summary.json
(subgroup_repeated_cv_per_fold.csv / _aggregate.csv come from File 4.)
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

from explainability.subgroup_analysis_common import (
    load_master_data, get_subgroup_mask, SUBGROUPS, TASKS, TASK_METRICS,
    SUBGROUP_FIELDS, DEEP_RESPONSE_CATEGORIES, PRIMARY_ALGORITHM, COMPARATOR_STAGE, RNA_STAGE,
    CLEAN_RESULTS_DIR, SUBGROUP_OUT_DIR, N_BOOTSTRAP, N_FOLDS, N_REPEATS, RANDOM_SEED,
)
from explainability.subgroup_counts_and_leakage_check import build_counts_report, OUT_PATH as COUNTS_PATH
from explainability.subgroup_paired_bootstrap import (
    run_primary_analysis, print_results, DIFFERENCES_PATH, INTERACTIONS_PATH,
)
from explainability.subgroup_repeated_cv import PER_FOLD_PATH, AGGREGATE_PATH

SUMMARY_PATH = f"{SUBGROUP_OUT_DIR}/subgroup_analysis_summary.json"
CANONICAL_WHOLE_COHORT_PATH = f"{CLEAN_RESULTS_DIR}/uncertainty/rna_subset_paired_differences.csv"


def locked_input_paths():
    paths = [f"{CLEAN_RESULTS_DIR}/{task}_{stage}_{PRIMARY_ALGORITHM}_{kind}"
             for task in TASKS for stage in (COMPARATOR_STAGE, RNA_STAGE)
             for kind in ("predictions.csv", "metrics.json")]
    return paths + [CANONICAL_WHOLE_COHORT_PATH, "data_pipeline/model_feature_sets.json",
                    "data/clinical/visit_pairs_with_rna.csv", "data/splits/visit_pair_splits.csv",
                    "data/splits/task_eligibility.csv"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_required_files():
    missing = [p for p in locked_input_paths() + [PER_FOLD_PATH, AGGREGATE_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))


def run_checks(df, counts_df, diff_df, inter_df, cv_fold_df, cv_agg_df):
    checks = {}

    # 1. "all" row reproduces the canonical whole-cohort XGBoost D - C point estimates
    canon = pd.read_csv(CANONICAL_WHOLE_COHORT_PATH)
    canon = canon[canon["comparison"] == "D - C"]
    for _, r in diff_df[diff_df["subgroup"] == "all"].iterrows():
        c = canon[(canon["task"] == r["task"]) & (canon["metric"] == r["metric"])]
        assert len(c) == 1, f"canonical D - C row missing for {r['task']}/{r['metric']}"
        c = c.iloc[0]
        assert np.isclose(r["observed_diff"], c["observed_diff"], atol=1e-10), \
            f"{r['task']}/{r['metric']}: whole-cohort D-C {r['observed_diff']} != canonical {c['observed_diff']}"
        assert (r["n_rows"], r["n_patients"]) == (c["n_rows"], c["n_patients"])
    checks["whole_cohort_matches_canonical_rna_subset_paired_differences"] = True

    # 2. complements partition the analysis set (treatment line: minus missing line rows)
    for task in TASKS:
        n = diff_df[(diff_df["task"] == task)].groupby("subgroup")["n_rows"].first()
        assert n["deep_responders"] + n["non_deep_responders"] == n["all"], f"{task}: deep split not a partition"
        assert n["frontline"] + n["later_line"] <= n["all"], f"{task}: line groups exceed analysis set"
    checks["complements_partition_analysis_set"] = True

    # 3. counts report agrees with the bootstrap's analysis set
    counts = counts_df.set_index(["task", "subgroup"])
    for _, r in diff_df.iterrows():
        assert counts.loc[(r["task"], r["subgroup"]), "n_rows_test_rna_available"] == r["n_rows"], \
            f"{r['task']}/{r['subgroup']}: counts report disagrees with bootstrap analysis set"
    checks["counts_report_consistent_with_bootstrap"] = True

    # 4. every bootstrap has a valid CI and reports skipped replicates
    for frame in (diff_df, inter_df):
        assert frame[["ci_lower", "ci_upper"]].notna().all().all(), "missing CI"
        assert (frame["n_boot_valid"] + frame["n_boot_skipped"] == N_BOOTSTRAP).all()
    checks["all_cis_defined_and_skips_reported"] = True

    # 5. only the primary C-vs-D XGBoost comparison, fixed metric set
    assert set(diff_df["comparison"]) == {f"D - C ({PRIMARY_ALGORITHM})"}
    for task in TASKS:
        assert set(diff_df.loc[diff_df["task"] == task, "metric"]) == set(TASK_METRICS[task])
    checks["primary_comparison_is_c_vs_d_xgboost_only"] = True

    # 6. secondary CV: expected groups/metrics, no CI columns
    assert not [c for c in cv_agg_df.columns if "ci" in c.lower().split("_")], "CV aggregate must not carry CIs"
    expected = {(s, t, m) for s in SUBGROUPS for t in TASKS for m in TASK_METRICS[t]}
    assert set(map(tuple, cv_agg_df[["subgroup", "task", "metric"]].values)) == expected, \
        "CV aggregate does not cover every subgroup x task x metric"
    assert set(cv_fold_df["repeat"]) == set(range(N_REPEATS)) and set(cv_fold_df["fold"]) <= set(range(N_FOLDS))
    checks["secondary_cv_complete_without_fold_bootstrap_ci"] = True

    # 7. subgroup membership re-derived on the master data matches the analysis-set counts
    rna_test = df[(df["split"] == "test") & df["days_since_rna_sample"].notna()]
    assert int(get_subgroup_mask(rna_test, "deep_responders").sum()) == \
        int(counts.loc[("multiclass", "deep_responders"), "n_rows_test_rna_available"])
    checks["membership_rederivation_consistent"] = True
    return checks


def _records(frame, cols):
    return json.loads(frame[cols].to_json(orient="records", double_precision=6))


if __name__ == "__main__":
    check_required_files()
    hashes_before = {p: sha256(p) for p in locked_input_paths()}
    os.makedirs(SUBGROUP_OUT_DIR, exist_ok=True)

    df = load_master_data()
    counts_df = build_counts_report(df)
    diff_df, inter_df, alignment_reports = run_primary_analysis(df)
    print_results(diff_df, inter_df)
    cv_fold_df, cv_agg_df = pd.read_csv(PER_FOLD_PATH), pd.read_csv(AGGREGATE_PATH)

    checks = run_checks(df, counts_df, diff_df, inter_df, cv_fold_df, cv_agg_df)
    hashes_after = {p: sha256(p) for p in locked_input_paths()}
    assert hashes_before == hashes_after, "A locked input changed during the run"
    checks["locked_inputs_unchanged_during_run"] = True

    counts_df.to_csv(COUNTS_PATH, index=False)
    diff_df.to_csv(DIFFERENCES_PATH, index=False)
    inter_df.to_csv(INTERACTIONS_PATH, index=False)

    n_subgroup_cis = int((diff_df["subgroup"] != "all").sum())
    summary = {
        "description": ("A priori, hypothesis-driven subgroup analysis of the incremental value of RNA "
                        "(Model D = Model C + RNA) for predicting next-visit response."),
        "hypotheses": {
            "H1_deep_responders": {
                "definition": f"vt_disease_response in {DEEP_RESPONSE_CATEGORIES}; complement = other non-missing Vt responses",
                "binary_note": "eligible_for_binary already excludes Vt = sCR",
                "test": "(D-C)_deep - (D-C)_non_deep"},
            "H2_treatment_line": {
                "definition": "frontline: current_line_number == 1; later_line: current_line_number >= 2",
                "test": "(D-C)_frontline - (D-C)_later_line"},
        },
        "membership_fields": sorted(set(SUBGROUP_FIELDS.values())) + ["eligible_for_binary (task filter)"],
        "primary_analysis": {
            "design": ("Locked clean-rerun whole-cohort test predictions, no retraining; paired patient-level "
                       "bootstrap; interactions use a joint patient bootstrap over both groups."),
            "comparison": f"{RNA_STAGE.upper()}-{PRIMARY_ALGORITHM} minus {COMPARATOR_STAGE.upper()}-{PRIMARY_ALGORITHM}",
            "analysis_set": "test split, task-eligible, RNA-available (days_since_rna_sample notna)",
            "n_bootstrap": N_BOOTSTRAP, "seed": RANDOM_SEED,
            "multiclass_macro_f1": "fixed 6-class MULTICLASS_CLASSES label set",
            "alignment": alignment_reports,
            "subgroup_differences": _records(diff_df, [
                "task", "subgroup", "metric", "n_rows", "n_patients", "C", "D",
                "observed_diff", "ci_lower", "ci_upper", "n_boot_skipped"]),
            "interactions": _records(inter_df, [
                "task", "metric", "group_a", "group_b", "diff_a", "diff_b", "observed_interaction",
                "ci_lower", "ci_upper", "n_boot_skipped", "n_patients_in_both"]),
        },
        "secondary_repeated_cv": {
            "design": (f"{N_FOLDS}-fold x {N_REPEATS}-repeat patient-grouped CV on subgroup train+val rows; "
                       "C/D RETRAINED within subgroup with locked best_params (different estimand from primary); "
                       "evaluated on RNA-available held-out patients."),
            "aggregate": _records(cv_agg_df, [
                "subgroup", "task", "metric", "n_fold_replicates", "mean_D_minus_C",
                "sd_D_minus_C", "min_D_minus_C", "max_D_minus_C", "frac_folds_D_better"]),
            "note": "Mean/SD over non-independent fold replicates; no CI or significance test by design.",
        },
        "multiplicity": {
            "n_interaction_cis": int(len(inter_df)), "n_within_subgroup_cis": n_subgroup_cis,
            "adjustment": "none; 95% CIs are unadjusted. The interaction CIs are the hypothesis tests; "
                          "within-subgroup CIs are descriptive.",
        },
        "checks": checks,
        "locked_input_sha256": hashes_after,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    for p in (COUNTS_PATH, DIFFERENCES_PATH, INTERACTIONS_PATH, PER_FOLD_PATH, AGGREGATE_PATH, SUMMARY_PATH):
        print(f"Saved/verified: {p}")
