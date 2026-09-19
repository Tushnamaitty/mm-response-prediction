"""
Subgroup analysis - File 2 of 5: row/patient/outcome count reporting and
runtime leakage checks for both a priori subgroup hypotheses (and their
complements), across both tasks.

"all" is the whole task-eligible cohort, for reference. The primary
analysis set (File 3) is the test split restricted to RNA-available
rows (days_since_rna_sample notna), the same convention as
rna_subset_paired_analysis_clean_rerun.py; its outcome prevalence /
distribution is reported here.

Does NOT modify or retrain any canonical model. Read-only.

Outputs (under artifacts/results_clean_rerun/subgroup_analysis/):
- subgroup_counts_report.csv
"""

import os

import pandas as pd

from explainability.subgroup_analysis_common import (
    load_master_data, build_subgroup_frame, apply_task_filter,
    assert_no_leakage_in_subgroup_definitions, outcome_summary,
    SUBGROUPS, TASKS, STALENESS_COL, BINARY_TARGET, MULTICLASS_TARGET, SUBGROUP_OUT_DIR,
)

OUT_PATH = f"{SUBGROUP_OUT_DIR}/subgroup_counts_report.csv"


def count_block(sub, task, subgroup_name):
    report = {"subgroup": subgroup_name, "task": task,
              "n_rows_total": int(len(sub)), "n_patients_total": int(sub["public_id"].nunique())}
    for split in ["train", "val", "test"]:
        split_df = sub[sub["split"] == split]
        rna_df = split_df[split_df[STALENESS_COL].notna()]
        report[f"n_rows_{split}"] = int(len(split_df))
        report[f"n_patients_{split}"] = int(split_df["public_id"].nunique())
        report[f"n_rows_{split}_rna_available"] = int(len(rna_df))
        report[f"n_patients_{split}_rna_available"] = int(rna_df["public_id"].nunique())

    analysis = sub[(sub["split"] == "test") & sub[STALENESS_COL].notna()]
    target = BINARY_TARGET if task == "binary" else MULTICLASS_TARGET
    report.update({f"analysis_set_{k}": v for k, v in outcome_summary(analysis[target], task).items()})
    return report


def build_counts_report(df):
    assert_no_leakage_in_subgroup_definitions(df)
    reports = []
    for task in TASKS:
        reports.append(count_block(apply_task_filter(df, task), task, "all"))
        for name in SUBGROUPS:
            reports.append(count_block(build_subgroup_frame(df, name, task), task, name))
    return pd.DataFrame(reports)


if __name__ == "__main__":
    os.makedirs(SUBGROUP_OUT_DIR, exist_ok=True)
    report_df = build_counts_report(load_master_data())
    for _, r in report_df.iterrows():
        print(f"{r['subgroup']:20s} {r['task']:10s} total {r['n_rows_total']:5d} rows / "
              f"{r['n_patients_total']:4d} pts | test RNA-avail {r['n_rows_test_rna_available']:4d} rows / "
              f"{r['n_patients_test_rna_available']:3d} pts")
    report_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
