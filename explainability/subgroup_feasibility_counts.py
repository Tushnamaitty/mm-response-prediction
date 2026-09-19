"""
Subgroup feasibility check ONLY - eligible patient counts for the two a
priori, hypothesis-driven subgroups (and complements), by task, split,
and RNA availability. NO modeling, NO performance analysis. Subgroup
membership comes from the shared subgroup_analysis_common filter chain.

Deep responders (vt_disease_response in VGPR/CR/sCR): for the binary
task the existing eligible_for_binary rule already excludes Vt=sCR
pairs (sCR cannot improve further), so binary counts show sCR rows
dropped by that rule, not by anything new added here.

Frontline (current_line_number == 1) vs later-line (>= 2).

Feasibility rule of thumb for the secondary repeated grouped CV: each
of 5 folds evaluates on that fold's RNA-available held-out train+val
patients. >=150 such patients (~30/fold) is workable; 80-150 is
marginal; <80 is too noisy to interpret.
"""

import os

import pandas as pd

from explainability.subgroup_analysis_common import (
    load_master_data, build_subgroup_frame, SUBGROUPS, TASKS, STALENESS_COL, CLEAN_RESULTS_DIR,
)

OUT_PATH = f"{CLEAN_RESULTS_DIR}/subgroup_feasibility_counts.csv"
CV_FEASIBLE_THRESHOLD = 150
CV_MARGINAL_THRESHOLD = 80


def describe_subgroup(sub, task, subgroup_name):
    rna = sub[sub[STALENESS_COL].notna()]

    def n_patients(frame, splits):
        return int(frame.loc[frame["split"].isin(splits), "public_id"].nunique())

    trainval_rna_patients = n_patients(rna, ["train", "val"])
    if trainval_rna_patients >= CV_FEASIBLE_THRESHOLD:
        cv_verdict = "FEASIBLE"
    elif trainval_rna_patients >= CV_MARGINAL_THRESHOLD:
        cv_verdict = "MARGINAL (feasible but noisy - report with caveat)"
    else:
        cv_verdict = "NOT FEASIBLE (single-split bootstrap only)"

    return {
        "subgroup": subgroup_name, "task": task,
        "trainval_patients": n_patients(sub, ["train", "val"]),
        "trainval_rna_patients": trainval_rna_patients,
        "test_patients": n_patients(sub, ["test"]),
        "test_rna_patients": n_patients(rna, ["test"]),
        "cv_feasibility": cv_verdict,
    }


if __name__ == "__main__":
    df = load_master_data()
    results = [describe_subgroup(build_subgroup_frame(df, name, task), task, name)
               for task in TASKS for name in SUBGROUPS]
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    os.makedirs(CLEAN_RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
