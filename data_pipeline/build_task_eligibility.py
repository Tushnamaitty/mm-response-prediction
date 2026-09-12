"""
Task eligibility: marks which pairs are usable for the BINARY
("improved") task vs. the MULTICLASS ("exact_next_response") task.

Why this exists: pairs where Vt = stringent_complete_response (sCR) are
STRUCTURALLY unable to ever be labeled "improved" - sCR is the top of
the IMWG response hierarchy, so there is no rank above it to improve to.
Including these in the binary task lets a model partly "cheat" by
learning "if already at sCR, predict Not Improved" - which is TRUE but
is an artifact of the target's definition, not a genuine clinical
prediction, and it dilutes the ablation study's ability to detect real
incremental value from temporal/treatment/RNA features.

The multiclass task has NO such ceiling problem - staying at sCR is a
perfectly valid, informative outcome to predict - so sCR pairs are kept
in full for multiclass.

This is a SAMPLE-SELECTION decision for the binary task specifically,
not a feature or preprocessing change:
  - Does NOT touch Step 10 (model_feature_sets.json) - no columns added
    or removed.
  - Does NOT touch Step 11 (patient_split_assignments.csv,
    visit_pair_splits.csv) - no patient moves between splits, no
    resplitting.
  - Does NOT touch Step 12 (preprocessors) - imputation/encoding
    statistics are about FEATURES, not about which rows enter a given
    task's loss function, so they're computed on ALL train rows exactly
    as before. This file is only consulted at Step 13 training time to
    build (X_binary, y_binary) as a ROW SUBSET of the already-preprocessed
    matrices.

Input:  data/clinical/visit_pairs_with_rna.csv
        data/splits/visit_pair_splits.csv
Output: data/splits/task_eligibility.csv  (pair_id, eligible_for_binary,
            eligible_for_multiclass, exclusion_reason)
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"
SPLITS_DIR = "data/splits"

SCR_LABEL = "stringent_complete_response"


def build_task_eligibility() -> pd.DataFrame:
    df = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    split_map = pd.read_csv(f"{SPLITS_DIR}/visit_pair_splits.csv")
    df = df.merge(split_map[["pair_id", "split"]], on="pair_id", how="left")

    is_scr = df["vt_disease_response"] == SCR_LABEL

    result = pd.DataFrame({
        "pair_id": df["pair_id"],
        "public_id": df["public_id"],
        "split": df["split"],
        "eligible_for_binary": ~is_scr,
        "eligible_for_multiclass": True,  # sCR is a valid, informative class here
        "exclusion_reason": is_scr.map({
            True: "vt_disease_response=stringent_complete_response - structurally cannot improve further",
            False: "",
        }),
    })
    return result


def validate(result: pd.DataFrame, original_df: pd.DataFrame) -> None:
    print("=== Task eligibility validation ===\n")

    if len(result) != len(original_df):
        raise ValueError(f"Row count mismatch: {len(result)} vs {len(original_df)}")
    if result["pair_id"].duplicated().sum() > 0:
        raise ValueError("Duplicate pair_id in task eligibility file.")
    print(f"Rows: {len(result)}, matches original table, zero duplicates. OK\n")

    n_excluded = (~result["eligible_for_binary"]).sum()
    n_included = result["eligible_for_binary"].sum()
    print(f"Binary task: {n_included} eligible, {n_excluded} excluded (sCR) "
          f"({n_excluded/len(result)*100:.1f}%)")
    print(f"Multiclass task: {result['eligible_for_multiclass'].sum()} eligible "
          f"(all rows, sCR retained)\n")

    print("Excluded-from-binary count per split:")
    excl_by_split = result[~result["eligible_for_binary"]]["split"].value_counts()
    for s in ["train", "val", "test"]:
        n = excl_by_split.get(s, 0)
        n_total_split = (result["split"] == s).sum()
        print(f"  {s:6s}: {n} excluded / {n_total_split} total "
              f"({n/n_total_split*100:.1f}%)")

    for s in ["train", "val", "test"]:
        n_eligible = ((result["split"] == s) & result["eligible_for_binary"]).sum()
        if n_eligible == 0:
            raise ValueError(f"Split '{s}' has ZERO eligible binary-task rows.")
    print("\nEvery split retains a usable binary-task sample. OK")


if __name__ == "__main__":
    original_df = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    result = build_task_eligibility()
    validate(result, original_df)

    out_path = f"{SPLITS_DIR}/task_eligibility.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")