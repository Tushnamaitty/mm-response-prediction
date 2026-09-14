"""
Step 11: Leakage-safe patient-level train/validation/test split.

Input:  data/clinical/visit_pairs_with_rna.csv
        data_pipeline/model_feature_sets.json  (used only to confirm the
            target column names - NOT modified, no feature sets altered)
Output: data/splits/patient_split_assignments.csv  (public_id, split)
        data/splits/visit_pair_splits.csv          (pair_id, public_id, split)

ONE split is produced and reused for all of Models A-D - no per-model
splitting, since the whole point of the shared master table is that every
model must be evaluated on identical patients/rows.

STRATIFICATION DECISION (see conversation for full reasoning):
Patients are split by public_id (never by row), stratified on a SINGLE
variable - whether the patient has >=1 RNA-matched pair (707 yes / 318 no).
This is the variable that matters MECHANICALLY (Model D needs adequate
RNA-covered patients in every split, and this doesn't reliably self-balance
by chance the way a target rate does). "Has >=1 improved pair" was
considered and rejected as a stratifier: 872/1025 patients (85%) have at
least one improved pair regardless of their true underlying rate (with
~13 pairs/patient, even a 17% per-pair rate means most patients get at
least one success by chance) - stratifying on it would not meaningfully
protect the row-level target balance and risks tiny/unstable strata if
combined with RNA status. Target distributions are VALIDATED after the
split (see checks below), not forced via multi-way stratification.

Split ratio: ~70/15/15, done independently within each RNA-status group
then combined, using a fixed random seed for reproducibility.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLINICAL_DIR = "data/clinical"
SPLITS_DIR = "data/splits"
FEATURE_SETS_PATH = "data_pipeline/model_feature_sets.json"

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def load_data() -> pd.DataFrame:
    return pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)


def get_target_names() -> tuple[str, str]:
    with open(FEATURE_SETS_PATH) as f:
        feature_sets = json.load(f)
    targets = feature_sets["targets"]
    binary_target = "improved"
    multiclass_target = "exact_next_response"
    assert binary_target in targets and multiclass_target in targets
    return binary_target, multiclass_target


def build_patient_level_table(df: pd.DataFrame, rna_indicator_col: str) -> pd.DataFrame:
    """One row per patient, with has_rna computed from whether ANY of
    that patient's pairs has a non-null value in a pathway-score column."""
    patient_df = df.groupby("public_id").agg(
        n_pairs=("pair_id", "count"),
        has_rna=(rna_indicator_col, lambda s: s.notna().any()),
    ).reset_index()
    return patient_df


def split_patients(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Stratify on has_rna only (see module docstring for reasoning).
    Split each RNA-status group independently at 70/15/15, then combine."""
    assignments = []

    for rna_status, group in patient_df.groupby("has_rna"):
        ids = group["public_id"].values

        # First split off the test set.
        train_val_ids, test_ids = train_test_split(
            ids, test_size=TEST_FRAC, random_state=RANDOM_SEED
        )
        # Then split the remainder into train/val, preserving the
        # relative proportion (VAL_FRAC of the ORIGINAL group).
        val_relative_frac = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_relative_frac, random_state=RANDOM_SEED
        )

        for pid in train_ids:
            assignments.append({"public_id": pid, "split": "train"})
        for pid in val_ids:
            assignments.append({"public_id": pid, "split": "val"})
        for pid in test_ids:
            assignments.append({"public_id": pid, "split": "test"})

    return pd.DataFrame(assignments)


def validate(
    df: pd.DataFrame,
    split_assignments: pd.DataFrame,
    patient_df: pd.DataFrame,
    binary_target: str,
    multiclass_target: str,
    rna_indicator_col: str,
) -> pd.DataFrame:
    """Every check specified in the requirements. Fails loudly (raises)
    on any real problem; prints distributional reports either way."""

    merged = df.merge(split_assignments, on="public_id", how="left")

    print(" Step 11: Patient-level split validation \n")

    # --- Check: every patient assigned exactly once ---
    n_patients_total = df["public_id"].nunique()
    n_patients_assigned = split_assignments["public_id"].nunique()
    dupe_assignments = split_assignments["public_id"].duplicated().sum()
    if dupe_assignments > 0:
        raise ValueError(f"{dupe_assignments} patients assigned to more than one split.")
    if n_patients_assigned != n_patients_total:
        raise ValueError(
            f"Patient count mismatch: {n_patients_total} in data, "
            f"{n_patients_assigned} assigned."
        )
    print(f"Patients: {n_patients_total} total, {n_patients_assigned} assigned exactly once. OK")

    # --- Check: zero patient overlap between splits ---
    split_sets = {
        s: set(split_assignments.loc[split_assignments["split"] == s, "public_id"])
        for s in ["train", "val", "test"]
    }
    for s1 in ["train", "val", "test"]:
        for s2 in ["train", "val", "test"]:
            if s1 < s2:
                overlap = split_sets[s1] & split_sets[s2]
                if overlap:
                    raise ValueError(f"Patient overlap between {s1} and {s2}: {overlap}")
    print("Zero patient overlap between train/val/test. OK\n")

    # --- Check: all rows assigned, no rows lost/duplicated, no unassigned rows ---
    if len(merged) != len(df):
        raise ValueError(f"Row count changed after merge: {len(df)} -> {len(merged)}")
    if merged["split"].isna().any():
        n_missing = merged["split"].isna().sum()
        raise ValueError(f"{n_missing} rows have no split assignment (orphaned public_id).")
    print(f"Rows: {len(df)} total, {len(merged)} after merge, 0 unassigned. OK")

    # --- Check: no duplicate pair_id ---
    dupe_pairs = merged["pair_id"].duplicated().sum()
    if dupe_pairs > 0:
        raise ValueError(f"{dupe_pairs} duplicate pair_id found after split merge.")
    print("Zero duplicate pair_id. OK\n")

    # --- Report: patient counts per split ---
    print("--- Patient counts per split ---")
    patient_counts = split_assignments["split"].value_counts()
    for s in ["train", "val", "test"]:
        pct = patient_counts.get(s, 0) / n_patients_total * 100
        print(f"  {s:6s}: {patient_counts.get(s, 0):4d} patients ({pct:.1f}%)")

    # Sanity check: split proportions not badly off from target ratios.
    target_fracs = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}
    for s, target_frac in target_fracs.items():
        actual_frac = patient_counts.get(s, 0) / n_patients_total
        if abs(actual_frac - target_frac) > 0.05:
            raise ValueError(
                f"Split '{s}' proportion {actual_frac:.3f} is badly off from "
                f"target {target_frac:.3f} (>5pp difference)."
            )
    print("  Proportions within 5pp of target 70/15/15. OK\n")

    # --- Report: row counts per split ---
    print("--- Row counts per split ---")
    row_counts = merged["split"].value_counts()
    for s in ["train", "val", "test"]:
        pct = row_counts.get(s, 0) / len(merged) * 100
        print(f"  {s:6s}: {row_counts.get(s, 0):5d} rows ({pct:.1f}%)")
    print()

    # --- Report: mean/median pairs per patient per split ---
    print("--- Pairs per patient per split ---")
    for s in ["train", "val", "test"]:
        sub = merged[merged["split"] == s]
        per_patient = sub.groupby("public_id").size()
        print(f"  {s:6s}: mean={per_patient.mean():.1f}, median={per_patient.median():.1f}")
    print()

    # --- Report: binary target distribution per split ---
    print("--- Binary target ('improved') distribution per split ---")
    overall_rate = df[binary_target].mean()
    print(f"  Overall (all data): {overall_rate*100:.1f}% improved")
    for s in ["train", "val", "test"]:
        sub = merged[merged["split"] == s]
        rate = sub[binary_target].mean()
        print(f"  {s:6s}: {rate*100:.1f}% improved (n={len(sub)})")
        if abs(rate - overall_rate) > 0.05:
            print(f"    WARNING: >5pp deviation from overall rate - review manually.")
    print()

    # --- Report: multiclass target distribution per split ---
    print("--- Multiclass target ('exact_next_response') distribution per split ---")
    overall_dist = df[multiclass_target].value_counts(normalize=True)
    for s in ["train", "val", "test"]:
        sub = merged[merged["split"] == s]
        sub_dist = sub[multiclass_target].value_counts(normalize=True)
        missing_classes = set(overall_dist.index) - set(sub_dist.index)
        if missing_classes:
            raise ValueError(
                f"Split '{s}' is missing target class(es) entirely: {missing_classes}"
            )
        print(f"  {s}:")
        for cls in overall_dist.index:
            print(f"    {cls:32s} overall={overall_dist[cls]*100:5.1f}% | {s}={sub_dist.get(cls, 0)*100:5.1f}%")
    print()

    # --- Report: RNA patient counts and row coverage per split ---
    print("--- RNA coverage per split ---")
    for s in ["train", "val", "test"]:
        sub_patients = split_assignments[split_assignments["split"] == s]["public_id"]
        sub_patient_df = patient_df[patient_df["public_id"].isin(sub_patients)]
        n_rna_patients = sub_patient_df["has_rna"].sum()
        n_total_patients = len(sub_patient_df)

        sub_rows = merged[merged["split"] == s]
        pct_rows_with_rna = sub_rows[rna_indicator_col].notna().mean() * 100

        print(
            f"  {s:6s}: {n_rna_patients}/{n_total_patients} patients have RNA "
            f"({n_rna_patients/n_total_patients*100:.1f}%) | "
            f"{pct_rows_with_rna:.1f}% of rows have a matched RNA sample"
        )
    print()

    return merged


if __name__ == "__main__":
    import os
    os.makedirs(SPLITS_DIR, exist_ok=True)

    df = load_data()
    binary_target, multiclass_target = get_target_names()
    rna_indicator_col = "Adipogenesis"  # any pathway column works as an indicator

    patient_df = build_patient_level_table(df, rna_indicator_col)
    split_assignments = split_patients(patient_df)

    merged = validate(df, split_assignments, patient_df, binary_target, multiclass_target, rna_indicator_col)

    split_assignments.to_csv(f"{SPLITS_DIR}/patient_split_assignments.csv", index=False)
    merged[["pair_id", "public_id", "split"]].to_csv(
        f"{SPLITS_DIR}/visit_pair_splits.csv", index=False
    )

    print(" STEP 11 VERIFIED ")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Saved: {SPLITS_DIR}/patient_split_assignments.csv")
    print(f"Saved: {SPLITS_DIR}/visit_pair_splits.csv")