"""
Add diagnosis-time baseline features to every visit pair, using
diagnosis_deid.csv.

Input:  data/clinical/visit_pairs_with_treatment.csv  (last output before
        the naming convention switches to one evolving master file)
Output: data/clinical/visit_pairs_master.csv   <- all subsequent scripts
        read AND overwrite this same file, adding more columns each time.

These are PATIENT-LEVEL CONSTANTS (fixed at diagnosis, before any Vt), so
this is a simple merge on public_id - no time-based matching needed, and
no leakage risk since diagnosis necessarily precedes every visit pair.

55 patients have more than one diagnosis_deid.csv row (e.g. a second
primary/reclassification). For those, we prefer the row flagged
diagnosis_is_primary_disease == True; if none is flagged, we take the row
with the earliest days_to_diagnosis; if that's also missing, we just take
the first row - so every patient ends up with exactly one diagnosis record.

Explicitly EXCLUDED from diagnosis_deid.csv (whole-course summary outcomes,
not baseline facts - including these would leak future information):
    best_overall_response, days_to_best_overall_response,
    days_to_last_followup, last_known_disease_status,
    days_to_last_known_disease_status, first_progressive_disease,
    days_to_first_progressive_disease
Also excluded as near-constant/uninformative across this MM-only cohort:
    morphology, primary_diagnosis, site_of_resection_or_biopsy,
    tissue_or_organ_of_origin, classification_of_tumor, icd_10_code,
    method_of_diagnosis, diagnosis_is_primary_disease (used only for
    row-selection above, not kept as a feature)
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

BASELINE_COLS = [
    "public_id",
    "iss_stage",
    "r_iss_stage",
    "age_at_diagnosis",
    "height_at_diagnosis_m",
    "weight_at_diagnosis_kg",
    "bmi_at_diagnosis",
]


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_treatment.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs


def load_diagnosis_baseline() -> pd.DataFrame:
    dx = pd.read_csv(f"{CLINICAL_DIR}/diagnosis_deid.csv")

    # Normalize the primary-disease flag (read as string "TRUE"/"FALSE" in
    # the raw CSV) to a real boolean for sorting.
    dx["is_primary"] = dx["diagnosis_is_primary_disease"].astype(str).str.upper() == "TRUE"

    # Pick one row per patient: primary flag first, then earliest diagnosis
    # date, then just whatever comes first as a final fallback.
    dx = dx.sort_values(
        ["public_id", "is_primary", "days_to_diagnosis"],
        ascending=[True, False, True],
    )
    dx_one_per_patient = dx.drop_duplicates(subset="public_id", keep="first")

    # Clean iss_stage/r_iss_stage: keep numeric stages as-is, treat
    # "not_reported" as missing rather than a fourth category.
    for col in ["iss_stage", "r_iss_stage"]:
        dx_one_per_patient[col] = dx_one_per_patient[col].replace(
            "not_reported", pd.NA
        )

    return dx_one_per_patient[BASELINE_COLS]


def summarize(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print(" Diagnosis-time baseline features ")
    print(f"Rows before merge: {len(before)} | after merge: {len(after)}")
    if len(before) != len(after):
        print(
            "WARNING: row count changed after merge - check for duplicate "
            "public_id values in the diagnosis baseline table."
        )
    for col in BASELINE_COLS[1:]:
        pct = after[col].notna().mean() * 100
        print(f"{col:25s} populated: {pct:5.1f}%")
    print()
    print("iss_stage distribution:")
    print(after["iss_stage"].value_counts(dropna=False))


if __name__ == "__main__":
    pairs = load_pairs()
    dx_baseline = load_diagnosis_baseline()

    result = pairs.merge(dx_baseline, on="public_id", how="left")

    summarize(pairs, result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")