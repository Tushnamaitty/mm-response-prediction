"""
Steps 2-4: Define the eligible cohort, build consecutive visit pairs,
and attach both prediction targets.

Input:  data/clinical/patient_timelines.csv  (output of Step 1)
Output: data/clinical/visit_pairs.csv

--- Step 2: Eligible cohort ---
Keep patients with >=2 visits flagged has_valid_response=True (from Step 1),
since only those visits can serve as pairing anchors. Audit missingness,
visit-gap variability, and RNA-seq availability while we're at it.

--- Step 3: Consecutive visit pairs ---
Within each eligible patient, take ONLY the valid-response visits (ignore
visits without a usable response, e.g. screening rows), sort by
days_to_visit, and pair each one with the next: (V_t -> V_t+1). Vt supplies
predictors, Vt+1 supplies the outcome. Record the time gap between them.

--- Step 4: Prediction targets ---
Task 1 (binary):    Improved (1) vs Not Improved (0), based on whether the
                     response at Vt+1 is ordinally better than at Vt.
Task 2 (multiclass): the exact disease_response at Vt+1.

Data-quality fix applied here: "stringent_complete_response complete_response"
(a malformed/duplicated label, ~1,300 rows) is collapsed to
"stringent_complete_response" before ranking.
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

# IMWG response scale, worst -> best. Used to determine "Improved" vs
# "Not Improved" and to catch/clean the malformed label.
RESPONSE_RANK = {
    "progressive_disease": 0,
    "stable_disease": 1,
    "partial_response": 2,
    "very_good_partial_response": 3,
    "complete_response": 4,
    "stringent_complete_response": 5,
}


def to_numeric_or_nan(val) -> float:
    """Convert a single scalar to float, returning NaN for anything that
    isn't a valid number (e.g. the string "not_reported"). Used instead
    of pd.to_numeric() here since that's meant for Series/array-like
    input - calling it on a single dict value works fine at runtime but
    doesn't match any of its declared type-checker overloads, which was
    flagging a (harmless but noisy) static-analysis warning."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def clean_response_label(x: str) -> str:
    """Fix the known malformed label before ranking."""
    if x == "stringent_complete_response complete_response":
        return "stringent_complete_response"
    return x


def load_timelines() -> pd.DataFrame:
    df = pd.read_csv(f"{CLINICAL_DIR}/patient_timelines.csv")
    df["disease_response"] = df["disease_response"].apply(clean_response_label)
    return df


def define_eligible_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: keep only patients with >=2 valid-response visits."""
    valid_counts = df.groupby("public_id")["has_valid_response"].sum()
    eligible_ids = valid_counts[valid_counts >= 2].index

    eligible = df[df["public_id"].isin(eligible_ids)].copy()

    # --- audit ---
    n_total_patients = df["public_id"].nunique()
    n_eligible = len(eligible_ids)
    print(" Step 2: Eligible cohort audit ")
    print(f"Total patients: {n_total_patients}")
    print(f"Eligible patients (>=2 valid response visits): {n_eligible}")
    print(f"Excluded: {n_total_patients - n_eligible}")

    return eligible


def build_visit_pairs(eligible: pd.DataFrame) -> pd.DataFrame:
    """Step 3: form consecutive valid-response visit pairs per patient."""
    valid_only = eligible[eligible["has_valid_response"]].sort_values(
        ["public_id", "days_to_visit"]
    )

    pairs = []
    for public_id, group in valid_only.groupby("public_id"):
        rows = group.to_dict("records")
        for i in range(len(rows) - 1):
            vt = rows[i]
            vt1 = rows[i + 1]
            pairs.append(
                {
                    "public_id": public_id,
                    "vt_study_visit": vt["study_visit"],
                    "vt_days_to_visit": vt["days_to_visit"],
                    "vt_disease_response": vt["disease_response"],
                    "vt_ecog_ps_numeric": to_numeric_or_nan(vt.get("ecog_ps_numeric")),
                    "vt_reported_signs_symptoms": vt.get("reported_signs_symptoms") == "yes",
                    "vt_reported_adverse_event": vt.get("reported_adverse_event") == "yes",
                    "vt_reported_clinical_event": vt.get("reported_clinical_event") == "yes",
                    "vt1_study_visit": vt1["study_visit"],
                    "vt1_days_to_visit": vt1["days_to_visit"],
                    "vt1_disease_response": vt1["disease_response"],
                    "time_gap_days": vt1["days_to_visit"] - vt["days_to_visit"],
                }
            )

    pairs_df = pd.DataFrame(pairs)

    print("\n Step 3: Visit pairs ")
    print(f"Total Vt -> Vt+1 pairs (before cleaning): {len(pairs_df)}")

    # Drop zero/negative time-gap pairs. Investigation confirmed these are
    # duplicate same-day followup records (e.g. a preliminary vs. confirmed
    # assessment of the same clinical visit), not genuine Vt -> Vt+1
    # transitions - keeping them would inject fake "instant improvement"
    # examples with time_gap_days == 0 into the training data.
    n_before = len(pairs_df)
    bad_gap = pairs_df["time_gap_days"] <= 0
    if bad_gap.sum() > 0:
        print(
            f"Dropping {bad_gap.sum()} pairs with zero/negative time gap "
            f"(duplicate same-day records, see project log for investigation)"
        )
        pairs_df = pairs_df[~bad_gap].reset_index(drop=True)

    print(f"Total Vt -> Vt+1 pairs (after cleaning): {len(pairs_df)}")
    print(f"Patients contributing pairs: {pairs_df['public_id'].nunique()}")
    print(
        f"Time gap (days) - median: {pairs_df['time_gap_days'].median():.0f}, "
        f"min: {pairs_df['time_gap_days'].min():.0f}, "
        f"max: {pairs_df['time_gap_days'].max():.0f}"
    )

    return pairs_df


def add_prediction_targets(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: attach the Improved/Not-Improved and exact-response targets."""
    pairs_df = pairs_df.copy()

    vt_rank = pairs_df["vt_disease_response"].map(RESPONSE_RANK)
    vt1_rank = pairs_df["vt1_disease_response"].map(RESPONSE_RANK)

    # Safety check: every response here should be a known, valid category
    # (invalid ones were already excluded via has_valid_response in Step 2/3).
    unmapped = pairs_df[vt_rank.isna() | vt1_rank.isna()]
    if len(unmapped) > 0:
        print(
            f"\nWARNING: {len(unmapped)} pairs have an unrecognized response "
            f"label not in RESPONSE_RANK - inspect these before proceeding:"
        )
        print(
            unmapped[["public_id", "vt_disease_response", "vt1_disease_response"]]
            .drop_duplicates()
            .to_string()
        )

    pairs_df["improved"] = (vt1_rank > vt_rank).astype("Int64")
    pairs_df.loc[vt_rank.isna() | vt1_rank.isna(), "improved"] = pd.NA

    # Task 2 target is just the cleaned vt1_disease_response column itself.
    pairs_df["exact_next_response"] = pairs_df["vt1_disease_response"]

    print("\n Step 4: Prediction targets ")
    print("Improved / Not Improved distribution:")
    print(pairs_df["improved"].value_counts(dropna=False))
    print("\nExact next response distribution:")
    print(pairs_df["exact_next_response"].value_counts(dropna=False))

    return pairs_df


if __name__ == "__main__":
    timelines = load_timelines()
    eligible = define_eligible_cohort(timelines)
    pairs = build_visit_pairs(eligible)
    pairs = add_prediction_targets(pairs)

    out_path = f"{CLINICAL_DIR}/visit_pairs.csv"
    pairs.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")