"""
Step 1: Reconstruct each patient's timeline.

Loads subject_deid.csv and followup_deid.csv, converts visit timing onto a
single days-from-baseline scale (days_to_visit is already relative to
index_date, so we use it directly), sorts each patient's visits
chronologically, and flags which visits have a VALID disease_response
(i.e. can serve as a "visit anchor" for Vt -> Vt+1 pairing later in Step 3).

Output: data/clinical/patient_timelines.csv
  One row per (patient, visit), sorted by public_id then days_to_visit,
  with an added `has_valid_response` boolean column.
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

# Responses that do NOT count as a valid assessment (per our data-quality
# check from the clinical verification step).
INVALID_RESPONSES = {"not_reported", "not_evaluable", ""}


def load_subjects() -> pd.DataFrame:
    subjects = pd.read_csv(f"{CLINICAL_DIR}/subject_deid.csv")
    # Keep just the identifying / demographic columns we need downstream.
    keep_cols = [
        "public_id",
        "race",
        "ethnicity",
        "gender",
        "vital_status",
        "age_at_index",
        "days_to_death",
    ]
    return subjects[keep_cols]


def load_followups() -> pd.DataFrame:
    followups = pd.read_csv(f"{CLINICAL_DIR}/followup_deid.csv")

    # Normalise disease_response for comparison (lowercase, strip whitespace).
    followups["disease_response"] = (
        followups["disease_response"].astype(str).str.strip().str.lower()
    )

    followups["has_valid_response"] = ~followups["disease_response"].isin(
        INVALID_RESPONSES
    )

    # A handful of rows (mostly "unscheduled" visits) are missing
    # days_to_visit. Where the row still has a genuinely valid response,
    # recover the timestamp from days_to_disease_response_assessment
    # instead of dropping the row outright. Rows that are missing
    # days_to_visit AND have no valid response are safe to drop later
    # (Step 2/3), since they were never usable anyway.
    missing_visit_day = followups["days_to_visit"].isna()
    recoverable = missing_visit_day & followups["has_valid_response"]
    followups.loc[recoverable, "days_to_visit"] = followups.loc[
        recoverable, "days_to_disease_response_assessment"
    ]

    return followups


def build_timelines() -> pd.DataFrame:
    subjects = load_subjects()
    followups = load_followups()

    # Attach subject-level info to every visit row.
    timeline = followups.merge(subjects, on="public_id", how="left")

    # Sort chronologically within each patient. days_to_visit is already
    # relative to index_date (first treatment), so it's a valid single
    # time axis across the whole cohort.
    timeline = timeline.sort_values(["public_id", "days_to_visit"]).reset_index(
        drop=True
    )

    return timeline


def summarize(timeline: pd.DataFrame) -> None:
    n_patients = timeline["public_id"].nunique()
    n_visits = len(timeline)
    avg_visits = n_visits / n_patients

    valid_counts = timeline.groupby("public_id")["has_valid_response"].sum()
    n_eligible = (valid_counts >= 2).sum()

    print(f"Patients: {n_patients}")
    print(f"Total visit rows: {n_visits}")
    print(f"Avg visits/patient: {avg_visits:.1f}")
    print(
        f"Patients with >=2 valid response assessments (eligible for pairing): "
        f"{n_eligible} / {n_patients}"
    )
    print(f"Missing days_to_visit: {timeline['days_to_visit'].isna().sum()}")


if __name__ == "__main__":
    timeline = build_timelines()
    summarize(timeline)

    out_path = f"{CLINICAL_DIR}/patient_timelines.csv"
    timeline.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")