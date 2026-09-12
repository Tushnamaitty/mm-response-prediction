"""
Add comorbidity (static, patient-level) and clinical event (per-visit)
features, using other_clinical_attributes_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

This file mixes two conceptually different things, handled differently:

1. COMORBIDITIES (nci_comorbidity_item codes) are treated as STATIC,
   PATIENT-LEVEL facts (assumption: chronic, pre-existing conditions
   typically captured at baseline/enrollment intake).

   IMPORTANT DATA-QUALITY FINDING (verified against clinical_event_diagnosis
   text before use): several of these codes are mislabeled relative to
   what they actually contain in this dataset:
     - "chf" is 99% (282/285) actual diagnosis = hypertension, not
       congestive heart failure. Renamed here to has_hypertension_or_chf
       to be honest about its real content.
     - "diabetes" is 100% (6/6) actual diagnosis = "borderline_diabetes"
       (pre-diabetes), not confirmed diabetes. Renamed to
       has_borderline_diabetes.
     - "copd" is mostly asthma/bronchitis, i.e. a broader "chronic
       pulmonary disease" bucket rather than COPD specifically. Renamed
       to has_chronic_pulmonary_disease.
     - "renal", "cva", "mi" were verified to match their labels
       accurately and are kept as-is.
   Do not rename these back to the raw code names without re-verifying
   against clinical_event_diagnosis - the raw NCI code names in this
   source file are not reliable as feature names on their own.

2. CLINICAL EVENTS (clinical_event_type: hosp_admission, ed) are
   genuinely time-varying - treating these as a static "ever" total would
   leak FUTURE hospitalizations into early visit pairs. This file has no
   days_to_* timestamp, only a study_visit label (same limitation as
   adverse_event_deid.csv), so - to stay leakage-safe without assuming a
   day-based ordering the data doesn't actually give us - these are
   matched with an EXACT (public_id, study_visit) join, same pattern as
   adverse_event/supportive_care: "did THIS specific visit have a
   hospitalization/ED event flagged", not a cumulative prior count.

Features added:
  has_hypertension_or_chf, has_chronic_pulmonary_disease,
  has_renal_comorbidity, has_borderline_diabetes, has_cva, has_mi
                              - static, patient-level
  comorbidity_count_total     - static, patient-level: total distinct
                                comorbidity items ever documented
  had_hospitalization         - per-visit, exact study_visit match
  had_ed_visit                - per-visit, exact study_visit match
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

TOP_COMORBIDITIES = {
    "chf": "has_hypertension_or_chf",
    "copd": "has_chronic_pulmonary_disease",
    "renal": "has_renal_comorbidity",
    "diabetes": "has_borderline_diabetes",
    "cva": "has_cva",
    "mi": "has_mi",
}
STATIC_COLS = list(TOP_COMORBIDITIES.values()) + ["comorbidity_count_total"]
EVENT_COLS = ["had_hospitalization", "had_ed_visit"]
NEW_COLS = STATIC_COLS + EVENT_COLS


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs.drop(columns=[c for c in NEW_COLS if c in pairs.columns])


def build_static_comorbidities(oc: pd.DataFrame) -> pd.DataFrame:
    per_patient = oc.groupby("public_id")["nci_comorbidity_item"].apply(
        lambda items: set(items.dropna())
    )
    static = pd.DataFrame(index=per_patient.index)
    for code, feat in TOP_COMORBIDITIES.items():
        static[feat] = per_patient.apply(lambda items: code in items)
    static["comorbidity_count_total"] = per_patient.apply(len)
    return static.reset_index()


def build_event_flags(oc: pd.DataFrame) -> pd.DataFrame:
    events = oc.dropna(subset=["clinical_event_type"])[
        ["public_id", "study_visit", "clinical_event_type"]
    ].copy()
    events["had_hospitalization"] = events["clinical_event_type"] == "hosp_admission"
    events["had_ed_visit"] = events["clinical_event_type"] == "ed"

    # A patient could have multiple event rows at the same visit; collapse
    # with "any" so duplicates don't cause a merge fan-out.
    events = events.groupby(["public_id", "study_visit"])[EVENT_COLS].any().reset_index()
    return events


def add_features(pairs: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    static = build_static_comorbidities(oc)
    events = build_event_flags(oc)

    result = pairs.merge(static, on="public_id", how="left")
    # Patients with no comorbidity file entry at all: 0 comorbidities, not missing.
    for feat in TOP_COMORBIDITIES.values():
        result[feat] = result[feat].fillna(False)
    result["comorbidity_count_total"] = result["comorbidity_count_total"].fillna(0).astype(int)

    result = result.merge(
        events,
        left_on=["public_id", "vt_study_visit"],
        right_on=["public_id", "study_visit"],
        how="left",
    )
    result = result.drop(columns=["study_visit"])

    return result


def summarize(df: pd.DataFrame) -> None:
    print(" Comorbidity (static) feature coverage ")
    for feat in TOP_COMORBIDITIES.values():
        print(f"{feat:24s} True rate: {df[feat].mean()*100:5.1f}%")
    print(f"comorbidity_count_total  mean: {df['comorbidity_count_total'].mean():.2f}")
    print()
    print(" Clinical event (per-visit) feature coverage ")
    for col in EVENT_COLS:
        pct_populated = df[col].notna().mean() * 100
        pct_true = df[col].mean(skipna=True) * 100
        print(f"{col:24s} populated: {pct_populated:5.1f}% | True rate: {pct_true:5.1f}%")


if __name__ == "__main__":
    pairs = load_pairs()
    oc = pd.read_csv(f"{CLINICAL_DIR}/other_clinical_attributes_deid.csv")

    result = add_features(pairs, oc)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")