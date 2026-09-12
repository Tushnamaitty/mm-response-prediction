"""
Add supportive care features to every visit pair, using
supportive_care_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

Same limitation as adverse_event_deid.csv: NO days_to_* timestamp, only a
study_visit label - so this uses an EXACT match on (public_id,
vt_study_visit), not an as-of join. Pairs whose Vt visit isn't covered by
this file get NaN.

Rationale: these supportive-care interventions are proxies for disease
severity that aren't directly captured elsewhere - e.g. transfusion
dependence implies severe anemia beyond what a single hemoglobin reading
shows, dialysis implies renal failure severe enough to need intervention
(not just an elevated creatinine value), and bisphosphonate use reflects
clinically-judged significant bone disease.

Features added (all boolean):
  had_transfusion
  had_dialysis
  had_bisphosphonate
  had_opioid                 - proxy for clinically significant bone pain
  had_radiation_therapy
  had_wbc_growth_factor       - proxy for treatment-induced neutropenia
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

SOURCE_TO_FEATURE = {
    "transfusion": "had_transfusion",
    "dialysis": "had_dialysis",
    "bisphosphonate_bone_directed": "had_bisphosphonate",
    "opioid": "had_opioid",
    "radiation_therapy": "had_radiation_therapy",
    "wbc_growth_factor": "had_wbc_growth_factor",
    "erythropoiesis_stimulating_agent": "had_esa",
    "meds_tumor_lysis_syndrome": "had_tumor_lysis_meds",
    "anticoagulation": "had_anticoagulation",
}
NEW_COLS = list(SOURCE_TO_FEATURE.values())


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs.drop(columns=[c for c in NEW_COLS if c in pairs.columns])


def load_supportive_care() -> pd.DataFrame:
    sc = pd.read_csv(f"{CLINICAL_DIR}/supportive_care_deid.csv")
    # "yes_c" appears once in erythropoiesis_stimulating_agent - a clear
    # typo variant of "yes", treated as such.
    YES_VALUES = {"yes", "yes_c"}
    for src, feat in SOURCE_TO_FEATURE.items():
        sc[feat] = sc[src].isin(YES_VALUES)
    return sc[["public_id", "study_visit"] + NEW_COLS]


def add_supportive_care_features(pairs: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    merged = pairs.merge(
        sc,
        left_on=["public_id", "vt_study_visit"],
        right_on=["public_id", "study_visit"],
        how="left",
    )
    return merged.drop(columns=["study_visit"])


def summarize(df: pd.DataFrame) -> None:
    print("=== Supportive care feature coverage ===")
    for col in NEW_COLS:
        pct_populated = df[col].notna().mean() * 100
        pct_true = df[col].mean(skipna=True) * 100
        print(f"{col:24s} populated: {pct_populated:5.1f}% | True rate: {pct_true:5.1f}%")


if __name__ == "__main__":
    pairs = load_pairs()
    sc = load_supportive_care()

    result = add_supportive_care_features(pairs, sc)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")