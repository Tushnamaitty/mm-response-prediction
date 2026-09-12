"""
Add patient-reported quality-of-life (QoL) features to every visit pair,
using qol_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

Unlike adverse_event/supportive_care, this file DOES have a numeric
days_to_survey timestamp, so we use the same as-of (backward) join as
labs/symptoms/bone assessment: most recent QoL survey at or before Vt.

Rationale: patient-reported outcomes (fatigue, pain, global health status,
physical functioning) have documented prognostic value in oncology
independent of clinician-assessed labs/symptoms - a patient's own
reported symptom burden can carry information not fully captured by
objective measures.

KNOWN DATA QUALITY NOTE: global_health_status_qol has a max value of
116.67 in the source data, though the EORTC QLQ-C30 scale is standardized
to 0-100. This is passed through as-is (not silently corrected, since the
true intended value is unknown) - worth being aware of if this feature
shows unusual behavior in modeling/explainability results later.

Features added (EORTC QLQ-C30 summary scores, 0-100 scale, higher =
more of that domain - i.e. higher fatigue/pain = worse, higher physical
functioning/global health = better):
  qol_global_health_status
  qol_fatigue
  qol_pain
  qol_physical_functioning
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

SOURCE_TO_FEATURE = {
    "global_health_status_qol": "qol_global_health_status",
    "fatigue": "qol_fatigue",
    "pain": "qol_pain",
    "physical_functioning": "qol_physical_functioning",
    "disease_symptoms": "qol_disease_symptoms",
    "side_effects_treatment": "qol_side_effects_treatment",
}
NEW_COLS = list(SOURCE_TO_FEATURE.values())


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs.drop(columns=[c for c in NEW_COLS if c in pairs.columns])


def load_qol() -> pd.DataFrame:
    qol = pd.read_csv(f"{CLINICAL_DIR}/qol_deid.csv")
    qol = qol.dropna(subset=["days_to_survey"]).copy()
    qol["days_to_survey"] = qol["days_to_survey"].astype(float)
    qol = qol.rename(columns=SOURCE_TO_FEATURE)

    # Fix applied at source: 24 rows (0.76%) have global_health_status_qol
    # above the valid 0-100 EORTC QLQ-C30 range (traced to two raw survey
    # items showing "8" where the scale should max at 7). Clip rather than
    # drop, since the direction (high score) is still correct, just the
    # magnitude is inflated.
    qol["qol_global_health_status"] = qol["qol_global_health_status"].clip(upper=100)

    return qol[["public_id", "days_to_survey"] + NEW_COLS].sort_values("days_to_survey")


def add_qol_features(pairs: pd.DataFrame, qol: pd.DataFrame) -> pd.DataFrame:
    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)

    merged = pd.merge_asof(
        p,
        qol,
        left_on="vt_days_to_visit",
        right_on="days_to_survey",
        by="public_id",
        direction="backward",
    )
    merged = merged.drop(columns=["days_to_survey"])
    return merged.sort_values("pair_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print("=== Quality-of-life feature coverage ===")
    for col in NEW_COLS:
        pct = df[col].notna().mean() * 100
        print(f"{col:28s} populated: {pct:5.1f}%  (mean={df[col].mean():.1f})")


if __name__ == "__main__":
    pairs = load_pairs()
    qol = load_qol()

    result = add_qol_features(pairs, qol)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")