"""
Add current CRAB-criteria clinical symptom features to every visit pair,
using signs_symptoms_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

signs_symptoms_deid.csv is visit-level, with days_to_signs_symptoms as its
time axis - same as-of (backward) matching pattern as labs: for each pair,
find the most recent signs/symptoms assessment at or before Vt.

CRAB = Calcium elevation, Renal insufficiency, Anemia, Bone lesions - the
defining clinical criteria for active (symptomatic) multiple myeloma.
Most of CRAB is already covered by labs (calcium, creatinine, hemoglobin)
and bone_assessment; this file adds the actual CLINICAL/SYMPTOM-level
confirmation of those same domains plus a few more (spinal cord
compression, recurrent infection, amyloidosis).

Features added (all "yes"/"no" in the source, converted to boolean):
  subject_had_signs_symptoms   - any signs/symptoms reported at all
  bone_pain
  hypercalcemia
  renal_insufficiency
  anemia
  lytic_bone_lesion
  spinal_cord_compression
  recurrent_bacteria_infection
  amyloidosis
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

SYMPTOM_COLS = [
    "subject_had_signs_symptoms",
    "bone_pain",
    "hypercalcemia",
    "renal_insufficiency",
    "anemia",
    "lytic_bone_lesion",
    "spinal_cord_compression",
    "recurrent_bacteria_infection",
    "amyloidosis",
    "soft_tissue_plasmacytoma_single",
    "soft_tissue_plasmacytoma_multi",
]


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs


def load_symptoms() -> pd.DataFrame:
    ss = pd.read_csv(f"{CLINICAL_DIR}/signs_symptoms_deid.csv")
    ss = ss.dropna(subset=["days_to_signs_symptoms"]).copy()
    ss["days_to_signs_symptoms"] = ss["days_to_signs_symptoms"].astype(float)

    for col in SYMPTOM_COLS:
        ss[col] = ss[col].map({"yes": True, "no": False})

    return ss.sort_values("days_to_signs_symptoms")


def add_symptom_features(pairs: pd.DataFrame, symptoms: pd.DataFrame) -> pd.DataFrame:
    # Make this safe to re-run: if these columns already exist (e.g. from a
    # previous attempt), drop them first so merge_asof doesn't silently
    # rename the new ones to *_x/*_y instead of using the plain names.
    pairs = pairs.drop(columns=[c for c in SYMPTOM_COLS if c in pairs.columns])

    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)

    merged = pd.merge_asof(
        p,
        symptoms[["public_id", "days_to_signs_symptoms"] + SYMPTOM_COLS],
        left_on="vt_days_to_visit",
        right_on="days_to_signs_symptoms",
        by="public_id",
        direction="backward",
    )
    merged = merged.drop(columns=["days_to_signs_symptoms"])
    return merged.sort_values("pair_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print(" CRAB-criteria clinical symptom feature coverage ")
    for col in SYMPTOM_COLS:
        pct_populated = df[col].notna().mean() * 100
        pct_true = df[col].mean(skipna=True) * 100
        print(f"{col:32s} populated: {pct_populated:5.1f}% | True rate: {pct_true:5.1f}%")


if __name__ == "__main__":
    pairs = load_pairs()
    symptoms = load_symptoms()

    result = add_symptom_features(pairs, symptoms)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")