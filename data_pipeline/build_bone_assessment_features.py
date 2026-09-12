"""
Add current bone-lesion assessment features to every visit pair, using
bone_assessment_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

Same as-of (backward) matching pattern as labs/symptoms, using
days_to_bone_assessment as the time axis.

This is the "Bone lesions" component of CRAB criteria, complementing the
signs_symptoms_deid.csv features already added (which include a simpler
lytic_bone_lesion yes/no from the symptom checklist - this file gives the
actual IMAGING-confirmed assessment, which is more clinically authoritative
than the symptom-report version).

Features added:
  bone_lytic_lesion_present   - boolean, imaging-confirmed lytic lesion
  n_lytic_lesions_ordinal     - 0/1/2/3 (3 = "three or more"), NaN if not
                                 reported. Ordinal encoding of a categorical
                                 field for direct numeric use in models.
  pathologic_fracture         - boolean
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

BONE_COLS = [
    "bone_lytic_lesion_present",
    "n_lytic_lesions_ordinal",
    "pathologic_fracture",
]

LESION_COUNT_MAP = {
    "none": 0,
    "one": 1,
    "two": 2,
    "greater_than_equal_to_three": 3,
    "not_reported": float("nan"),
}


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    # Idempotency: drop these columns if a previous run already added them.
    return pairs.drop(columns=[c for c in BONE_COLS if c in pairs.columns])


def load_bone_assessments() -> pd.DataFrame:
    ba = pd.read_csv(f"{CLINICAL_DIR}/bone_assessment_deid.csv")
    ba = ba.dropna(subset=["days_to_bone_assessment"]).copy()
    ba["days_to_bone_assessment"] = ba["days_to_bone_assessment"].astype(float)

    ba["bone_lytic_lesion_present"] = ba["lytic_bone_lesion_present"].map(
        {"yes": True, "no": False}
    )
    ba["n_lytic_lesions_ordinal"] = ba["number_of_lytic_lesions"].map(LESION_COUNT_MAP)
    ba["pathologic_fracture"] = ba["pathologic_fracture"].map(
        {"yes": True, "no": False}
    )  # "not_reported" -> NaN automatically, not in the map

    return ba.sort_values("days_to_bone_assessment")


def add_bone_features(pairs: pd.DataFrame, bone: pd.DataFrame) -> pd.DataFrame:
    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)

    merged = pd.merge_asof(
        p,
        bone[["public_id", "days_to_bone_assessment"] + BONE_COLS],
        left_on="vt_days_to_visit",
        right_on="days_to_bone_assessment",
        by="public_id",
        direction="backward",
    )
    merged = merged.drop(columns=["days_to_bone_assessment"])
    return merged.sort_values("pair_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print(" Bone assessment feature coverage ")
    for col in BONE_COLS:
        pct_populated = df[col].notna().mean() * 100
        print(f"{col:28s} populated: {pct_populated:5.1f}%")
    print()
    print("n_lytic_lesions_ordinal distribution:")
    print(df["n_lytic_lesions_ordinal"].value_counts(dropna=False))


if __name__ == "__main__":
    pairs = load_pairs()
    bone = load_bone_assessments()

    result = add_bone_features(pairs, bone)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")