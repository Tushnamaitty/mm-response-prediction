"""
Add family cancer history features to every visit pair, using
family_history_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
        data/clinical/family_history_deid.csv
        data/clinical/subject_deid.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

Every row in this file represents an actual cancer diagnosis in a
relative (there is no "no cancer" category recorded) - so a patient's
presence in this file at all means a positive family cancer history.
Patients with no rows here are treated as no family history (rather than
missing) - the file only records occurrences, not confirmed absences, so
this carries a small assumption: a patient truly not asked would look
identical to one confirmed negative. Treated as static, patient-level
(family history doesn't change over the course of treatment).

Features added (static, patient-level):
  has_family_history_of_cancer          - any relative with any cancer
  has_family_history_of_myeloma         - any relative specifically with
                                           multiple myeloma (relevant given
                                           MM's known familial clustering)
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

NEW_COLS = [
    "has_family_history_of_cancer",
    "has_family_history_of_myeloma",
    "count_relatives_with_cancer",
]


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs.drop(columns=[c for c in NEW_COLS if c in pairs.columns])


def build_family_history_flags() -> pd.DataFrame:
    fh = pd.read_csv(f"{CLINICAL_DIR}/family_history_deid.csv")

    has_cancer = fh.groupby("public_id").size() > 0
    has_myeloma = fh.groupby("public_id")["relationship_primary_diagnosis"].apply(
        lambda diag: (diag == "multiple_myeloma").any()
    )

    result = pd.DataFrame(
        {
            "has_family_history_of_cancer": has_cancer,
            "has_family_history_of_myeloma": has_myeloma,
        }
    ).reset_index()

    # count_relatives_with_cancer comes from subject_deid.csv, a separate
    # (patient-reported, numeric/ordinal) field distinct from the
    # per-relative rows above - "not_reported" treated as missing, "none"
    # as 0.
    subj = pd.read_csv(f"{CLINICAL_DIR}/subject_deid.csv")
    count_map = subj["count_relatives_with_cancer"].replace("none", "0")
    count_map = pd.to_numeric(count_map, errors="coerce")
    subj["count_relatives_with_cancer"] = count_map

    result = result.merge(
        subj[["public_id", "count_relatives_with_cancer"]], on="public_id", how="outer"
    )
    return result


def add_features(pairs: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    result = pairs.merge(flags, on="public_id", how="left")
    for col in ["has_family_history_of_cancer", "has_family_history_of_myeloma"]:
        result[col] = result[col].fillna(False)
    # count_relatives_with_cancer stays nullable numeric - a patient
    # genuinely marked "not_reported" should stay missing, not become 0.
    return result


def summarize(df: pd.DataFrame) -> None:
    print(" Family history feature coverage ")
    for col in ["has_family_history_of_cancer", "has_family_history_of_myeloma"]:
        print(f"{col:34s} True rate: {df[col].mean()*100:5.1f}%")
    print(
        f"count_relatives_with_cancer         populated: "
        f"{df['count_relatives_with_cancer'].notna().mean()*100:5.1f}% "
        f"(mean={df['count_relatives_with_cancer'].mean():.2f})"
    )


if __name__ == "__main__":
    pairs = load_pairs()
    flags = build_family_history_flags()

    result = add_features(pairs, flags)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")