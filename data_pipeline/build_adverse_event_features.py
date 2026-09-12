"""
Add peripheral neuropathy (treatment toxicity) features to every visit
pair, using adverse_event_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

IMPORTANT DIFFERENCE from other feature scripts: adverse_event_deid.csv has
NO days_to_* timestamp - only a study_visit label (e.g. "month_3"). So
this uses an EXACT match on (public_id, study_visit) against the pair's
vt_study_visit, rather than an as-of (backward-in-time) join like every
other script. Consequence: pairs whose Vt visit has no corresponding AE
record (e.g. "baseline", "screening" - this file only covers on-treatment
follow-up visits) get NaN here, not a "carried forward" value. This is a
real structural limitation of the source file, not a bug.

69 duplicate (public_id, study_visit) rows exist in the source; we keep
the WORST (highest-grade / most clinically significant) record per
visit rather than an arbitrary one.

Why this matters: peripheral neuropathy is a common, dose-limiting
toxicity of two of the most-used drug classes in this cohort (PI and
IMiD - see build_drug_exposure_features.py). Its presence/severity can
affect subsequent treatment intensity, which in turn affects response.

Features added:
  has_peripheral_neuropathy       - boolean, sensory OR motor present
  peripheral_neuropathy_max_grade - 1/2/3 (worst of sensory/motor grade),
                                     NaN if not applicable/not reported
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

NEW_COLS = [
    "has_peripheral_neuropathy",
    "peripheral_neuropathy_max_grade",
    "had_other_ae",
]

# The 'ae' column lists the specific adverse event type per row - we only
# used the dedicated peripheral neuropathy columns before, missing this
# entirely. 4,852 of 5,025 rows are peripheral_neuropathy; the remaining
# ~170 cover a long tail of distinct, individually-rare events (pneumonia,
# sepsis, neutropenic fever, acute kidney injury, etc. - each only
# single-digit occurrences, too few to model individually). Rather than
# ignore them, they're combined into one "had some other documented AE"
# flag - some of these (sepsis, neutropenic fever, respiratory failure)
# are genuinely serious toxicities worth knowing about even in aggregate.
NON_INFORMATIVE_AE_VALUES = {"peripheral_neuropathy", "not_reported", "none"}

GRADE_MAP = {
    "grade_1_asymptomatic": 1,
    "grade_2_moderate_symptoms": 2,
    "grade_3_severe_symptoms": 3,
}


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs.drop(columns=[c for c in NEW_COLS if c in pairs.columns])


def load_adverse_events() -> pd.DataFrame:
    ae = pd.read_csv(f"{CLINICAL_DIR}/adverse_event_deid.csv")

    # BUG FIX: peripheral_sensory/motor_neuropathy_present have values
    # beyond yes/no - "not_applicable" (both columns) and "not_done"
    # (motor only). A naive == "yes" comparison silently treats these as
    # False (confirmed NO neuropathy), when they actually mean "not
    # assessed" - a meaningful difference. 174 rows had BOTH sensory and
    # motor as not_applicable, which would have been wrongly recorded as
    # a confirmed negative. Explicitly map to True/False/missing instead.
    PRESENT_MAP = {"yes": True, "no": False}  # not_applicable/not_done -> NaN
    sensory_present = ae["peripheral_sensory_neuropathy_present"].map(PRESENT_MAP)
    motor_present = ae["peripheral_motor_neuropathy_present"].map(PRESENT_MAP)
    # True if either is confirmed True; False only if BOTH are confirmed
    # False; missing if neither side gives a confirmed answer.
    ae["has_peripheral_neuropathy"] = None
    ae.loc[(sensory_present == True) | (motor_present == True), "has_peripheral_neuropathy"] = True
    ae.loc[(sensory_present == False) & (motor_present == False), "has_peripheral_neuropathy"] = False

    sensory_grade = ae["peripheral_sensory_neuropathy_grade"].map(GRADE_MAP)
    motor_grade = ae["peripheral_motor_neuropathy_grade"].map(GRADE_MAP)
    ae["peripheral_neuropathy_max_grade"] = pd.concat(
        [sensory_grade, motor_grade], axis=1
    ).max(axis=1)

    ae["had_other_ae"] = ~ae["ae"].isin(NON_INFORMATIVE_AE_VALUES) & ae["ae"].notna()

    # BUG FIX: pandas' built-in "any" aggregation on an object-dtype
    # column treats None as falsy, so a (patient, visit) group where
    # EVERY row is None (fully unknown) would silently become False
    # (confirmed negative) instead of staying missing - reintroducing
    # the exact class of bug just fixed above, one step later. Use an
    # explicit tri-state aggregator instead: True wins if any row is
    # True; else False if any row is explicitly False; else None
    # (nothing resolved either way).
    def tri_state_any(vals):
        vals = list(vals)
        if any(v is True for v in vals):
            return True
        if any(v is False for v in vals):
            return False
        return None

    # Aggregate (not just dedupe) per (public_id, study_visit): a patient
    # can have genuinely separate rows for the same visit covering
    # different AE types (e.g. one row for neuropathy, another for
    # pneumonia) - taking "first row sorted by grade" would silently drop
    # the had_other_ae signal from a second row. Aggregate across all
    # same-visit rows instead so no signal is lost.
    ae = ae.groupby(["public_id", "study_visit"]).agg(
        has_peripheral_neuropathy=("has_peripheral_neuropathy", tri_state_any),
        peripheral_neuropathy_max_grade=("peripheral_neuropathy_max_grade", "max"),
        had_other_ae=("had_other_ae", tri_state_any),
    ).reset_index()

    return ae[["public_id", "study_visit"] + NEW_COLS]


def add_ae_features(pairs: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    merged = pairs.merge(
        ae,
        left_on=["public_id", "vt_study_visit"],
        right_on=["public_id", "study_visit"],
        how="left",
    )
    return merged.drop(columns=["study_visit"])


def summarize(df: pd.DataFrame) -> None:
    print("=== Peripheral neuropathy feature coverage ===")
    for col in NEW_COLS:
        pct = df[col].notna().mean() * 100
        print(f"{col:32s} populated: {pct:5.1f}%")
    print()
    print("peripheral_neuropathy_max_grade distribution:")
    print(df["peripheral_neuropathy_max_grade"].value_counts(dropna=False))


if __name__ == "__main__":
    pairs = load_pairs()
    ae = load_adverse_events()

    result = add_ae_features(pairs, ae)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")