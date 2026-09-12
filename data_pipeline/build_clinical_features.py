"""
Steps 5-6: Add current clinical state and temporal/trajectory features
to each visit pair, using labs_deid.csv.

Input:  data/clinical/visit_pairs.csv   (output of Steps 2-4)
        data/clinical/labs_deid.csv
Output: data/clinical/visit_pairs_with_clinical.csv

labs_deid.csv is in LONG format: one row per (patient, lab_test, collection
date). For each of the 5 key labs your project doc calls out (M-protein,
free light chains x2, calcium, creatinine, hemoglobin), we attach, AT Vt:

--- Step 5: current clinical state ---
  {lab}_current            - most recent value at or before Vt
  {lab}_current_day        - the day that value was drawn on

--- Step 6: temporal / trajectory features ---
  {lab}_previous           - the value drawn before that one
  {lab}_change             - current - previous (absolute change)
  {lab}_pct_change         - (current - previous) / previous
  {lab}_days_since_previous - days between the previous and current draws

Also carries forward vt_days_to_visit itself as a simple
"time since baseline" temporal feature (baseline = first treatment date,
per subject_deid.csv's index_date convention - a reasonable proxy for
"time since diagnosis" until we confirm an exact diagnosis date elsewhere).
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

# CRITICAL FIX: labs_deid.csv reports the SAME lab test in multiple
# different units across rows (e.g. m-spike in both g/dl and g/l - a 10x
# numeric difference for the same real concentration). Without
# harmonizing to one canonical unit per test BEFORE computing
# current/previous/change, a patient whose unit happened to differ
# between two visits would show a fake, huge "change" that has nothing
# to do with their actual health. Each test is converted to the unit
# listed below; units too rare/ambiguous to safely convert (each is a
# single occurrence in the source, verified individually before this
# decision) are dropped rather than guessed at.
LAB_UNIT_CONVERSIONS = {
    "m-spike": {  # target: g/dl
        "g/dl": 1.0,
        "g/l": 0.1,
        "mg/dl": 0.001,
        # "mg/l" (n=1) dropped - too rare/ambiguous to safely convert
    },
    "kappa_light_chain": {  # target: mg/l (standard FLC assay unit)
        "mg/l": 1.0,
        "mg/dl": 10.0,
        # "percent" (n=1) dropped - not a valid concentration unit here
    },
    "lambda_light_chains": {  # target: mg/l
        "mg/l": 1.0,
        "mg/dl": 10.0,
    },
    "serum_calcium": {  # target: mg/dl
        "mg/dl": 1.0,
        "mmol/l": 4.008,
    },
    "serum_creatinine": {  # target: mg/dl
        "mg/dl": 1.0,
        "umol/l": 1 / 88.4,
        # "mmol/l" (n=1) dropped - too rare/ambiguous to safely convert
    },
    "hemoglobin": {  # target: g/dl
        "g/dl": 1.0,
        "g/l": 0.1,
        # "x10^3/mcl" (n=1) dropped - wrong unit label for hemoglobin,
        # not safely correctable
    },
    "b2m": {  # target: mg/l
        "mg/l": 1.0,
        "mcg/ml": 1.0,  # 1 mcg/mL == 1 mg/L
        "mg/dl": 10.0,
        "g/l": 1000.0,
        "ng/ml": 0.001,
        "mcg/l": 0.001,
        "mg/ml": 1000.0,
        "mcg/dl": 0.01,
        # "g/dl" (n=194) intentionally EXCLUDED, not dropped as rare -
        # this is actually a large group, but g/dL for b2m (normally
        # ~mg/L range) would imply implausibly huge values, suggesting
        # a systematic unit-recording error rather than a real
        # measurement. Treated as missing rather than converted, since
        # converting via the textbook g/dl->mg/l factor (x10000) would
        # produce physiologically implausible values.
        # "percent" (n=1) dropped
    },
    "serum_albumin": {  # target: g/dl
        "g/dl": 1.0,
        "g/l": 0.1,
        # "percent" (n=1) dropped
    },
    "serum_ldh": {  # target: u/l (u/l and iu/l are clinically equivalent
        # for this enzymatic assay)
        "u/l": 1.0,
        "iu/l": 1.0,
        # bare "l" (n=73) dropped - genuinely ambiguous, NOT assumed to
        # be a truncated "u/l" without evidence
        # "mg/dl" (n=1) dropped - not a valid unit for an enzyme activity
    },
}


def harmonize_lab_units(labs: pd.DataFrame) -> pd.DataFrame:
    """Convert each lab test's values to one canonical unit, dropping
    rows whose unit isn't in the conversion table for that test (either
    too rare to safely convert, or a suspected recording error - see
    comments above)."""
    labs = labs.copy()
    labs["lab_test_result_unit"] = labs["lab_test_result_unit"].str.lower()

    keep_masks = []
    for test, conversions in LAB_UNIT_CONVERSIONS.items():
        test_mask = labs["lab_test"] == test
        unit_ok_mask = labs["lab_test_result_unit"].isin(conversions.keys())
        keep_masks.append(test_mask & unit_ok_mask)
        # Apply the conversion factor in place for this test's rows.
        for unit, factor in conversions.items():
            row_mask = test_mask & (labs["lab_test_result_unit"] == unit)
            labs.loc[row_mask, "lab_test_result"] = (
                labs.loc[row_mask, "lab_test_result"] * factor
            )

    # Only keep rows for tests we harmonize, with a convertible unit.
    combined_mask = pd.concat(keep_masks, axis=1).any(axis=1)
    return labs[combined_mask]

# Maps project-doc clinical concept -> actual lab_test value in the data.
KEY_LABS = {
    "m_protein": "m-spike",
    "kappa_flc": "kappa_light_chain",
    "lambda_flc": "lambda_light_chains",
    "calcium": "serum_calcium",
    "creatinine": "serum_creatinine",
    "hemoglobin": "hemoglobin",
    "b2m": "b2m",
    "albumin": "serum_albumin",
    "ldh": "serum_ldh",
}


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs.csv")
    pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs


def load_labs() -> pd.DataFrame:
    labs = pd.read_csv(f"{CLINICAL_DIR}/labs_deid.csv")
    labs["lab_test_result"] = pd.to_numeric(labs["lab_test_result"], errors="coerce")
    labs = labs.dropna(subset=["lab_test_result", "days_to_specimen_collection"])
    labs = harmonize_lab_units(labs)
    return labs


def add_lab_features(
    pairs: pd.DataFrame, labs: pd.DataFrame, lab_test: str, prefix: str
) -> pd.DataFrame:
    """Attach current/previous/change columns for one lab test via an
    as-of (backward) merge on days_to_specimen_collection <= vt_days_to_visit.
    """
    sub = labs[labs["lab_test"] == lab_test][
        ["public_id", "days_to_specimen_collection", "lab_test_result"]
    ].copy()
    sub = sub.sort_values(["public_id", "days_to_specimen_collection"])

    # Attach, to each lab record, what the PREVIOUS reading (for the same
    # patient/test) was - so a backward as-of merge on the CURRENT reading
    # also brings the previous one along in a single pass.
    sub[f"{prefix}_previous"] = sub.groupby("public_id")["lab_test_result"].shift(1)
    sub["prev_day"] = sub.groupby("public_id")["days_to_specimen_collection"].shift(1)

    sub = sub.rename(
        columns={
            "lab_test_result": f"{prefix}_current",
            "days_to_specimen_collection": "lab_day",
        }
    )
    sub["lab_day"] = sub["lab_day"].astype(float)

    pairs_sorted = pairs.sort_values("vt_days_to_visit").copy()
    pairs_sorted["vt_days_to_visit"] = pairs_sorted["vt_days_to_visit"].astype(float)

    merged = pd.merge_asof(
        pairs_sorted,
        sub.sort_values("lab_day"),
        left_on="vt_days_to_visit",
        right_on="lab_day",
        by="public_id",
        direction="backward",
    )

    merged[f"{prefix}_change"] = merged[f"{prefix}_current"] - merged[f"{prefix}_previous"]

    # pct_change = change / previous is mathematically undefined when
    # previous == 0. Two distinct cases, handled explicitly rather than
    # left as a silent division:
    #   previous == 0 AND current == 0  -> pct_change = 0 (a defined,
    #       meaningful "no change" - e.g. M-protein stayed undetectable)
    #   previous == 0 AND current != 0  -> pct_change stays missing
    #       (genuinely undefined - "went from 0 to X" has no finite
    #       percent change; a raw division would produce +/-inf here)
    prev = merged[f"{prefix}_previous"]
    curr = merged[f"{prefix}_current"]
    pct_change = merged[f"{prefix}_change"] / prev
    pct_change = pct_change.replace([float("inf"), float("-inf")], pd.NA)
    zero_to_zero = (prev == 0) & (curr == 0)
    pct_change = pct_change.mask(zero_to_zero, 0.0)
    merged[f"{prefix}_pct_change"] = pct_change

    merged[f"{prefix}_days_since_previous"] = merged["lab_day"] - merged["prev_day"]

    merged = merged.drop(columns=["lab_day", "prev_day"])
    return merged.sort_values("pair_id").reset_index(drop=True)


def add_all_lab_features(pairs: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    result = pairs.copy()
    for prefix, lab_test in KEY_LABS.items():
        enriched = add_lab_features(result, labs, lab_test, prefix)
        new_cols = [c for c in enriched.columns if c not in result.columns]
        result = result.merge(enriched[["pair_id"] + new_cols], on="pair_id", how="left")
    return result


def summarize(df: pd.DataFrame) -> None:
    print("=== Steps 5-6: Clinical + temporal feature coverage ===")
    for prefix in KEY_LABS:
        current_col = f"{prefix}_current"
        prev_col = f"{prefix}_previous"
        pct_have_current = df[current_col].notna().mean() * 100
        pct_have_previous = df[prev_col].notna().mean() * 100
        print(
            f"{prefix:12s} current: {pct_have_current:5.1f}% populated | "
            f"previous: {pct_have_previous:5.1f}% populated"
        )


if __name__ == "__main__":
    pairs = load_pairs()
    labs = load_labs()

    enriched = add_all_lab_features(pairs, labs)
    summarize(enriched)

    out_path = f"{CLINICAL_DIR}/visit_pairs_with_clinical.csv"
    enriched.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {enriched.shape})")