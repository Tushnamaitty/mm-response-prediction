"""
Add circulating multiple myeloma cell (CMMC) count features to every
visit pair, using sample_cmmc_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

CMMC count is a liquid-biopsy-type biomarker (circulating tumor cells) -
potentially a strong signal, but coverage is expected to be low/sparse
since it's only drawn at specific milestones (post-transplant
confirmation, remission confirmation, etc.), not routinely. Included
anyway per the "don't pre-exclude based on a guess about usefulness"
principle - if coverage turns out too sparse to be useful, that will show
up naturally in Step 19's missingness-cutoff robustness check, rather
than being decided upfront here.

This file DOES have a proper day timestamp (days_to_cmmc_procurement),
so uses the standard as-of (backward) join, same as labs.

Features added:
  cmmc_number       - most recent circulating tumor cell count at/before Vt
                       (raw value; highly right-skewed in the source -
                       median 208, max 47,146 - consider a log transform
                       at modeling time if used)
  cmmc_detected     - boolean, cmmc_number > 0 (a count of exactly 0 is a
                       meaningful "no circulating cells detected" result,
                       not a missing value)
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

NEW_COLS = ["cmmc_number", "cmmc_detected"]


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs.drop(columns=[c for c in NEW_COLS if c in pairs.columns])


def load_cmmc() -> pd.DataFrame:
    cm = pd.read_csv(f"{CLINICAL_DIR}/sample_cmmc_deid.csv")
    cm = cm.dropna(subset=["days_to_cmmc_procurement", "cmmc_number"]).copy()
    cm["days_to_cmmc_procurement"] = cm["days_to_cmmc_procurement"].astype(float)
    cm["cmmc_detected"] = cm["cmmc_number"] > 0
    return cm[["public_id", "days_to_cmmc_procurement"] + NEW_COLS].sort_values(
        "days_to_cmmc_procurement"
    )


def add_cmmc_features(pairs: pd.DataFrame, cmmc: pd.DataFrame) -> pd.DataFrame:
    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)

    merged = pd.merge_asof(
        p,
        cmmc,
        left_on="vt_days_to_visit",
        right_on="days_to_cmmc_procurement",
        by="public_id",
        direction="backward",
    )
    merged = merged.drop(columns=["days_to_cmmc_procurement"])
    return merged.sort_values("pair_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print("=== CMMC feature coverage ===")
    for col in NEW_COLS:
        pct = df[col].notna().mean() * 100
        print(f"{col:16s} populated: {pct:5.1f}%")


if __name__ == "__main__":
    pairs = load_pairs()
    cmmc = load_cmmc()

    result = add_cmmc_features(pairs, cmmc)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")