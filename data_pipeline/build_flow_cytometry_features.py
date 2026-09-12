"""
Add flow cytometry / immunophenotyping features to every visit pair,
using sample_features_deid.csv, matched to actual collection dates via
sample_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
        data/clinical/sample_features_deid.csv
        data/clinical/sample_deid.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

This is a genuinely distinct molecular data modality from RNA-seq -
plasma-cell surface marker expression and ploidy from bone marrow/blood
flow cytometry - not covered by anything else in the pipeline so far.

JOIN CHALLENGE: sample_features_deid.csv has NO direct day/timestamp
column, only a `study_visit_id` (format "mmrf_XXXX_N"). This matches
sample_deid.csv's `sample_visit_id` field, which DOES have
days_to_sample_procurement. sample_deid.csv has 2 rows per visit_id (one
per specimen type, e.g. bone marrow aspirate + peripheral blood) - their
collection days were verified consistent within each visit_id (only 3/1404
groups differ, and those are fully-missing groups, not real conflicts),
so we take one day value per visit_id safely.

Once dated, the same as-of (backward) matching pattern as labs/RNA-seq is
used: most recent flow cytometry sample at or before Vt.

Features added:
  cd38_detected, cd56_detected, cd138_detected  - boolean (True/False),
        NaN if not tested. CD38 is notable given anti-CD38 antibody
        (daratumumab) exposure is already tracked separately - lets us
        cross-check whether CD38 expression and anti-CD38 drug exposure
        show a meaningful relationship in later explainability analysis.
  cd38_pct, cd56_pct                            - % of plasma cells positive
  ig_heavy_chain_type, ig_light_chain_type      - categorical (igg/iga/igm/
        kappa/lambda/etc.), "unknown"/"not_recorded" mapped to missing
        since they carry no real information
  has_braf_mutation                             - boolean, NaN if not
        tested (a stray "99" sentinel value in the source is also
        treated as missing, not a real category)
  dna_index                                     - ploidy marker (high
        missingness in source, ~64%, but genuinely informative when
        present)
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

NEW_COLS = [
    "cd38_detected",
    "cd56_detected",
    "cd138_detected",
    "cd45_typical_detected",
    "cd38_pct",
    "cd56_pct",
    "pct_plasma_cells_bm",
    "pct_plasma_cells_pb",
    "ig_heavy_chain_type",
    "ig_light_chain_type",
    "has_braf_mutation",
    "dna_index",
]


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs.drop(columns=[c for c in NEW_COLS if c in pairs.columns])


def build_visit_day_lookup() -> pd.DataFrame:
    s = pd.read_csv(f"{CLINICAL_DIR}/sample_deid.csv")
    day_lookup = (
        s.groupby("sample_visit_id")["days_to_sample_procurement"]
        .first()  # verified consistent across duplicate rows per visit_id
        .reset_index()
        .rename(columns={"sample_visit_id": "study_visit_id"})
    )
    return day_lookup


def load_sample_features(day_lookup: pd.DataFrame) -> pd.DataFrame:
    sf = pd.read_csv(f"{CLINICAL_DIR}/sample_features_deid.csv")
    sf = sf.merge(day_lookup, on="study_visit_id", how="left")
    sf = sf.dropna(subset=["days_to_sample_procurement", "public_id"]).copy()

    sf["cd38_detected"] = sf["flowcyt_cd38_detected"].map({1.0: True, 0.0: False})
    sf["cd56_detected"] = sf["flowcyt_cd56_detected"].map({1.0: True, 0.0: False})
    sf["cd138_detected"] = sf["flowcyt_cd138_detected"].map({1.0: True, 0.0: False})
    sf["cd45_typical_detected"] = sf["flowcyt_cd45_plasma_cells_typical_detected"].map(
        {1.0: True, 0.0: False}
    )
    # Bone marrow plasma cell % is a core WHO/IMWG diagnostic criterion for
    # myeloma. Prefer the flow cytometry version (31/1504 missing) over the
    # morphology version (179/1504 missing) since it has better coverage;
    # fall back to morphology where flow cytometry wasn't done.
    flowcyt_bm_pct = sf[
        ["flowcyt_pct_plasma_cells_in_bm_low", "flowcyt_pct_plasma_cells_in_bm_high"]
    ].mean(axis=1)
    sf["pct_plasma_cells_bm"] = flowcyt_bm_pct.fillna(sf["morphology_pct_plasma_cells_in_bm"])
    # Peripheral blood plasma cells (circulating disease) - same
    # flow-cytometry-preferred, morphology-fallback approach.
    flowcyt_pb_pct = sf[
        ["flowcyt_pct_plasma_cells_in_pb_low", "flowcyt_pct_plasma_cells_in_pb_high"]
    ].mean(axis=1)
    sf["pct_plasma_cells_pb"] = flowcyt_pb_pct.fillna(sf["morphology_pct_plasma_cells_in_pb"])
    # A percentage cannot exceed 100. flowcyt_cd38_plasma_cells_percent has
    # one row at 1010% in the source (clearly a data-entry error, e.g. a
    # stray digit) - unlike the QoL >100 case, there's no clear mechanism
    # to infer the intended value here, so it's treated as missing rather
    # than guessed at or clipped.
    sf["cd38_pct"] = sf["flowcyt_cd38_plasma_cells_percent"].where(
        sf["flowcyt_cd38_plasma_cells_percent"] <= 100
    )
    sf["cd56_pct"] = sf["flowcyt_cd56_plasma_cells_percent"].where(
        sf["flowcyt_cd56_plasma_cells_percent"] <= 100
    )

    unreliable = {"unknown", "not_recorded"}
    sf["ig_heavy_chain_type"] = sf["ig_heavy_institution"].where(
        ~sf["ig_heavy_institution"].isin(unreliable)
    )
    sf["ig_light_chain_type"] = sf["ig_light_institution"].where(
        ~sf["ig_light_institution"].isin(unreliable)
    )

    # braf_status: 0/1 are real, 99 is a sentinel/error code -> missing.
    sf["has_braf_mutation"] = sf["braf_status"].where(sf["braf_status"].isin([0.0, 1.0]))
    sf["has_braf_mutation"] = sf["has_braf_mutation"].map({1.0: True, 0.0: False})

    sf["dna_index"] = sf["flowcyt_dna_index"]

    sf["days_to_sample_procurement"] = sf["days_to_sample_procurement"].astype(float)
    return sf[["public_id", "days_to_sample_procurement"] + NEW_COLS].sort_values(
        "days_to_sample_procurement"
    )


def add_flow_features(pairs: pd.DataFrame, sf: pd.DataFrame) -> pd.DataFrame:
    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)

    merged = pd.merge_asof(
        p,
        sf,
        left_on="vt_days_to_visit",
        right_on="days_to_sample_procurement",
        by="public_id",
        direction="backward",
    )
    merged = merged.drop(columns=["days_to_sample_procurement"])
    return merged.sort_values("pair_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print("=== Flow cytometry / immunophenotyping feature coverage ===")
    for col in NEW_COLS:
        pct = df[col].notna().mean() * 100
        print(f"{col:24s} populated: {pct:5.1f}%")


if __name__ == "__main__":
    pairs = load_pairs()
    day_lookup = build_visit_day_lookup()
    sf = load_sample_features(day_lookup)

    result = add_flow_features(pairs, sf)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")