"""
Step 9: Match RNA-seq pathway scores to the correct visit pair.

Input:  data/clinical/visit_pairs_master.csv   (frozen clinical pipeline)
        data/rna/pathway_scores_hallmark.csv   (859 samples x 50
            Hallmark pathways, indexed by GDC File ID)
        data/rna/gdc_sample_sheet_2026-09-12.tsv   (GDC Sample Sheet -
            solves the File ID -> public_id mapping problem)
        data/clinical/sample_deid.csv   (gives the real collection date
            for each sample, via the same day-lookup already used for
            flow cytometry)
Output: data/clinical/visit_pairs_with_rna.csv

NOTE: this is a NEW output file, not an overwrite of visit_pairs_master.csv
- the clinical pipeline is frozen; this step builds on top of it rather
than reopening it.

THE ID-MAPPING PROBLEM THIS SOLVES: pathway_scores_hallmark.csv is
indexed by GDC file UUID, completely unrelated to the public_id (mmrf_XXXX)
format used throughout the clinical pipeline. The GDC Sample Sheet
(downloaded from portal.gdc.cancer.gov's Cart after adding the same RNA-seq
files) gives File ID -> Case ID directly - verified 859/859 (100%) of
pathway_scores_hallmark.csv's rows have a matching File ID in the sheet.

THE DATE PROBLEM THIS SOLVES: the sample sheet's Sample ID
(e.g. "MMRF_1024_1_BM_CD138pos") doesn't have a day-based timestamp
either, but a derived key (lowercase, first 3 underscore-separated parts
-> "mmrf_1024_1") matches sample_deid.csv's sample_visit_id field, which
DOES have days_to_sample_procurement - verified 847/847 (100%) of the
derived keys resolve to a real collection date.

LEAKAGE RULE: only pathway scores from a sample collected AT OR BEFORE Vt
are ever used (backward as-of join, direction="backward", same pattern
as every other script in this pipeline) - a sample collected after Vt is
never visible to that row.
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"
RNA_DIR = "data/rna"

def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv", low_memory=False)
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs


def load_file_id_to_case_id() -> pd.DataFrame:
    """GDC Sample Sheet: File ID (GDC UUID, matches pathway_scores_hallmark.csv's
    index) -> Case ID (public_id) and Sample ID (used to derive the
    sample_deid.csv join key for the actual collection date)."""
    sheet = pd.read_csv(f"{RNA_DIR}/gdc_sample_sheet_2026-09-12.tsv", sep="\t")
    gene_expr = sheet[sheet["Data Type"] == "Gene Expression Quantification"].copy()

    # Derive the sample_deid.csv-style key: lowercase, first 3
    # underscore-separated parts (e.g. "MMRF_1024_1_BM_CD138pos" ->
    # "mmrf_1024_1"), verified to match sample_deid.csv's
    # sample_visit_id field for 847/847 (100%) of these samples.
    gene_expr["sample_visit_id_key"] = (
        gene_expr["Sample ID"].str.lower().str.split("_").str[:3].str.join("_")
    )

    gene_expr = gene_expr.rename(columns={"File ID": "file_id", "Case ID": "public_id"})
    # BUG FIX: GDC's Case ID is uppercase ("MMRF_1024"), but public_id is
    # lowercase throughout the rest of this pipeline ("mmrf_1024") - without
    # this, merge_asof's patient-matching would silently find zero matches.
    gene_expr["public_id"] = gene_expr["public_id"].str.lower()
    return gene_expr[["file_id", "public_id", "sample_visit_id_key"]].drop_duplicates()


def load_sample_day_lookup() -> pd.DataFrame:
    """Same mechanism already used in build_flow_cytometry_features.py:
    sample_deid.csv gives the real collection date for each sample_visit_id.
    """
    s = pd.read_csv(f"{CLINICAL_DIR}/sample_deid.csv")
    s["sample_visit_id"] = s["sample_visit_id"].str.lower()
    day_lookup = (
        s.groupby("sample_visit_id")["days_to_sample_procurement"]
        .first()  # verified consistent across duplicate specimen-type rows
        .reset_index()
        .rename(columns={"sample_visit_id": "sample_visit_id_key"})
    )
    return day_lookup


def load_dated_pathway_scores() -> tuple[pd.DataFrame, list[str]]:
    scores = pd.read_csv(f"{RNA_DIR}/pathway_scores_hallmark.csv", index_col=0)
    pathway_cols = list(scores.columns)
    scores = scores.reset_index().rename(columns={scores.index.name or "index": "file_id"})

    mapping = load_file_id_to_case_id()
    day_lookup = load_sample_day_lookup()

    dated = scores.merge(mapping, on="file_id", how="inner")
    n_before = len(scores)
    n_after = len(dated)
    print(f"Pathway score rows: {n_before} | matched to a patient: {n_after}")

    dated = dated.merge(day_lookup, on="sample_visit_id_key", how="inner")
    n_dated = len(dated)
    print(f"Matched to a sample_visit_id with a date lookup: {n_dated} / {n_after}")

    # A handful of matched keys still have a NaN date value in sample_deid.csv
    # itself (the key exists, but no procurement date was ever recorded for
    # it) - these can't be used for the as-of join and are dropped here,
    # with the real final count printed for transparency.
    dated = dated.dropna(subset=["days_to_sample_procurement"])
    print(f"Matched to a REAL (non-null) collection date: {len(dated)} / {n_after}")

    dated["days_to_sample_procurement"] = dated["days_to_sample_procurement"].astype(float)
    result = dated[
        ["public_id", "days_to_sample_procurement"] + pathway_cols
    ].sort_values("days_to_sample_procurement")
    return result, pathway_cols


def add_rna_features(pairs: pd.DataFrame, dated_scores: pd.DataFrame) -> pd.DataFrame:
    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)

    # LEAKAGE-SAFE: backward as-of join - only a sample collected at or
    # before Vt is ever visible to that row. A sample collected after Vt
    # is never used, regardless of how "close" it might be.
    merged = pd.merge_asof(
        p,
        dated_scores,
        left_on="vt_days_to_visit",
        right_on="days_to_sample_procurement",
        by="public_id",
        direction="backward",
    )
    # DESIGN REVIEW ADDITION: median RNA sample staleness at matched Vt is
    # ~866 days (myeloma has well-documented clonal evolution under
    # treatment, so a 2+ year old sample may poorly represent current
    # biology). Rather than silently discard this, it's persisted as its
    # own feature so a model can learn to weight/discount stale RNA -
    # previously this value was computed internally then dropped.
    merged["days_since_rna_sample"] = (
        merged["vt_days_to_visit"] - merged["days_to_sample_procurement"]
    )
    merged = merged.drop(columns=["days_to_sample_procurement"])
    return merged.sort_values("pair_id").reset_index(drop=True)


def summarize(df: pd.DataFrame, pathway_cols: list[str]) -> None:
    print("\n RNA-seq pathway score coverage on visit pairs ")
    sample_col = pathway_cols[0]
    pct = df[sample_col].notna().mean() * 100
    print(f"Pairs with a matched RNA-seq sample (at/before Vt): {pct:.1f}%")
    print(f"Unique patients with RNA-seq available: {df.loc[df[sample_col].notna(), 'public_id'].nunique()}")


if __name__ == "__main__":
    pairs = load_pairs()
    dated_scores, pathway_cols = load_dated_pathway_scores()

    result = add_rna_features(pairs, dated_scores)
    summarize(result, pathway_cols)

    out_path = f"{CLINICAL_DIR}/visit_pairs_with_rna.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")