"""
Step 7: Add treatment-context features to each visit pair, using
administered_line_deid.csv, administered_regimen_line_deid.csv, and
administered_sc_transplant_deid.csv.

Input:  data/clinical/visit_pairs_with_clinical.csv  (output of Steps 5-6)
        data/clinical/administered_line_deid.csv
        data/clinical/administered_regimen_line_deid.csv
        data/clinical/administered_sc_transplant_deid.csv
Output: data/clinical/visit_pairs_with_treatment.csv

Unlike labs (point-in-time values), treatment records are INTERVALS
(days_to_*_start -> days_to_*_end). "Active at Vt" means:
    start <= vt_day  AND  (end is missing OR end >= vt_day)

Features added at Vt:
  current_line_number         - line of therapy active at Vt (NaN if in a gap)
  current_line_is_active      - True if a line was genuinely ongoing at Vt
                                 (vs. just the most recently started one, in
                                 case of a gap between lines)
  line_duration_so_far        - vt_day - start of the current line
  n_prior_lines_completed     - lines that had already ENDED before Vt

  current_regimen_name/type/categories - regimen active at Vt
  regimen_duration_so_far     - vt_day - start of the current regimen
  n_prior_regimens            - regimens that had already STARTED before Vt
  prior_treatment_failure     - True if any PRIOR regimen (ended before Vt)
                                 was discontinued for disease_progression_relapse
                                 or lack_of_response

  has_prior_transplant        - any transplant strictly before Vt
  n_prior_transplants         - count of such transplants
  days_since_last_transplant  - Vt_day - most recent prior transplant day

NOTE: regimen_type had messy/inconsistent values in the source data.
These are now cleaned via clean_regimen_type() (see REGIMEN_TYPE_CLEANUP
above): genuine typos and the "main"/"maintenance" duplicate labeling are
merged into their correct canonical category; single-occurrence entries
too ambiguous to confidently correct ("?", "y", "pre", "conditioning",
"main?") are mapped to missing rather than guessed at.
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

# regimen_type has known messy values in the source data: typos, an
# inconsistent-labeling duplicate ("main" vs "maintenance" for the same
# concept), and a handful of single-occurrence corrupted entries. Rather
# than let each variant become its own fake category, genuine typos are
# merged into their correct canonical label; entries that are too
# ambiguous to confidently correct (a bare "?", "y", "pre", "conditioning",
# and "main?" - note the "?" is the SOURCE itself flagging uncertainty,
# not something we're choosing to second-guess) are mapped to missing
# rather than guessed at.
REGIMEN_TYPE_CLEANUP = {
    "main": "maintenance",
    "consolidatio": "consolidation",
    "sct_p": "sct_prep",
    "tand_sct": "tandem_sct",
    "main?": None,
    "pre": None,
    "conditioning": None,
    "?": None,
    "y": None,
}


def clean_regimen_type(series: pd.Series) -> pd.Series:
    return series.replace(REGIMEN_TYPE_CLEANUP)


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_clinical.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))
    return pairs


def _prep_pairs_for_asof(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)
    return p


def add_current_line_features(pairs: pd.DataFrame) -> pd.DataFrame:
    line = pd.read_csv(f"{CLINICAL_DIR}/administered_line_deid.csv")
    line = line.sort_values("days_to_line_start").copy()
    line["days_to_line_start"] = line["days_to_line_start"].astype(float)
    line["days_to_line_end"] = line["days_to_line_end"].astype(float)

    p = _prep_pairs_for_asof(pairs)

    merged = pd.merge_asof(
        p,
        line[["public_id", "days_to_line_start", "days_to_line_end", "line"]],
        left_on="vt_days_to_visit",
        right_on="days_to_line_start",
        by="public_id",
        direction="backward",
    )

    still_active = (
        merged["days_to_line_end"].isna()
        | (merged["days_to_line_end"] >= merged["vt_days_to_visit"])
    )
    merged["current_line_is_active"] = still_active.fillna(False)
    merged["current_line_number"] = merged["line"]
    merged["line_duration_so_far"] = (
        merged["vt_days_to_visit"] - merged["days_to_line_start"]
    )
    # BUG FIX: previously computed unconditionally, so a patient in a gap
    # between lines (current_line_is_active=False) still showed "time
    # since the last line started" as if still in that line. Blank it in
    # a gap, matching how regimen_duration_so_far already behaves.
    merged.loc[~merged["current_line_is_active"], "line_duration_so_far"] = None

    # n_prior_lines_completed: count lines whose end was strictly before Vt.
    line_ended = line.dropna(subset=["days_to_line_end"]).sort_values(
        ["public_id", "days_to_line_end"]
    ).copy()
    line_ended["n_lines_ended_so_far"] = line_ended.groupby("public_id").cumcount() + 1
    line_ended = line_ended.sort_values("days_to_line_end")

    merged = pd.merge_asof(
        merged.sort_values("vt_days_to_visit"),
        line_ended[["public_id", "days_to_line_end", "n_lines_ended_so_far"]],
        left_on="vt_days_to_visit",
        right_on="days_to_line_end",
        by="public_id",
        direction="backward",
        allow_exact_matches=False,  # strictly BEFORE Vt, not on the same day
    )
    merged["n_prior_lines_completed"] = merged["n_lines_ended_so_far"].fillna(0).astype(int)

    keep_cols = [
        "pair_id",
        "current_line_number",
        "current_line_is_active",
        "line_duration_so_far",
        "n_prior_lines_completed",
    ]
    return merged[keep_cols]


def add_current_regimen_features(pairs: pd.DataFrame) -> pd.DataFrame:
    reg = pd.read_csv(f"{CLINICAL_DIR}/administered_regimen_line_deid.csv")
    reg = reg.dropna(subset=["public_id", "days_to_regimen_start"])
    reg["regimen_type"] = clean_regimen_type(reg["regimen_type"])
    reg = reg.sort_values("days_to_regimen_start").copy()
    reg["days_to_regimen_start"] = reg["days_to_regimen_start"].astype(float)
    reg["days_to_regimen_end"] = reg["days_to_regimen_end"].astype(float)

    p = _prep_pairs_for_asof(pairs)

    merged = pd.merge_asof(
        p,
        reg[
            [
                "public_id",
                "days_to_regimen_start",
                "days_to_regimen_end",
                "regimen_name",
                "regimen_type",
                "regimen_categories",
            ]
        ],
        left_on="vt_days_to_visit",
        right_on="days_to_regimen_start",
        by="public_id",
        direction="backward",
    )

    still_active = (
        merged["days_to_regimen_end"].isna()
        | (merged["days_to_regimen_end"] >= merged["vt_days_to_visit"])
    )
    # Blank out regimen name/type for rows where the matched regimen had
    # already ended before Vt (i.e. we're in a gap between regimens).
    for col in ["regimen_name", "regimen_type", "regimen_categories"]:
        merged.loc[~still_active.fillna(False), col] = None

    merged["current_regimen_name"] = merged["regimen_name"]
    merged["current_regimen_type"] = merged["regimen_type"]
    merged["current_regimen_categories"] = merged["regimen_categories"]
    merged["regimen_duration_so_far"] = (
        merged["vt_days_to_visit"] - merged["days_to_regimen_start"]
    )
    merged.loc[~still_active.fillna(False), "regimen_duration_so_far"] = None

    # n_prior_regimens: regimens that had already STARTED before Vt.
    reg_started = reg.sort_values(["public_id", "days_to_regimen_start"]).copy()
    reg_started["n_regimens_started_so_far"] = (
        reg_started.groupby("public_id").cumcount() + 1
    )
    reg_started = reg_started.sort_values("days_to_regimen_start")
    merged = pd.merge_asof(
        merged.sort_values("vt_days_to_visit"),
        reg_started[["public_id", "days_to_regimen_start", "n_regimens_started_so_far"]],
        left_on="vt_days_to_visit",
        right_on="days_to_regimen_start",
        by="public_id",
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_dup"),
    )
    merged["n_prior_regimens"] = merged["n_regimens_started_so_far"].fillna(0).astype(int)

    # prior_treatment_failure: any PRIOR regimen (ended strictly before Vt)
    # discontinued for disease progression/relapse or lack of response.
    FAILURE_REASONS = {"disease_progression_relapse", "lack_of_response"}
    reg_ended = reg.dropna(subset=["days_to_regimen_end"]).sort_values(
        ["public_id", "days_to_regimen_end"]
    ).copy()
    reg_ended["is_failure"] = reg_ended["reason_for_discontinuation"].isin(
        FAILURE_REASONS
    )
    reg_ended["n_failures_so_far"] = reg_ended.groupby("public_id")["is_failure"].cumsum()
    reg_ended = reg_ended.sort_values("days_to_regimen_end")

    merged = pd.merge_asof(
        merged.sort_values("vt_days_to_visit"),
        reg_ended[["public_id", "days_to_regimen_end", "n_failures_so_far"]],
        left_on="vt_days_to_visit",
        right_on="days_to_regimen_end",
        by="public_id",
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_dup2"),
    )
    merged["prior_treatment_failure"] = (
        merged["n_failures_so_far"].fillna(0) > 0
    )

    keep_cols = [
        "pair_id",
        "current_regimen_name",
        "current_regimen_type",
        "current_regimen_categories",
        "regimen_duration_so_far",
        "n_prior_regimens",
        "prior_treatment_failure",
    ]
    return merged[keep_cols]


def add_transplant_features(pairs: pd.DataFrame) -> pd.DataFrame:
    sct = pd.read_csv(f"{CLINICAL_DIR}/administered_sc_transplant_deid.csv")
    sct = sct.sort_values("days_to_sct").copy()
    sct["days_to_sct"] = sct["days_to_sct"].astype(float)
    sct["n_transplants_so_far"] = sct.groupby("public_id").cumcount() + 1
    # type_of_sct distinguishes autologous (patient's own cells, standard
    # of care) from allogeneic (donor cells - a fundamentally different,
    # higher-risk procedure, rare in this cohort but clinically important
    # when present).
    sct["is_allo"] = sct["type_of_sct"].str.startswith("allo", na=False)
    sct["n_allo_so_far"] = sct.groupby("public_id")["is_allo"].cumsum()

    p = _prep_pairs_for_asof(pairs)

    merged = pd.merge_asof(
        p,
        sct[["public_id", "days_to_sct", "n_transplants_so_far", "n_allo_so_far"]],
        left_on="vt_days_to_visit",
        right_on="days_to_sct",
        by="public_id",
        direction="backward",
        allow_exact_matches=False,  # strictly BEFORE Vt
    )

    merged["n_prior_transplants"] = merged["n_transplants_so_far"].fillna(0).astype(int)
    merged["has_prior_transplant"] = merged["n_prior_transplants"] > 0
    merged["has_prior_allo_transplant"] = merged["n_allo_so_far"].fillna(0) > 0
    merged["days_since_last_transplant"] = (
        merged["vt_days_to_visit"] - merged["days_to_sct"]
    )

    keep_cols = [
        "pair_id",
        "has_prior_transplant",
        "has_prior_allo_transplant",
        "n_prior_transplants",
        "days_since_last_transplant",
    ]
    return merged[keep_cols]


def summarize(df: pd.DataFrame) -> None:
    print("=== Step 7: Treatment-context feature coverage ===")
    print(f"current_line_number populated:      {df['current_line_number'].notna().mean()*100:.1f}%")
    print(f"current_line_is_active True:         {df['current_line_is_active'].mean()*100:.1f}%")
    print(f"current_regimen_name populated:      {df['current_regimen_name'].notna().mean()*100:.1f}%")
    print(f"has_prior_transplant True:           {df['has_prior_transplant'].mean()*100:.1f}%")
    print(f"has_prior_allo_transplant True:      {df['has_prior_allo_transplant'].mean()*100:.1f}%")
    print(f"prior_treatment_failure True:        {df['prior_treatment_failure'].mean()*100:.1f}%")
    print(f"n_prior_lines_completed - mean:      {df['n_prior_lines_completed'].mean():.2f}")
    print(f"n_prior_regimens - mean:             {df['n_prior_regimens'].mean():.2f}")


if __name__ == "__main__":
    pairs = load_pairs()

    line_feats = add_current_line_features(pairs)
    reg_feats = add_current_regimen_features(pairs)
    sct_feats = add_transplant_features(pairs)

    result = pairs.merge(line_feats, on="pair_id", how="left")
    result = result.merge(reg_feats, on="pair_id", how="left")
    result = result.merge(sct_feats, on="pair_id", how="left")

    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_with_treatment.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")