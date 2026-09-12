"""
Add drug-class exposure features to every visit pair, using
administered_agent_deid.csv.

Input:  data/clinical/visit_pairs_master.csv
Output: data/clinical/visit_pairs_master.csv   (overwritten, columns added)

Unlike administered_regimen_line_deid.csv (which gives a generic combined
category string like "imid_chemo_steroid"), this file has actual per-drug
records - so we can tell exactly which DRUG CLASSES a patient is currently
on / has ever been on, at each Vt. This matters because, e.g., anti-CD38
antibody exposure (daratumumab) is a much more specific and clinically
meaningful signal than a generic regimen-category label.

Same interval-based "active at Vt" logic as Step 7's line/regimen
features: active = start <= Vt AND (end is missing OR end >= Vt).

Drug classes tracked (from agent_category):
  pi     - proteasome inhibitor (bortezomib, carfilzomib, ixazomib)
  imid   - immunomodulatory drug (lenalidomide, pomalidomide, thalidomide)
  cd38   - anti-CD38 antibody (daratumumab)
  slamf7 - anti-SLAMF7 antibody (elotuzumab)
  bcma   - BCMA-targeted therapy

For each class, two features:
  currently_on_{class}   - True if a drug of this class is active at Vt
  prior_exposure_{class} - True if any drug of this class STARTED before Vt
                           (regardless of whether it's still active) -
                           i.e. "has this patient ever been exposed to this
                           drug class by this point in their treatment"
"""

import pandas as pd

CLINICAL_DIR = "data/clinical"

DRUG_CLASSES = ["pi", "imid", "cd38", "slamf7", "bcma", "steroid", "chemo"]


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_master.csv")
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(len(pairs)))

    drop_cols = [f"currently_on_{c}" for c in DRUG_CLASSES] + [
        f"prior_exposure_{c}" for c in DRUG_CLASSES
    ]
    return pairs.drop(columns=[c for c in drop_cols if c in pairs.columns])


def load_agents() -> pd.DataFrame:
    a = pd.read_csv(f"{CLINICAL_DIR}/administered_agent_deid.csv")
    a["days_to_agent_start"] = a["days_to_agent_start"].astype(float)
    a["days_to_agent_end"] = a["days_to_agent_end"].astype(float)
    return a


def add_drug_class_features(pairs: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    p = pairs.sort_values("vt_days_to_visit").copy()
    p["vt_days_to_visit"] = p["vt_days_to_visit"].astype(float)

    for drug_class in DRUG_CLASSES:
        cls_agents = agents[agents["agent_category"] == drug_class].sort_values(
            "days_to_agent_start"
        )

        # --- currently_on: active at Vt ---
        merged = pd.merge_asof(
            p[["pair_id", "public_id", "vt_days_to_visit"]],
            cls_agents[["public_id", "days_to_agent_start", "days_to_agent_end"]],
            left_on="vt_days_to_visit",
            right_on="days_to_agent_start",
            by="public_id",
            direction="backward",
        )
        still_active = merged["days_to_agent_end"].isna() | (
            merged["days_to_agent_end"] >= merged["vt_days_to_visit"]
        )
        currently_on = still_active.fillna(False) & merged["days_to_agent_start"].notna()
        p[f"currently_on_{drug_class}"] = currently_on.values

        # --- prior_exposure: ANY drug of this class started before Vt ---
        cls_started = cls_agents.copy()
        cls_started["n_started_so_far"] = cls_started.groupby("public_id").cumcount() + 1
        merged2 = pd.merge_asof(
            p[["pair_id", "public_id", "vt_days_to_visit"]],
            cls_started[["public_id", "days_to_agent_start", "n_started_so_far"]],
            left_on="vt_days_to_visit",
            right_on="days_to_agent_start",
            by="public_id",
            direction="backward",
            allow_exact_matches=True,  # exposure counts even if started ON Vt
        )
        p[f"prior_exposure_{drug_class}"] = (
            merged2["n_started_so_far"].fillna(0) > 0
        ).values

    return p.sort_values("pair_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print("=== Drug-class exposure feature coverage ===")
    for drug_class in DRUG_CLASSES:
        cur_pct = df[f"currently_on_{drug_class}"].mean() * 100
        prior_pct = df[f"prior_exposure_{drug_class}"].mean() * 100
        print(
            f"{drug_class:8s} currently_on: {cur_pct:5.1f}% | "
            f"ever_exposed_by_Vt: {prior_pct:5.1f}%"
        )


if __name__ == "__main__":
    pairs = load_pairs()
    agents = load_agents()

    result = add_drug_class_features(pairs, agents)
    summarize(result)

    out_path = f"{CLINICAL_DIR}/visit_pairs_master.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (shape: {result.shape})")