"""
Step 10 (XAI): Reliability and stability checks.

Four checks, all reusing existing Step 3/7/8 outputs and Step 2's
X_test arrays - no new SHAP computation, no retraining.

1. Missingness dominance: formalizes the already-observed
   rna_available = 0 result (Step 4/8) as an explicit pass/fail record.
2. Repeated-measures sensitivity: recomputes Step 3's global importance
   using ONE row per patient (their latest eligible test visit) and
   compares top-10 rankings against the full-row version. Large
   disagreement would mean a small number of patients' repeated visits
   are driving the full-row ranking.
3. Correlated-feature stability: pairwise correlation among raw values
   for three groups flagged during Steps 5/7/8 as needing this check -
   the M-protein trio (current/previous/change, central to the PD
   finding), the kappa FLC trio, and the 50 Hallmark pathway scores.
   Any pair above CORRELATION_FLAG_THRESHOLD is flagged for cautious
   individual interpretation.
4. Cross-model consistency: C vs D's non-RNA top-10 features, since D
   is a strict superset of C - should show substantial overlap.

Outputs (under artifacts/results_clean_rerun/xai/):
- reliability_checks_summary.json
- repeated_measures_importance_{key}.csv (per-patient-latest-visit ranking)
- correlation_flags.csv
"""

import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "modeling")
from modeling.train_models import load_master_data, load_preprocessor_and_meta, get_binary_xyz, get_multiclass_xyz

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
SHAP_DIR = f"{OUT_DIR}/shap_artifacts"

D_MODELS = [
    {"task": "binary", "stage": "d", "algorithm": "lightgbm"},
    {"task": "multiclass", "stage": "d", "algorithm": "xgboost"},
]
CD_PAIRS = [
    {"c_key": "binary_c_lightgbm", "d_key": "binary_d_lightgbm"},
    {"c_key": "multiclass_a_xgboost", "d_key": "multiclass_d_xgboost"},  # A is the non-RNA reference here
]

CORRELATION_FLAG_THRESHOLD = 0.7
TOP_N_OVERLAP = 10

FEATURE_GROUPS_TO_CHECK = {
    "m_protein_trio": ["m_protein_current", "m_protein_previous", "m_protein_change"],
    "kappa_flc_trio": ["kappa_flc_current", "kappa_flc_previous", "kappa_flc_change"],
}


def check_missingness_dominance():
    print("\n" + "=" * 70)
    print("CHECK 1: Missingness dominance (rna_available)")
    print("=" * 70)
    results = {}
    for spec in D_MODELS:
        key = f"{spec['task']}_{spec['stage']}_{spec['algorithm']}"
        df = pd.read_csv(f"{OUT_DIR}/global_importance_{key}.csv")
        row = df[df["original_column"] == "rna_available"]
        importance = float(row["importance"].iloc[0]) if len(row) else None
        rank = int(row["rank"].iloc[0]) if len(row) else None
        passed = importance is not None and importance < 1e-6
        results[key] = {"importance": importance, "rank": rank, "passed": passed}
        print(f"  {key}: rna_available importance={importance}, rank={rank} -> "
              f"{'PASS' if passed else 'FLAG'}")
    return results


def check_repeated_measures_sensitivity():
    print("\n" + "=" * 70)
    print("CHECK 2: Repeated-measures sensitivity (one row per patient)")
    print("=" * 70)
    results = {}
    for spec in D_MODELS:
        task, stage, algorithm = spec["task"], spec["stage"], spec["algorithm"]
        key = f"{task}_{stage}_{algorithm}"

        full_importance = pd.read_csv(f"{OUT_DIR}/global_importance_{key}.csv")
        full_top10 = set(full_importance.head(TOP_N_OVERLAP)["original_column"])

        shap_values = np.load(f"{SHAP_DIR}/{key}_shap_values.npy")
        with open(f"{SHAP_DIR}/{key}_feature_names.json") as f:
            feature_names = json.load(f)
        row_ids = pd.read_csv(f"{SHAP_DIR}/{key}_row_ids.csv")
        rollup_df = pd.read_csv(f"{OUT_DIR}/rollup_mapping_{key}.csv")
        fn_to_original = dict(zip(rollup_df["output_feature"], rollup_df["original_column"]))

        df = load_master_data()
        preprocessor, meta = load_preprocessor_and_meta(stage)
        getter = get_binary_xyz if task == "binary" else get_multiclass_xyz
        _, _, _, pid_te_full, _ = getter(df, stage, preprocessor, meta, "test")
        master = load_master_data().set_index("pair_id")
        vt_days = master.reindex(row_ids["pair_id"])["vt_days_to_visit"].to_numpy()

        one_per_patient = (pd.DataFrame({"pair_id": row_ids["pair_id"], "public_id": row_ids["public_id"],
                                          "vt_days_to_visit": vt_days, "row_pos": np.arange(len(row_ids))})
                            .sort_values("vt_days_to_visit").groupby("public_id").tail(1))
        keep_pos = one_per_patient["row_pos"].to_numpy()
        print(f"  {key}: {len(row_ids)} rows -> {len(keep_pos)} rows (one per patient, latest visit)")

        if task == "binary":
            sub_shap = shap_values[keep_pos]
            per_output = np.mean(np.abs(sub_shap), axis=0)
        else:
            sub_shap = shap_values[:, keep_pos, :]
            per_output = np.mean(np.mean(np.abs(sub_shap), axis=1), axis=0)

        sub_df = pd.DataFrame({"output_feature": feature_names,
                                "original_column": [fn_to_original[f] for f in feature_names],
                                "importance": per_output})
        sub_rolled = (sub_df.groupby("original_column")["importance"].sum()
                      .reset_index().sort_values("importance", ascending=False).reset_index(drop=True))
        sub_rolled.insert(0, "rank", range(1, len(sub_rolled) + 1))
        sub_rolled.to_csv(f"{OUT_DIR}/repeated_measures_importance_{key}.csv", index=False)

        sub_top10 = set(sub_rolled.head(TOP_N_OVERLAP)["original_column"])
        overlap = full_top10 & sub_top10
        jaccard = len(overlap) / len(full_top10 | sub_top10)
        results[key] = {"full_row_top10": sorted(full_top10), "one_per_patient_top10": sorted(sub_top10),
                         "overlap": sorted(overlap), "jaccard": jaccard}
        print(f"  Full-row top10:        {sorted(full_top10)}")
        print(f"  One-per-patient top10: {sorted(sub_top10)}")
        print(f"  Jaccard overlap: {jaccard:.3f}")
    return results


def check_correlated_feature_stability():
    print("\n" + "=" * 70)
    print("CHECK 3: Correlated-feature stability")
    print("=" * 70)
    master = load_master_data()

    flags = []
    for group_name, cols in FEATURE_GROUPS_TO_CHECK.items():
        available = [c for c in cols if c in master.columns]
        corr = master[available].corr(method="spearman")
        print(f"\n  {group_name}:")
        print(corr.round(3).to_string())
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                r = corr.iloc[i, j]
                flagged = abs(r) > CORRELATION_FLAG_THRESHOLD
                flags.append({"group": group_name, "feature_1": available[i], "feature_2": available[j],
                              "spearman_r": float(r), "flagged": bool(flagged)})
                if flagged:
                    print(f"    FLAGGED: {available[i]} vs {available[j]}: r={r:.3f}")

    with open("data_pipeline/model_feature_sets.json") as f:
      fs = json.load(f)
    rna_additions = set(fs["model_d"]) - set(fs["model_c"])  # pathways + days_since_rna_sample
    known_pathways = rna_additions - {"days_since_rna_sample"}  # rna_available isn't in this JSON at all
    known_pathways = {p for p in known_pathways if p in master.columns}
    pathway_corr = master[list(known_pathways)].corr(method="spearman")
    high_pairs = []
    for i, c1 in enumerate(pathway_corr.columns):
        for j, c2 in enumerate(pathway_corr.columns):
            if j <= i:
                continue
            r = pathway_corr.loc[c1, c2]
            if abs(r) > CORRELATION_FLAG_THRESHOLD:
                high_pairs.append({"group": "hallmark_pathways", "feature_1": c1, "feature_2": c2,
                                    "spearman_r": float(r), "flagged": True})
    flags.extend(high_pairs)
    print(f"\n  Hallmark pathways ({len(known_pathways)} checked): "
          f"{len(high_pairs)} pairs above |r|={CORRELATION_FLAG_THRESHOLD}")
    if high_pairs:
        for p in sorted(high_pairs, key=lambda x: -abs(x["spearman_r"]))[:10]:
            print(f"    {p['feature_1']} vs {p['feature_2']}: r={p['spearman_r']:.3f}")

    pd.DataFrame(flags).to_csv(f"{OUT_DIR}/correlation_flags.csv", index=False)
    return flags


def check_cross_model_consistency():
    print("\n" + "=" * 70)
    print("CHECK 4: Cross-model consistency (C/A vs D, non-RNA features)")
    print("=" * 70)
    results = {}
    for pair in CD_PAIRS:
        c_df = pd.read_csv(f"{OUT_DIR}/global_importance_{pair['c_key']}.csv")
        d_df = pd.read_csv(f"{OUT_DIR}/global_importance_{pair['d_key']}.csv")

        rollup_ref = pd.read_csv(f"{OUT_DIR}/rollup_mapping_{pair['d_key']}.csv")
        rna_cols = set(rollup_ref["original_column"]) - set(pd.read_csv(f"{OUT_DIR}/rollup_mapping_{pair['c_key']}.csv")["original_column"])

        d_non_rna = d_df[~d_df["original_column"].isin(rna_cols)]
        c_top10 = set(c_df.head(TOP_N_OVERLAP)["original_column"])
        d_top10_non_rna = set(d_non_rna.head(TOP_N_OVERLAP)["original_column"])
        overlap = c_top10 & d_top10_non_rna
        jaccard = len(overlap) / len(c_top10 | d_top10_non_rna)

        label = f"{pair['c_key']}_vs_{pair['d_key']}"
        results[label] = {"c_top10": sorted(c_top10), "d_top10_non_rna": sorted(d_top10_non_rna),
                           "overlap": sorted(overlap), "jaccard": jaccard}
        print(f"\n  {label}:")
        print(f"    {pair['c_key']} top10: {sorted(c_top10)}")
        print(f"    {pair['d_key']} top10 (non-RNA): {sorted(d_top10_non_rna)}")
        print(f"    Jaccard: {jaccard:.3f}")
    return results


if __name__ == "__main__":
    r1 = check_missingness_dominance()
    r2 = check_repeated_measures_sensitivity()
    r3 = check_correlated_feature_stability()
    r4 = check_cross_model_consistency()

    with open(f"{OUT_DIR}/reliability_checks_summary.json", "w") as f:
        json.dump({"missingness_dominance": r1, "repeated_measures_sensitivity": r2,
                    "correlated_feature_stability": r3, "cross_model_consistency": r4}, f, indent=2, default=str)

    print(f"\n{'='*70}\nSTEP 10 COMPLETE\n{'='*70}")
    print(f"Summary: {OUT_DIR}/reliability_checks_summary.json")
    print(f"Correlation flags: {OUT_DIR}/correlation_flags.csv")