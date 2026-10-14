"""
Direction 2 follow-up -- STAGE 1 of the TRAIN-ONLY pathway-selection redesign (see PATHWAY_TRAINONLY_PROTOCOL.md).

Responds to the concern that validation/test patients influenced pathway selection. The earlier analysis chose its five-pathway
family as the five smallest p-values of the pooled S3 ordinal regression (all retained patients) and z-scored pathway scores over
all 751 RNA samples. Here, using ONLY patients in the original training partition (split == "train"):

  * pathway scores are z-scored with the distinct-sample means / SDs (ddof = 1) of TRAIN RNA-available rows only;
  * the frozen D2 `regression_analysis` (S3_adjusted: ordinal GEE, 16 continuous Model-C covariates + log1p RNA age + exact strata;
    covariate transformation / winsorisation / standardisation computed on the final TRAIN rows) is run on TRAIN rows only;
  * SELECTION RULE (unchanged): the five pathways with the smallest robust Wald p-values. No direction, q-value or
    biological-coherence filter is added or removed.
  * Stop conditions: the regression is not `complete`, or fewer than five pathways have valid estimates. No alternative rule,
    model or pathway count is tried.

Validation and test patients are removed from the frame before any pathway statistic is computed (cohort-wide count checks of the
frozen loaders still run first). The selected family is written to a hashed artifact that Stage 2
(direction2_proliferation_replication_trainonly.py) reads. Existing outputs are never overwritten.

POST HOC: the earlier pooled results (which included validation and test patients) were seen before this redesign, so this is
not a blind selection and not external validation.

Run from the repo root:
    python -m explainability.direction2_pathway_selection_trainonly
Outputs: artifacts/results_clean_rerun/direction2_replication_trainonly/
"""

import datetime
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from explainability import direction2_association as d2a
from explainability.direction2_common import (
    EXPECTED, load_feature_sets, package_versions, pathway_columns, require, select_universe, sha256_file, write_csv,
    write_json,
)

OUT = "artifacts/results_clean_rerun/direction2_replication_trainonly"
PROTOCOL = "explainability/PATHWAY_TRAINONLY_PROTOCOL.md"
PATIENT_SPLITS = "data/splits/patient_split_assignments.csv"
DEV_SPLITS = ("train",)
N_SELECT = 5
ANALYSIS = "S3_trainonly"
RULE = ("the five pathways with the smallest robust Wald p-values in the TRAIN-ONLY frozen D2 S3 ordinal regression "
        "(no direction, q-value or biological-coherence filter); unchanged from the earlier analysis")
LABEL = ("DIRECTION 2 FOLLOW-UP (TRAIN-ONLY SELECTION): pathway selection using original-training-partition patients only; "
         "post hoc, exploratory, not external validation")
OLD_FAMILY = ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle", "Myc Targets V1", "Myc Targets V2"]  # comparison only, after selection


def build_pathway_z_dev(m_dev, path_cols):
    """z-score each pathway across DISTINCT (patient, sample) vectors among DEVELOPMENT rows only; Z indexed by pair_id."""
    r = m_dev[m_dev["rna_available"]]
    require(not r[path_cols].isna().any().any(), "RNA-available development rows with missing pathway values")
    h = pd.util.hash_pandas_object(r[path_cols].round(8), index=False).to_numpy()
    pid = pd.factorize(r["public_id"])[0]
    first = ~pd.DataFrame({"pid": pid, "h": h}).duplicated().to_numpy()
    distinct = r.loc[first, path_cols]
    mu, sd = distinct.mean(axis=0), distinct.std(axis=0, ddof=1)
    require((sd > 0).all(), "a pathway has zero variance across distinct development samples")
    Z = (r[path_cols] - mu) / sd
    Z.index = r["pair_id"].to_numpy()
    return Z, {"n_distinct_samples": int(first.sum()), "n_rna_rows": int(len(r)),
               "pathway_mean": {p: float(mu[p]) for p in path_cols}, "pathway_sd": {p: float(sd[p]) for p in path_cols}}


def main():
    t0 = time.time()
    print(LABEL)
    marker = f"{OUT}/selection_artifact.json"
    if os.path.exists(marker):
        raise SystemExit(f"[pathway_selection_trainonly] {marker} already exists: existing outputs are never overwritten. "
                         f"Delete the folder deliberately to rerun.")
    protocol_sha = sha256_file(PROTOCOL)
    fs = load_feature_sets()
    path_cols = pathway_columns(fs)
    master = d2a.load_master_full()                         # cohort-wide count and split-disjointness checks (frozen)
    ps = pd.read_csv(PATIENT_SPLITS).set_index("public_id")["split"]
    require((master["public_id"].map(ps) == master["split"]).all(), "pair-level and patient-level splits disagree")

    # ---- development frame: validation / test rows are removed BEFORE any pathway statistic ----
    U_full = select_universe(master, "ordinal")             # frozen whole-universe count checks
    dev_mask_master = master["split"].isin(DEV_SPLITS)
    U = U_full[U_full["split"].isin(DEV_SPLITS)].reset_index(drop=True)
    m_dev = master[dev_mask_master].copy()
    n_excluded_patients = int(master.loc[~dev_mask_master, "public_id"].nunique())
    n_excluded_rows = int((~dev_mask_master).sum())
    dev_ids = set(m_dev["public_id"])
    del master, U_full                                      # nothing outside the development set is used below
    require(set(U["split"]) == {"train"} and set(m_dev["split"]) == {"train"}, "a non-train row entered the development frame")
    require(set(U["public_id"]) <= dev_ids and all(ps[p] == "train" for p in dev_ids), "development ids are not all original-train")

    Z, zinfo = build_pathway_z_dev(m_dev, path_cols)
    require(set(U["pair_id"]) <= set(Z.index), "universe rows without development-only z-scores")
    print(f"Development set: {len(U)} universe rows / {U['public_id'].nunique()} patients "
          f"({n_excluded_patients} validation/test patients, {n_excluded_rows} rows removed); "
          f"{zinfo['n_distinct_samples']} distinct RNA samples z-scored")

    # ---- frozen D2 S3 ordinal regression, development rows only ----
    df, acc = d2a.regression_analysis("ordinal", U, Z, path_cols, ANALYSIS, False)
    print(f"{ANALYSIS}: status={acc['analysis_status']} rows {acc['universe_rows']} -> {acc['final_rows']} "
          f"({acc['final_patients']} patients, {acc['final_strata']} strata)")
    ok = df[~df["status"].str.startswith("failed")]
    os.makedirs(OUT, exist_ok=True)
    files = []
    p = f"{OUT}/S3_trainonly_regression_results.csv"
    write_csv(df.assign(rank_by_p=df["p_wald_robust"].rank(method="first")), p, float_format="%.10g")
    files.append(p)
    p = f"{OUT}/S3_trainonly_row_accounting.json"
    write_json(acc, p)
    files.append(p)

    stop = None
    if acc["analysis_status"] != "complete" or len(ok) < N_SELECT:
        stop = (f"STOP: S3 train-only regression status {acc['analysis_status']!r}, {len(ok)} valid pathways "
                f"(need a complete analysis and >= {N_SELECT}); no alternative rule or model is tried")
        print(stop)
    ranked = ok.sort_values("p_wald_robust", kind="mergesort").reset_index(drop=True)
    sel = list(ranked["pathway"][:N_SELECT]) if stop is None else []
    ranking = [{"rank": int(i + 1), "pathway": r.pathway, "estimate": float(r.estimate), "odds_ratio_per_SD": float(r.odds_ratio_per_SD),
                "p_wald_robust": float(r.p_wald_robust), "q_bh": float(r.q_bh), "selected": bool(i < N_SELECT)}
               for i, r in ranked.iterrows()]
    artifact = {"label": LABEL, "rule": RULE, "analysis": ANALYSIS, "n_selected": N_SELECT, "family": sel,
                "ranking_top_15": ranking[:15], "stop_reason": stop,
                "development_splits": list(DEV_SPLITS),
                "development": {"universe_rows": acc["universe_rows"], "universe_patients": acc["universe_patients"],
                                "final_rows": acc["final_rows"], "final_patients": acc["final_patients"],
                                "final_strata": acc["final_strata"], "rows_lost_sparse_strata": acc.get("rows_lost_sparse_strata"),
                                "n_distinct_rna_samples_zscored": zinfo["n_distinct_samples"]},
                "excluded_from_selection": {"validation_and_test_patients": n_excluded_patients, "rows": n_excluded_rows},
                "development_patient_ids_sha256": hashlib.sha256("\n".join(sorted(dev_ids)).encode()).hexdigest(),
                "protocol_sha256": protocol_sha, "code_sha256": sha256_file(os.path.abspath(__file__)),
                "input_master_sha256": sha256_file(d2a.MASTER_PATH), "versions": package_versions()}
    p = f"{OUT}/selection_artifact.json"
    write_json(artifact, p)
    files.append(p)
    art_sha = sha256_file(p)                                # hashed BEFORE any comparison / Stage 2

    # ---- comparison with the earlier family (reporting only; written after the artifact; never fed back) ----
    comp = {"label": LABEL, "selection_artifact_sha256": art_sha, "selected_family": sel, "earlier_family": OLD_FAMILY,
            "shared": sorted(set(sel) & set(OLD_FAMILY)), "newly_selected": sorted(set(sel) - set(OLD_FAMILY)),
            "no_longer_selected": sorted(set(OLD_FAMILY) - set(sel)), "same_set": bool(set(sel) == set(OLD_FAMILY)),
            "note": "reporting only; does not influence selection"}
    p = f"{OUT}/selection_comparison_with_earlier_family.json"
    write_json(comp, p)
    files.append(p)
    hashes = {os.path.relpath(f, OUT).replace("\\", "/"): sha256_file(f) for f in sorted(files)}
    write_json({"label": LABEL, "files": hashes, "selection_artifact_sha256": art_sha, "protocol_sha256": protocol_sha,
                "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "runtime_seconds": round(time.time() - t0, 1)}, f"{OUT}/selection_manifest.json")
    print("\nTop 10 by p (development set only):")
    for r in ranking[:10]:
        print(f"  {r['rank']:2d} {r['pathway']:<28s} OR/SD {r['odds_ratio_per_SD']:.3f}  p {r['p_wald_robust']:.3g}  q {r['q_bh']:.3g}"
              f"{'  <- selected' if r['selected'] else ''}")
    print(f"Selected: {sel}\nArtifact SHA256: {art_sha}\nSame set as earlier family: {comp['same_set']}")


if __name__ == "__main__":
    main()
