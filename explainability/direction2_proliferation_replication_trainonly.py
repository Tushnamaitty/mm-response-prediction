"""
Direction 2 follow-up -- STAGE 2 of the TRAIN-ONLY pathway-selection redesign (see PATHWAY_TRAINONLY_PROTOCOL.md).

*** POST HOC, EXPLORATORY. NOT untouched confirmation and NOT external validation. ***
The earlier pooled results, the earlier family and the earlier test result were seen before this redesign.

Differences from explainability/direction2_proliferation_replication.py (which is unchanged and whose outputs are kept):
  * the family is NOT hard-coded: it is read from the hashed Stage-1 artifact (five smallest p-values of the TRAIN-ONLY S3 ordinal
    regression; selection rule unchanged);
  * composite construction, covariate-transformation parameters and the discovery model use TRAIN patients only (the earlier
    version used train + validation);
  * the single primary directional test is run in the HELD-OUT patients = validation + test (the earlier version used test only);
  * held-out outcomes are blanked in the frame used for composite construction and discovery and are attached only after the frozen
    discovery artifact has been written and hashed (blinding by data, not only by code order).
Everything else -- composite (PC1 of the family, loadings > 0 gate), frozen S3 ordinal formulation, acceptance criteria (all
loadings positive; all six classes in discovery; design full rank; model converged; composite coefficient negative with one-sided
p < 0.025), support checks (>= 80 held-out patients; all six classes; eta non-degenerate), validity flag, single one-sided test,
interpretation rule and the descriptive 2,000-draw bootstrap -- is the earlier code, imported unchanged where it is family-independent.

STOPPING RULE: if any discovery gate fails the script stops BEFORE any held-out outcome is read; nothing is retried.

Run from the repo root (after direction2_pathway_selection_trainonly):
    python -m explainability.direction2_proliferation_replication_trainonly
Outputs: artifacts/results_clean_rerun/direction2_replication_trainonly/ (never overwrites).
"""

import datetime
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from explainability import direction2_proliferation_replication as base
from explainability.direction2_association import strata_key
from explainability.direction2_common import (
    RESPONSE_RANK, S3_MIN_STRATUM_ROWS, STALENESS_COL, continuous_matrix, S3_CONTINUOUS_FEATURES,
)
from explainability.direction3_common import (
    RANDOM_SEED, load_feature_sets, load_master, pathway_columns, require, sha256_file, write_csv, write_json,
    write_output_manifest,
)
from explainability.direction3_molecular import gee_fit, locate_terms

OUT = "artifacts/results_clean_rerun/direction2_replication_trainonly"
PROTOCOL = "explainability/PATHWAY_TRAINONLY_PROTOCOL.md"
LABEL = ("DIRECTION 2 FOLLOW-UP (TRAIN-ONLY SELECTION AND DISCOVERY, HELD-OUT VALIDATION+TEST EVALUATION): post hoc, exploratory; "
         "NOT untouched confirmation; NOT external validation")
FAMILY_STATEMENT = ("family = five smallest p-values of the TRAIN-ONLY S3 ordinal regression (rule unchanged from the earlier "
                    "analysis); composite construction and discovery use train patients only; the earlier pooled results, which "
                    "included validation and test patients, were seen before this redesign")
MULTIPLICITY_STATEMENT = ("one primary test within this analysis; project-level post hoc selection remains (earlier pooled looks, "
                          "earlier replication) and must be acknowledged")
DISC_SPLITS = ("train",)
HELDOUT_SPLITS = ("val", "test")
ALPHA_ONE_SIDED, MIN_TEST_PATIENTS, N_BOOT, N_CLASSES = base.ALPHA_ONE_SIDED, base.MIN_TEST_PATIENTS, base.N_BOOT, base.N_CLASSES
COV_COLS, OUTCOME_COLS = base.COV_COLS, base.OUTCOME_COLS
BLANK_COLS = ["improved", "exact_next_response", "time_gap_days"]


def feature_cols(family):
    return (["pair_id", "public_id", "vt_disease_response", "current_line_number", "line_group",
             "currently_on_pi", "currently_on_imid", "currently_on_cd38", STALENESS_COL] + COV_COLS + list(family))


def build_composite(rna_rows, path_cols, family):
    require(all(p in path_cols for p in family), "a family pathway is missing from the pathway columns")
    first = base.distinct_sample_mask(rna_rows, path_cols)
    S = rna_rows.loc[first, list(family)]
    mu, sd = S.mean(axis=0), S.std(axis=0, ddof=1)
    require((sd > 0).all(), "a family pathway has zero variance")
    Z = ((S - mu) / sd).to_numpy(dtype=float)
    R = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(R)
    lam, vec = float(w[-1]), V[:, -1].copy()
    if vec.sum() < 0:
        vec = -vec
    comp = (Z @ vec) / np.sqrt(lam)
    return {"names": list(family), "mean": [float(x) for x in mu], "sd": [float(x) for x in sd],
            "loadings": [float(x) for x in vec], "lambda1": lam, "variance_explained": lam / len(family),
            "correlation": R.tolist(), "n_distinct_samples": int(first.sum()),
            "composite_variance_in_discovery": float(comp.var(ddof=1)),
            "all_loadings_positive": bool((vec > 0).all())}


def run_discovery(master, path_cols, family, disc_splits):
    """Composite + discovery model on rows whose split is in `disc_splits`; same logic as the earlier run_discovery."""
    disc_mask = master["split"].isin(disc_splits)
    require(bool(master.loc[disc_mask, "split"].isin(disc_splits).all()), "a non-discovery row entered the discovery mask")
    rna_d = master[disc_mask & master["rna_avail"]]
    comp_art = build_composite(rna_d, path_cols, family)            # outcome-blind, discovery rows only

    disc = base.prepare_rows(master, disc_mask)
    acc = {"discovery_rows_before_exclusions": int(len(disc)), "discovery_patients_before": int(disc["public_id"].nunique())}
    miss = disc[COV_COLS].isna().any(axis=1).to_numpy()
    acc["rows_lost_incomplete_covariates"] = int(miss.sum())
    cc = disc[~miss].reset_index(drop=True)
    y = cc["exact_next_response"].map(RESPONSE_RANK).to_numpy(dtype=int)
    key = strata_key(cc)
    sparse = (key.map(key.value_counts()).to_numpy() < S3_MIN_STRATUM_ROWS)
    acc["rows_lost_sparse_strata"] = int(sparse.sum())
    cc, key, y = cc[~sparse].reset_index(drop=True), key[~sparse].reset_index(drop=True), y[~sparse]
    novar = pd.Series(y).groupby(key).transform("nunique").to_numpy() < 2
    acc["rows_lost_no_outcome_variation_strata"] = int(novar.sum())
    cc, key, y = cc[~novar].reset_index(drop=True), key[~novar].reset_index(drop=True), y[~novar]
    acc.update({"discovery_rows": int(len(cc)), "discovery_patients": int(cc["public_id"].nunique()),
                "discovery_strata": int(key.nunique()), "outcome_class_counts": base.class_counts(y)})

    cov_params = base.fit_covariate_params(cc)
    Zc, stale_z = base.apply_covariates(cc, cov_params)
    require(np.allclose(Zc, continuous_matrix(cc, features=S3_CONTINUOUS_FEATURES), atol=1e-12, rtol=0),
            "covariate transformation differs from the frozen D2 continuous_matrix")
    comp = base.composite_values(cc, comp_art)
    D = pd.get_dummies(key, drop_first=True, dtype=float)
    dummy_names = [str(c) for c in D.columns]
    reference_stratum = sorted(set(key))[0]
    X = np.column_stack([comp, Zc, stale_z[:, None], D.to_numpy(dtype=float)])
    names = ["composite"] + COV_COLS + ["log1p_days_since_rna_sample"] + [f"stratum[{c}]" for c in dummy_names]

    gates = {"all_five_loadings_positive": comp_art["all_loadings_positive"],
             "all_six_outcome_classes_represented": bool(set(np.unique(y)) == set(range(N_CLASSES))),
             "design_full_rank": bool(int(np.linalg.matrix_rank(X)) == X.shape[1])}
    result = {"beta_disc": None}
    if all(gates.values()):
        try:
            mod, r = gee_fit("ordinal", y, X, pd.factorize(cc["public_id"])[0])
            params = np.asarray(r.params)
            n_thr = N_CLASSES - 1
            require(len(params) == n_thr + X.shape[1], "parameter vector length differs from thresholds + design columns")
            cont_names = ["composite"] + COV_COLS + ["log1p_days_since_rna_sample"]
            idx = locate_terms(mod, {nm: X[:, j] for j, nm in enumerate(cont_names)})
            for j, nm in enumerate(cont_names):
                require(idx[nm] == n_thr + j, f"unexpected parameter layout for {nm}")
            ic = idx["composite"]
            beta, se = float(params[ic]), float(np.asarray(r.bse)[ic])
            p1 = float(norm.cdf(beta / se))
            gates["model_identifiable_and_converged"] = True
            gates["composite_negative_one_sided_p_lt_0.025"] = bool(beta < 0 and p1 < ALPHA_ONE_SIDED)
            result = {"beta_disc": beta, "se_disc": se, "z_disc": beta / se, "one_sided_p_disc": p1,
                      "ci95_disc": [float(beta - 1.959963984540054 * se), float(beta + 1.959963984540054 * se)],
                      "thresholds": [float(x) for x in params[:n_thr]],
                      "clinical_coefficients": [float(x) for x in params[n_thr + 1:]],
                      "clinical_coefficient_names": names[1:]}
        except Exception as e:                                # no fallback model
            gates["model_identifiable_and_converged"] = False
            gates["failure_reason"] = f"{type(e).__name__}: {e}"
    all_ok = all(v for k, v in gates.items() if k != "failure_reason") and "composite_negative_one_sided_p_lt_0.025" in gates
    art = {"label": LABEL, "family": list(family), "family_statement": FAMILY_STATEMENT, "composite": comp_art,
           "covariate_params": cov_params, "dummy_names": dummy_names, "reference_stratum": reference_stratum,
           "retained_strata": sorted(set(key)), "discovery": result,
           "discovery_splits": list(disc_splits),
           "discovery_patient_ids_sha256": hashlib.sha256(
               "\n".join(sorted(cc["public_id"].astype(str).unique())).encode()).hexdigest(),
           "min_stratum_rows": S3_MIN_STRATUM_ROWS}
    return art, gates, acc, all_ok, cc


def score_heldout_features(master, art, heldout_splits):
    """Feature-only frame (NO outcome columns) of held-out rows, scored with the frozen artifact."""
    held = base.prepare_rows(master, master["split"].isin(heldout_splits))
    tf = held[feature_cols(art["family"])].copy()
    require(not (OUTCOME_COLS & set(tf.columns)), "outcome column present in the held-out feature frame")
    acc = {"heldout_rows_before_exclusions": int(len(tf)), "heldout_patients_before": int(tf["public_id"].nunique())}
    miss = tf[COV_COLS].isna().any(axis=1).to_numpy()
    acc["rows_lost_incomplete_covariates"] = int(miss.sum())
    tf = tf[~miss].reset_index(drop=True)
    key = strata_key(tf)
    keep = key.isin(art["retained_strata"]).to_numpy()
    acc["rows_lost_strata_not_retained_in_discovery"] = int((~keep).sum())
    tf, key = tf[keep].reset_index(drop=True), key[keep].reset_index(drop=True)
    Zc, stale_z = base.apply_covariates(tf, art["covariate_params"])
    Dm = np.column_stack([(key == nm).to_numpy(dtype=float) for nm in art["dummy_names"]]) if art["dummy_names"] \
        else np.zeros((len(tf), 0))
    coef = np.asarray(art["discovery"]["clinical_coefficients"])
    require(len(coef) == Zc.shape[1] + 1 + Dm.shape[1], "clinical coefficient vector length mismatch")
    eta = Zc @ coef[:Zc.shape[1]] + stale_z * coef[Zc.shape[1]] + Dm @ coef[Zc.shape[1] + 1:]
    comp = base.composite_values(tf, art["composite"])
    acc.update({"heldout_rows": int(len(tf)), "heldout_patients": int(tf["public_id"].nunique())})
    return tf, comp, eta, acc


def load_selection():
    """Read the Stage-1 artifact and check it against the manifest, the protocol and the stop rule."""
    p = f"{OUT}/selection_artifact.json"
    require(os.path.exists(p), "Stage-1 selection artifact not found: run direction2_pathway_selection_trainonly first")
    sha = sha256_file(p)
    man = json.load(open(f"{OUT}/selection_manifest.json", encoding="utf-8"))
    require(man["selection_artifact_sha256"] == sha, "selection artifact hash differs from the Stage-1 manifest")
    art = json.load(open(p, encoding="utf-8"))
    require(art["stop_reason"] is None and len(art["family"]) == art["n_selected"] == 5, "Stage 1 did not select a family")
    require(art["protocol_sha256"] == sha256_file(PROTOCOL), "protocol file changed since Stage 1")
    require(art["development_splits"] == list(DISC_SPLITS), "Stage-1 development splits differ from Stage-2 discovery splits")
    return art["family"], sha, art


def main():
    t0 = time.time()
    print(LABEL)
    marker = f"{OUT}/output_manifest.json"
    if os.path.exists(marker):
        raise SystemExit(f"[proliferation_replication_trainonly] {marker} already exists: existing outputs are never overwritten. "
                         f"Delete the folder deliberately to rerun.")
    family, sel_sha, sel_art = load_selection()
    print(f"Family (from Stage-1 artifact {sel_sha[:12]}...): {family}")
    fs = load_feature_sets()
    path_cols = pathway_columns(fs)
    master = load_master()                                   # frozen cohort counts and patient-disjoint splits asserted
    heldout_ids = set(master.loc[master["split"].isin(HELDOUT_SPLITS), "public_id"])
    blind_cols = BLANK_COLS + [c for c in master.columns if c.startswith("vt1_")]
    blind = master.copy()
    blind.loc[~blind["split"].isin(DISC_SPLITS), blind_cols] = np.nan   # held-out outcomes are not available to discovery
    require(blind.loc[~blind["split"].isin(DISC_SPLITS), "exact_next_response"].isna().all(), "held-out outcomes not blanked")

    # ---------------- discovery (train only; held-out outcomes blanked) ----------------
    art, gates, disc_acc, discovery_ok, cc = run_discovery(blind, path_cols, family, DISC_SPLITS)
    require(not (set(cc["public_id"]) & heldout_ids), "a held-out patient entered discovery")
    del blind
    art["selection_artifact_sha256"] = sel_sha
    art["protocol_sha256"] = sha256_file(PROTOCOL)
    os.makedirs(OUT, exist_ok=True)
    files = []
    p = f"{OUT}/discovery_composite_loadings.csv"
    write_csv(pd.DataFrame({"pathway": family, "loading": art["composite"]["loadings"], "mean": art["composite"]["mean"],
                            "sd": art["composite"]["sd"]}), p, float_format="%.12g")
    files.append(p)
    p = f"{OUT}/frozen_composite_artifact.json"
    write_json(art, p)
    files.append(p)
    art_sha = sha256_file(p)                                  # hashed BEFORE any held-out row is scored
    with open(p, encoding="utf-8") as f:
        art = json.load(f)                                    # the held-out step uses ONLY what is in the frozen file
    discovery_report = {"label": LABEL, "family_statement": FAMILY_STATEMENT, "family": family, "gates": gates, "accounting": disc_acc,
                        "composite": {k: art["composite"][k] for k in ("lambda1", "variance_explained", "loadings", "correlation",
                                                                       "n_distinct_samples", "composite_variance_in_discovery")},
                        "discovery_estimate": {k: v for k, v in art["discovery"].items()
                                               if k in ("beta_disc", "se_disc", "z_disc", "one_sided_p_disc", "ci95_disc")},
                        "frozen_artifact_sha256": art_sha, "selection_artifact_sha256": sel_sha,
                        "discovery_gates_passed": bool(discovery_ok)}
    p = f"{OUT}/discovery_report.json"
    write_json(discovery_report, p)
    files.append(p)
    print(f"Discovery: {disc_acc.get('discovery_rows')} rows / {disc_acc.get('discovery_patients')} patients; gates: {gates}")
    print(f"Frozen artifact SHA256: {art_sha}")

    summary = {"label": LABEL, "family": family, "family_statement": FAMILY_STATEMENT, "multiplicity": MULTIPLICITY_STATEMENT,
               "frozen_artifact_sha256": art_sha, "selection_artifact_sha256": sel_sha, "discovery_gates_passed": bool(discovery_ok),
               "discovery_splits": list(DISC_SPLITS), "heldout_splits": list(HELDOUT_SPLITS),
               "interpretation_limit": "association given clinical state; not predictive improvement; not external validation",
               "completed_utc": None}
    if not discovery_ok:
        summary.update({"status": "STOPPED AT DISCOVERY GATE: held-out outcomes were never touched; no alternative composite tried",
                        "test_result": None})
        print("STOPPED: a discovery gate failed. Held-out outcomes were not touched.")
    else:
        # ---------------- single held-out test (outcomes attached only now) ----------------
        tf, comp, eta, held_acc = score_heldout_features(master, art, HELDOUT_SPLITS)
        require(not (set(tf["public_id"]) & set(cc["public_id"])), "held-out and discovery patients overlap")
        held_acc["heldout_patients_by_split"] = {s: int(tf.loc[tf["public_id"].isin(set(master.loc[master["split"] == s, "public_id"])),
                                                              "public_id"].nunique()) for s in HELDOUT_SPLITS}
        y = master.set_index("pair_id").loc[tf["pair_id"].to_numpy(), "exact_next_response"].map(RESPONSE_RANK).to_numpy(dtype=int)
        counts = base.class_counts(y)
        support = {"heldout_rows": held_acc["heldout_rows"], "heldout_patients": held_acc["heldout_patients"],
                   "heldout_patients_by_split": held_acc["heldout_patients_by_split"], "outcome_class_counts": counts,
                   "all_six_classes_represented": bool(set(np.unique(y)) == set(range(N_CLASSES))),
                   "patients_at_least_80": bool(held_acc["heldout_patients"] >= MIN_TEST_PATIENTS),
                   "eta_non_degenerate": bool(np.std(eta) > 0)}
        print(f"Held-out support: {held_acc['heldout_rows']} rows / {held_acc['heldout_patients']} patients; classes: {counts}")
        result, status = None, None
        if not (support["all_six_classes_represented"] and support["patients_at_least_80"] and support["eta_non_degenerate"]):
            status = "HELD-OUT SUPPORT CHECK FAILED: test model not run (inconclusive)"
        else:
            groups = pd.factorize(tf["public_id"])[0]
            try:
                fit = base.fit_test_model(y, comp, eta, groups)
                lo, hi = fit["ci95"]
                z = fit["beta"] / fit["se"]
                valid = bool(fit["b_eta"] > 0)
                boots, failed = base.bootstrap_beta(y, comp, eta, groups, N_BOOT, RANDOM_SEED, fit["n_thr"])
                beta_disc = art["discovery"]["beta_disc"]
                result = {"beta_heldout": fit["beta"], "se_robust": fit["se"], "z": z,
                          "one_sided_p_H1_beta_lt_0": float(norm.cdf(z)), "ci95_wald": [lo, hi],
                          "odds_ratio_per_discovery_SD": float(np.exp(fit["beta"])),
                          "odds_ratio_ci95": [float(np.exp(lo)), float(np.exp(hi))],
                          "recalibration_slope_b_eta": fit["b_eta"], "validity_flag_b_eta_positive": valid,
                          "discovery_beta": beta_disc, "discovery_ci95": art["discovery"]["ci95_disc"],
                          "bootstrap_ci95_descriptive": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                                                         if len(boots) > 0 else None),
                          "bootstrap_n_ok": int(len(boots)), "bootstrap_n_failed": int(failed),
                          "bootstrap_seed": RANDOM_SEED, "interpretation": base.interpret(fit["beta"], lo, hi, beta_disc, valid)}
                status = "COMPLETED: single primary directional test run once"
            except Exception as e:                            # no fallback model
                status = f"HELD-OUT MODEL FAILED: {type(e).__name__}: {e} (reported; no alternative model is tried)"
        test_report = {"label": LABEL, "accounting": held_acc, "support": support, "status": status, "result": result,
                       "hypothesis": "H1: beta < 0 (higher composite -> lower next-response class)",
                       "primary_rule": "replicated if the two-sided 95% CI of beta is entirely below 0 (= one-sided p < 0.025)"}
        p = f"{OUT}/heldout_report.json"
        write_json(test_report, p)
        files.append(p)
        if result is not None:
            p = f"{OUT}/heldout_primary_result.csv"
            write_csv(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in result.items()}]),
                      p, float_format="%.10g")
            files.append(p)
        summary.update({"status": status, "heldout_result": result, "heldout_support": support})
        print(f"{status}")
        if result is not None:
            print(f"beta_heldout = {result['beta_heldout']:+.4f} (95% CI {result['ci95_wald'][0]:+.4f} to {result['ci95_wald'][1]:+.4f}); "
                  f"discovery beta = {result['discovery_beta']:+.4f}\n=> {result['interpretation']}")

    summary["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    p = f"{OUT}/run_summary.json"
    write_json(summary, p)
    files.append(p)
    man = write_output_manifest(OUT, files, {"label": LABEL, "frozen_artifact_sha256": art_sha, "selection_artifact_sha256": sel_sha,
                                             "protocol_sha256": sha256_file(PROTOCOL),
                                             "input_master_sha256": sha256_file("data/clinical/visit_pairs_with_rna.csv"),
                                             "code_sha256": sha256_file(os.path.abspath(__file__))})
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT}\nCombined SHA256: {man['combined_sha256']}")
    print("POST HOC, EXPLORATORY -- not untouched confirmation, not external validation.")


if __name__ == "__main__":
    main()
