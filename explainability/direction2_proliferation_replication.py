"""
Direction 2 follow-up -- POST-HOC INTERNAL SPLIT-SAMPLE REPLICATION of the proliferation / MYC axis.

*** This is NOT untouched confirmation and NOT external validation. ***
The pooled D2 (S3/S5) and D3 (B1) association results that motivated it already INCLUDED the test patients, and the five
pathways below were chosen AFTER seeing those pooled results. They are a post-hoc, biologically coherent family motivated
by the previously observed pooled proliferation/MYC findings. Independence here is procedural only: from the moment the
family is fixed, the discovery step uses train+val patients only, the composite construction is outcome-blind, and the
test patients play no role in pathway or composite selection, sign selection, preprocessing, or model specification.

FAMILY (fixed, post-hoc): E2F Targets, G2-M Checkpoint, Mitotic Spindle, Myc Targets V1, Myc Targets V2.

STEP 1 -- COMPOSITE (train+val only, outcome-blind, deterministic)
  * z-score the five scores with the means / SDs (ddof=1) of the DISTINCT (patient, RNA-sample) vectors among train+val
    RNA-available rows;
  * composite = PC1 of their 5x5 correlation matrix, oriented so that the loadings sum to a positive value,
    composite = (sum_j loading_j * z_j) / sqrt(lambda_1)  (unit variance in discovery);
  * structural gate: all five loadings must be positive (else stop: no alternative composite is tried).

STEP 2 -- DISCOVERY MODEL (frozen D2 S3 ordinal formulation, train+val only)
  OrdinalGEE (cumulative logit, independence working correlation, patient-clustered robust SEs) of the next-response class
  (PD<SD<PR<VGPR<CR<sCR) on: composite + 16 continuous Model-C covariates + log1p(days_since_rna_sample), with exact-strata
  dummies (Vt response x line group x PI/IMiD/CD38). Frozen handling: complete-case; strata with < 20 rows excluded;
  strata without outcome variation excluded; covariates transformed / winsorised (1st-99th) / z-scored on the final
  discovery rows. Terms are located in the model by content, never by position.
  Gates (any failure stops the analysis before test OUTCOMES are touched): all five loadings positive; all six outcome
  classes represented in discovery; design full rank; model identifiable and converged; composite coefficient negative
  with one-sided p < 0.025.
  Everything needed for the test is written to a frozen, hashed artifact BEFORE the test step and re-read from that file.

STEP 3 -- ONE PRIMARY DIRECTIONAL TEST in the RNA-available test patients
  * clinical linear predictor eta = discovery coefficients (covariates + strata; no thresholds, no composite) applied to the
    test rows with the discovery transformation parameters; rows with incomplete covariates or in strata not retained in
    discovery are excluded and counted;
  * OrdinalGEE (independence, patient-clustered robust SEs) of the next class on  composite + eta  (thresholds free);
  * estimand: beta = cumulative log-odds per discovery-SD of the composite, conditional on the recalibrated clinical
    predictor; H1: beta < 0 (higher proliferation/MYC -> lower next-response class);
  * one-sided robust Wald p; replicated if the two-sided 95% CI is entirely below 0. A patient-cluster bootstrap CI
    (2000 draws, seed 42) is a DESCRIPTIVE cross-check and does not change the rule.
  Support checks (from the frozen D3 tier-2 minimum): >= 80 test patients; all six outcome classes represented (exact
  counts reported); eta non-degenerate; model identifiable and converged. Validity flag: the recalibration slope of eta
  must be positive, otherwise the result is reported as INCONCLUSIVE.

INTERPRETATION (fixed in advance), with [L, U] the 95% CI of beta in test and beta_disc the discovery estimate:
  replicated          U < 0   (if L > beta_disc: replicated but attenuated)
  contradicted/weaker U >= 0 and L > beta_disc
  inconclusive        otherwise, or validity/support flag failed
  This describes an ASSOCIATION given clinical state; it says nothing about predictive improvement.

MULTIPLICITY: one primary test within this analysis, so no additional within-analysis multiplicity adjustment; broader
post-hoc / project-level selection remains (the family follows pooled looks that included the test patients, and this
follows many earlier D1-D4 analyses) and must be acknowledged when reporting.

STOPPING RULE: run discovery, apply the gates, hash the frozen artifact; if any gate fails stop without touching test
outcomes; otherwise run the single test once and stop. No reruns, alternative composites, cutoffs, models or endpoints.

Outputs only under artifacts/results_clean_rerun/direction2_replication/proliferation_axis/ . Existing outputs are never
overwritten. Nothing in D1-D4 or earlier outputs is modified.

Run from the repo root:
    python -m explainability.direction2_proliferation_replication
"""

import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from explainability.direction2_association import strata_key
from explainability.direction2_common import (
    CONTINUOUS_FEATURES, RESPONSE_RANK, S3_CONTINUOUS_FEATURES, S3_MIN_STRATUM_ROWS, STALENESS_COL,
    WINSOR_PERCENTILES, continuous_matrix, derive_line_group, transform_series,
)
from explainability.direction3_common import (
    RANDOM_SEED, load_feature_sets, load_master, pathway_columns, require, sha256_file, write_csv, write_json,
    write_output_manifest,
)
from explainability.direction3_molecular import gee_fit, locate_terms

OUT = "artifacts/results_clean_rerun/direction2_replication/proliferation_axis"
LABEL = ("DIRECTION 2 FOLLOW-UP: POST-HOC INTERNAL SPLIT-SAMPLE REPLICATION of the proliferation/MYC axis "
         "(NOT untouched confirmation; NOT external validation)")
FAMILY = ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle", "Myc Targets V1", "Myc Targets V2"]
FAMILY_STATEMENT = ("post-hoc, biologically coherent family motivated by the previously observed pooled D2/D3 proliferation/MYC "
                    "findings, which included test patients; PC1 construction thereafter is outcome-blind and train+val-only")
MULTIPLICITY_STATEMENT = ("one primary test within this analysis, so no additional within-analysis multiplicity adjustment; "
                          "broader post-hoc / project-level selection remains and must be acknowledged")
ALPHA_ONE_SIDED = 0.025
MIN_TEST_PATIENTS = 80          # frozen D3 tier-2 minimum
N_BOOT = 2000
N_CLASSES = len(RESPONSE_RANK)
COV_COLS = list(S3_CONTINUOUS_FEATURES)
OUTCOME_COLS = {"improved", "exact_next_response", "vt1_disease_response", "time_gap_days"}
FEATURE_COLS = (["pair_id", "public_id", "vt_disease_response", "current_line_number", "line_group",
                 "currently_on_pi", "currently_on_imid", "currently_on_cd38", STALENESS_COL] + COV_COLS + FAMILY)


# ---------------------------------------------------------------------------
# Composite (outcome-blind)
# ---------------------------------------------------------------------------
def distinct_sample_mask(df, path_cols):
    """True for the first row of every distinct (patient, RNA-sample) vector."""
    h = pd.util.hash_pandas_object(df[path_cols].round(8), index=False).to_numpy()
    pid = pd.factorize(df["public_id"])[0]
    return ~pd.DataFrame({"pid": pid, "h": h}).duplicated().to_numpy()


def build_composite(rna_rows, path_cols):
    require(all(p in path_cols for p in FAMILY), "a family pathway is missing from the pathway columns")
    first = distinct_sample_mask(rna_rows, path_cols)
    S = rna_rows.loc[first, FAMILY]
    mu, sd = S.mean(axis=0), S.std(axis=0, ddof=1)
    require((sd > 0).all(), "a family pathway has zero variance")
    Z = ((S - mu) / sd).to_numpy(dtype=float)
    R = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(R)
    lam, vec = float(w[-1]), V[:, -1].copy()
    if vec.sum() < 0:
        vec = -vec
    comp = (Z @ vec) / np.sqrt(lam)
    return {"names": FAMILY, "mean": [float(x) for x in mu], "sd": [float(x) for x in sd],
            "loadings": [float(x) for x in vec], "lambda1": lam, "variance_explained": lam / len(FAMILY),
            "correlation": R.tolist(), "n_distinct_samples": int(first.sum()),
            "composite_variance_in_discovery": float(comp.var(ddof=1)),
            "all_loadings_positive": bool((vec > 0).all())}


def composite_values(df, art):
    z = (df[art["names"]].to_numpy(dtype=float) - np.asarray(art["mean"])) / np.asarray(art["sd"])
    return (z @ np.asarray(art["loadings"])) / np.sqrt(art["lambda1"])


# ---------------------------------------------------------------------------
# Covariates: parameters fitted on discovery rows, applied unchanged to test rows
# ---------------------------------------------------------------------------
def fit_covariate_params(cc):
    params = {}
    for name in COV_COLS:
        x = transform_series(CONTINUOUS_FEATURES[name], cc[name]).to_numpy(dtype=float)
        lo, hi = np.nanpercentile(x, WINSOR_PERCENTILES[0]), np.nanpercentile(x, WINSOR_PERCENTILES[1])
        xc = np.clip(x, lo, hi)
        sd = float(np.nanstd(xc))
        params[name] = {"kind": CONTINUOUS_FEATURES[name], "lo": float(lo), "hi": float(hi),
                        "mean": float(np.nanmean(xc)), "sd": sd if sd > 0 else 1.0}
    st = np.log1p(cc[STALENESS_COL].to_numpy(dtype=float))
    params["__log1p_stale__"] = {"mean": float(st.mean()), "sd": float(st.std(ddof=0))}
    return params


def apply_covariates(df, params):
    cols = []
    for name in COV_COLS:
        p = params[name]
        x = transform_series(p["kind"], df[name]).to_numpy(dtype=float)
        cols.append((np.clip(x, p["lo"], p["hi"]) - p["mean"]) / p["sd"])
    st = np.log1p(df[STALENESS_COL].to_numpy(dtype=float))
    p = params["__log1p_stale__"]
    return np.column_stack(cols), (st - p["mean"]) / p["sd"]


def prepare_rows(master, split_mask):
    U = master[split_mask & master["rna_avail"] & master["current_line_number"].notna()]
    U = U.sort_values("pair_id").reset_index(drop=True)
    U["line_group"] = derive_line_group(U["current_line_number"])
    return U


def class_counts(y):
    return {c: int((y == r).sum()) for c, r in RESPONSE_RANK.items()}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def run_discovery(master, path_cols):
    trainval = master["split"] != "test"
    require(bool((master.loc[trainval, "split"] != "test").all()), "a test row entered the discovery mask")
    rna_tv = master[trainval & master["rna_avail"]]
    comp_art = build_composite(rna_tv, path_cols)                      # outcome-blind, train+val only

    disc = prepare_rows(master, trainval)
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
                "discovery_strata": int(key.nunique()), "outcome_class_counts": class_counts(y)})

    cov_params = fit_covariate_params(cc)
    Zc, stale_z = apply_covariates(cc, cov_params)
    require(np.allclose(Zc, continuous_matrix(cc, features=S3_CONTINUOUS_FEATURES), atol=1e-12, rtol=0),
            "covariate transformation differs from the frozen D2 continuous_matrix")
    comp = composite_values(cc, comp_art)
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
            idx = locate_terms(mod, {nm: X[:, j] for j, nm in enumerate(cont_names)})     # content-based
            for j, nm in enumerate(cont_names):                                              # layout proof for the tail
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
    art = {"label": LABEL, "family": FAMILY, "family_statement": FAMILY_STATEMENT, "composite": comp_art,
           "covariate_params": cov_params, "dummy_names": dummy_names, "reference_stratum": reference_stratum,
           "retained_strata": sorted(set(key)), "discovery": result,
           "discovery_patient_ids_sha256": __import__("hashlib").sha256(
               "\n".join(sorted(cc["public_id"].astype(str).unique())).encode()).hexdigest(),
           "min_stratum_rows": S3_MIN_STRATUM_ROWS}
    return art, gates, acc, all_ok, cc


# ---------------------------------------------------------------------------
# Test (single directional test)
# ---------------------------------------------------------------------------
def score_test_features(master, art):
    """Feature-only frame (NO outcome columns) scored with the frozen artifact."""
    test = prepare_rows(master, master["split"] == "test")
    tf = test[FEATURE_COLS].copy()
    require(not (OUTCOME_COLS & set(tf.columns)), "outcome column present in the test feature frame")
    acc = {"test_rows_before_exclusions": int(len(tf)), "test_patients_before": int(tf["public_id"].nunique())}
    miss = tf[COV_COLS].isna().any(axis=1).to_numpy()
    acc["rows_lost_incomplete_covariates"] = int(miss.sum())
    tf = tf[~miss].reset_index(drop=True)
    key = strata_key(tf)
    keep = key.isin(art["retained_strata"]).to_numpy()
    acc["rows_lost_strata_not_retained_in_discovery"] = int((~keep).sum())
    tf, key = tf[keep].reset_index(drop=True), key[keep].reset_index(drop=True)
    Zc, stale_z = apply_covariates(tf, art["covariate_params"])
    Dm = np.column_stack([(key == nm).to_numpy(dtype=float) for nm in art["dummy_names"]]) if art["dummy_names"] \
        else np.zeros((len(tf), 0))
    coef = np.asarray(art["discovery"]["clinical_coefficients"])
    require(len(coef) == Zc.shape[1] + 1 + Dm.shape[1], "clinical coefficient vector length mismatch")
    eta = Zc @ coef[:Zc.shape[1]] + stale_z * coef[Zc.shape[1]] + Dm @ coef[Zc.shape[1] + 1:]
    comp = composite_values(tf, art["composite"])
    acc.update({"test_rows": int(len(tf)), "test_patients": int(tf["public_id"].nunique())})
    return tf, comp, eta, acc


def fit_test_model(y, comp, eta, groups):
    X = np.column_stack([comp, eta])
    mod, r = gee_fit("ordinal", y, X, groups)
    idx = locate_terms(mod, {"composite": comp, "eta": eta})
    params = np.asarray(r.params)
    n_thr = N_CLASSES - 1
    require(len(params) == n_thr + 2, "unexpected parameter count in the test model")
    ci = np.asarray(r.conf_int())
    ic, ie = idx["composite"], idx["eta"]
    require(ic == n_thr and ie == n_thr + 1, "unexpected parameter layout in the test model (expected thresholds, composite, eta)")
    return {"beta": float(params[ic]), "se": float(np.asarray(r.bse)[ic]), "ci95": [float(ci[ic][0]), float(ci[ic][1])],
            "b_eta": float(params[ie]), "se_eta": float(np.asarray(r.bse)[ie]), "n_thr": n_thr}


def bootstrap_beta(y, comp, eta, pid_codes, n_boot, seed, n_thr):
    rng = np.random.RandomState(seed)
    uniq = np.unique(pid_codes)
    rows_by = {p: np.flatnonzero(pid_codes == p) for p in uniq}
    betas, failed = [], 0
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by[p] for p in drawn])
        grp = np.concatenate([np.full(len(rows_by[p]), k) for k, p in enumerate(drawn)])
        try:
            if len(np.unique(y[idx])) < N_CLASSES:
                raise ValueError("a class is missing from the resample")
            _, r = gee_fit("ordinal", y[idx], np.column_stack([comp[idx], eta[idx]]), grp)
            params = np.asarray(r.params)
            require(len(params) == n_thr + 2, "parameter count")
            betas.append(float(params[n_thr]))            # layout verified on the original fit: [thresholds, composite, eta]
        except Exception:
            failed += 1
    return np.asarray(betas), failed


def interpret(beta, lo, hi, beta_disc, valid):
    if not valid:
        return "INCONCLUSIVE (validity flag: recalibration slope of the clinical predictor is not positive)"
    if hi < 0:
        if lo > beta_disc:
            return "REPLICATED but ATTENUATED (95% CI excludes 0 and excludes the discovery effect)"
        return "REPLICATED (95% CI entirely below 0, in the hypothesised direction)"
    if lo > beta_disc:
        return "CONTRADICTED / weaker than discovery (CI includes 0 and excludes the discovery effect)"
    return "INCONCLUSIVE (95% CI compatible with both zero and the discovery effect)"


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(LABEL)
    marker = f"{OUT}/output_manifest.json"
    if os.path.exists(marker):
        raise SystemExit(f"[proliferation_replication] {marker} already exists: existing outputs are never overwritten. "
                         f"Delete the folder deliberately to rerun.")
    fs = load_feature_sets()
    path_cols = pathway_columns(fs)
    master = load_master()                                   # frozen cohort counts and patient-disjoint splits asserted
    test_ids = set(master.loc[master["split"] == "test", "public_id"])

    # ---------------- discovery ----------------
    art, gates, disc_acc, discovery_ok, cc = run_discovery(master, path_cols)
    require(not (set(cc["public_id"]) & test_ids), "a test patient entered discovery")
    os.makedirs(OUT, exist_ok=True)
    files = []
    p = f"{OUT}/discovery_composite_loadings.csv"
    write_csv(pd.DataFrame({"pathway": FAMILY, "loading": art["composite"]["loadings"], "mean": art["composite"]["mean"],
                            "sd": art["composite"]["sd"]}), p, float_format="%.12g")
    files.append(p)
    p = f"{OUT}/frozen_composite_artifact.json"
    write_json(art, p)
    files.append(p)
    art_sha = sha256_file(p)                                  # hashed BEFORE any test row is scored
    with open(p, encoding="utf-8") as f:
        art = json.load(f)                                    # the test step uses ONLY what is in the frozen file
    discovery_report = {"label": LABEL, "family_statement": FAMILY_STATEMENT, "gates": gates, "accounting": disc_acc,
                        "composite": {k: art["composite"][k] for k in ("lambda1", "variance_explained", "loadings",
                                                                       "n_distinct_samples", "composite_variance_in_discovery")},
                        "discovery_estimate": {k: v for k, v in art["discovery"].items()
                                               if k in ("beta_disc", "se_disc", "z_disc", "one_sided_p_disc", "ci95_disc")},
                        "frozen_artifact_sha256": art_sha, "discovery_gates_passed": bool(discovery_ok)}
    p = f"{OUT}/discovery_report.json"
    write_json(discovery_report, p)
    files.append(p)
    print(f"Discovery: {disc_acc.get('discovery_rows')} rows / {disc_acc.get('discovery_patients')} patients; gates: {gates}")
    print(f"Frozen artifact SHA256: {art_sha}")

    summary = {"label": LABEL, "family_statement": FAMILY_STATEMENT, "multiplicity": MULTIPLICITY_STATEMENT,
               "frozen_artifact_sha256": art_sha, "discovery_gates_passed": bool(discovery_ok),
               "interpretation_limit": "association given clinical state; not predictive improvement; not external validation",
               "completed_utc": None}
    if not discovery_ok:
        summary.update({"status": "STOPPED AT DISCOVERY GATE: test outcomes were never touched; no alternative composite tried",
                        "test_result": None})
        print("STOPPED: a discovery gate failed. Test outcomes were not touched.")
    else:
        # ---------------- single test (outcomes attached only now) ----------------
        tf, comp, eta, test_acc = score_test_features(master, art)
        require(not (set(tf["public_id"]) & set(cc["public_id"])), "test and discovery patients overlap")
        y = master.set_index("pair_id").loc[tf["pair_id"].to_numpy(), "exact_next_response"].map(RESPONSE_RANK)\
                  .to_numpy(dtype=int)
        counts = class_counts(y)
        support = {"test_rows": test_acc["test_rows"], "test_patients": test_acc["test_patients"],
                   "outcome_class_counts": counts,
                   "all_six_classes_represented": bool(set(np.unique(y)) == set(range(N_CLASSES))),
                   "patients_at_least_80": bool(test_acc["test_patients"] >= MIN_TEST_PATIENTS),
                   "eta_non_degenerate": bool(np.std(eta) > 0)}
        print(f"Test support: {test_acc['test_rows']} rows / {test_acc['test_patients']} patients; classes: {counts}")
        result, status = None, None
        if not (support["all_six_classes_represented"] and support["patients_at_least_80"] and support["eta_non_degenerate"]):
            status = "TEST SUPPORT CHECK FAILED: test model not run (inconclusive)"
        else:
            groups = pd.factorize(tf["public_id"])[0]
            try:
                fit = fit_test_model(y, comp, eta, groups)
                lo, hi = fit["ci95"]
                z = fit["beta"] / fit["se"]
                valid = bool(fit["b_eta"] > 0)
                boots, failed = bootstrap_beta(y, comp, eta, groups, N_BOOT, RANDOM_SEED, fit["n_thr"])
                beta_disc = art["discovery"]["beta_disc"]
                result = {"beta_test": fit["beta"], "se_robust": fit["se"], "z": z,
                          "one_sided_p_H1_beta_lt_0": float(norm.cdf(z)), "ci95_wald": [lo, hi],
                          "odds_ratio_per_discovery_SD": float(np.exp(fit["beta"])),
                          "recalibration_slope_b_eta": fit["b_eta"], "validity_flag_b_eta_positive": valid,
                          "discovery_beta": beta_disc, "discovery_ci95": art["discovery"]["ci95_disc"],
                          "bootstrap_ci95_descriptive": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                                                         if len(boots) > 0 else None),
                          "bootstrap_n_ok": int(len(boots)), "bootstrap_n_failed": int(failed),
                          "bootstrap_seed": RANDOM_SEED, "interpretation": interpret(fit["beta"], lo, hi, beta_disc, valid)}
                status = "COMPLETED: single primary directional test run once"
            except Exception as e:                            # no fallback model
                status = f"TEST MODEL FAILED: {type(e).__name__}: {e} (reported; no alternative model is tried)"
        test_report = {"label": LABEL, "accounting": test_acc, "support": support, "status": status, "result": result,
                       "hypothesis": "H1: beta < 0 (higher proliferation/MYC composite -> lower next-response class)",
                       "primary_rule": "replicated if the two-sided 95% CI of beta is entirely below 0 (= one-sided p < 0.025)"}
        p = f"{OUT}/test_report.json"
        write_json(test_report, p)
        files.append(p)
        if result is not None:
            p = f"{OUT}/test_primary_result.csv"
            write_csv(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in result.items()}]),
                      p, float_format="%.10g")
            files.append(p)
        summary.update({"status": status, "test_result": result, "test_support": support})
        print(f"{status}")
        if result is not None:
            print(f"beta_test = {result['beta_test']:+.4f} (95% CI {result['ci95_wald'][0]:+.4f} to {result['ci95_wald'][1]:+.4f}); "
                  f"discovery beta = {result['discovery_beta']:+.4f}\n=> {result['interpretation']}")

    summary["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    p = f"{OUT}/run_summary.json"
    write_json(summary, p)
    files.append(p)
    man = write_output_manifest(OUT, files, {"label": LABEL, "frozen_artifact_sha256": art_sha,
                                             "input_master_sha256": sha256_file("data/clinical/visit_pairs_with_rna.csv"),
                                             "code_sha256": sha256_file(os.path.abspath(__file__))})
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT}\nCombined SHA256: {man['combined_sha256']}")
    print("POST-HOC INTERNAL SPLIT-SAMPLE REPLICATION -- not untouched confirmation, not external validation.")


if __name__ == "__main__":
    main()
