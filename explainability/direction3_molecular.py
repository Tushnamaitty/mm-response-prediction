"""
Direction 3 -- PART B: RETROSPECTIVE molecular association and pathway x treatment interaction.

*** RETROSPECTIVE MOLECULAR ASSOCIATION / INTERACTION -- NOT PREDICTIVE VALIDATION ***
Outcome is the Vt+1 response. RNA = baseline / as-of-sample tumour biology (latest sample at or before Vt),
NOT current-state RNA and NOT causal. Treatment = drug classes active AT Vt.

Universe (binary, PRIMARY): pooled RNA-available rows with eligible_for_binary and non-missing line (8,261 rows /
703 patients). Outcome = improved. Pathway scores are z-scored over the DISTINCT (patient, sample) vectors (751).

Per pathway (all 50 Hallmark pathways), patient-clustered GEE (binomial logit, independence working correlation,
robust patient-cluster SEs), TWO models:
  M0 (B1) : pathway + PI + IMiD + steroid + covariates                       -> B1 pathway MAIN effect
  M1 (B2) : pathway + PI + IMiD + steroid + pathway x PI + pathway x IMiD
            + pathway x steroid + covariates (ALL THREE treatments in the SAME model; classes overlap and are
            entered additively, never as exclusive groups)                   -> B2 joint 3-df Wald of the
                                                                               three interaction terms
  NOTE: in M1 the pathway coefficient is the slope when none of PI/IMiD/steroid is on, so B1 uses the
  no-interaction model M0 (as pre-specified). M1's reference slope and the slopes among rows on each class
  (pathway + average-mix interaction contrast) are reported as DESCRIPTIVE.
BH over the 50 pathways separately for B1 and for B2. GLOBAL test: patient-level permutation of the WHOLE RNA
vector (whole vectors permuted across distinct (patient, sample) units), statistic = sum over the 50 pathways of the
B1 Wald chi-square (and, separately, of the B2 3-df Wald), 1000 permutations (fast clustered-logit implementation,
cross-checked against the statsmodels GEE before it is used).

Covariates (Vt-side Model-C fields): current response (Vt response dummies), line group 1/2/3+, CD38, chemo,
SLAMF7-or-BCMA flag, log1p RNA staleness (days_since_rna_sample), log1p m-protein, log1p LDH, hemoglobin, albumin,
log1p creatinine, log1p B2M, ISS stage, age at diagnosis (all continuous covariates z-scored on the analysis rows).
Missing covariates: COMPLETE-CASE (no imputation, no zero-filling, no pooling); every row lost is reported in
partB_row_accounting.csv/.json.

Secondary outcome: worse / same / better vs the current response (computed from exact_next_response and
vt_disease_response) with OrdinalGEE on all RNA-available rows (same M0/M1 structure, BH per family, no
global permutation). The pathway / interaction terms are located in the model by CONTENT (never by position).
Sensitivities (binary): +pct_plasma_cells_bm ; +prior treatment exposure (prior_exposure_* for PI, IMiD, steroid,
CD38, chemo, SLAMF7-or-BCMA). Rank-deficient designs are reported as failed, never silently reduced.

EXPLORATORY (Tier 2, separate family): pathway x CD38 interaction. One extra model per pathway, fitted on the SAME
analysis rows as the primary binary analysis: pathway + PI + IMiD + steroid + pathway x PI + pathway x IMiD +
pathway x steroid (these three interaction terms are ADJUSTMENT terms only) + pathway x CD38 (the ONLY target) +
covariates (CD38 is already a main-effect covariate). Only the pathway x CD38 term is tested/reported as the result,
with its own BH correction across the 50 pathways; it is never added to the primary B1/B2 families and is labelled
Tier 2 / exploratory (CD38: < 150 pooled RNA patients / < 150 events). Chemo, SLAMF7 and BCMA enter as
main-effect covariates only. The reported slopes among rows with CD38 off / on use the average PI/IMiD/steroid mix of
those rows (contrast over the pathway, three adjustment-interaction and CD38-interaction coefficients).

Run from the repo root:
    python -m explainability.direction3_molecular
    python -m explainability.direction3_molecular --smoke [--force]   (5 pathways, few permutations)
"""

import argparse
import datetime
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from explainability.direction2_common import EXPECTED, RESPONSE_RANK, derive_line_group, transform_series
from explainability.direction3_common import (
    CONT_COVARIATES, GLOBAL_PERM_SEED, LABEL_B, N_PERM_GLOBAL, PLASMA_COL, PRIMARY_TREATMENTS,
    PRIOR_EXPOSURE_COLS, RNA_INTERPRETATION, STALENESS_COL, TREATMENT_COLS, assert_predictors_clean,
    assert_support_matches_declared_roles, assert_treatment_fields_clean, bh_qvalues, build_pathway_z, expect,
    get_dirs3, guard_no_overwrite, load_feature_sets, load_master, pathway_columns, require, sha256_file,
    support_table, write_csv, write_json, write_output_manifest,
)

N_PATHWAYS_SMOKE = 5
N_PERM_GLOBAL_SMOKE = 20


# ---------------------------------------------------------------------------
# Universe + design (complete-case, fully reported)
# ---------------------------------------------------------------------------
def select_universe_b(master, family):
    m = master["rna_avail"] & master["current_line_number"].notna()
    if family == "binary":
        m = m & master["eligible_for_binary"]
    u = master[m].sort_values("pair_id").reset_index(drop=True)
    u["line_group"] = derive_line_group(u["current_line_number"])
    key = "binary" if family == "binary" else "ordinal"       # ordinal universe = all RNA-available rows
    expect(f"Part B universe rows [{key}]", len(u), EXPECTED["universe_rows"][key])
    expect(f"Part B universe patients [{key}]", int(u["public_id"].nunique()), EXPECTED["universe_patients"][key])
    return u


def _z(x):
    sd = x.std(ddof=0)
    require(sd > 0, "a covariate is constant on the analysis rows")
    return (x - x.mean()) / sd


def make_design(U, family, sens, label):
    """Complete-case design. sens in {primary, plasma, prior}. Returns dict (ok flag + accounting)."""
    require(sens in ("primary", "plasma", "prior"), f"unknown sensitivity {sens!r}")
    miss_cols = list(CONT_COVARIATES) + ([PLASMA_COL] if sens == "plasma" else [])
    acc = {"analysis": label, "family": family, "sensitivity": sens, "universe_rows": int(len(U)),
           "universe_patients": int(U["public_id"].nunique()),
           "rows_missing_by_covariate": json.dumps({c: int(U[c].isna().sum()) for c in miss_cols})}
    any_miss = U[miss_cols].isna().any(axis=1).to_numpy()
    cc = U[~any_miss].reset_index(drop=True)
    acc.update({"rows_lost_incomplete_covariates": int(any_miss.sum()), "final_rows": int(len(cc)),
                "final_patients": int(cc["public_id"].nunique()), "share_rows_retained": float(len(cc) / len(U))})

    if family == "binary":
        y = cc["improved"].to_numpy(dtype=int)
    else:                                              # worse(0) / same(1) / better(2) vs the current response
        rn = cc["exact_next_response"].map(RESPONSE_RANK).to_numpy(dtype=float)
        rv = cc["vt_disease_response"].map(RESPONSE_RANK).to_numpy(dtype=float)
        require(np.isfinite(rn).all() and np.isfinite(rv).all(), "unmapped response labels")
        y = (np.sign(rn - rv) + 1).astype(int)
    acc["outcome_counts"] = json.dumps({int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})

    cols, names = [], []

    def add(a, nm):
        a = np.asarray(a, dtype=float)
        cols.append(a if a.ndim == 2 else a[:, None])
        names.extend(nm)

    if family == "binary":
        add(np.ones(len(cc)), ["const"])
    D = pd.get_dummies(cc["vt_disease_response"].astype(str), drop_first=True, dtype=float)
    add(D.to_numpy(), [f"vt_response[{c}]" for c in D.columns])
    L = pd.get_dummies(cc["line_group"].astype(str), drop_first=True, dtype=float)
    add(L.to_numpy(), [f"line_group[{c}]" for c in L.columns])
    for nm, col in [("on_cd38", "currently_on_cd38"), ("on_chemo", "currently_on_chemo"),
                    ("on_slamf7_or_bcma", "currently_on_other")]:
        add(cc[col].to_numpy(dtype=float), [nm])
    stale = np.log1p(cc[STALENESS_COL].to_numpy(dtype=float))
    require(np.isfinite(stale).all(), "missing days_since_rna_sample in analysis rows")
    add(_z(stale), ["log1p_rna_staleness"])
    for name, kind in CONT_COVARIATES.items():
        add(_z(transform_series(kind, cc[name]).to_numpy(dtype=float)), [name])
    if sens == "plasma":
        add(_z(cc[PLASMA_COL].to_numpy(dtype=float)), [PLASMA_COL])
    if sens == "prior":
        for t in ["pi", "imid", "steroid", "cd38", "chemo"]:
            add(cc[PRIOR_EXPOSURE_COLS[t]].to_numpy(dtype=float), [f"prior_exposure_{t}"])
        add(cc["prior_exposure_other"].to_numpy(dtype=float), ["prior_exposure_slamf7_or_bcma"])
    X_cov = np.column_stack(cols)
    T = np.column_stack([cc[TREATMENT_COLS[t]].to_numpy(dtype=float) for t in PRIMARY_TREATMENTS])
    rank = int(np.linalg.matrix_rank(np.column_stack([T, X_cov])))
    acc["design_columns_without_pathway"] = int(T.shape[1] + X_cov.shape[1])
    acc["design_full_rank"] = bool(rank == T.shape[1] + X_cov.shape[1])
    groups = pd.factorize(cc["public_id"])[0]
    mix = {t: T[T[:, k] == 1].mean(axis=0) for k, t in enumerate(PRIMARY_TREATMENTS)}
    return {"ok": acc["design_full_rank"], "acc": acc, "cc": cc, "y": y, "X_cov": X_cov, "cov_names": names,
            "T": T, "groups": groups, "mix": mix}


# ---------------------------------------------------------------------------
# statsmodels GEE (primary estimation) with content-based term location
# ---------------------------------------------------------------------------
def gee_fit(family, y, X, groups):
    from statsmodels.genmod.generalized_estimating_equations import GEE, OrdinalGEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Independence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "binary":
            mod = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Independence())
        else:
            mod = OrdinalGEE(y, X, groups=groups, cov_struct=Independence())
        r = mod.fit(maxiter=100)
    if getattr(r, "converged", None) is False:
        raise RuntimeError("not_converged")
    return mod, r


def locate_terms(mod, term_vectors):
    """Find each tested term in the model's (expanded) design matrix by CONTENT, never by position. OrdinalGEE
    prepends threshold parameters and expands rows per cut-point, so positional indexing is unsafe."""
    exog = np.asarray(mod.exog, dtype=float)
    n = len(next(iter(term_vectors.values())))
    require(exog.shape[0] % n == 0, "design row count is not a multiple of the number of observations")
    rep = exog.shape[0] // n
    sexog = np.sort(exog, axis=0)
    idx = {}
    for name, v in term_vectors.items():
        sv = np.sort(np.repeat(np.asarray(v, dtype=float), rep))
        hits = np.flatnonzero(np.all(np.isclose(sexog, sv[:, None], atol=1e-10, rtol=0.0), axis=0))
        require(len(hits) == 1, f"term {name!r}: expected exactly one matching design column, found {len(hits)}")
        j = int(hits[0])
        require(np.unique(exog[:, j]).size > 2, f"term {name!r} matched a binary column (threshold/indicator)")
        idx[name] = j
    require(len(set(idx.values())) == len(idx), "two tested terms matched the same design column")
    return idx


def fit_pathway(family, design, z, interactions):
    T, Xc = design["T"], design["X_cov"]
    terms = {"pathway": z}
    blocks = [z[:, None], T]
    if interactions:
        ZT = z[:, None] * T
        blocks.append(ZT)
        for k, t in enumerate(PRIMARY_TREATMENTS):
            terms[f"pathway_x_{t}"] = ZT[:, k]
    blocks.append(Xc)
    X = np.column_stack(blocks)
    mod, r = gee_fit(family, design["y"], X, design["groups"])
    idx = locate_terms(mod, terms)
    params, cov = np.asarray(r.params), np.asarray(r.cov_params())
    require(len(params) == np.asarray(mod.exog).shape[1], "parameter vector length differs from the design matrix")
    return {"params": params, "cov": cov, "bse": np.asarray(r.bse), "p": np.asarray(r.pvalues),
            "ci": np.asarray(r.conf_int()), "idx": idx}


def _row_from_fits(family, design, pw, m0, m1):
    i0 = m0["idx"]["pathway"]
    est, se, p = float(m0["params"][i0]), float(m0["bse"][i0]), float(m0["p"][i0])
    ci = m0["ci"][i0]
    row = {"B1_estimate_per_SD": est, "B1_se_robust": se, "B1_p": p, "B1_odds_ratio_per_SD": float(np.exp(est)),
           "B1_ci_lo": float(ci[0]), "B1_ci_hi": float(ci[1])}
    ix = m1["idx"]
    iI = [ix[f"pathway_x_{t}"] for t in PRIMARY_TREATMENTS]
    bI, VI = m1["params"][iI], m1["cov"][np.ix_(iI, iI)]
    W = float(bI @ np.linalg.solve(VI, bI))
    row.update({"B2_wald_chi2": W, "B2_df": 3, "B2_p": float(chi2.sf(W, 3))})
    g = ix["pathway"]
    row.update({"M1_reference_slope_none_of_PI_IMiD_steroid": float(m1["params"][g]),
                "M1_reference_slope_p": float(m1["p"][g])})
    for k, t in enumerate(PRIMARY_TREATMENTS):
        row[f"M1_interaction_pathway_x_{t}"] = float(m1["params"][iI[k]])
        row[f"M1_interaction_pathway_x_{t}_p"] = float(m1["p"][iI[k]])
    th = [g] + iI
    covth = m1["cov"][np.ix_(th, th)]
    thv = m1["params"][th]
    for t in PRIMARY_TREATMENTS:
        c = np.r_[1.0, design["mix"][t]]
        e, s = float(c @ thv), float(np.sqrt(c @ covth @ c))
        row[f"slope_among_rows_on_{t}"] = e
        row[f"slope_among_rows_on_{t}_se"] = s
        row[f"slope_among_rows_on_{t}_p"] = float(2 * norm.sf(abs(e / s))) if s > 0 else np.nan
        row[f"slope_among_rows_on_{t}_odds_ratio"] = float(np.exp(e))
    return row


def analyze(family, design, zp, path_cols, label):
    rows = []
    for j, pw in enumerate(path_cols):
        row = {"family": family, "analysis": label, "pathway": pw, "status": "ok"}
        try:
            m0 = fit_pathway(family, design, zp[:, j], False)
            m1 = fit_pathway(family, design, zp[:, j], True)
            row.update(_row_from_fits(family, design, pw, m0, m1))
        except AssertionError as e:
            if str(e).startswith("[direction2]"):
                raise                                   # structural failures are never swallowed
            row["status"] = f"failed: AssertionError: {e}"
        except Exception as e:                          # no fallback model
            row["status"] = f"failed: {type(e).__name__}: {e}"
        rows.append(row)
        if (j + 1) % 10 == 0:
            print(f"    {label}: {j + 1}/{len(path_cols)} pathways", flush=True)
    df = pd.DataFrame(rows)
    for c in ["B1_p", "B2_p"]:
        if c not in df.columns:
            df[c] = np.nan
    df["B1_q_bh"] = bh_qvalues(df["B1_p"].to_numpy())
    df["B2_q_bh"] = bh_qvalues(df["B2_p"].to_numpy())
    df["B1_significant_q_lt_0.05"] = df["B1_q_bh"] < 0.05
    df["B2_significant_q_lt_0.05"] = df["B2_q_bh"] < 0.05
    n_fail = int(df["status"].str.startswith("failed").sum())
    return df, {"n_pathways_ok": int(len(df) - n_fail), "n_pathways_failed": n_fail,
                "failure_reasons": json.dumps(sorted(set(df.loc[df["status"].str.startswith("failed"), "status"])))}


def fit_pathway_cd38(family, design, z):
    """EXPLORATORY Tier-2 model: pathway + PI + IMiD + steroid + pathway x PI + pathway x IMiD + pathway x steroid
    (ADJUSTMENT terms) + pathway x CD38 (the ONLY target) + covariates. Separate from the primary M0/M1 models."""
    T, Xc = design["T"], design["X_cov"]
    cd38 = design["cc"]["currently_on_cd38"].to_numpy(dtype=float)
    ZT = z[:, None] * T
    zc = z * cd38
    terms = {"pathway": z, "pathway_x_cd38": zc}
    for k, t in enumerate(PRIMARY_TREATMENTS):
        terms[f"pathway_x_{t}"] = ZT[:, k]
    X = np.column_stack([z[:, None], T, ZT, zc[:, None], Xc])
    mod, r = gee_fit(family, design["y"], X, design["groups"])
    idx = locate_terms(mod, terms)
    params, cov = np.asarray(r.params), np.asarray(r.cov_params())
    require(len(params) == np.asarray(mod.exog).shape[1], "parameter vector length differs from the design matrix")
    return {"params": params, "cov": cov, "bse": np.asarray(r.bse), "p": np.asarray(r.pvalues),
            "ci": np.asarray(r.conf_int()), "idx": idx}


def analyze_cd38(family, design, zp, path_cols, label, support_row):
    rows = []
    cd38v = design["cc"]["currently_on_cd38"].to_numpy(dtype=float)
    mix_on = design["T"][cd38v == 1].mean(axis=0)            # average PI/IMiD/steroid mix among CD38-on rows
    mix_off = design["T"][cd38v == 0].mean(axis=0)           # ... and among CD38-off rows
    for j, pw in enumerate(path_cols):
        row = {"family": family, "analysis": label, "role": "exploratory_tier2", "pathway": pw, "status": "ok",
               "cd38_pooled_rna_patients": int(support_row["group_pts"]),
               "cd38_pooled_events": int(support_row["group_events"]),
               "cd38_support_tier": int(support_row["tier"]),
               "bh_family": "exploratory Tier-2 pathway x CD38 interaction, 50 pathways (separate from primary B1/B2)"}
        try:
            m = fit_pathway_cd38(family, design, zp[:, j])
            g, d = m["idx"]["pathway"], m["idx"]["pathway_x_cd38"]
            est, se, pv, ci = float(m["params"][d]), float(m["bse"][d]), float(m["p"][d]), m["ci"][d]
            th = [g] + [m["idx"][f"pathway_x_{t}"] for t in PRIMARY_TREATMENTS] + [d]
            thv, covth = m["params"][th], m["cov"][np.ix_(th, th)]
            c_on, c_off = np.r_[1.0, mix_on, 1.0], np.r_[1.0, mix_off, 0.0]
            on, on_se = float(c_on @ thv), float(np.sqrt(c_on @ covth @ c_on))
            off, off_se = float(c_off @ thv), float(np.sqrt(c_off @ covth @ c_off))
            row.update({"CD38_interaction_estimate": est, "CD38_interaction_se_robust": se, "CD38_interaction_p": pv,
                        "CD38_interaction_ci_lo": float(ci[0]), "CD38_interaction_ci_hi": float(ci[1]),
                        "CD38_interaction_ratio_of_odds_ratios": float(np.exp(est)),
                        "slope_CD38_off": off, "slope_CD38_off_se": off_se,
                        "slope_CD38_off_p": float(2 * norm.sf(abs(off / off_se))) if off_se > 0 else np.nan,
                        "slope_CD38_on": on, "slope_CD38_on_se": on_se,
                        "slope_CD38_on_p": float(2 * norm.sf(abs(on / on_se))) if on_se > 0 else np.nan,
                        "slope_CD38_on_odds_ratio": float(np.exp(on)),
                        "slope_definition": "pathway slope among rows with CD38 off/on at those rows' average "
                                            "PI/IMiD/steroid mix (descriptive); TARGET = CD38_interaction_*"})
        except AssertionError as e:
            if str(e).startswith("[direction2]"):
                raise
            row["status"] = f"failed: AssertionError: {e}"
        except Exception as e:                          # no fallback model
            row["status"] = f"failed: {type(e).__name__}: {e}"
        rows.append(row)
        if (j + 1) % 10 == 0:
            print(f"    {label}: {j + 1}/{len(path_cols)} pathways", flush=True)
    df = pd.DataFrame(rows)
    if "CD38_interaction_p" not in df.columns:
        df["CD38_interaction_p"] = np.nan
    df["CD38_interaction_q_bh"] = bh_qvalues(df["CD38_interaction_p"].to_numpy())
    df["CD38_interaction_q_lt_0.05_exploratory"] = df["CD38_interaction_q_bh"] < 0.05
    n_fail = int(df["status"].str.startswith("failed").sum())
    return df, {"n_pathways_ok": int(len(df) - n_fail), "n_pathways_failed": n_fail,
                "failure_reasons": json.dumps(sorted(set(df.loc[df["status"].str.startswith("failed"), "status"])))}


# ---------------------------------------------------------------------------
# Fast clustered logistic (independence GEE == logistic MLE + cluster-robust sandwich) for the global test
# ---------------------------------------------------------------------------
def np_logit(Xs, ys, starts, beta0, max_iter=50, tol=1e-9):
    beta, converged = beta0.copy(), False
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ beta, -35, 35)))
        W = p * (1.0 - p)
        H = (Xs * W[:, None]).T @ Xs
        step = np.linalg.solve(H, Xs.T @ (ys - p))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    require(converged, "numpy clustered-logit did not converge")
    p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ beta, -35, 35)))
    W = p * (1.0 - p)
    H = (Xs * W[:, None]).T @ Xs
    Sg = np.add.reduceat(Xs * (ys - p)[:, None], starts, axis=0)
    bread = np.linalg.inv(H)
    return beta, bread @ (Sg.T @ Sg) @ bread


class NumpyClusterLogit:
    def __init__(self, design):
        order = np.argsort(design["groups"], kind="stable")
        gs = design["groups"][order]
        self.order = order
        self.starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        self.y = design["y"][order].astype(float)
        self.Xrest = np.column_stack([design["T"], design["X_cov"]])[order]
        self.k3 = len(PRIMARY_TREATMENTS)
        self.beta_rest, _ = np_logit(self.Xrest, self.y, self.starts, np.zeros(self.Xrest.shape[1]))

    def stats(self, zs):
        """(B1 Wald chi2 of the pathway term in M0, B2 3-df Wald of the interactions in M1, full fits)"""
        X0 = np.column_stack([zs, self.Xrest])
        b0, c0 = np_logit(X0, self.y, self.starts, np.r_[0.0, self.beta_rest])
        X1 = np.column_stack([zs, self.Xrest, zs[:, None] * self.Xrest[:, :self.k3]])
        b1, c1 = np_logit(X1, self.y, self.starts, np.r_[0.0, self.beta_rest, np.zeros(self.k3)])
        bi, vi = b1[-self.k3:], c1[-self.k3:, -self.k3:]
        return float(b0[0] ** 2 / c0[0, 0]), float(bi @ np.linalg.solve(vi, bi)), (b0, c0, b1, c1)


def crosscheck_numpy_vs_gee(design, zp, n_check=3):
    """STOP unless the fast implementation reproduces the statsmodels GEE (estimates and robust SEs)."""
    eng = NumpyClusterLogit(design)
    for j in range(min(n_check, zp.shape[1])):
        m0 = fit_pathway("binary", design, zp[:, j], False)
        m1 = fit_pathway("binary", design, zp[:, j], True)
        _, _, (b0, c0, b1, c1) = eng.stats(zp[eng.order, j])
        g0 = m0["idx"]["pathway"]
        require(np.isclose(b0[0], m0["params"][g0], rtol=1e-3, atol=1e-4), "numpy vs GEE: M0 coefficient mismatch")
        require(np.isclose(np.sqrt(c0[0, 0]), m0["bse"][g0], rtol=5e-3), "numpy vs GEE: M0 robust SE mismatch")
        iI = [m1["idx"][f"pathway_x_{t}"] for t in PRIMARY_TREATMENTS]
        require(np.allclose(b1[-3:], m1["params"][iI], rtol=1e-3, atol=1e-4), "numpy vs GEE: M1 interaction mismatch")
        require(np.allclose(np.sqrt(np.diag(c1)[-3:]), m1["bse"][iI], rtol=5e-3), "numpy vs GEE: M1 interaction SE mismatch")
    return True


def global_permutation(design, zp, n_perm, seed):
    eng = NumpyClusterLogit(design)
    zs = zp[eng.order]
    K = zp.shape[1]
    h = pd.util.hash_pandas_object(pd.DataFrame(np.round(zp, 8)), index=False).to_numpy()
    key = pd.DataFrame({"g": design["groups"], "h": h})
    sample_idx = key.groupby(["g", "h"], sort=False).ngroup().to_numpy()
    first = ~key.duplicated().to_numpy()
    Vs = zp[first]
    require(Vs.shape[0] == sample_idx.max() + 1, "sample index / vector count mismatch")
    sidx_sorted = sample_idx[eng.order]
    obs = [eng.stats(zs[:, j])[:2] for j in range(K)]
    G1_obs, G2_obs = float(sum(o[0] for o in obs)), float(sum(o[1] for o in obs))
    rng = np.random.RandomState(seed)
    G1, G2 = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        Zb = Vs[rng.permutation(Vs.shape[0])][sidx_sorted]
        s1 = s2 = 0.0
        for j in range(K):
            c1, w2, _ = eng.stats(Zb[:, j])
            s1, s2 = s1 + c1, s2 + w2
        G1[b], G2[b] = s1, s2
        if (b + 1) % 50 == 0:
            print(f"    global permutation {b + 1}/{n_perm}", flush=True)
    obs_df = pd.DataFrame({"B1_wald_chi2_numpy": [o[0] for o in obs], "B2_wald_chi2_numpy": [o[1] for o in obs]})
    return {"n_permutations": n_perm, "seed": seed, "n_distinct_samples_permuted": int(Vs.shape[0]),
            "B1_global_stat_observed": G1_obs, "B1_global_p": float((1 + (G1 >= G1_obs).sum()) / (n_perm + 1)),
            "B2_global_stat_observed": G2_obs, "B2_global_p": float((1 + (G2 >= G2_obs).sum()) / (n_perm + 1)),
            "statistic": "sum over pathways of Wald chi-square (B1: 1 df; B2: 3 df); whole RNA vectors permuted "
                         "across distinct (patient, sample) units"}, obs_df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(f"DIRECTION 3 / PART B  |  {LABEL_B}\n{RNA_INTERPRETATION}")
    dirs = get_dirs3(args.smoke)
    out_dir = dirs["partB"]
    guard_no_overwrite(f"{out_dir}/output_manifest.json", args.smoke, args.force)
    os.makedirs(out_dir, exist_ok=True)
    n_perm = N_PERM_GLOBAL_SMOKE if args.smoke else N_PERM_GLOBAL

    fs = load_feature_sets()
    assert_predictors_clean(fs)
    assert_treatment_fields_clean(fs)
    path_all = pathway_columns(fs)
    master = load_master()
    tbl = support_table(master)
    assert_support_matches_declared_roles(tbl)
    Z = build_pathway_z(master, path_all)
    path_cols = path_all[:N_PATHWAYS_SMOKE] if args.smoke else path_all
    files = []
    p = f"{out_dir}/partB_group_support_used.csv"
    write_csv(tbl[tbl["universe"] == "partB_pooled"], p)
    files.append(p)

    U_bin, U_ord = select_universe_b(master, "binary"), select_universe_b(master, "ordinal")
    accounting, tables = [], {}

    def run(label, family, U, sens):
        d = make_design(U, family, sens, label)
        accounting.append(d["acc"])
        if not d["ok"]:
            d["acc"]["analysis_status"] = "failed: design matrix is rank deficient"
            print(f"  {label}: FAILED (rank-deficient design) -- reported, not silently reduced")
            return d, None
        zp = Z.loc[d["cc"]["pair_id"].to_numpy(), path_cols].to_numpy(dtype=float)
        d["zp"] = zp
        print(f"\n[{label}] rows {d['acc']['universe_rows']} -> {d['acc']['final_rows']} "
              f"(incomplete covariates lost {d['acc']['rows_lost_incomplete_covariates']})", flush=True)
        df, info = analyze(family, d, zp, path_cols, label)
        d["acc"].update(info)
        d["acc"]["analysis_status"] = "complete" if info["n_pathways_failed"] == 0 else (
            "partial" if info["n_pathways_ok"] > 0 else "failed")
        tables[label] = df
        return d, df

    # ---- PRIMARY: binary improved (GEE) ----
    dp, df_primary = run("binary_primary", "binary", U_bin, "primary")
    require(dp["ok"] and df_primary is not None, "primary binary design failed: STOP")
    require(df_primary["status"].eq("ok").all(), "primary binary analysis has failed pathways: STOP and inspect")

    # ---- fast implementation cross-check, then GLOBAL permutation test ----
    print("\n[cross-check] numpy clustered logit vs statsmodels GEE...", flush=True)
    crosscheck_numpy_vs_gee(dp, dp["zp"])
    print(f"[global] {n_perm} whole-RNA-vector permutations (B1 and B2)...", flush=True)
    glob, obs_df = global_permutation(dp, dp["zp"], n_perm, GLOBAL_PERM_SEED)
    glob["smoke"] = bool(args.smoke)
    glob["n_pathways"] = len(path_cols)
    write_json(glob, f"{out_dir}/partB_global_permutation.json")
    files.append(f"{out_dir}/partB_global_permutation.json")
    df_primary["B1_wald_chi2_numpy"] = obs_df["B1_wald_chi2_numpy"].to_numpy()
    df_primary["B2_wald_chi2_numpy"] = obs_df["B2_wald_chi2_numpy"].to_numpy()
    print(f"  GLOBAL: B1 p={glob['B1_global_p']:.4g}  B2 p={glob['B2_global_p']:.4g}")

    # ---- SECONDARY: worse/same/better (OrdinalGEE) ----
    do, df_ord = run("ordinal_secondary_worse_same_better", "ordinal", U_ord, "primary")
    # ---- SENSITIVITIES (binary) ----
    run("binary_sens_plasma_cell_pct", "binary", U_bin, "plasma")
    run("binary_sens_prior_treatment_exposure", "binary", U_bin, "prior")

    # ---- EXPLORATORY (Tier 2): separate pathway x CD38 interaction analysis (own BH family) ----
    cd = tbl[(tbl["treatment"] == "cd38") & (tbl["universe"] == "partB_pooled")].iloc[0]
    require(int(cd["tier"]) == 2, "CD38 must be Tier 2 (exploratory) in the pooled universe: STOP")
    label_cd = "binary_exploratory_tier2_cd38_interaction"
    print(f"\n[{label_cd}] exploratory, same analysis rows as the primary binary analysis "
          f"({dp['acc']['final_rows']} rows); own BH family over {len(path_cols)} pathways", flush=True)
    df_cd, info_cd = analyze_cd38("binary", dp, dp["zp"], path_cols, label_cd, cd)
    acc_cd = {**dp["acc"], "analysis": label_cd, "role": "exploratory_tier2", **info_cd}
    acc_cd["analysis_status"] = "complete" if info_cd["n_pathways_failed"] == 0 else (
        "partial" if info_cd["n_pathways_ok"] > 0 else "failed")
    accounting.append(acc_cd)
    tables[label_cd] = df_cd

    # ---- write ----
    for label, df in tables.items():
        p = f"{out_dir}/partB_results_{label}.csv"
        write_csv(df, p, float_format="%.8g")
        files.append(p)
    acc_df = pd.DataFrame(accounting)
    for ext, writer in (("csv", lambda path: write_csv(acc_df, path)),
                        ("json", lambda path: write_json(accounting, path))):
        p = f"{out_dir}/partB_row_accounting.{ext}"
        writer(p)
        files.append(p)
    summary = {"label": LABEL_B, "interpretation": RNA_INTERPRETATION, "smoke": bool(args.smoke),
               "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "runtime_seconds": round(time.time() - t0, 1), "n_pathways": len(path_cols),
               "global_permutation": glob,
               "models": {"B1": "M0: pathway + PI + IMiD + steroid + covariates (main effect)",
                          "B2": "M1: M0 + pathway x PI + pathway x IMiD + pathway x steroid; joint 3-df Wald",
                          "estimator": "GEE independence, patient-clustered robust SEs (statsmodels); OrdinalGEE for the secondary"},
               "fdr": "BH over the pathways separately for B1 and B2 (per analysis)",
               "covariates_complete_case": "no imputation / zero-filling / pooling; row attrition in partB_row_accounting",
               "analysis_status": [{k: a.get(k) for k in ("analysis", "analysis_status", "final_rows",
                                                          "rows_lost_incomplete_covariates", "failure_reasons")}
                                   for a in accounting],
               "primary_counts": {"B1_q_lt_0.05": int((tables["binary_primary"]["B1_q_bh"] < 0.05).sum()),
                                  "B2_q_lt_0.05": int((tables["binary_primary"]["B2_q_bh"] < 0.05).sum())},
               "exploratory_tier2_cd38": {
                   "model": "M0 + pathway x PI/IMiD/steroid (adjustment terms only) + pathway x CD38 (sole target, 1 df); separate from B1/B2",
                   "bh_family": "own BH over the pathways, NOT mixed into the primary families",
                   "interaction_q_lt_0.05": int((tables[label_cd]["CD38_interaction_q_bh"] < 0.05).sum()),
                   "support": {"pooled_rna_patients": int(cd["group_pts"]), "pooled_events": int(cd["group_events"]),
                               "tier": int(cd["tier"])},
                   "interpretation": "exploratory / hypothesis-generating (Tier 2); RNA is baseline as-of-sample biology"}}
    p = f"{out_dir}/run_summary.json"
    write_json(summary, p)
    files.append(p)
    man = write_output_manifest(out_dir, files, {"label": LABEL_B, "smoke": bool(args.smoke),
                                                 "input_master_sha256": sha256_file("data/clinical/visit_pairs_with_rna.csv"),
                                                 "global_permutation_seed": GLOBAL_PERM_SEED})
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs in {out_dir}\nCombined output SHA256: {man['combined_sha256']}")
    print(LABEL_B)


if __name__ == "__main__":
    main()
