"""
Direction 2 -- STEP 2: frozen RNA association analysis on the frozen matched pairs.

*** RETROSPECTIVE MOLECULAR ASSOCIATION -- NOT PREDICTIVE VALIDATION ***
Discordance is defined by the Vt+1 response. Results describe association
between baseline-biased RNA (one sample per patient for ~95% of patients) and
the next-visit response among clinically similar visits. They are NOT evidence
that RNA predicts response.

Order of operations (enforced in code):
  1. Verify the SHA256 manifest written by direction2_matching.py. ABORT on any
     mismatch (pair tables, spec, or input files changed).
  2. Only THEN read the full master table (outcomes + 50 pathway columns).
  3. Orient pairs by outcome, build pair-level deltas of z-scored pathway scores.
  4. Run the frozen analyses.

FAMILIES (BH-FDR is applied WITHIN each family x analysis over the 50 pathways)
  ordinal  PRIMARY    M1 pairs (multiclass universe) whose Vt+1 classes differ.
                      delta = z(pathway of higher-Vt+1-rank member) - z(lower-rank member)
  binary   SECONDARY  M1 pairs (binary universe) discordant on 'improved'.
                      delta = z(improved member) - z(non-improved member)
Pathway scores are z-scored over the DISTINCT (patient, RNA-sample) vectors
(expected 751), not over visit rows.

PRIMARY TEST (per pathway): exact sign-flip permutation (exact enumeration if
n_pairs <= 20, else 10,000 Monte Carlo flips with add-one correction) on the
studentised mean paired difference; pair-level percentile bootstrap CIs (10,000);
BH q; single-step max-T FWER p; GLOBAL test = sum of squared d_z across the 50
pathways under the SAME flips (pathway correlation preserved).
Seeds: per-analysis sign-flip seed = 42 + crc32('<family>|<analysis>') % 100000,
bootstrap seed = that + 1 (both written to every results row); S4 draw k = 1042 + k.

SENSITIVITIES (declared in advance; none is used to choose a primary result)
  S1  calipers p50 / p90 (frozen pair tables)
  S2  RNA-age GAP between matched members: |days_since_rna_sample(a) - days_since_rna_sample(b)| <= 365.
      This is the pre-specified between-member DIFFERENCE, NOT a cap on either member's own RNA age.
  S3  regression-adjusted analysis on all rows (rules below; no pooling, no imputation, no model fallback)
  S4  200 randomised patient-once matchings (seeded random visiting order): median effect, IQR, sign share
  S5  S3 + pct_plasma_cells_bm
  S6  leave-one-Vt-response-stratum-out; line-1-only; line-2+ only
  S7  discovery (both members train/val) vs holdout (both test): DESCRIPTIVE ONLY

S3/S5 HANDLING RULES (deterministic, documented, all counts reported in
S3_S5_row_accounting.csv / .json):
  0. COVARIATE SET: the 17 matching features MINUS line_duration_so_far (16 continuous features) plus
     log1p(days_since_rna_sample). line_duration_so_far is STRUCTURALLY missing when no treatment line is
     active at Vt (treatment state, not ordinary missingness); keeping it under complete-case handling
     would restrict S3/S5 to actively treated visits, so it is removed from the S3/S5 covariates ONLY.
     Treatment context stays adjusted through the exact strata (line group x PI/IMiD/CD38) and
     n_prior_lines_completed. The primary matching and its 17-feature distance are NOT changed.
  1. COMPLETE-CASE on the remaining (genuinely missing) covariates: rows missing any of the 16 retained
     continuous features (S5: and pct_plasma_cells_bm) are excluded. NO imputation, NO zero-filling.
     Missing counts per covariate and rows lost are reported; rows with missing line_duration_so_far
     are NOT excluded and their number is reported.
  2. Exact-strata fixed effects: one dummy per stratum (Vt response x line group x PI/IMiD/CD38).
     Strata with < 20 complete-case rows are EXCLUDED with their rows (never pooled).
  3. Strata whose outcome has no variation among the remaining rows are EXCLUDED (their dummy is not
     estimable); counts reported.
  4. Covariates are transformed / winsorised (1st-99th) / z-scored on the FINAL analysis rows.
  5. Models: binary = GEE binomial-logit; ordinal = OrdinalGEE (cumulative logit); both with
     independence working correlation and patient-clustered robust SEs. There is NO fallback model:
     an exception, non-convergence or a rank-deficient design marks that pathway (or the whole
     analysis) as FAILED with the reason. A failed analysis is never replaced by a different model.

Run from the repo root:
    python -m explainability.direction2_association
Options:  --smoke (tiny, separate output dir)  --skip-regression (smoke only)
          --force (smoke only: overwrite existing association outputs)
          --n-perm N --n-boot N (frozen values 10000 / 10000)
"""

import argparse
import datetime
import json
import os
import time
import warnings
import zlib

import numpy as np
import pandas as pd

from explainability.direction2_common import (
    ANALYSIS_LABEL, ANALYSIS_SPEC, AUDIT_REFERENCE, CONTINUOUS_FEATURES, ELIGIBILITY_PATH, EXACT_KEYS,
    EXPECTED, FAMILIES, FAMILY_ROLE, MASTER_PATH, RANDOM_SEED, RESPONSE_RANK, S2_RNA_AGE_GAP_CAP_DAYS,
    S3_CONTINUOUS_FEATURES, S3_EXCLUDED_FEATURES, S3_MIN_STRATUM_ROWS, S4_SEED_BASE, SPLIT_PATH,
    STALENESS_COL, assert_similarity_inputs_clean,
    bh_qvalues, combined_hash, continuous_matrix, expect, forbidden_columns, get_dirs,
    load_feature_sets, package_versions, pathway_columns, require, select_universe, sha256_file,
    to_bool, write_csv, write_json,
)

N_PERM_DEFAULT = 10000
N_BOOT_DEFAULT = 10000
EXACT_MAX_N = 20
MIN_INFORMATIVE_PAIRS = 5
S2_NAME = "S2_rna_age_gap_le_365d"

S3_COVARIATE_SET = list(S3_CONTINUOUS_FEATURES) + ["log1p_days_since_rna_sample"]
S5_COVARIATE_SET = S3_COVARIATE_SET + ["pct_plasma_cells_bm"]

PAIR_DESIGN_FILES = {"primary": "primary_p75_M1", "S1_p50": "S1_p50_M1", "S1_p90": "S1_p90_M1"}


def analysis_seed(family, name):
    """Deterministic per-analysis seed (recorded in every results row)."""
    return RANDOM_SEED + zlib.crc32(f"{family}|{name}".encode("utf-8")) % 100000


# ---------------------------------------------------------------------------
# 1. Verify the freeze BEFORE any RNA/outcome value is read
# ---------------------------------------------------------------------------
def verify_freeze(dirs):
    mpath = f"{dirs['frozen']}/frozen_pair_manifest.json"
    require(os.path.exists(mpath), f"{mpath} not found: run direction2_matching first")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    for rel, expected in manifest["files"].items():
        p = f"{dirs['frozen']}/{rel}"
        require(os.path.exists(p), f"frozen file missing: {p}")
        actual = sha256_file(p)
        require(actual == expected, f"FROZEN FILE CHANGED since freeze: {rel}\n  expected {expected}\n  actual   {actual}")
    require(combined_hash(manifest["files"]) == manifest["combined_sha256"], "combined frozen hash mismatch")
    for p, expected in manifest["input_files_sha256"].items():
        require(sha256_file(p) == expected, f"INPUT FILE CHANGED since freeze: {p}")
    print(f"Freeze verified. Combined SHA256 = {manifest['combined_sha256']}")
    return manifest


def recheck_pair_structure(P, label):
    """Structural re-check of a loaded frozen pair table (fail loudly)."""
    groups = [P] if (P["draw"] == -1).all() else [g for _, g in P.groupby("draw")]
    for g in groups:
        require((g["public_id_a"].to_numpy() != g["public_id_b"].to_numpy()).all(),
                f"{label}: same-patient pair in frozen table")
        pats = pd.concat([g["public_id_a"], g["public_id_b"]])
        require(pats.is_unique, f"{label}: patient appears in more than one pair")
        require((g["pair_id_a"].to_numpy() < g["pair_id_b"].to_numpy()).all(), f"{label}: pair orientation")


def load_frozen_pairs(dirs, fs):
    """Load the frozen pair tables; the tables must contain no outcome / RNA / vt1_* column."""
    forbidden = forbidden_columns(fs)
    pairs = {}
    for family in FAMILIES:
        for key, design in list(PAIR_DESIGN_FILES.items()) + [("S4", "S4_random_M1")]:
            P = pd.read_csv(f"{dirs['frozen']}/pairs_{family}_{design}.csv")
            bad = sorted(set(P.columns) & forbidden)
            require(not bad, f"frozen pair table {family}/{design} contains forbidden columns: {bad}")
            require(not any(c.startswith("vt1_") for c in P.columns), "frozen pair table has a vt1_* column")
            recheck_pair_structure(P, f"{family}/{design}")
            pairs[(family, key)] = P
    return pairs


# ---------------------------------------------------------------------------
# 2. Merge outcomes + pathway scores (first point RNA values are read)
# ---------------------------------------------------------------------------
def load_master_full():
    m = pd.read_csv(MASTER_PATH, low_memory=False)
    expect("raw master column count", m.shape[1], EXPECTED["master_columns"])
    sp = pd.read_csv(SPLIT_PATH, usecols=["pair_id", "split"])
    el = pd.read_csv(ELIGIBILITY_PATH, usecols=["pair_id", "eligible_for_binary"])
    m = m.merge(sp, on="pair_id", how="left", validate="one_to_one")
    m = m.merge(el, on="pair_id", how="left", validate="one_to_one")
    require(m["split"].notna().all() and m["eligible_for_binary"].notna().all(), "split/eligibility merge gaps")
    m["eligible_for_binary"] = to_bool(m["eligible_for_binary"])
    m["rna_available"] = m[STALENESS_COL].notna()

    expect("master rows", len(m), EXPECTED["master_rows"])
    expect("master patients", int(m["public_id"].nunique()), EXPECTED["master_patients"])
    expect("RNA-available rows", int(m["rna_available"].sum()), EXPECTED["rna_rows"])
    expect("RNA-available patients", int(m.loc[m["rna_available"], "public_id"].nunique()), EXPECTED["rna_patients"])
    expect("binary-eligible rows", int(m["eligible_for_binary"].sum()), EXPECTED["binary_eligible_rows"])
    sp_pat = m.drop_duplicates("public_id").groupby("split").size().to_dict()
    for s, n in EXPECTED["split_patients"].items():
        expect(f"patients in split {s}", int(sp_pat.get(s, 0)), n)
    return m


def build_pathway_z(m, path_cols):
    """z-score each pathway across DISTINCT (patient, sample) vectors; return Z indexed by pair_id (RNA rows)."""
    r = m[m["rna_available"]].copy()
    miss = r[path_cols].isna()
    require(not miss.any().any(), "RNA-available rows with missing pathway values")
    nonr = m[~m["rna_available"]]
    require(nonr[path_cols].isna().all(axis=1).all(), "pathway values present on rows without RNA availability")

    h = pd.util.hash_pandas_object(r[path_cols].round(8), index=False).to_numpy()
    pid = pd.factorize(r["public_id"])[0]
    key = pd.DataFrame({"pid": pid, "h": h})
    first = ~key.duplicated().to_numpy()
    expect("distinct (patient, RNA sample) vectors", int(first.sum()), EXPECTED["distinct_rna_samples"])
    distinct = r.loc[first, path_cols]
    mu = distinct.mean(axis=0)
    sd = distinct.std(axis=0, ddof=1)
    require((sd > 0).all(), "a pathway has zero variance across distinct samples")
    Z = (r[path_cols] - mu) / sd
    Z.index = r["pair_id"].to_numpy()
    return Z, {"n_distinct_samples": int(first.sum()), "n_rna_rows": int(len(r))}


def check_pair_identity(P, master_pid, label):
    """Frozen pair table ids must still agree with the master table."""
    a = master_pid.loc[P["pair_id_a"].to_numpy()].astype(str).to_numpy()
    b = master_pid.loc[P["pair_id_b"].to_numpy()].astype(str).to_numpy()
    require((a == P["public_id_a"].astype(str).to_numpy()).all() and
            (b == P["public_id_b"].astype(str).to_numpy()).all(),
            f"{label}: public_id in frozen pairs disagrees with the master table")


# ---------------------------------------------------------------------------
# 3. Orient pairs by outcome
# ---------------------------------------------------------------------------
def orient_pairs(pairs, O, family):
    a = pairs["pair_id_a"].to_numpy()
    b = pairs["pair_id_b"].to_numpy()
    if family == "binary":
        require(bool(O.loc[a, "eligible_for_binary"].all()) and bool(O.loc[b, "eligible_for_binary"].all()),
                "binary pair includes a row that is not eligible_for_binary")
        va = O.loc[a, "improved"].to_numpy(dtype=float)
        vb = O.loc[b, "improved"].to_numpy(dtype=float)
    else:
        va = O.loc[a, "exact_next_response"].map(RESPONSE_RANK).to_numpy(dtype=float)
        vb = O.loc[b, "exact_next_response"].map(RESPONSE_RANK).to_numpy(dtype=float)
    require(np.isfinite(va).all() and np.isfinite(vb).all(), f"[{family}] unmapped outcome values in pairs")
    a_hi = va > vb
    out = pairs.copy()
    out["outcome_a"] = va
    out["outcome_b"] = vb
    out["informative"] = va != vb
    out["pair_id_hi"] = np.where(a_hi, a, b)
    out["pair_id_lo"] = np.where(a_hi, b, a)
    pa = pairs["public_id_a"].to_numpy(dtype=object)
    pb = pairs["public_id_b"].to_numpy(dtype=object)
    out["public_id_hi"] = np.where(a_hi, pa, pb)
    out["public_id_lo"] = np.where(a_hi, pb, pa)
    sa = O.loc[a, STALENESS_COL].to_numpy(dtype=float)
    sb = O.loc[b, STALENESS_COL].to_numpy(dtype=float)
    require(np.isfinite(sa).all() and np.isfinite(sb).all(), "pair member without RNA timing")
    out["rna_age_a"], out["rna_age_b"] = sa, sb
    out["rna_age_gap"] = np.abs(sa - sb)          # S2 uses THIS (between-member difference)
    require(np.allclose(out["rna_age_gap"].to_numpy(), np.abs(out["rna_age_a"].to_numpy() - out["rna_age_b"].to_numpy())),
            "rna_age_gap is not the absolute between-member difference")
    out["split_a"] = O.loc[a, "split"].to_numpy(dtype=object)
    out["split_b"] = O.loc[b, "split"].to_numpy(dtype=object)
    return out


def delta_matrix(inf, Z):
    hi = Z.loc[inf["pair_id_hi"].to_numpy()].to_numpy(dtype=float)
    lo = Z.loc[inf["pair_id_lo"].to_numpy()].to_numpy(dtype=float)
    d = hi - lo
    require(np.isfinite(d).all(), "non-finite pathway delta")
    return d


# ---------------------------------------------------------------------------
# 4. Paired statistics
# ---------------------------------------------------------------------------
def _flip_chunks(n, n_perm, seed):
    if n <= EXACT_MAX_N:
        total = 1 << n
        step = 1 << 15
        bits = np.arange(n, dtype=np.int64)
        for start in range(0, total, step):
            ids = np.arange(start, min(start + step, total), dtype=np.int64)
            yield (1 - 2 * ((ids[:, None] >> bits[None, :]) & 1)).astype(np.float64)
    else:
        rng = np.random.RandomState(seed)
        done = 0
        while done < n_perm:
            c = min(2000, n_perm - done)
            yield (rng.randint(0, 2, size=(c, n)) * 2 - 1).astype(np.float64)
            done += c


def pair_analysis(delta, seed, n_perm, n_boot):
    """delta: (n_pairs x 50). Returns per-pathway arrays + global statistics."""
    n, k = delta.shape
    mean = delta.mean(axis=0)
    m2 = (delta ** 2).mean(axis=0)
    var = np.maximum((m2 - mean ** 2) * n / (n - 1), 1e-24)
    sd = np.sqrt(var)
    dz = mean / sd
    t_obs = mean / np.sqrt(var / n)
    res = {"n": n, "mean": mean, "sd": sd, "dz": dz, "t_obs": t_obs}

    # --- sign-flip permutation (shared flips across pathways => correlation preserved) ---
    abs_t = np.abs(t_obs)
    tol = 1.0 - 1e-12
    g_obs = float((dz ** 2).sum())
    count = np.zeros(k)
    count_max = np.zeros(k)
    count_glob = 0
    total = 0
    for S in _flip_chunks(n, n_perm, seed):
        mb = S @ delta / n
        vb = np.maximum((m2[None, :] - mb ** 2) * n / (n - 1), 1e-24)
        tb = np.abs(mb / np.sqrt(vb / n))
        count += (tb >= abs_t[None, :] * tol).sum(axis=0)
        mx = tb.max(axis=1)
        count_max += (mx[:, None] >= abs_t[None, :] * tol).sum(axis=0)
        gb = (mb ** 2 / vb).sum(axis=1)
        count_glob += int((gb >= g_obs * tol).sum())
        total += S.shape[0]
    exact = n <= EXACT_MAX_N
    if exact:
        p, p_max, p_glob = count / total, count_max / total, count_glob / total
    else:
        p, p_max, p_glob = (count + 1) / (total + 1), (count_max + 1) / (total + 1), (count_glob + 1) / (total + 1)
    res.update({"p": p, "p_maxT": p_max, "global_dz2": g_obs, "global_p": float(p_glob),
                "p_method": "exact_enumeration" if exact else "monte_carlo_add_one", "n_flips": int(total)})

    # --- pair-level percentile bootstrap (chunked) ---
    rng = np.random.RandomState(seed + 1)
    means, dzs = [], []
    for s0 in range(0, n_boot, 250):
        c = min(250, n_boot - s0)
        idx = rng.randint(0, n, size=(c, n))
        X = delta[idx]
        mu = X.mean(axis=1)
        sdv = np.maximum(X.std(axis=1, ddof=1), 1e-12)
        means.append(mu)
        dzs.append(mu / sdv)
    means = np.vstack(means)
    dzs = np.vstack(dzs)
    res["mean_ci"] = (np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0))
    res["dz_ci"] = (np.percentile(dzs, 2.5, axis=0), np.percentile(dzs, 97.5, axis=0))
    return res


def results_table(res, path_cols, family, analysis, role, seed):
    df = pd.DataFrame({
        "family": family, "family_role": FAMILY_ROLE[family], "analysis": analysis, "analysis_role": role,
        "pathway": path_cols, "n_pairs": res["n"],
        "mean_delta": res["mean"], "sd_delta": res["sd"], "d_z": res["dz"], "t_obs": res["t_obs"],
    })
    df["mean_ci_lo"], df["mean_ci_hi"] = res["mean_ci"]
    df["dz_ci_lo"], df["dz_ci_hi"] = res["dz_ci"]
    df["p_signflip"] = res["p"]
    df["q_bh"] = bh_qvalues(res["p"])
    df["p_maxT"] = res["p_maxT"]
    df["significant_q_lt_0.05"] = df["q_bh"] < 0.05
    df["suggestive_0.05_le_q_lt_0.10"] = (df["q_bh"] >= 0.05) & (df["q_bh"] < 0.10)
    df["p_method"] = res["p_method"]
    df["n_flips"] = res["n_flips"]
    df["seed_signflip"] = seed
    df["seed_bootstrap"] = seed + 1
    return df


def global_row(res, family, analysis, role, seed):
    q = bh_qvalues(res["p"])
    return {"family": family, "analysis": analysis, "analysis_role": role, "n_informative_pairs": res["n"],
            "global_sum_dz2": res["global_dz2"], "global_p": res["global_p"],
            "p_method": res["p_method"], "n_flips": res["n_flips"],
            "min_p_signflip": float(np.min(res["p"])), "min_q_bh": float(np.min(q)),
            "n_q_lt_0.05": int((q < 0.05).sum()), "n_q_lt_0.10": int((q < 0.10).sum()),
            "n_maxT_lt_0.05": int((res["p_maxT"] < 0.05).sum()),
            "seed_signflip": seed, "seed_bootstrap": seed + 1}


# ---------------------------------------------------------------------------
# S3 / S5: regression-adjusted analysis on ALL universe rows (rules in module docstring)
# ---------------------------------------------------------------------------
def strata_key(U):
    return (U["vt_disease_response"].astype(str) + "|" + U["line_group"].astype(str) + "|"
            + U["currently_on_pi"].astype(int).astype(str) + U["currently_on_imid"].astype(int).astype(str)
            + U["currently_on_cd38"].astype(int).astype(str))


PATHWAY_TERM = "pathway_z"      # name of the tested pathway predictor in every S3/S5 model


def ordinal_pathway_param_index(mod, pathway_col):
    """OrdinalGEE's parameter vector is NOT just the supplied covariates: statsmodels expands the
    data to one binary row per (observation, cut-point) and PREPENDS one threshold (cut-point)
    parameter per cumulative split (K-1 of them) to the coefficients. Taking params[0] therefore
    returns the first THRESHOLD, not the pathway effect (the bug fixed here).

    The pathway term is located by CONTENT, not by a hard-coded position: it is the unique column of
    the model's expanded design matrix that equals the tested pathway predictor (once per cut-point).
    Fails loudly unless (a) exactly one such column exists, (b) it is a continuous (non-binary)
    column, i.e. not a threshold/cut-point indicator, and (c) its name is the pathway term.
    Returns (parameter index, parameter name)."""
    exog = np.asarray(mod.exog, dtype=float)
    pathway_col = np.asarray(pathway_col, dtype=float)
    n = len(pathway_col)
    matches = []
    for j in range(exog.shape[1]):
        col = exog[:, j]
        if col.shape[0] % n != 0:
            continue
        rep = col.shape[0] // n                      # rows per observation = number of cut-points
        if np.allclose(np.sort(col), np.sort(np.repeat(pathway_col, rep)), atol=1e-10, rtol=0.0):
            matches.append(j)
    require(len(matches) == 1,
            f"expected exactly one expanded-design column equal to the tested pathway predictor, "
            f"found {len(matches)}")
    idx = matches[0]
    require(np.unique(exog[:, idx]).size > 2,
            "matched design column is binary: it is a threshold/cut-point indicator, not the pathway predictor")
    names = [f"term_{j}" for j in range(exog.shape[1])]
    names[idx] = PATHWAY_TERM
    name = names[idx]
    require(name == PATHWAY_TERM and not name.lower().startswith(("threshold", "cut", "const", "intercept", "i(")),
            f"extracted parameter {name!r} is not the pathway predictor")
    return idx, name


def _failed_frame(family, analysis, path_cols, reason, acc, covariate_set):
    df = pd.DataFrame({"family": family, "family_role": FAMILY_ROLE[family], "analysis": analysis,
                       "analysis_role": "sensitivity", "covariate_set": covariate_set, "pathway": path_cols,
                       "estimate": np.nan, "se_robust": np.nan, "p_wald_robust": np.nan, "ci_lo": np.nan,
                       "ci_hi": np.nan, "odds_ratio_per_SD": np.nan, "method_used": "",
                       "status": f"failed: {reason}", "n_rows": acc.get("final_rows", 0),
                       "n_patients": acc.get("final_patients", 0), "q_bh": np.nan})
    return df


def regression_analysis(family, U, Z, path_cols, analysis, add_purity):
    # Direct imports of only what is used. `statsmodels.api` is deliberately NOT imported: it pulls in
    # nonparametric/lowess extension modules that Windows Application Control can block.
    from statsmodels.genmod.generalized_estimating_equations import GEE, OrdinalGEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Independence
    covariate_set = "S5" if add_purity else "S3"
    require("line_duration_so_far" not in S3_CONTINUOUS_FEATURES,
            "line_duration_so_far must not be in the S3/S5 covariate set")
    cov_cols = list(S3_CONTINUOUS_FEATURES) + (["pct_plasma_cells_bm"] if add_purity else [])
    acc = {"family": family, "analysis": analysis, "covariate_set": covariate_set,
           "n_continuous_covariates": len(S3_CONTINUOUS_FEATURES),
           "features_excluded_from_S3_S5": json.dumps(S3_EXCLUDED_FEATURES),
           "universe_rows": int(len(U)), "universe_patients": int(U["public_id"].nunique()),
           "universe_rows_with_structurally_missing_line_duration_so_far":
               int(U["line_duration_so_far"].isna().sum())}

    # 1. complete-case on the frozen covariates (no imputation, no zero-filling)
    acc["rows_missing_by_covariate"] = json.dumps({c: int(U[c].isna().sum()) for c in cov_cols})
    any_miss = U[cov_cols].isna().any(axis=1).to_numpy()
    acc["rows_lost_incomplete_covariates"] = int(any_miss.sum())
    cc = U[~any_miss].reset_index(drop=True)
    acc["rows_after_complete_case"] = int(len(cc))
    if family == "binary":
        y = cc["improved"].to_numpy(dtype=int)
    else:
        y = cc["exact_next_response"].map(RESPONSE_RANK).to_numpy(dtype=int)

    # 2. exact strata: exclude (never pool) strata with < S3_MIN_STRATUM_ROWS complete-case rows
    key = strata_key(cc)
    sizes = key.map(key.value_counts()).to_numpy()
    sparse = sizes < S3_MIN_STRATUM_ROWS
    acc["n_strata_complete_case"] = int(key.nunique())
    acc["n_sparse_strata_excluded"] = int(key[sparse].nunique())
    acc["rows_lost_sparse_strata"] = int(sparse.sum())
    cc = cc[~sparse].reset_index(drop=True)
    key = key[~sparse].reset_index(drop=True)
    y = y[~sparse]

    # 3. exclude strata with no outcome variation among the remaining rows
    nun = pd.Series(y).groupby(key).transform("nunique").to_numpy()
    novar = nun < 2
    acc["n_strata_no_outcome_variation_excluded"] = int(key[novar].nunique())
    acc["rows_lost_no_outcome_variation"] = int(novar.sum())
    cc = cc[~novar].reset_index(drop=True)
    key = key[~novar].reset_index(drop=True)
    y = y[~novar]

    acc["final_rows"] = int(len(cc))
    acc["final_patients"] = int(cc["public_id"].nunique())
    acc["final_strata"] = int(key.nunique())
    acc["rows_lost_total"] = int(len(U) - len(cc))
    acc["share_rows_retained"] = float(len(cc) / len(U))
    if len(cc) < 200 or key.nunique() < 2:
        acc["analysis_status"] = "failed"
        acc["failure_reasons"] = "too few rows/strata after the frozen exclusion rules"
        return _failed_frame(family, analysis, path_cols, acc["failure_reasons"], acc, covariate_set), acc

    acc["final_rows_with_structurally_missing_line_duration_so_far_retained"] = \
        int(cc["line_duration_so_far"].isna().sum())

    # 4. design on the FINAL rows (16 retained features; line_duration_so_far is NOT a covariate)
    Zc = continuous_matrix(cc, features=S3_CONTINUOUS_FEATURES)
    require(Zc.shape[1] == len(S3_CONTINUOUS_FEATURES), "unexpected number of S3/S5 continuous covariates")
    require(np.isfinite(Zc).all(), "NaN remains in covariates after complete-case")
    stale = np.log1p(cc[STALENESS_COL].to_numpy(dtype=float))
    require(np.isfinite(stale).all(), "missing days_since_rna_sample in regression rows")
    stale = (stale - stale.mean()) / stale.std(ddof=0)
    blocks = [Zc, stale[:, None]]
    if add_purity:
        pur = cc["pct_plasma_cells_bm"].to_numpy(dtype=float)
        require(np.isfinite(pur).all(), "NaN pct_plasma_cells_bm after complete-case")
        blocks.append(((pur - pur.mean()) / pur.std(ddof=0))[:, None])
    D = pd.get_dummies(key, drop_first=True, dtype=float)
    blocks.append(D.to_numpy(dtype=float))
    X_base = np.column_stack(blocks)
    n = len(cc)
    if family == "binary":
        X_base = np.column_stack([np.ones(n), X_base])
    rank_ok = int(np.linalg.matrix_rank(X_base)) == X_base.shape[1]
    acc["design_columns"] = int(X_base.shape[1])
    acc["design_full_rank"] = bool(rank_ok)
    if not rank_ok:
        acc["analysis_status"] = "failed"
        acc["failure_reasons"] = "design matrix is rank deficient"
        return _failed_frame(family, analysis, path_cols, acc["failure_reasons"], acc, covariate_set), acc

    groups = pd.factorize(cc["public_id"])[0]
    zp = Z.loc[cc["pair_id"].to_numpy()].to_numpy(dtype=float)
    rows = []
    for j, pw in enumerate(path_cols):
        X = np.column_stack([zp[:, j], X_base])
        status, method = "ok", ""
        est = se = pval = lo = hi = np.nan
        pw_idx, pw_name = -1, ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if family == "binary":
                    method = "GEE_binomial_logit_independence_robust"
                    mod = GEE(y, X, groups=groups, family=Binomial(),
                              cov_struct=Independence())
                else:
                    method = "OrdinalGEE_cumulative_logit_independence_robust"
                    mod = OrdinalGEE(y, X, groups=groups, cov_struct=Independence())
                r = mod.fit(maxiter=100)
            if getattr(r, "converged", None) is False:
                status = "failed: not_converged"
            else:
                params = np.asarray(r.params)
                if family == "binary":
                    # binary GEE has no threshold parameters: design is [pathway, intercept, ...]
                    pw_idx, pw_name = 0, PATHWAY_TERM
                else:
                    # OrdinalGEE params = [thresholds..., coefficients...]: locate the pathway term
                    # explicitly (by content/name), never by position, and assert it is not a threshold.
                    pw_idx, pw_name = ordinal_pathway_param_index(mod, X[:, 0])
                    require(len(params) == np.asarray(mod.exog).shape[1],
                            "OrdinalGEE parameter vector length differs from its expanded design matrix")
                require(pw_name == PATHWAY_TERM, "extracted parameter is not the tested pathway predictor")
                est = float(params[pw_idx])
                se = float(np.asarray(r.bse)[pw_idx])
                pval = float(np.asarray(r.pvalues)[pw_idx])
                ci = np.asarray(r.conf_int())[pw_idx]
                lo, hi = float(ci[0]), float(ci[1])
                if not np.isfinite([est, se, pval]).all():
                    status = "failed: non_finite_estimate"
                    est = se = pval = lo = hi = np.nan
        except AssertionError as e:
            if str(e).startswith("[direction2]"):
                raise                                            # structural check failures are never swallowed
            status = f"failed: AssertionError: {e}"
        except Exception as e:                                   # NO fallback model
            status = f"failed: {type(e).__name__}: {e}"
        rows.append({"family": family, "family_role": FAMILY_ROLE[family], "analysis": analysis,
                     "analysis_role": "sensitivity", "covariate_set": covariate_set, "pathway": pw,
                     "estimate": est, "se_robust": se, "p_wald_robust": pval, "ci_lo": lo, "ci_hi": hi,
                     "odds_ratio_per_SD": float(np.exp(est)) if np.isfinite(est) else np.nan,
                     "method_used": method, "status": status, "n_rows": n,
                     "n_patients": int(len(np.unique(groups))),
                     "extracted_param_name": pw_name, "extracted_param_index": pw_idx})
    df = pd.DataFrame(rows)
    df["q_bh"] = bh_qvalues(df["p_wald_robust"].to_numpy())
    n_fail = int(df["status"].str.startswith("failed").sum())
    acc["n_pathways_ok"] = int(len(df) - n_fail)
    acc["n_pathways_failed"] = n_fail
    acc["analysis_status"] = "complete" if n_fail == 0 else ("partial" if n_fail < len(df) else "failed")
    acc["failure_reasons"] = json.dumps(sorted(set(df.loc[df["status"].str.startswith("failed"), "status"])))
    return df, acc


# ---------------------------------------------------------------------------
# Balance / attrition / count QA
# ---------------------------------------------------------------------------
def balance_table(inf, U):
    Zc = continuous_matrix(U)
    zdf = pd.DataFrame(Zc, columns=list(CONTINUOUS_FEATURES), index=U["pair_id"].to_numpy())
    stale = np.log1p(U[STALENESS_COL].to_numpy(dtype=float))
    zdf["log1p_days_since_rna_sample"] = (stale - stale.mean()) / stale.std(ddof=0)
    pur = U["pct_plasma_cells_bm"].to_numpy(dtype=float)
    zdf["pct_plasma_cells_bm"] = (pur - np.nanmean(pur)) / np.nanstd(pur)
    hi = zdf.loc[inf["pair_id_hi"].to_numpy()].to_numpy(dtype=float)
    lo = zdf.loc[inf["pair_id_lo"].to_numpy()].to_numpy(dtype=float)
    diff = hi - lo
    sd = np.nanstd(np.vstack([hi, lo]), axis=0)
    smd = np.nanmean(diff, axis=0) / np.where(sd == 0, np.nan, sd)
    return pd.DataFrame({"feature": zdf.columns, "n_pairs_observed": (~np.isnan(diff)).sum(axis=0),
                         "paired_SMD_hi_minus_lo": smd, "abs_SMD_gt_0.25": np.abs(smd) > 0.25})


def attrition_row(family, oriented, U):
    inf = oriented[oriented["informative"]]
    row = {"family": family, "role": FAMILY_ROLE[family], "universe_rows": len(U),
           "universe_patients": int(U["public_id"].nunique()), "n_pairs": len(oriented),
           "n_patients_in_pairs": int(2 * len(oriented)), "n_informative_pairs": len(inf),
           "n_concordant_pairs": int((~oriented["informative"]).sum()),
           "n_informative_patients": int(2 * len(inf)),
           "informative_S2_rna_age_gap_le_365d": int((inf["rna_age_gap"] <= S2_RNA_AGE_GAP_CAP_DAYS).sum()),
           "pairs_trainval_trainval": int(((oriented["split_a"] != "test") & (oriented["split_b"] != "test")).sum()),
           "pairs_test_test": int(((oriented["split_a"] == "test") & (oriented["split_b"] == "test")).sum()),
           "pairs_cross_split": int(((oriented["split_a"] == "test") != (oriented["split_b"] == "test")).sum()),
           "informative_trainval_trainval": int(((inf["split_a"] != "test") & (inf["split_b"] != "test")).sum()),
           "informative_test_test": int(((inf["split_a"] == "test") & (inf["split_b"] == "test")).sum()),
           "median_rna_age_gap_days_informative": float(inf["rna_age_gap"].median()) if len(inf) else np.nan}
    if family == "binary":
        row["concordant_both_improved"] = int(((oriented["outcome_a"] == 1) & (oriented["outcome_b"] == 1)).sum())
        row["concordant_both_not_improved"] = int(((oriented["outcome_a"] == 0) & (oriented["outcome_b"] == 0)).sum())
    return row


def count_qa_row(family, n_pairs, n_informative, matching_attrition):
    """EXACT comparison with the earlier audit; explains any difference (no tolerance)."""
    ref = AUDIT_REFERENCE[family]
    mq = matching_attrition["designs"]["primary_p75_M1"]["count_qa"]
    pairs_equal = n_pairs == ref["pairs"]
    inf_equal = n_informative == ref["informative_pairs"]
    same_sel = mq["pair_set_identical_to_legacy_tie_order"]
    if pairs_equal and inf_equal:
        why = "identical to the audit counts"
    elif same_sel:
        why = ("UNEXPLAINED BY TIE-BREAKING: the frozen selection is identical to the legacy tie-order selection, "
               "so the difference must come from data/code differences versus the audit run. INVESTIGATE before "
               "interpreting any result.")
    else:
        why = (f"pair selection differs from the legacy tie order ({mq['n_pairs_only_in_frozen_selection']} pairs only "
               f"in the frozen selection, {mq['n_pairs_only_in_legacy_selection']} only in the legacy selection); "
               f"{mq['n_candidate_pairs_sharing_an_exactly_tied_distance']} candidate pairs share an exactly tied "
               f"distance, which is where explicit pair_id tie-breaking can legitimately change the selection. The "
               f"informative count additionally depends on the outcomes of whichever tied pairs were chosen.")
    return {"family": family, "exact_pairs": int(n_pairs), "audit_pairs": ref["pairs"],
            "pairs_difference": int(n_pairs - ref["pairs"]),
            "exact_informative_pairs": int(n_informative), "audit_informative_pairs": ref["informative_pairs"],
            "informative_difference": int(n_informative - ref["informative_pairs"]),
            "identical_to_audit": bool(pairs_equal and inf_equal),
            "selection_identical_to_legacy_tie_order": bool(same_sel), "explanation": why}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-regression", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    args = ap.parse_args()
    if args.smoke:
        n_perm, n_boot = 500, 300
    else:
        require(not args.skip_regression, "--skip-regression is only allowed together with --smoke")
        require(not args.force, "--force is only allowed together with --smoke")
        require(args.n_perm == N_PERM_DEFAULT and args.n_boot == N_BOOT_DEFAULT,
                f"frozen design requires --n-perm {N_PERM_DEFAULT} and --n-boot {N_BOOT_DEFAULT}")
        n_perm, n_boot = args.n_perm, args.n_boot

    t0 = time.time()
    print(f"DIRECTION 2 / STEP 2: association  |  {ANALYSIS_LABEL}")
    dirs = get_dirs(args.smoke)
    if os.path.exists(f"{dirs['assoc']}/run_summary.json") and not args.force:
        raise SystemExit(f"[direction2] {dirs['assoc']}/run_summary.json exists: existing results are never "
                         f"overwritten silently. Delete them deliberately (or use --smoke --force for smoke runs).")
    os.makedirs(dirs["assoc"], exist_ok=True)
    with open(f"{dirs['assoc']}/README_RETROSPECTIVE_ASSOCIATION.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(ANALYSIS_LABEL + "\n\nEvery file in this folder describes a retrospective association between "
                "pathway scores of the latest RNA sample at or before Vt and the Vt+1 response among "
                "clinically similar visits. Discordance is defined by Vt+1. None of these results is a "
                "predictive validation.\n")
    write_json({**ANALYSIS_SPEC, "n_perm": n_perm, "n_boot": n_boot, "smoke": bool(args.smoke),
                "s2_cap_days": S2_RNA_AGE_GAP_CAP_DAYS, "s3_min_stratum_rows": S3_MIN_STRATUM_ROWS,
                "s3_covariate_set": S3_COVARIATE_SET, "s5_covariate_set": S5_COVARIATE_SET,
                "s3_s5_excluded_features": S3_EXCLUDED_FEATURES,
                "matching_distance_features_unchanged": list(CONTINUOUS_FEATURES),
                "s4_seed_base": S4_SEED_BASE}, f"{dirs['assoc']}/analysis_spec.json")

    fs = load_feature_sets()
    assert_similarity_inputs_clean(fs)
    path_cols = pathway_columns(fs)

    # 1. verify freeze, 2. load frozen pairs (still no RNA/outcome values read)
    manifest = verify_freeze(dirs)
    frozen_pairs = load_frozen_pairs(dirs, fs)

    # 3. FIRST read of outcomes and pathway values
    master = load_master_full()
    Z, z_info = build_pathway_z(master, path_cols)
    O = master.set_index("pair_id")[["improved", "exact_next_response", "split", STALENESS_COL,
                                     "eligible_for_binary"]]
    master_pid = master.set_index("pair_id")["public_id"]
    for (fam, key), P in frozen_pairs.items():
        check_pair_identity(P, master_pid, f"{fam}/{key}")

    all_results, global_rows, attrition_rows, skipped = [], [], [], []
    reg_results, reg_accounting, s4_summaries, count_qa_rows = [], [], [], []

    for family in FAMILIES:
        print(f"\n[{family}] ({FAMILY_ROLE[family]})")
        with open(f"{dirs['frozen']}/matching_attrition_{family}.json", encoding="utf-8") as f:
            matching_attrition = json.load(f)
        U = select_universe(master, family)

        ori = {key: orient_pairs(frozen_pairs[(family, key)], O, family)
               for key in ["primary", "S1_p50", "S1_p90"]}
        prim = ori["primary"]
        inf_prim = prim[prim["informative"]]
        require(len(inf_prim) >= MIN_INFORMATIVE_PAIRS, f"[{family}] too few informative primary pairs")

        qa = count_qa_row(family, len(prim), len(inf_prim), matching_attrition)
        count_qa_rows.append(qa)
        print(f"  EXACT counts: pairs={qa['exact_pairs']} (audit {qa['audit_pairs']}), "
              f"informative={qa['exact_informative_pairs']} (audit {qa['audit_informative_pairs']}); "
              f"identical to audit: {qa['identical_to_audit']}")
        print(f"  count QA: {qa['explanation']}")

        keep_cols = ["pair_index", "pair_id_a", "pair_id_b", "public_id_a", "public_id_b", "distance",
                     "n_joint_features", "vt_disease_response", "line_group", "on_pi", "on_imid", "on_cd38",
                     "outcome_a", "outcome_b", "informative", "pair_id_hi", "pair_id_lo",
                     "public_id_hi", "public_id_lo", "rna_age_a", "rna_age_b", "rna_age_gap", "split_a", "split_b"]
        write_csv(prim[keep_cols], f"{dirs['assoc']}/pairs_with_outcomes_{family}_primary.csv", float_format="%.10f")
        delta_prim = delta_matrix(inf_prim, Z)
        ddf = pd.DataFrame(delta_prim, columns=path_cols)
        ddf.insert(0, "pair_index", inf_prim["pair_index"].to_numpy())
        ddf.insert(1, "pair_id_hi", inf_prim["pair_id_hi"].to_numpy())
        ddf.insert(2, "pair_id_lo", inf_prim["pair_id_lo"].to_numpy())
        write_csv(ddf, f"{dirs['assoc']}/pair_deltas_{family}_primary.csv", float_format="%.8f")
        write_csv(balance_table(inf_prim, U), f"{dirs['assoc']}/balance_informative_pairs_{family}.csv",
                  float_format="%.6f")
        attrition_rows.append(attrition_row(family, prim, U))

        def run_set(name, inf, role):
            if len(inf) < MIN_INFORMATIVE_PAIRS:
                skipped.append({"family": family, "analysis": name, "n_informative_pairs": int(len(inf)),
                                "reason": f"fewer than {MIN_INFORMATIVE_PAIRS} informative pairs"})
                return None
            seed = analysis_seed(family, name)
            res = pair_analysis(delta_matrix(inf, Z), seed, n_perm, n_boot)
            all_results.append(results_table(res, path_cols, family, name, role, seed))
            global_rows.append(global_row(res, family, name, role, seed))
            return res

        # ---- PRIMARY (GLOBAL test first in the log) ----
        res_primary = run_set("primary", inf_prim, "primary" if family == "ordinal" else "secondary")
        print(f"  primary: informative pairs={len(inf_prim)}  GLOBAL sum d_z^2 p={res_primary['global_p']:.4g} "
              f"({res_primary['p_method']}, flips={res_primary['n_flips']})")

        # ---- S1 ----
        for key in ["S1_p50", "S1_p90"]:
            run_set(key, ori[key][ori[key]["informative"]], "sensitivity")
        # ---- S2: between-member RNA-age GAP <= 365 d (frozen definition) ----
        run_set(S2_NAME, inf_prim[inf_prim["rna_age_gap"] <= S2_RNA_AGE_GAP_CAP_DAYS], "sensitivity")
        # ---- S6 ----
        for lvl in sorted(inf_prim["vt_disease_response"].unique()):
            run_set(f"S6_leave_out_{lvl}", inf_prim[inf_prim["vt_disease_response"] != lvl], "sensitivity")
        run_set("S6_line1_only", inf_prim[inf_prim["line_group"] == "1"], "sensitivity")
        run_set("S6_line2plus_only", inf_prim[inf_prim["line_group"].isin(["2", "3+"])], "sensitivity")
        # ---- S7 (descriptive only: no confirmatory claim) ----
        tv = inf_prim[(inf_prim["split_a"] != "test") & (inf_prim["split_b"] != "test")]
        te = inf_prim[(inf_prim["split_a"] == "test") & (inf_prim["split_b"] == "test")]
        run_set("S7_discovery_trainval_pairs", tv, "descriptive")
        run_set("S7_holdout_test_pairs", te, "descriptive")

        # ---- S4: randomised matchings (effects only; draw k seed = S4_SEED_BASE + k) ----
        P4 = orient_pairs(frozen_pairs[(family, "S4")], O, family)
        rows4 = []
        for draw, g in P4.groupby("draw"):
            gi = g[g["informative"]]
            if len(gi) < MIN_INFORMATIVE_PAIRS:
                continue
            d = delta_matrix(gi, Z)
            mu = d.mean(axis=0)
            sdv = np.maximum(d.std(axis=0, ddof=1), 1e-12)
            rows4.append((len(gi), mu, mu / sdv))
        if rows4:
            ns = np.array([r[0] for r in rows4])
            MU = np.vstack([r[1] for r in rows4])
            DZ = np.vstack([r[2] for r in rows4])
            psign = np.sign(res_primary["mean"])
            s4_summaries.append(pd.DataFrame({
                "family": family, "analysis": "S4_random_matchings", "pathway": path_cols,
                "n_draws": len(rows4), "seed_base": S4_SEED_BASE, "median_informative_pairs": float(np.median(ns)),
                "median_mean_delta": np.median(MU, axis=0),
                "iqr_lo_mean_delta": np.percentile(MU, 25, axis=0),
                "iqr_hi_mean_delta": np.percentile(MU, 75, axis=0),
                "median_d_z": np.median(DZ, axis=0),
                "share_draws_same_sign_as_primary": np.where(
                    psign == 0, np.nan, (np.sign(MU) == psign[None, :]).mean(axis=0))}))

        # ---- S3 / S5 regression (ALL universe rows; frozen handling rules; no fallback) ----
        if args.skip_regression:
            skipped.append({"family": family, "analysis": "S3/S5", "reason": "--skip-regression (smoke only)"})
        else:
            for name, add_p in [("S3_adjusted", False), ("S5_adjusted_plus_purity", True)]:
                print(f"  running {name} regression (50 pathways)...")
                df_reg, acc = regression_analysis(family, U, Z, path_cols, name, add_p)
                reg_results.append(df_reg)
                reg_accounting.append(acc)
                print(f"    {name}: status={acc['analysis_status']}  rows {acc['universe_rows']} -> {acc['final_rows']} "
                      f"(complete-case lost {acc['rows_lost_incomplete_covariates']}, sparse strata lost "
                      f"{acc.get('rows_lost_sparse_strata', 'NA')}, no-variation strata lost "
                      f"{acc.get('rows_lost_no_outcome_variation', 'NA')})")

    # ---- write combined outputs ----
    long_df = pd.concat(all_results, ignore_index=True)
    write_csv(long_df, f"{dirs['assoc']}/all_pathway_results_long.csv", float_format="%.8g")
    for fam in FAMILIES:
        for ana in long_df.loc[long_df["family"] == fam, "analysis"].unique():
            sub = long_df[(long_df["family"] == fam) & (long_df["analysis"] == ana)]
            write_csv(sub, f"{dirs['assoc']}/per_analysis/pathway_results_{fam}_{ana}.csv", float_format="%.8g")
    write_csv(pd.DataFrame(global_rows), f"{dirs['assoc']}/global_tests.csv", float_format="%.8g")
    write_csv(pd.DataFrame(attrition_rows), f"{dirs['assoc']}/attrition_summary.csv")
    write_csv(pd.DataFrame(count_qa_rows), f"{dirs['assoc']}/matching_count_qa.csv")
    if skipped:
        write_csv(pd.DataFrame(skipped), f"{dirs['assoc']}/skipped_analyses.csv")
    reg_df = pd.concat(reg_results, ignore_index=True) if reg_results else pd.DataFrame()
    if len(reg_df):
        write_csv(reg_df, f"{dirs['assoc']}/S3_S5_regression_results.csv", float_format="%.8g")
        write_csv(pd.DataFrame(reg_accounting), f"{dirs['assoc']}/S3_S5_row_accounting.csv")
        write_json(reg_accounting, f"{dirs['assoc']}/S3_S5_row_accounting.json")
    s4_df = pd.concat(s4_summaries, ignore_index=True) if s4_summaries else pd.DataFrame()
    if len(s4_df):
        write_csv(s4_df, f"{dirs['assoc']}/S4_randomized_matching_summary.csv", float_format="%.8g")

    # ---- robustness summary: sign agreement across S1, S2, S3 (pre-declared rule) ----
    rob_frames = []
    for fam in FAMILIES:
        prim = long_df[(long_df["family"] == fam) & (long_df["analysis"] == "primary")].set_index("pathway")
        out = prim[["n_pairs", "mean_delta", "d_z", "p_signflip", "q_bh", "p_maxT"]].copy()
        s0 = np.sign(out["mean_delta"])
        for ana, col in [("S1_p50", "sign_S1_p50"), ("S1_p90", "sign_S1_p90"), (S2_NAME, "sign_S2")]:
            sub = long_df[(long_df["family"] == fam) & (long_df["analysis"] == ana)]
            out[col] = np.sign(sub.set_index("pathway")["mean_delta"].reindex(out.index)) if len(sub) else np.nan
        if len(reg_df):
            s3 = reg_df[(reg_df["family"] == fam) & (reg_df["analysis"] == "S3_adjusted")].set_index("pathway")["estimate"]
            out["sign_S3"] = np.sign(s3.reindex(out.index))
        else:
            out["sign_S3"] = np.nan
        cons = ((out["sign_S1_p50"] == s0) & (out["sign_S1_p90"] == s0) & (out["sign_S2"] == s0)
                & (out["sign_S3"] == s0) & (s0 != 0))
        out["consistent_direction_S1_S2_S3"] = cons
        out["robust_flag_q_lt_0.05_and_consistent"] = cons & (out["q_bh"] < 0.05)
        out.insert(0, "family", fam)
        rob_frames.append(out.reset_index())
    rob = pd.concat(rob_frames, ignore_index=True)
    write_csv(rob, f"{dirs['assoc']}/robustness_summary.csv", float_format="%.8g")

    # ---- machine-readable run summary ----
    run_summary = {
        "label": ANALYSIS_LABEL,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 1),
        "smoke": bool(args.smoke),
        "frozen_combined_sha256_verified": manifest["combined_sha256"],
        "seeds": {"base": RANDOM_SEED, "n_perm": n_perm, "n_boot": n_boot, "s4_seed_base": S4_SEED_BASE,
                  "per_analysis": "42 + crc32('<family>|<analysis>') % 100000 (see seed_signflip / seed_bootstrap columns)"},
        "rna_z_scoring": z_info,
        "families": {fam: {"role": FAMILY_ROLE[fam],
                           **next(r for r in attrition_rows if r["family"] == fam)} for fam in FAMILIES},
        "exact_count_qa_vs_audit": count_qa_rows,
        "global_tests": [r for r in global_rows if r["analysis"] == "primary"],
        "n_pathways": len(path_cols),
        "s3_s5_analysis_status": [{k: a[k] for k in ("family", "analysis", "analysis_status", "final_rows",
                                                     "rows_lost_total", "failure_reasons") if k in a}
                                  for a in reg_accounting],
        "skipped_analyses": skipped,
        "covariate_sets": {"S3": S3_COVARIATE_SET, "S5": S5_COVARIATE_SET,
                           "strata_fixed_effects": EXACT_KEYS, "min_stratum_rows": S3_MIN_STRATUM_ROWS},
        "claim_gate": ("Report the GLOBAL test before any single-pathway statement; call a pathway "
                       "robust only if q<0.05 AND direction agrees in S1 (p50,p90), S2 and S3."),
        "interpretation": ANALYSIS_LABEL,
        "versions": package_versions(),
    }
    write_json(run_summary, f"{dirs['assoc']}/run_summary.json")
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs in {dirs['assoc']}")
    print(ANALYSIS_LABEL)


if __name__ == "__main__":
    main()
