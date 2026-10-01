"""
Post-hoc exploratory longer-term endpoint extension for the MMRF CoMMpass project.

FROZEN QUESTION (specified before any RNA-survival association is examined):
    Does pre-treatment RNA pathway information add prognostic discrimination for
    PFS beyond baseline clinical characteristics and initial treatment?

Primary cohort:
    - one patient row
    - RNA sample day <= 0 relative to first treatment/index date
    - expected n = 674
Primary endpoint:
    - official supplied MMRF PFS (censor_pfs==1 means event)
Secondary endpoint:
    - official supplied MMRF OS (censor_os==1 means death)
Comparator:
    - age_at_index + gender + ISS + exact first_line regimen
RNA model:
    - comparator + the same fixed 50 Hallmark pathway scores used in Model D
Model:
    - ridge-penalized Cox PH, implemented here with scipy (Breslow ties)
Validation:
    - 5 repeats x 5 outer folds, patient-level (one row/patient)
    - 5-fold inner CV chooses ridge lambda by Harrell C-index
Primary metric:
    - pooled out-of-fold Harrell C-index; delta = RNA - comparator (positive favors RNA)
Inference:
    - patient bootstrap (2000) on per-patient OOF predictions averaged across repeats
    - one-sided whole-RNA-vector permutation null (200), statistic = delta C-index
Secondary descriptive metric:
    - IPCW integrated Brier score over 1-3 years; delta = RNA - comparator (negative favors RNA)
Prespecified timing sensitivity:
    - RNA day in [-30, 0]

IMPORTANT:
    - This is exploratory internal validation, NOT external validation.
    - No pathway selection, subgroup search, alternate timing cutoffs, or model switching.
    - This script never reconstructs PFS/OS; it uses the official supplied endpoints.
    - Real outputs are never overwritten. Checkpoints permit safe resume.

Run from repository root:
    python -m explainability.survival_incremental_analysis --gates-only
    python -m explainability.survival_incremental_analysis --timing-pilot
    python -m explainability.survival_incremental_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from explainability.direction2_common import load_feature_sets, pathway_columns

# -----------------------------------------------------------------------------
# Frozen constants
# -----------------------------------------------------------------------------
SEED = 42
N_REPEATS = 5
N_FOLDS = 5
N_INNER = 5
N_BOOT = 2000
N_PERM = 200

# Ridge strength is on the mean negative partial log-likelihood scale.
# Includes a practically-unpenalized option (0) and a wide shrinkage range.
LAMBDA_GRID = np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=float)

PRIMARY_ENDPOINT = "pfs"
SENS_WINDOW = (-30.0, 0.0)
IBS_T0 = 365.0
IBS_T1 = 1095.0
IBS_GRID_N = 50

EXPECTED_N = 674
EXPECTED_PFS_EVENTS = 459
EXPECTED_OS_EVENTS = 247
EXPECTED_SENS_N = 624
EXPECTED_SENS_PFS_EVENTS = 423
EXPECTED_SENS_OS_EVENTS = 229
EXPECTED_PATHWAYS = 50

ROOT = Path("artifacts/results_clean_rerun/survival_incremental")
CHECKPOINT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "cache"

MASTER_PATH = Path("data/clinical/visit_pairs_with_rna.csv")
SURV_PATH = Path("data/clinical/survival_pfs_response_deid.csv")
SUBJECT_PATH = Path("data/clinical/subject_deid.csv")
DIAG_PATH = Path("data/clinical/diagnosis_deid.csv")
TX_PATH = Path("data/clinical/administered_regimen_line_deid.csv")

SPEC = {
    "label": "POST-HOC EXPLORATORY PRETREATMENT-RNA LONG-TERM PROGNOSIS EXTENSION",
    "primary_endpoint": "official supplied PFS",
    "secondary_endpoint": "official supplied OS",
    "primary_cohort": "one pre-treatment RNA unit per patient; inferred RNA day <= 0",
    "timing_sensitivity": "inferred RNA day in [-30, 0]",
    "comparator": ["age_at_index", "gender", "ISS", "exact first_line regimen"],
    "rna_increment": "fixed 50 Hallmark pathways used by existing Model D",
    "model": "ridge Cox PH; Breslow ties",
    "outer_cv": "5 repeats x 5 folds; shuffled KFold at patient level",
    "inner_cv": "5 folds; choose lambda by mean validation Harrell C-index",
    "lambda_grid": LAMBDA_GRID.tolist(),
    "preprocessing": (
        "fit within training fold only; age median imputation + missing indicator; "
        "ISS not_reported treated as missing category; gender/ISS/first_line one-hot; "
        "unknown validation categories map to all-zero; all resulting columns standardized "
        "using training-fold mean/SD before ridge fitting"
    ),
    "primary_metric": "OOF Harrell C-index; delta = RNA - comparator",
    "secondary_metric": "IPCW IBS over 365-1095 days; delta = RNA - comparator",
    "bootstrap": "2000 patient resamples; repeat-averaged OOF predictions",
    "permutation": (
        "200 one-sided whole-RNA-vector patient permutations; clinical/treatment/outcomes/timing fixed; "
        "outer-fold RNA-model lambdas are frozen to the values selected in the observed nested-CV run; "
        "success direction is null delta C-index >= observed"
    ),
    "interpretation": "exploratory internal validation; not external validation; no causal claim",
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def require(cond, msg):
    if not cond:
        raise AssertionError(f"[survival_incremental] {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def spec_sha() -> str:
    return sha256_text(json.dumps(SPEC, sort_keys=True, separators=(",", ":")))


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ["numpy", "pandas", "scipy", "scikit-learn"]:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def guard_real_outputs():
    marker = ROOT / "DONE.json"
    if marker.exists():
        raise SystemExit(f"[survival_incremental] {marker} exists; real outputs are never overwritten")


# -----------------------------------------------------------------------------
# Cohort construction
# -----------------------------------------------------------------------------
def _numeric_dx(s):
    # survival dx_id is numeric; diagnosis dx_id is e.g. mmrf_1007-mmrf_commpass-d10
    out = pd.to_numeric(s, errors="coerce")
    miss = out.isna()
    if miss.any():
        ext = s.astype(str).str.extract(r"-d(\d+)$")[0]
        out.loc[miss] = pd.to_numeric(ext.loc[miss], errors="coerce")
    return out


def build_cohort():
    fs = load_feature_sets()
    paths = pathway_columns(fs)
    require(len(paths) == EXPECTED_PATHWAYS, f"expected 50 pathways, got {len(paths)}")

    use = ["public_id", "vt_days_to_visit", "days_since_rna_sample"] + paths
    vp = pd.read_csv(MASTER_PATH, usecols=use, low_memory=False)
    r = vp.dropna(subset=["vt_days_to_visit", "days_since_rna_sample"]).copy()
    r["rna_day"] = r["vt_days_to_visit"] - r["days_since_rna_sample"]
    require((r["days_since_rna_sample"] >= 0).all(), "negative days_since_rna_sample found")

    # RNA-available rows should carry complete pathway vectors.
    require(not r[paths].isna().any().any(), "RNA-timed row has missing pathway value")

    # Collapse repeated visit reuse of the same pre-treatment sample to one patient row.
    pre = r.loc[r["rna_day"] <= 0, ["public_id", "rna_day"] + paths].copy()
    timing_n = pre.groupby("public_id")["rna_day"].nunique()
    require((timing_n == 1).all(), "a patient has >1 distinct pre-treatment RNA day")

    # Also require one unique 50-pathway vector per pre-treatment patient.
    hashes = pd.util.hash_pandas_object(pre[paths].round(8), index=False)
    pre["_rna_hash"] = hashes.astype(str)
    vec_n = pre.groupby("public_id")["_rna_hash"].nunique()
    require((vec_n == 1).all(), "a patient has >1 distinct pre-treatment RNA vector")
    pre = pre.drop_duplicates("public_id", keep="first").drop(columns="_rna_hash")
    require(len(pre) == EXPECTED_N, f"baseline RNA cohort expected {EXPECTED_N}, got {len(pre)}")

    surv = pd.read_csv(SURV_PATH, usecols=[
        "public_id", "dx_id", "survival_time_pfs", "censor_pfs", "surivival_time_os", "censor_os"
    ], low_memory=False)
    require(not surv["public_id"].duplicated().any(), "survival table has duplicate public_id")
    surv["_dx_num"] = _numeric_dx(surv["dx_id"])

    subj = pd.read_csv(SUBJECT_PATH, usecols=["public_id", "age_at_index", "gender"], low_memory=False)
    require(not subj["public_id"].duplicated().any(), "subject table has duplicate public_id")

    diag = pd.read_csv(DIAG_PATH, usecols=["public_id", "dx_id", "iss_stage"], low_memory=False)
    diag["_dx_num"] = _numeric_dx(diag["dx_id"])
    require(not diag[["public_id", "_dx_num"]].duplicated().any(), "diagnosis public_id/dx key duplicated")
    d = surv[["public_id", "_dx_num"]].merge(
        diag[["public_id", "_dx_num", "iss_stage"]], on=["public_id", "_dx_num"], how="left", validate="one_to_one"
    )

    tx = pd.read_csv(TX_PATH, usecols=["public_id", "first_line"], low_memory=False)
    tx = tx[tx["public_id"].isin(pre["public_id"])].copy()
    # Exactly one nonmissing first_line value per patient, even though it repeats across treatment rows.
    first = tx.groupby("public_id")["first_line"].agg(lambda s: sorted(set(s.dropna().astype(str))))
    require(len(first) == EXPECTED_N, "not all baseline-RNA patients occur in treatment table")
    require((first.str.len() == 1).all(), "first_line is absent or non-unique for a baseline-RNA patient")
    first = first.str[0].rename("first_line").reset_index()

    x = (pre.merge(surv.drop(columns="dx_id"), on="public_id", how="left", validate="one_to_one")
            .merge(subj, on="public_id", how="left", validate="one_to_one")
            .merge(d[["public_id", "iss_stage"]], on="public_id", how="left", validate="one_to_one")
            .merge(first, on="public_id", how="left", validate="one_to_one"))

    require(len(x) == EXPECTED_N, "cohort row count changed after merges")
    req = ["survival_time_pfs", "censor_pfs", "surivival_time_os", "censor_os", "gender", "iss_stage", "first_line"]
    require(x[req].notna().all().all(), "unexpected missing endpoint/categorical baseline field")
    require((x["survival_time_pfs"] > 0).all() and (x["surivival_time_os"] > 0).all(), "non-positive survival time")
    require(int((x["censor_pfs"] == 1).sum()) == EXPECTED_PFS_EVENTS, "PFS event count changed")
    require(int((x["censor_os"] == 1).sum()) == EXPECTED_OS_EVENTS, "OS event count changed")
    require(x["public_id"].is_unique, "cohort is not one row per patient")

    # Treat explicit not_reported as missing-category information, not as an ISS level.
    x["iss_stage"] = x["iss_stage"].astype(str).str.strip().replace({"not_reported": "__MISSING__", "nan": "__MISSING__"})
    x["gender"] = x["gender"].astype(str).str.strip().str.lower()
    x["first_line"] = x["first_line"].astype(str).str.strip()

    # Frozen sensitivity cohort.
    sens = x[(x["rna_day"] >= SENS_WINDOW[0]) & (x["rna_day"] <= SENS_WINDOW[1])].copy()
    require(len(sens) == EXPECTED_SENS_N, f"sensitivity cohort expected {EXPECTED_SENS_N}, got {len(sens)}")
    require(int((sens["censor_pfs"] == 1).sum()) == EXPECTED_SENS_PFS_EVENTS, "sensitivity PFS event count changed")
    require(int((sens["censor_os"] == 1).sum()) == EXPECTED_SENS_OS_EVENTS, "sensitivity OS event count changed")

    return x.sort_values("public_id").reset_index(drop=True), paths


# -----------------------------------------------------------------------------
# Fold-local preprocessing
# -----------------------------------------------------------------------------
@dataclass
class Preprocessor:
    age_median: float
    categories: dict
    mean: np.ndarray
    sd: np.ndarray
    feature_names: list
    include_rna: bool
    pathways: list


def _raw_design(df, age_median, categories, pathways, include_rna):
    age = pd.to_numeric(df["age_at_index"], errors="coerce")
    age_missing = age.isna().astype(float).to_numpy()[:, None]
    age_imp = age.fillna(age_median).to_numpy(float)[:, None]
    blocks = [age_imp, age_missing]
    names = ["age_at_index", "age_missing"]

    for col in ["gender", "iss_stage", "first_line"]:
        vals = df[col].astype(str).to_numpy()
        cats = categories[col]
        # Full one-hot; unseen validation category -> all zero. Cox has no intercept.
        M = np.column_stack([(vals == c).astype(float) for c in cats]) if cats else np.zeros((len(df), 0))
        blocks.append(M)
        names.extend([f"{col}={c}" for c in cats])

    if include_rna:
        Z = df[pathways].to_numpy(float)
        require(np.isfinite(Z).all(), "non-finite RNA pathway value")
        blocks.append(Z)
        names.extend(pathways)

    X = np.column_stack(blocks).astype(float)
    return X, names


def fit_preprocessor(train, pathways, include_rna):
    age = pd.to_numeric(train["age_at_index"], errors="coerce")
    med = float(age.median())
    require(np.isfinite(med), "cannot estimate age median")
    cats = {}
    for col in ["gender", "iss_stage", "first_line"]:
        cats[col] = sorted(train[col].astype(str).unique().tolist())
    X, names = _raw_design(train, med, cats, pathways, include_rna)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    # Columns absent/constant in a training fold carry no information; leave them at zero after centering.
    sd = np.where(sd > 1e-12, sd, 1.0)
    return Preprocessor(med, cats, mu, sd, names, include_rna, pathways)


def transform(df, pp):
    X, names = _raw_design(df, pp.age_median, pp.categories, pp.pathways, pp.include_rna)
    require(names == pp.feature_names, "design feature order changed")
    X = (X - pp.mean) / pp.sd
    require(np.isfinite(X).all(), "non-finite standardized design")
    return X


# -----------------------------------------------------------------------------
# Ridge Cox PH (Breslow ties)
# -----------------------------------------------------------------------------
def cox_nll_grad(beta, X, time_, event, lam):
    """Mean negative Breslow partial log-likelihood + lam/2 ||beta||^2."""
    beta = np.asarray(beta, float)
    eta = X @ beta
    # Shift for stable exponentiation; Cox PL invariant to a common shift.
    shift = float(np.max(eta)) if len(eta) else 0.0
    ee = np.exp(np.clip(eta - shift, -50, 50))

    order = np.argsort(-time_, kind="mergesort")
    t = np.asarray(time_)[order]
    e = np.asarray(event, int)[order]
    Xo = X[order]
    etao = eta[order]
    eeo = ee[order]

    cum_risk = np.cumsum(eeo)
    cum_xrisk = np.cumsum(eeo[:, None] * Xo, axis=0)

    # Breslow: all events at the same time use the risk set including everyone with time >= t.
    # In descending order, the last index of each tied time is that risk-set endpoint.
    _, first_rev = np.unique(t[::-1], return_index=True)
    last_idx = (len(t) - 1 - first_rev)
    risk_end = np.empty(len(t), dtype=int)
    # map each unique time to last index
    for j in last_idx:
        risk_end[t == t[j]] = j

    ev_idx = np.flatnonzero(e == 1)
    require(len(ev_idx) > 0, "training fold has no events")
    re = risk_end[ev_idx]
    logden = np.log(np.maximum(cum_risk[re], 1e-300)) + shift
    pll = np.sum(etao[ev_idx] - logden)
    grad_pll = np.sum(Xo[ev_idx] - cum_xrisk[re] / np.maximum(cum_risk[re, None], 1e-300), axis=0)

    n_ev = float(len(ev_idx))
    nll = -pll / n_ev + 0.5 * float(lam) * float(beta @ beta)
    grad = -grad_pll / n_ev + float(lam) * beta
    return float(nll), grad


def fit_cox(X, time_, event, lam):
    p = X.shape[1]
    b0 = np.zeros(p, dtype=float)

    def fun(b):
        return cox_nll_grad(b, X, time_, event, lam)

    res = minimize(fun, b0, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-11, "maxls": 50})
    if not res.success:
        # Numerical rescue only: same objective, penalty and start; more line-search allowance.
        res = minimize(fun, b0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-11, "maxls": 200})
    require(res.success, f"ridge Cox solver failed: {res.message}")
    require(np.isfinite(res.x).all(), "non-finite Cox coefficients")
    return res.x


def harrell_c(time_, event, risk):
    """Harrell C: higher risk predicts earlier event. Tied risks count 0.5."""
    t = np.asarray(time_, float)
    e = np.asarray(event, int)
    r = np.asarray(risk, float)
    require(len(t) == len(e) == len(r), "C-index length mismatch")
    concord = 0.0
    comparable = 0.0
    n = len(t)
    for i in range(n):
        if e[i] != 1:
            continue
        mask = t > t[i]
        if not np.any(mask):
            continue
        rr = r[mask]
        comparable += len(rr)
        concord += np.sum(r[i] > rr) + 0.5 * np.sum(r[i] == rr)
    return float(concord / comparable) if comparable > 0 else np.nan


# -----------------------------------------------------------------------------
# Survival probabilities / IPCW Brier score
# -----------------------------------------------------------------------------
def breslow_baseline(time_, event, lp):
    """Return event times and cumulative baseline hazard H0(t)."""
    t = np.asarray(time_, float)
    e = np.asarray(event, int)
    lp = np.asarray(lp, float)
    shift = float(np.max(lp))
    rr = np.exp(np.clip(lp - shift, -50, 50))
    event_times = np.sort(np.unique(t[e == 1]))
    h = []
    for u in event_times:
        d = np.sum((t == u) & (e == 1))
        denom = np.sum(rr[t >= u])
        h.append(d / max(denom, 1e-300) * math.exp(-shift))
    return event_times, np.cumsum(np.asarray(h, float))


def step_eval(times, vals, q, left=0.0):
    idx = np.searchsorted(times, q, side="right") - 1
    out = np.full(np.shape(q), left, dtype=float)
    ok = idx >= 0
    out[ok] = vals[idx[ok]]
    return out


def km_censoring(train_time, train_event):
    """KM of censoring distribution G(t): censoring is the event here."""
    t = np.asarray(train_time, float)
    censor = 1 - np.asarray(train_event, int)
    uniq = np.sort(np.unique(t))
    surv = []
    s = 1.0
    for u in uniq:
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & (censor == 1))
        if at_risk > 0:
            s *= (1.0 - d / at_risk)
        surv.append(s)
    return uniq, np.asarray(surv, float)


def km_g_at(km_t, km_s, q, left_limit=False):
    q = np.asarray(q, float)
    if left_limit:
        # G(t-) excludes censoring jumps exactly at t.
        idx = np.searchsorted(km_t, q, side="left") - 1
    else:
        idx = np.searchsorted(km_t, q, side="right") - 1
    out = np.ones_like(q, dtype=float)
    ok = idx >= 0
    out[ok] = km_s[idx[ok]]
    return out


def predict_survival(lp_test, base_t, base_H, grid):
    H = step_eval(base_t, base_H, np.asarray(grid, float), left=0.0)
    return np.exp(-np.exp(np.clip(lp_test, -50, 50))[:, None] * H[None, :])


def ipcw_brier_rows(test_time, test_event, surv_pred, grid, km_t, km_s):
    """Return n x T weighted squared-error contributions; NaN where G unsupported."""
    t = np.asarray(test_time, float)
    e = np.asarray(test_event, int)
    grid = np.asarray(grid, float)
    n, m = surv_pred.shape
    out = np.full((n, m), np.nan, float)
    G_tminus = km_g_at(km_t, km_s, t, left_limit=True)
    for j, u in enumerate(grid):
        G_u = float(km_g_at(km_t, km_s, np.array([u]), left_limit=False)[0])
        if G_u <= 1e-6:
            continue
        y = (t > u).astype(float)
        w = np.zeros(n, float)
        # event observed by u
        a = (t <= u) & (e == 1) & (G_tminus > 1e-6)
        w[a] = 1.0 / G_tminus[a]
        # still at risk after u
        b = t > u
        w[b] = 1.0 / G_u
        out[:, j] = w * (y - surv_pred[:, j]) ** 2
    return out


# -----------------------------------------------------------------------------
# CV machinery
# -----------------------------------------------------------------------------
def endpoint_arrays(df, endpoint):
    if endpoint == "pfs":
        return df["survival_time_pfs"].to_numpy(float), df["censor_pfs"].to_numpy(int)
    if endpoint == "os":
        return df["surivival_time_os"].to_numpy(float), df["censor_os"].to_numpy(int)
    raise ValueError(endpoint)


def make_outer_folds(n, repeat):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
    fold = np.empty(n, int)
    for f, (_, va) in enumerate(kf.split(np.arange(n))):
        fold[va] = f
    return fold


def make_inner_folds(n, repeat, outer_fold):
    kf = KFold(n_splits=N_INNER, shuffle=True, random_state=SEED + 10000 + 10 * repeat + outer_fold)
    return list(kf.split(np.arange(n)))


def choose_lambda(train_df, pathways, include_rna, endpoint, repeat, outer_fold):
    t_all, e_all = endpoint_arrays(train_df, endpoint)
    scores = {float(l): [] for l in LAMBDA_GRID}
    for tr_idx, va_idx in make_inner_folds(len(train_df), repeat, outer_fold):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        pp = fit_preprocessor(tr, pathways, include_rna)
        Xtr = transform(tr, pp)
        Xva = transform(va, pp)
        ttr, etr = endpoint_arrays(tr, endpoint)
        tva, eva = endpoint_arrays(va, endpoint)
        for lam in LAMBDA_GRID:
            b = fit_cox(Xtr, ttr, etr, float(lam))
            c = harrell_c(tva, eva, Xva @ b)
            require(np.isfinite(c), "non-finite inner C-index")
            scores[float(lam)].append(c)
    means = {l: float(np.mean(v)) for l, v in scores.items()}
    best_val = max(means.values())
    # deterministic tie rule: strongest regularization within numerical tie.
    tied = [l for l, v in means.items() if abs(v - best_val) <= 1e-12]
    best = max(tied)
    return best, means


def fit_outer(train_df, test_df, pathways, include_rna, endpoint, repeat, outer_fold):
    lam, inner = choose_lambda(train_df, pathways, include_rna, endpoint, repeat, outer_fold)
    pp = fit_preprocessor(train_df, pathways, include_rna)
    Xtr = transform(train_df, pp)
    Xte = transform(test_df, pp)
    ttr, etr = endpoint_arrays(train_df, endpoint)
    beta = fit_cox(Xtr, ttr, etr, lam)
    lp_tr = Xtr @ beta
    lp_te = Xte @ beta
    bt, bh = breslow_baseline(ttr, etr, lp_tr)
    km_t, km_s = km_censoring(ttr, etr)
    return lp_te, bt, bh, km_t, km_s, lam, inner


def run_oof(df, pathways, endpoint, label, save_detail=True):
    """Observed OOF run for comparator and RNA model."""
    rows = []
    tune_rows = []
    grid = np.linspace(IBS_T0, IBS_T1, IBS_GRID_N)
    for rep in range(N_REPEATS):
        fold = make_outer_folds(len(df), rep)
        for f in range(N_FOLDS):
            te_idx = np.flatnonzero(fold == f)
            tr_idx = np.flatnonzero(fold != f)
            tr = df.iloc[tr_idx]
            te = df.iloc[te_idx]
            tte, ete = endpoint_arrays(te, endpoint)

            out = {}
            for model_name, inc_rna in [("C0", False), ("C0_RNA", True)]:
                start = time.time()
                lp, bt, bh, km_t, km_s, lam, inner = fit_outer(
                    tr, te, pathways, inc_rna, endpoint, rep, f
                )
                S = predict_survival(lp, bt, bh, grid)
                br = ipcw_brier_rows(tte, ete, S, grid, km_t, km_s)
                out[model_name] = (lp, br)
                tune_rows.append({
                    "analysis": label, "endpoint": endpoint, "repeat": rep, "fold": f,
                    "model": model_name, "lambda": lam,
                    "inner_scores_json": json.dumps(inner, sort_keys=True),
                    "seconds": time.time() - start,
                })
                print(f"[{label} {endpoint}] repeat {rep+1}/{N_REPEATS} fold {f+1}/{N_FOLDS} "
                      f"{model_name} lambda={lam:g}", flush=True)

            for j, idx in enumerate(te_idx):
                rec = {
                    "analysis": label, "endpoint": endpoint, "repeat": rep, "fold": f,
                    "public_id": df.iloc[idx]["public_id"], "time": tte[j], "event": ete[j],
                    "risk_C0": out["C0"][0][j], "risk_C0_RNA": out["C0_RNA"][0][j],
                }
                # Store per-patient integrated Brier contribution; mean over supported grid.
                rec["ibsrow_C0"] = float(np.nanmean(out["C0"][1][j]))
                rec["ibsrow_C0_RNA"] = float(np.nanmean(out["C0_RNA"][1][j]))
                rows.append(rec)

    pred = pd.DataFrame(rows)
    tune = pd.DataFrame(tune_rows)
    if save_detail:
        write_csv(pred, ROOT / f"{label}_{endpoint}_oof_predictions.csv")
        write_csv(tune, ROOT / f"{label}_{endpoint}_tuning.csv")
    return pred, tune


def summarize_oof(pred):
    by_rep = []
    for rep, d in pred.groupby("repeat"):
        c0 = harrell_c(d["time"], d["event"], d["risk_C0"])
        cr = harrell_c(d["time"], d["event"], d["risk_C0_RNA"])
        ib0 = float(d["ibsrow_C0"].mean())
        ibr = float(d["ibsrow_C0_RNA"].mean())
        by_rep.append({"repeat": int(rep), "c_C0": c0, "c_C0_RNA": cr, "delta_c": cr-c0,
                       "ibs_C0": ib0, "ibs_C0_RNA": ibr, "delta_ibs": ibr-ib0})
    by_rep = pd.DataFrame(by_rep)

    # Primary point estimate: average repeat-specific metric difference.
    summary = {
        "c_C0": float(by_rep["c_C0"].mean()),
        "c_C0_RNA": float(by_rep["c_C0_RNA"].mean()),
        "delta_c": float(by_rep["delta_c"].mean()),
        "ibs_C0": float(by_rep["ibs_C0"].mean()),
        "ibs_C0_RNA": float(by_rep["ibs_C0_RNA"].mean()),
        "delta_ibs": float(by_rep["delta_ibs"].mean()),
    }
    return summary, by_rep


def bootstrap_delta_c(pred, B=N_BOOT, seed=SEED):
    # Resample patients; retain all five repeat predictions for each sampled patient.
    ids = pred["public_id"].drop_duplicates().to_numpy()
    by_id = {pid: pred[pred["public_id"] == pid] for pid in ids}
    rng = np.random.RandomState(seed)
    vals = []
    for b in range(B):
        samp = rng.choice(ids, size=len(ids), replace=True)
        # Duplicate patient clusters need unique bootstrap identities; metrics only need concatenated rows.
        chunks = [by_id[pid] for pid in samp]
        z = pd.concat(chunks, ignore_index=True)
        ds = []
        for _, d in z.groupby("repeat"):
            c0 = harrell_c(d["time"], d["event"], d["risk_C0"])
            cr = harrell_c(d["time"], d["event"], d["risk_C0_RNA"])
            ds.append(cr - c0)
        vals.append(float(np.mean(ds)))
    vals = np.asarray(vals)
    return {
        "n_boot": int(B), "seed": int(seed),
        "ci_lower": float(np.quantile(vals, 0.025)),
        "ci_upper": float(np.quantile(vals, 0.975)),
        "bootstrap_mean": float(vals.mean()),
    }


# -----------------------------------------------------------------------------
# Negative-control permutation
# -----------------------------------------------------------------------------
def permute_rna(df, pathways, seed):
    rng = np.random.RandomState(seed)
    out = df.copy()
    V = out[pathways].to_numpy(copy=True)
    out.loc[:, pathways] = V[rng.permutation(len(out))]
    return out


def fit_outer_fixed_lambda(train_df, test_df, pathways, endpoint, lam):
    """Fit the RNA model with a pre-frozen ridge lambda (used only for permutation nulls)."""
    pp = fit_preprocessor(train_df, pathways, True)
    Xtr = transform(train_df, pp)
    Xte = transform(test_df, pp)
    ttr, etr = endpoint_arrays(train_df, endpoint)
    beta = fit_cox(Xtr, ttr, etr, float(lam))
    return Xte @ beta


def permutation_stat(df_perm, pathways, endpoint, observed_c0_pred, lambda_map):
    """
    Refit only the RNA model under scrambled RNA. Comparator OOF risks and the
    observed-run outer-fold ridge lambdas are frozen. This mirrors a negative
    control of the already-selected architecture without re-tuning to each null.
    """
    rows = []
    c0_map = observed_c0_pred.set_index(["repeat", "public_id"])["risk_C0"]
    for rep in range(N_REPEATS):
        fold = make_outer_folds(len(df_perm), rep)
        for f in range(N_FOLDS):
            te_idx = np.flatnonzero(fold == f)
            tr_idx = np.flatnonzero(fold != f)
            tr = df_perm.iloc[tr_idx]
            te = df_perm.iloc[te_idx]
            lam = float(lambda_map[(rep, f)])
            lp = fit_outer_fixed_lambda(tr, te, pathways, endpoint, lam)
            tte, ete = endpoint_arrays(te, endpoint)
            for j, idx in enumerate(te_idx):
                pid = df_perm.iloc[idx]["public_id"]
                rows.append({"repeat": rep, "public_id": pid, "time": tte[j], "event": ete[j],
                             "risk_C0": float(c0_map.loc[(rep, pid)]), "risk_C0_RNA": lp[j]})
    z = pd.DataFrame(rows)
    ds = []
    for _, d in z.groupby("repeat"):
        ds.append(harrell_c(d["time"], d["event"], d["risk_C0_RNA"]) -
                  harrell_c(d["time"], d["event"], d["risk_C0"]))
    return float(np.mean(ds))


def run_permutations(df, pathways, endpoint, observed_pred, observed_tune, observed_delta):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Freeze the RNA-model lambda chosen by the observed nested CV for each outer fold.
    rt = observed_tune[observed_tune["model"] == "C0_RNA"].copy()
    require(len(rt) == N_REPEATS * N_FOLDS, "observed RNA tuning table has wrong fold count")
    lambda_map = {(int(r.repeat), int(r.fold)): float(r.lambda_) for r in
                  rt.rename(columns={"lambda": "lambda_"}).itertuples(index=False)}

    vals = []
    for p in range(N_PERM):
        ck = CHECKPOINT_DIR / f"perm_{endpoint}_{p:03d}.json"
        if ck.exists():
            with open(ck, encoding="utf-8") as f:
                obj = json.load(f)
            require(obj["spec_sha256"] == spec_sha(), f"checkpoint {p} belongs to a different spec")
            vals.append(float(obj["delta_c"]))
            continue
        dperm = permute_rna(df, pathways, SEED + 20000 + p)
        start = time.time()
        stat = permutation_stat(dperm, pathways, endpoint, observed_pred, lambda_map)
        atomic_json({"perm": p, "delta_c": stat, "spec_sha256": spec_sha(), "seconds": time.time()-start}, ck)
        vals.append(stat)
        print(f"[permutation] {p+1}/{N_PERM} checkpointed", flush=True)
    vals = np.asarray(vals, float)
    # More positive delta is more favorable to RNA.
    k = int(np.sum(vals >= observed_delta))
    pval = (1 + k) / (N_PERM + 1)
    return {"B": N_PERM, "k_null_ge_observed": k, "one_sided_p": float(pval),
            "null_mean": float(vals.mean()), "null_sd": float(vals.std(ddof=1))}


# -----------------------------------------------------------------------------
# Gates / self-checks
# -----------------------------------------------------------------------------
def finite_difference_gradient_check():
    rng = np.random.RandomState(7)
    n, p = 40, 6
    X = rng.normal(size=(n, p))
    t = rng.randint(1, 20, size=n).astype(float)
    e = (rng.rand(n) < 0.6).astype(int)
    e[0] = 1
    b = rng.normal(scale=0.1, size=p)
    lam = 0.3
    f, g = cox_nll_grad(b, X, t, e, lam)
    gn = np.zeros(p)
    eps = 1e-6
    for j in range(p):
        bp, bm = b.copy(), b.copy()
        bp[j] += eps; bm[j] -= eps
        fp = cox_nll_grad(bp, X, t, e, lam)[0]
        fm = cox_nll_grad(bm, X, t, e, lam)[0]
        gn[j] = (fp-fm)/(2*eps)
    rel = np.linalg.norm(g-gn) / max(1e-12, np.linalg.norm(g)+np.linalg.norm(gn))
    require(rel < 1e-5, f"Cox gradient self-check failed: rel={rel}")
    return float(rel)


def metric_selfcheck():
    # Perfect ordering should give C=1; reverse should give 0.
    t = np.array([1., 2., 3., 4.])
    e = np.array([1, 1, 1, 0])
    require(abs(harrell_c(t, e, np.array([4.,3.,2.,1.])) - 1.0) < 1e-12, "C-index perfect-order check failed")
    require(abs(harrell_c(t, e, np.array([1.,2.,3.,4.])) - 0.0) < 1e-12, "C-index reverse-order check failed")


def run_gates():
    print("=== SURVIVAL INCREMENTAL GATES ===")
    print("Spec SHA256:", spec_sha())
    x, paths = build_cohort()
    print("Primary cohort:", len(x), "patients")
    print("PFS events:", int(x["censor_pfs"].sum()))
    print("OS deaths:", int(x["censor_os"].sum()))
    print("Pathways:", len(paths))
    sens = x[(x["rna_day"] >= SENS_WINDOW[0]) & (x["rna_day"] <= SENS_WINDOW[1])]
    print("Sensitivity cohort:", len(sens))
    print("Sensitivity PFS events:", int(sens["censor_pfs"].sum()))
    print("Sensitivity OS deaths:", int(sens["censor_os"].sum()))
    rel = finite_difference_gradient_check()
    metric_selfcheck()
    print("Cox gradient relative error:", rel)

    # Small solver sanity check only; not a scientific result.
    rng = np.random.RandomState(9)
    X = rng.normal(size=(80, 8))
    tt = rng.exponential(10, size=80) + 1
    ee = (rng.rand(80) < 0.65).astype(int)
    b = fit_cox(X, tt, ee, 0.1)
    require(np.isfinite(b).all(), "solver sanity check failed")
    print("Solver sanity check: PASS")
    print("ALL GATES PASSED")
    return x, paths


def timing_pilot(df, paths):
    # One outer fold. No scientific performance is printed or saved.
    fold = make_outer_folds(len(df), 0)
    tr = df.iloc[np.flatnonzero(fold != 0)]
    te = df.iloc[np.flatnonzero(fold == 0)]

    start = time.time()
    observed_fit = fit_outer(tr, te, paths, True, "pfs", 0, 0)
    nested_sec = time.time() - start
    lam = float(observed_fit[5])

    start = time.time()
    _ = fit_outer_fixed_lambda(tr, te, paths, "pfs", lam)
    fixed_sec = time.time() - start

    # Main PFS + OS + sensitivity each fit two observed models over 25 outer folds.
    projected_observed = nested_sec * N_FOLDS * N_REPEATS * 2 * 3
    # Permutation is PFS-primary only and uses frozen observed-run lambdas: one fit/fold.
    projected_perm = fixed_sec * N_FOLDS * N_REPEATS * N_PERM
    obj = {"one_nested_outer_rna_fold_seconds": nested_sec,
           "one_fixed_lambda_outer_rna_fold_seconds": fixed_sec,
           "rough_all_observed_analyses_seconds": projected_observed,
           "rough_primary_permutation_seconds": projected_perm,
           "rough_total_hours": (projected_observed + projected_perm)/3600,
           "spec_sha256": spec_sha()}
    atomic_json(obj, ROOT / "timing_pilot.json")
    print(json.dumps(obj, indent=2))


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def endpoint_analysis(df, paths, endpoint, label, do_permutation=False):
    pred, tune = run_oof(df, paths, endpoint, label)
    summary, by_rep = summarize_oof(pred)
    write_csv(by_rep, ROOT / f"{label}_{endpoint}_repeat_metrics.csv")
    boot = bootstrap_delta_c(pred, N_BOOT, SEED + (0 if endpoint == "pfs" else 1000))
    out = {"analysis": label, "endpoint": endpoint, **summary, "bootstrap_delta_c": boot}
    if do_permutation:
        out["permutation_delta_c"] = run_permutations(df, paths, endpoint, pred, tune, summary["delta_c"])
    atomic_json(out, ROOT / f"{label}_{endpoint}_summary.json")
    return out


def manifest(files):
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in files if p.exists()}
    combo = sha256_text("\n".join(f"{k}:{hashes[k]}" for k in sorted(hashes)))
    return {"spec": SPEC, "spec_sha256": spec_sha(), "versions": versions(), "files": hashes,
            "combined_sha256": combo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--timing-pilot", action="store_true")
    args = ap.parse_args()

    x, paths = run_gates()
    if args.gates_only:
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    if args.timing_pilot:
        timing_pilot(x, paths)
        return

    guard_real_outputs()
    require((ROOT / "timing_pilot.json").exists(), "run --timing-pilot before the full analysis")

    # Freeze spec on disk before any outcome/RNA model result is printed.
    spec_path = ROOT / "analysis_spec.json"
    if spec_path.exists():
        with open(spec_path, encoding="utf-8") as f:
            old = json.load(f)
        require(old.get("spec_sha256") == spec_sha(), "analysis_spec.json does not match current frozen spec")
    else:
        atomic_json({"spec": SPEC, "spec_sha256": spec_sha()}, spec_path)

    print("\n=== PRIMARY PFS ===", flush=True)
    pfs = endpoint_analysis(x, paths, "pfs", "primary", do_permutation=True)

    print("\n=== SECONDARY OS ===", flush=True)
    osres = endpoint_analysis(x, paths, "os", "secondary", do_permutation=False)

    print("\n=== PRESPECIFIED -30 TO 0 DAY PFS SENSITIVITY ===", flush=True)
    sens = x[(x["rna_day"] >= SENS_WINDOW[0]) & (x["rna_day"] <= SENS_WINDOW[1])].copy().reset_index(drop=True)
    sensres = endpoint_analysis(sens, paths, "pfs", "sensitivity_m30_0", do_permutation=False)

    final = {"primary_pfs": pfs, "secondary_os": osres, "sensitivity_pfs": sensres,
             "interpretation": SPEC["interpretation"], "spec_sha256": spec_sha()}
    atomic_json(final, ROOT / "final_summary.json")

    files = [p for p in ROOT.rglob("*") if p.is_file() and "checkpoints" not in p.parts and p.name != "DONE.json"]
    man = manifest(files)
    atomic_json(man, ROOT / "output_manifest.json")
    atomic_json({"status": "complete", "spec_sha256": spec_sha(), "combined_sha256": man["combined_sha256"]}, ROOT / "DONE.json")

    print("\n=== DONE ===")
    print("Primary PFS delta C-index:", pfs["delta_c"])
    print("Primary PFS bootstrap 95% CI:", pfs["bootstrap_delta_c"]["ci_lower"], pfs["bootstrap_delta_c"]["ci_upper"])
    print("Primary PFS permutation p:", pfs["permutation_delta_c"]["one_sided_p"])
    print("Secondary OS delta C-index:", osres["delta_c"])
    print("Sensitivity PFS delta C-index:", sensres["delta_c"])
    print("Output:", ROOT)
    print("Combined SHA256:", man["combined_sha256"])


if __name__ == "__main__":
    main()
