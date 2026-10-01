"""
Post-hoc exploratory low-variance RNA incremental-value analysis over locked Model C.

Scientific question
-------------------
Does the 50-Hallmark RNA profile add predictive information beyond the locked
multiclass Model C when the RNA increment is estimated with ridge shrinkage
rather than the original high-variance Model D/XGBoost formulation?

This is POST-HOC and EXPLORATORY. It is not confirmatory and not external
validation. The design was frozen before this implementation:
  * train+val patients only; multiclass task only; held-out RNA-available rows.
  * locked 5x5 patient folds and locked D3 Model-C OOF probabilities.
  * Model C enters as five row-specific cumulative-logit offsets. A common
    ridge-penalized RNA shift is applied across the five cut-points.
  * 50 Hallmark pathways, no pathway selection, fold-local standardization.
  * lambda grid {1,10,100,1000,10000,100000, infinity}; nested patient CV.
  * primary estimand: mean-over-repeats delta RPS = RNA - Model C (lower better).
  * patient-cluster bootstrap: 2000 draws, seed 42.
  * whole-RNA-vector permutation null: B=200 unless the prespecified timing
    pilot projects >8 hours, in which case B=100. No interim statistical looks.
  * success requires observed delta<0, bootstrap 95% upper<0, and one-sided
    permutation p<=0.05. No variants after this analysis.

Important implementation details
--------------------------------
The primary nested comparator is the minimally floored Model-C distribution:
    p_tilde = (p + 1e-6) / (1 + 6e-6)
The raw locked Model-C comparison is descriptive; the floor must change RPS by
<1e-5.

RNA permutations operate on DISTINCT (patient, RNA-sample) units. Because an
explicit RNA sample id is not retained in the analysis table, the same frozen
unit definition as Direction 3 is used: patient id + hash of the rounded
50-pathway vector. A permuted vector is propagated to every visit row sharing
that original unit. Visit rows are NEVER permuted independently.

This script writes only under:
  artifacts/results_clean_rerun/rna_low_variance_incremental/
It never modifies existing D1-D4 outputs.

Suggested order from repo root:
  python -m explainability.rna_low_variance_incremental --gates-only
  python -m explainability.rna_low_variance_incremental --timing-pilot
  python -m explainability.rna_low_variance_incremental

The full run is checkpointed per permutation. Re-running the exact same full
command resumes completed permutations after verifying the frozen config.
"""

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, "modeling")
sys.path.insert(0, "data_pipeline")
from train_models import build_model, MULTICLASS_CLASSES as TM_CLASSES, RANDOM_SEED as TM_SEED  # noqa: E402
from repeated_cv_c_vs_d_clean_rerun import make_repeat_folds, fit_fold_preprocessor, transform_rows  # noqa: E402
from explainability.direction3_common import (  # noqa: E402
    LOCKED_DIR, MULTICLASS_CLASSES, RANDOM_SEED, STALENESS_COL,
    assert_predictors_clean, load_feature_sets, load_master, pathway_columns,
    require, rps_rows, multiclass_logloss_rows, sha256_file,
)

# ----------------------------- frozen constants -----------------------------
N_FOLDS = 5
N_REPEATS = 5
N_BOOT = 2000
N_PERM_DEFAULT = 200
N_PERM_FALLBACK = 100
BOOT_SEED = 42
PERM_SEED_BASE = 30042
PROB_EPS = 1e-6
LAMBDA_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, math.inf]
TIE_TOL = 1e-12
GRAD_REL_TOL = 1e-5
IDENTITY_TOL = 1e-6
FLOOR_RPS_TOL = 1e-5
RAW_RPS_REPRO_TOL = 1e-9
# Explicit planted-null numerical gate, frozen before real analysis.
# The gate checks probability identity at beta=0, not a noisy fitted sample effect.
PLANTED_NULL_PROB_TOL = 1e-12
TIMING_FALLBACK_HOURS = 8.0

D3_DIR = Path("artifacts/results_clean_rerun/direction3/partA_predictive")
OOF_PATH = D3_DIR / "oof_predictions" / "oof_multiclass.csv"
FOLD_PATH = D3_DIR / "oof_predictions" / "oof_fold_assignments.csv"
D3_REPEAT_METRICS = D3_DIR / "partA_repeat_level_metrics.csv"
D3_MANIFEST = D3_DIR / "output_manifest.json"
LOCKED_OOF_SHA256 = "9701df1fd3d2728f085f508daf1a472cfae263af48bf640cecfd62d600ee7716"
LOCKED_FOLD_SHA256 = "9601302e2d770eee2e833093c7c63b833b016d2060c3059b9b7b671af852dee6"
LOCKED_C_METRICS_SHA256 = "2f3b9a500ccf86bfc968ad184ce2bf8c508e2076289942e6d45e6fd5250ff285"
C_METRICS_PATH = Path(LOCKED_DIR) / "multiclass_c_xgboost_metrics.json"
OUT_DIR = Path("artifacts/results_clean_rerun/rna_low_variance_incremental")
CACHE_DIR = OUT_DIR / "cache_inner_model_c"
CHECKPOINT_DIR = OUT_DIR / "permutation_checkpoints"
CONFIG_PATH = OUT_DIR / "frozen_config.json"
GATES_PATH = OUT_DIR / "gates_report.json"
TIMING_PATH = OUT_DIR / "timing_pilot.json"
OBS_PATH = OUT_DIR / "observed_predictions.csv"
OBS_SUMMARY_PATH = OUT_DIR / "observed_summary.json"
FINAL_PATH = OUT_DIR / "run_summary.json"
MANIFEST_PATH = OUT_DIR / "output_manifest.json"


def _sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def _csv_atomic(df, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def _config():
    return {
        "analysis": "post-hoc exploratory low-variance RNA increment over locked Model C",
        "classes": list(MULTICLASS_CLASSES),
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "n_bootstrap": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "n_perm_default": N_PERM_DEFAULT,
        "n_perm_fallback": N_PERM_FALLBACK,
        "perm_seed_base": PERM_SEED_BASE,
        "prob_floor_eps": PROB_EPS,
        "lambda_grid": ["inf" if math.isinf(x) else x for x in LAMBDA_GRID],
        "tie_tol": TIE_TOL,
        "gradient_relative_tolerance": GRAD_REL_TOL,
        "identity_tolerance": IDENTITY_TOL,
        "floor_rps_tolerance": FLOOR_RPS_TOL,
        "raw_rps_reproduction_tolerance": RAW_RPS_REPRO_TOL,
        "planted_null_probability_tolerance": PLANTED_NULL_PROB_TOL,
        "timing_fallback_hours": TIMING_FALLBACK_HOURS,
        "locked_oof_sha256": LOCKED_OOF_SHA256,
        "locked_fold_sha256": LOCKED_FOLD_SHA256,
        "locked_model_c_metrics_sha256": LOCKED_C_METRICS_SHA256,
        "primary_metric": "RPS",
        "primary_estimand": "mean over 5 repeats of mean-row RPS(RNA)-RPS(floored Model C)",
        "success_rule": "delta<0 AND patient-bootstrap 95% CI upper<0 AND one-sided permutation p<=0.05",
        "permutation_unit": "whole 50-pathway vector across distinct (patient, RNA-sample) units; propagated to all rows sharing unit",
        "multiplicity": "one primary test within this analysis; broader post-hoc/project-level selection remains",
    }


def ensure_frozen_config():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _config()
    txt = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    h = _sha_text(txt)
    if CONFIG_PATH.exists():
        old = json.load(open(CONFIG_PATH, encoding="utf-8"))
        require(old.get("config_sha256") == h, "frozen config differs from existing run; refusing to resume")
    else:
        _json_dump({**cfg, "config_sha256": h}, CONFIG_PATH)
    return cfg, h


def floor_probs(P):
    P = np.asarray(P, dtype=float)
    require(P.ndim == 2 and P.shape[1] == 6, "probability matrix must be n x 6")
    require(np.isfinite(P).all() and (P >= 0).all(), "invalid Model-C probabilities")
    rs = P.sum(axis=1)
    require(np.max(np.abs(rs - 1.0)) < 1e-5, "Model-C probabilities do not sum to one")
    Q = (P + PROB_EPS) / (1.0 + 6.0 * PROB_EPS)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q


def probs_to_offsets(P):
    F = np.cumsum(P, axis=1)[:, :-1]
    require(((F > 0) & (F < 1)).all(), "cumulative probabilities not strictly inside (0,1)")
    require((np.diff(F, axis=1) > 0).all(), "cumulative probabilities not strictly increasing")
    return np.log(F) - np.log1p(-F)


def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def shifted_probs(offsets, Z, beta):
    s = np.asarray(Z, float) @ np.asarray(beta, float)
    F = sigmoid(np.asarray(offsets, float) - s[:, None])
    q = np.empty((len(s), 6), dtype=float)
    q[:, 0] = F[:, 0]
    q[:, 1:5] = F[:, 1:] - F[:, :-1]
    q[:, 5] = 1.0 - F[:, 4]
    require(np.isfinite(q).all(), "non-finite shifted probabilities")
    # Mathematically positive because the offsets are ordered; allow tiny numerical dust only.
    require(float(q.min()) > -1e-12, "shifted probabilities became materially negative")
    q = np.clip(q, 1e-15, None)
    q /= q.sum(axis=1, keepdims=True)
    return q


def nll_grad(beta, offsets, Z, y, lam):
    beta = np.asarray(beta, float)
    Z = np.asarray(Z, float)
    y = np.asarray(y, int)
    s = Z @ beta
    F = sigmoid(offsets - s[:, None])
    dF = -F * (1.0 - F)  # derivative wrt scalar shift s
    n = len(y)
    q = np.empty(n, float)
    dq = np.empty(n, float)
    m0 = y == 0
    m5 = y == 5
    mm = ~(m0 | m5)
    q[m0], dq[m0] = F[m0, 0], dF[m0, 0]
    q[m5], dq[m5] = 1.0 - F[m5, 4], -dF[m5, 4]
    idx = np.flatnonzero(mm)
    yy = y[idx]
    q[idx] = F[idx, yy] - F[idx, yy - 1]
    dq[idx] = dF[idx, yy] - dF[idx, yy - 1]
    q = np.clip(q, 1e-15, None)
    loss = -np.log(q).sum() + 0.5 * lam * float(beta @ beta)
    dlds = -(dq / q)
    grad = Z.T @ dlds + lam * beta
    return float(loss), np.asarray(grad, float)


def fit_beta(offsets, Z, y, lam, x0=None):
    p = Z.shape[1]

    if math.isinf(lam):
        return np.zeros(p, float), {
            "success": True,
            "status": 0,
            "message": "lambda=infinity; beta fixed to zero",
            "nit": 0,
            "retry_used": False,
        }

    if x0 is None:
        x0 = np.zeros(p, float)

    x0 = np.asarray(x0, float)

    # Original locked numerical fit.
    res = minimize(
        lambda b: nll_grad(b, offsets, Z, y, lam),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
            "gtol": 1e-7,
        },
    )

    retry_used = False

    # Numerical safeguard only:
    # if L-BFGS-B's default line search terminates abnormally,
    # retry the SAME objective, penalty and starting point with
    # a larger line-search allowance.
    if not res.success:
        retry_used = True
        first_message = str(res.message)

        res = minimize(
            lambda b: nll_grad(b, offsets, Z, y, lam),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": 1000,
                "ftol": 1e-12,
                "gtol": 1e-7,
                "maxls": 200,
            },
        )

        if res.success:
            print(
                f"  numerical safeguard: L-BFGS-B retry succeeded "
                f"for lambda={lam:g} after initial '{first_message}'",
                flush=True,
            )

    require(
        bool(res.success),
        f"ridge ordinal solver failed after numerical retry: {res.message}",
    )
    require(
        np.isfinite(res.x).all(),
        "ridge ordinal solver returned non-finite beta",
    )

    return np.asarray(res.x, float), {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.nit),
        "fun": float(res.fun),
        "retry_used": retry_used,
    }

def finite_difference_gradient_check():
    rng = np.random.RandomState(12345)
    n, p = 31, 7
    raw = rng.dirichlet(np.ones(6), size=n)
    off = probs_to_offsets(floor_probs(raw))
    Z = rng.normal(size=(n, p))
    y = rng.randint(0, 6, size=n)
    b = rng.normal(scale=0.05, size=p)
    lam = 10.0
    _, g = nll_grad(b, off, Z, y, lam)
    h = 1e-6
    gn = np.zeros(p)
    for j in range(p):
        bp, bm = b.copy(), b.copy()
        bp[j] += h; bm[j] -= h
        fp, _ = nll_grad(bp, off, Z, y, lam)
        fm, _ = nll_grad(bm, off, Z, y, lam)
        gn[j] = (fp - fm) / (2 * h)
    rel = float(np.linalg.norm(g - gn) / max(1.0, np.linalg.norm(g), np.linalg.norm(gn)))
    require(rel < GRAD_REL_TOL, f"analytic gradient check failed: relative error={rel}")
    return rel


def planted_effect_selfcheck():
    rng = np.random.RandomState(54321)
    n, p = 1200, 50
    base = rng.dirichlet(np.ones(6) * 2.0, size=n)
    P0 = floor_probs(base)
    off = probs_to_offsets(P0)
    Z = rng.normal(size=(n, p))
    true_b = np.zeros(p); true_b[:5] = np.array([0.35, -0.25, 0.20, 0.15, -0.10])
    P1 = shifted_probs(off, Z, true_b)
    y = np.array([rng.choice(6, p=P1[i]) for i in range(n)])
    bhat, _ = fit_beta(off, Z, y, 10.0)
    delta = float(np.mean(rps_rows(shifted_probs(off, Z, bhat), y) - rps_rows(P0, y)))
    require(delta < 0, f"planted-effect recovery failed: delta RPS={delta}")
    # Exact planted-null numerical identity: beta=0 must reproduce P0 to machine precision.
    Pnull = shifted_probs(off, Z, np.zeros(p))
    maxdiff = float(np.max(np.abs(Pnull - P0)))
    require(maxdiff < PLANTED_NULL_PROB_TOL,
            f"planted-null beta=0 identity failed: max probability diff={maxdiff}")
    return {"planted_effect_delta_rps": delta, "planted_null_max_probability_diff": maxdiff}


def load_locked_inputs():
    require(sha256_file(str(OOF_PATH)) == LOCKED_OOF_SHA256, "locked D3 multiclass OOF SHA256 mismatch")
    require(sha256_file(str(FOLD_PATH)) == LOCKED_FOLD_SHA256, "locked D3 fold-assignment SHA256 mismatch")
    require(sha256_file(str(C_METRICS_PATH)) == LOCKED_C_METRICS_SHA256, "locked Model-C metrics SHA256 mismatch")
    man = json.load(open(D3_MANIFEST, encoding="utf-8"))
    require(man.get("oof_sha256", {}).get("oof_multiclass.csv") == LOCKED_OOF_SHA256,
            "D3 manifest does not pin expected multiclass OOF")
    with open(C_METRICS_PATH, encoding="utf-8") as f:
        cparams = json.load(f)["best_params"]
    oof = pd.read_csv(OOF_PATH)
    folds = pd.read_csv(FOLD_PATH)
    folds = folds[folds["task"] == "multiclass"].copy()
    require(len(oof) > 0 and len(folds) > 0, "locked OOF/folds empty")
    return oof, folds, cparams


def prepare_master_and_units():
    fs = load_feature_sets()
    assert_predictors_clean(fs)
    paths = pathway_columns(fs)
    require(len(paths) == 50, f"expected 50 Hallmark pathways, found {len(paths)}")
    m = load_master()
    m = m[m["split"] != "test"].reset_index(drop=True)
    require((m["split"] != "test").all(), "test split entered analysis universe")
    m["_row"] = np.arange(len(m))
    r = m[STALENESS_COL].notna().to_numpy()
    require(not m.loc[r, paths].isna().any().any(), "RNA-available row has missing pathway score")
    vec = m.loc[r, paths]
    h = pd.util.hash_pandas_object(vec.round(8), index=False).to_numpy()
    pid_codes = pd.factorize(m.loc[r, "public_id"])[0]
    key = pd.DataFrame({"pid": pid_codes, "h": h})
    unit = key.groupby(["pid", "h"], sort=False).ngroup().to_numpy()
    first = ~key.duplicated().to_numpy()
    V = vec.to_numpy(float)[first]
    require(V.shape[0] == int(unit.max()) + 1, "RNA unit indexing mismatch")
    unit_full = np.full(len(m), -1, dtype=int)
    unit_full[np.flatnonzero(r)] = unit
    m["_rna_unit"] = unit_full
    return m, fs, paths, V


def validate_locked_oof(master, oof, folds):
    pcols = [f"pC_{c}" for c in MULTICLASS_CLASSES]
    require(all(c in oof.columns for c in pcols), "locked OOF missing Model-C probability columns")
    require(set(oof["repeat"].unique()) == set(range(N_REPEATS)), "locked OOF repeats drifted")
    counts = oof.groupby("repeat").size().to_dict()
    require(len(set(counts.values())) == 1, "locked OOF repeats have different row counts")
    first_ids = None
    for r in range(N_REPEATS):
        d = oof[oof["repeat"] == r].sort_values("pair_id")
        ids = d["pair_id"].to_numpy()
        if first_ids is None: first_ids = ids
        else: require(np.array_equal(ids, first_ids), "locked OOF repeats disagree on evaluated rows")
    mm = master.set_index("pair_id")
    require(set(first_ids) <= set(mm.index), "locked OOF contains pair absent from train+val master")
    require(mm.loc[first_ids, STALENESS_COL].notna().all(), "locked OOF includes non-RNA row")
    # Re-derive every outer patient fold and compare with saved assignments.
    pats = master["public_id"].unique()
    for r in range(N_REPEATS):
        derived = make_repeat_folds(pats, N_FOLDS, seed=RANDOM_SEED + r)
        sf = folds[folds["repeat"] == r]
        for f in range(N_FOLDS):
            a = set(map(str, derived[f]))
            b = set(map(str, sf.loc[sf["fold"] == f, "public_id"]))
            require(a == b, f"locked outer fold mismatch repeat={r} fold={f}")
    # OOF row fold must agree with patient fold.
    fmap = {(int(x.repeat), str(x.public_id)): int(x.fold) for x in folds.itertuples()}
    bad = [i for i, x in enumerate(oof.itertuples()) if fmap[(int(x.repeat), str(x.public_id))] != int(x.fold)]
    require(not bad, "OOF row fold disagrees with locked patient fold")
    return pcols


def model_c_reproduction_gate(oof, pcols):
    vals_raw, vals_floor = [], []
    for r in range(N_REPEATS):
        d = oof[oof["repeat"] == r].sort_values("pair_id")
        y = d["y_true"].to_numpy(int)
        P = d[pcols].to_numpy(float)
        Pf = floor_probs(P)
        vals_raw.append(float(rps_rows(P, y).mean()))
        vals_floor.append(float(rps_rows(Pf, y).mean()))
    raw_mean = float(np.mean(vals_raw)); floor_mean = float(np.mean(vals_floor))
    # Compare raw OOF recomputation to committed D3 repeat-level all/C/RPS rows.
    rep = pd.read_csv(D3_REPEAT_METRICS)
    z = rep[(rep["task"] == "multiclass") & (rep["model"] == "C") &
            (rep["group"] == "all") & (rep["metric"] == "rps")].sort_values("repeat")
    require(len(z) == N_REPEATS, "could not locate five committed D3 Model-C all-row RPS values")
    committed = z["value"].to_numpy(float)
    max_rep_diff = float(np.max(np.abs(np.array(vals_raw) - committed)))
    require(max_rep_diff < RAW_RPS_REPRO_TOL,
            f"raw Model-C RPS does not reproduce committed D3 values: max diff={max_rep_diff}")
    floor_diff = float(abs(floor_mean - raw_mean))
    require(floor_diff < FLOOR_RPS_TOL, f"probability floor changes mean RPS by {floor_diff}, >= {FLOOR_RPS_TOL}")
    # beta=0 identity on all locked OOF rows
    max_identity = 0.0
    for r in range(N_REPEATS):
        d = oof[oof["repeat"] == r]
        Pf = floor_probs(d[pcols].to_numpy(float))
        off = probs_to_offsets(Pf)
        Pz = shifted_probs(off, np.zeros((len(d), 50)), np.zeros(50))
        max_identity = max(max_identity, float(np.max(np.abs(Pz - Pf))))
    require(max_identity < IDENTITY_TOL, f"beta=0 identity gate failed: {max_identity}")
    return {"raw_rps_by_repeat": vals_raw, "floored_rps_by_repeat": vals_floor,
            "raw_rps_mean": raw_mean, "floored_rps_mean": floor_mean,
            "floor_absolute_rps_change": floor_diff,
            "max_raw_vs_committed_repeat_rps_diff": max_rep_diff,
            "max_beta0_probability_identity_diff": max_identity}


def run_gates(write=True):
    cfg, cfg_hash = ensure_frozen_config()
    require(TM_SEED == RANDOM_SEED, "train_models seed differs")
    require(list(TM_CLASSES) == list(MULTICLASS_CLASSES), "multiclass class order differs")
    oof, folds, cparams = load_locked_inputs()
    master, fs, paths, V = prepare_master_and_units()
    pcols = validate_locked_oof(master, oof, folds)
    repro = model_c_reproduction_gate(oof, pcols)
    grad_rel = finite_difference_gradient_check()
    sim = planted_effect_selfcheck()
    report = {
        "status": "PASSED",
        "config_sha256": cfg_hash,
        "locked_oof_sha256": sha256_file(str(OOF_PATH)),
        "locked_fold_sha256": sha256_file(str(FOLD_PATH)),
        "locked_model_c_metrics_sha256": sha256_file(str(C_METRICS_PATH)),
        "n_trainval_rows": int(len(master)),
        "n_trainval_patients": int(master["public_id"].nunique()),
        "n_rna_rows": int(master[STALENESS_COL].notna().sum()),
        "n_rna_patients": int(master.loc[master[STALENESS_COL].notna(), "public_id"].nunique()),
        "n_distinct_rna_units": int(V.shape[0]),
        "n_pathways": len(paths),
        "model_c_best_params": cparams,
        "gradient_relative_error": grad_rel,
        **sim,
        **repro,
    }
    if write: _json_dump(report, GATES_PATH)
    return report, master, fs, paths, V, oof, folds, cparams, pcols


def sample_stats(master, paths, unit_vectors, patient_mask):
    rows = master[patient_mask & master[STALENESS_COL].notna()]
    units = np.unique(rows["_rna_unit"].to_numpy(int))
    require(len(units) > 1, "too few RNA sample units for standardization")
    X = unit_vectors[units]
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    require(np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all(), "invalid pathway standardization")
    return mu, sd


def z_rows(master, paths, unit_vectors, row_mask, mu, sd):
    rows = master[row_mask]
    units = rows["_rna_unit"].to_numpy(int)
    require((units >= 0).all(), "non-RNA row requested for RNA transform")
    return (unit_vectors[units] - mu) / sd


def locked_eval_probs(oof, repeat, fold, pair_ids, pcols):
    d = oof[(oof["repeat"] == repeat) & (oof["fold"] == fold)].set_index("pair_id")
    require(set(pair_ids) == set(d.index), f"locked OOF eval row mismatch repeat={repeat} fold={fold}")
    d = d.loc[pair_ids]
    return d[pcols].to_numpy(float), d["y_true"].to_numpy(int)


def inner_cache_path(repeat, fold):
    return CACHE_DIR / f"inner_model_c_r{repeat}_f{fold}.npz"


def build_or_load_inner_offsets(master, fs, cparams, repeat, fold, outer_train_patients, paths):
    """Cross-fit Model C within an outer-training patient set. Cache only Model-C predictions.
    The cache is independent of RNA values and therefore shared by observed/permuted passes."""
    path = inner_cache_path(repeat, fold)
    outer_train_patients = np.asarray(outer_train_patients, dtype=object)
    outer_train_rna = master[master["public_id"].isin(outer_train_patients) & master[STALENESS_COL].notna()]
    target_pairs = np.sort(outer_train_rna["pair_id"].to_numpy())
    if path.exists():
        z = np.load(path, allow_pickle=False)
        require(str(z["config_sha256"].item()) == json.load(open(CONFIG_PATH, encoding="utf-8"))["config_sha256"],
                "inner cache config mismatch")
        require(np.array_equal(z["pair_id"], target_pairs), f"inner cache pair ids mismatch r{repeat} f{fold}")
        return z["pair_id"], z["probs"], z["inner_fold"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    inner_folds = make_repeat_folds(outer_train_patients, N_FOLDS, seed=RANDOM_SEED + 10000 + 10 * repeat + fold)
    rec = []
    model_c_cols = list(fs["model_c"])
    for j, val_pats in enumerate(inner_folds):
        tr_pats = np.setdiff1d(outer_train_patients, val_pats)
        tr_mask = master["public_id"].isin(tr_pats)
        va_mask = master["public_id"].isin(val_pats) & master[STALENESS_COL].notna()
        require(int(va_mask.sum()) > 0, f"inner fold has no RNA eval rows r{repeat} f{fold} inner{j}")
        prep, wdf, cols = fit_fold_preprocessor(master, model_c_cols, tr_mask, is_model_d=False)
        Xtr = transform_rows(prep, wdf, cols, tr_mask)
        Xva = transform_rows(prep, wdf, cols, va_mask)
        ytr = np.array([MULTICLASS_CLASSES.index(v) for v in master.loc[tr_mask, "exact_next_response"]])
        mc = build_model("xgboost", "multiclass", cparams)
        mc.fit(Xtr, ytr)
        P = mc.predict_proba(Xva)
        require(P.shape[1] == 6, "inner Model C did not return six classes")
        rec.append(pd.DataFrame({"pair_id": master.loc[va_mask, "pair_id"].to_numpy(), "inner_fold": j,
                                 **{f"p{k}": P[:, k] for k in range(6)}}))
    d = pd.concat(rec, ignore_index=True).sort_values("pair_id")
    require(np.array_equal(d["pair_id"].to_numpy(), target_pairs), f"inner cross-fit did not cover outer-training RNA rows r{repeat} f{fold}")
    probs = d[[f"p{k}" for k in range(6)]].to_numpy(float)
    inner_id = d["inner_fold"].to_numpy(int)
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, pair_id=target_pairs, probs=probs, inner_fold=inner_id,
                        config_sha256=np.array(json.load(open(CONFIG_PATH, encoding="utf-8"))["config_sha256"]))
    os.replace(tmp, path)
    return target_pairs, probs, inner_id


def choose_lambda_and_fit(master, paths, unit_vectors, pair_ids_train, P_train, inner_fold_id):
    idx = master.set_index("pair_id")
    train_rows = idx.loc[pair_ids_train]
    y = np.array([MULTICLASS_CLASSES.index(v) for v in train_rows["exact_next_response"]])
    public = train_rows["public_id"].astype(str).to_numpy()
    # Each inner fold is patient-disjoint by construction; verify.
    for j in range(N_FOLDS):
        require(set(public[inner_fold_id == j]).isdisjoint(set(public[inner_fold_id != j])), "inner patient leakage")
    scores = []
    for lam in LAMBDA_GRID:
        fold_scores = []
        for j in range(N_FOLDS):
            tr = inner_fold_id != j; va = ~tr
            tr_pats = set(public[tr])
            mask_stats = master["public_id"].astype(str).isin(tr_pats)
            mu, sd = sample_stats(master, paths, unit_vectors, mask_stats)
            units_tr = train_rows.loc[tr, "_rna_unit"].to_numpy(int)
            units_va = train_rows.loc[va, "_rna_unit"].to_numpy(int)
            Ztr = (unit_vectors[units_tr] - mu) / sd
            Zva = (unit_vectors[units_va] - mu) / sd
            Ptr = floor_probs(P_train[tr]); Pva = floor_probs(P_train[va])
            off_tr = probs_to_offsets(Ptr); off_va = probs_to_offsets(Pva)
            b, _ = fit_beta(off_tr, Ztr, y[tr], lam)
            pred = shifted_probs(off_va, Zva, b)
            fold_scores.append(float(rps_rows(pred, y[va]).mean()))
        scores.append({"lambda": lam, "mean_inner_rps": float(np.mean(fold_scores)), "fold_rps": fold_scores})
    best_val = min(x["mean_inner_rps"] for x in scores)
    eligible = [x for x in scores if x["mean_inner_rps"] <= best_val + TIE_TOL]
    # Larger lambda wins ties; infinity is largest.
    chosen = max(eligible, key=lambda x: x["lambda"])["lambda"]
    # Final standardization on distinct RNA units of ALL outer-training patients represented here.
    outer_pats = set(public)
    mask_stats = master["public_id"].astype(str).isin(outer_pats)
    mu, sd = sample_stats(master, paths, unit_vectors, mask_stats)
    units = train_rows["_rna_unit"].to_numpy(int)
    Z = (unit_vectors[units] - mu) / sd
    b, info = fit_beta(probs_to_offsets(floor_probs(P_train)), Z, y, chosen)
    return chosen, b, mu, sd, scores, info


def one_pass(master, fs, paths, unit_vectors, oof, folds, cparams, pcols, save_rows=False, progress_prefix="observed"):
    all_rows = []
    selected = []
    pats_all = master["public_id"].unique()
    for r in range(N_REPEATS):
        outer = make_repeat_folds(pats_all, N_FOLDS, seed=RANDOM_SEED + r)
        for f, eval_pats in enumerate(outer):
            t0 = time.time()
            train_pats = np.setdiff1d(pats_all, eval_pats)
            pair_tr, Ptrain, inner_id = build_or_load_inner_offsets(master, fs, cparams, r, f, train_pats, paths)
            chosen, beta, mu, sd, inner_scores, fitinfo = choose_lambda_and_fit(
                master, paths, unit_vectors, pair_tr, Ptrain, inner_id)
            eval_mask = master["public_id"].isin(eval_pats) & master[STALENESS_COL].notna()
            eval_df = master[eval_mask].sort_values("pair_id")
            pair_ev = eval_df["pair_id"].to_numpy()
            Praw, y_locked = locked_eval_probs(oof, r, f, pair_ev, pcols)
            y_master = np.array([MULTICLASS_CLASSES.index(v) for v in eval_df["exact_next_response"]])
            require(np.array_equal(y_locked, y_master), "locked OOF outcome disagrees with master")
            Pbase = floor_probs(Praw)
            Zev = (unit_vectors[eval_df["_rna_unit"].to_numpy(int)] - mu) / sd
            Prna = shifted_probs(probs_to_offsets(Pbase), Zev, beta)
            drow = rps_rows(Prna, y_master) - rps_rows(Pbase, y_master)
            drow_raw = rps_rows(Prna, y_master) - rps_rows(Praw, y_master)
            llrow = multiclass_logloss_rows(Prna, y_master) - multiclass_logloss_rows(Pbase, y_master)
            for i in range(len(eval_df)):
                all_rows.append({"repeat": r, "fold": f, "pair_id": int(pair_ev[i]),
                                 "public_id": str(eval_df.iloc[i]["public_id"]), "y_true": int(y_master[i]),
                                 "delta_rps": float(drow[i]), "delta_rps_vs_raw_model_c": float(drow_raw[i]),
                                 "delta_logloss": float(llrow[i])})
            selected.append({"repeat": r, "fold": f,
                             "lambda": "inf" if math.isinf(chosen) else float(chosen),
                             "beta_l2": float(np.linalg.norm(beta)), "fit_iterations": fitinfo["nit"],
                             "inner_scores": [{"lambda": "inf" if math.isinf(x["lambda"]) else x["lambda"],
                                               "mean_inner_rps": x["mean_inner_rps"], "fold_rps": x["fold_rps"]}
                                              for x in inner_scores]})
            lamtxt = "inf" if math.isinf(chosen) else f"{chosen:g}"
            print(f"  {progress_prefix}: repeat {r} fold {f} done; lambda={lamtxt}; {time.time()-t0:.1f}s", flush=True)
    rows = pd.DataFrame(all_rows)
    # Same evaluation rows once per repeat.
    n_per = rows.groupby("repeat").size().to_numpy()
    require(len(n_per) == N_REPEATS and len(set(n_per)) == 1, "pass has inconsistent repeat row counts")
    rep = rows.groupby("repeat").agg(delta_rps=("delta_rps", "mean"),
                                        delta_rps_vs_raw_model_c=("delta_rps_vs_raw_model_c", "mean"),
                                        delta_logloss=("delta_logloss", "mean")).reset_index()
    stat = float(rep["delta_rps"].mean())
    return stat, rows, rep, selected


def patient_bootstrap(rows):
    # Average each row's delta over repeats first, then cluster-resample patients.
    x = rows.groupby(["pair_id", "public_id"], as_index=False)["delta_rps"].mean()
    pats = np.array(sorted(x["public_id"].unique()), dtype=object)
    by = {p: x.loc[x["public_id"] == p, "delta_rps"].to_numpy(float) for p in pats}
    rng = np.random.RandomState(BOOT_SEED)
    vals = np.empty(N_BOOT, float)
    for b in range(N_BOOT):
        samp = rng.choice(pats, size=len(pats), replace=True)
        arr = np.concatenate([by[p] for p in samp])
        vals[b] = arr.mean()
    return {"n_bootstrap": N_BOOT, "seed": BOOT_SEED, "mean": float(vals.mean()),
            "ci_lower": float(np.percentile(vals, 2.5)), "ci_upper": float(np.percentile(vals, 97.5))}


def permute_unit_vectors(V, seed):
    perm = np.random.RandomState(seed).permutation(V.shape[0])
    return V[perm].copy()


def timing_pilot():
    report, master, fs, paths, V, oof, folds, cparams, pcols = run_gates(write=True)
    # Build/cache one outer fold's Model-C inner offsets, then time one observed RNA ridge fold.
    pats = master["public_id"].unique(); r = 0; f = 0
    outer = make_repeat_folds(pats, N_FOLDS, seed=RANDOM_SEED)
    train_pats = np.setdiff1d(pats, outer[f])
    t0 = time.time()
    pair_tr, Ptrain, inner_id = build_or_load_inner_offsets(master, fs, cparams, r, f, train_pats, paths)
    cache_seconds = time.time() - t0
    t1 = time.time()
    choose_lambda_and_fit(master, paths, V, pair_tr, Ptrain, inner_id)
    ridge_seconds = time.time() - t1
    # A permutation pass reuses all Model-C caches; approximate 25 folds x measured ridge fold.
    perm_pass_seconds = 25.0 * ridge_seconds
    projected_200_hours = N_PERM_DEFAULT * perm_pass_seconds / 3600.0
    chosen_B = N_PERM_FALLBACK if projected_200_hours > TIMING_FALLBACK_HOURS else N_PERM_DEFAULT
    out = {"status": "TIMING ONLY; NO PRIMARY STATISTIC COMPUTED", "cache_build_seconds_one_outer_fold": cache_seconds,
           "ridge_seconds_one_outer_fold": ridge_seconds, "projected_seconds_per_permutation": perm_pass_seconds,
           "projected_hours_B200": projected_200_hours, "fallback_threshold_hours": TIMING_FALLBACK_HOURS,
           "chosen_permutation_count": chosen_B,
           "note": "Projection is a prespecified timing heuristic from one outer fold; no outcome metric is reported."}
    _json_dump(out, TIMING_PATH)
    print(json.dumps(out, indent=2))
    return out


def checkpoint_path(p):
    return CHECKPOINT_DIR / f"perm_{p:03d}.json"


def determine_B():
    require(TIMING_PATH.exists(), "run --timing-pilot before the full analysis")
    t = json.load(open(TIMING_PATH, encoding="utf-8"))
    B = int(t["chosen_permutation_count"])
    require(B in (N_PERM_DEFAULT, N_PERM_FALLBACK), "timing pilot selected unexpected permutation count")
    return B


def full_run():
    require(not FINAL_PATH.exists() and not MANIFEST_PATH.exists(), "final outputs already exist; refusing to rerun")
    report, master, fs, paths, V, oof, folds, cparams, pcols = run_gates(write=True)
    B = determine_B()
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8")); cfg_hash = cfg["config_sha256"]

    if OBS_SUMMARY_PATH.exists() and OBS_PATH.exists():
        obs_summary = json.load(open(OBS_SUMMARY_PATH, encoding="utf-8"))
        require(obs_summary["config_sha256"] == cfg_hash, "observed checkpoint config mismatch")
        obs_rows = pd.read_csv(OBS_PATH)
        obs = float(obs_summary["delta_rps"])
        boot = obs_summary["patient_bootstrap"]
        print(f"Observed pass already complete: delta RPS={obs:.10g}; resuming permutations only.", flush=True)
    else:
        print("Running observed pass...", flush=True)
        obs, obs_rows, rep, selected = one_pass(master, fs, paths, V, oof, folds, cparams, pcols,
                                                 save_rows=True, progress_prefix="observed")
        boot = patient_bootstrap(obs_rows)
        raw_desc = float(obs_rows.groupby("repeat")["delta_logloss"].mean().mean())
        raw_c_rps_desc = float(obs_rows.groupby("repeat")["delta_rps_vs_raw_model_c"].mean().mean())
        share_inf = float(np.mean([x["lambda"] == "inf" for x in selected]))
        _csv_atomic(obs_rows, OBS_PATH, float_format="%.10g")
        obs_summary = {"config_sha256": cfg_hash, "delta_rps": obs,
                       "delta_rps_vs_raw_unfloored_model_c_descriptive": raw_c_rps_desc,
                       "delta_logloss_descriptive": raw_desc,
                       "patient_bootstrap": boot, "share_outer_folds_lambda_infinity": share_inf,
                       "selected_lambda_by_outer_fold": selected,
                       "per_repeat": rep.to_dict("records"),
                       "note": "Primary comparator uses minimally floored Model-C probabilities; raw locked comparator is descriptive."}
        _json_dump(obs_summary, OBS_SUMMARY_PATH)
        print(f"Observed pass complete and frozen: delta RPS={obs:.10g}. No decision is made until all permutations finish.", flush=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in range(B):
        cp = checkpoint_path(p)
        if cp.exists():
            d = json.load(open(cp, encoding="utf-8"))
            require(d["config_sha256"] == cfg_hash and int(d["B"]) == B and int(d["permutation"]) == p,
                    f"permutation checkpoint mismatch at {cp}")
            existing.append(p)
    if existing:
        print(f"Found {len(existing)}/{B} valid permutation checkpoints; resuming at first missing permutation.", flush=True)

    start_all = time.time()
    for p in range(B):
        cp = checkpoint_path(p)
        if cp.exists():
            continue
        seed = PERM_SEED_BASE + p
        t0 = time.time()
        Vp = permute_unit_vectors(V, seed)
        stat, _, rep, selected = one_pass(master, fs, paths, Vp, oof, folds, cparams, pcols,
                                           save_rows=False, progress_prefix=f"perm {p+1}/{B}")
        _json_dump({"config_sha256": cfg_hash, "B": B, "permutation": p, "seed": seed,
                    "delta_rps": stat, "runtime_seconds": time.time() - t0,
                    "per_repeat": rep.to_dict("records"),
                    "selected_lambda": [x["lambda"] for x in selected]}, cp)
        done = sum(checkpoint_path(j).exists() for j in range(B))
        print(f"PERMUTATION {done}/{B} checkpointed ({time.time()-t0:.1f}s). No interim p-value shown.", flush=True)

    null = np.array([json.load(open(checkpoint_path(p), encoding="utf-8"))["delta_rps"] for p in range(B)], float)
    k = int(np.sum(null <= obs))
    p_perm = float((1 + k) / (B + 1))
    success_components = {"observed_delta_negative": bool(obs < 0),
                          "bootstrap_ci_upper_below_zero": bool(float(boot["ci_upper"]) < 0),
                          "permutation_p_le_0.05": bool(p_perm <= 0.05)}
    n_ok = sum(success_components.values())
    if n_ok == 3: decision = "SUPPORTED (EXPLORATORY)"
    elif n_ok == 0: decision = "NO INCREMENTAL VALUE DETECTED"
    else: decision = "SUGGESTIVE OR INCONCLUSIVE"
    final = {"label": "POST-HOC EXPLORATORY low-variance RNA incremental-value analysis",
             "completed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "config_sha256": cfg_hash,
             "n_permutations": B, "observed_delta_rps": obs, "patient_bootstrap": boot,
             "permutation_null_mean": float(null.mean()), "permutation_null_sd": float(null.std(ddof=1)),
             "permutation_null_p2_5": float(np.percentile(null, 2.5)), "permutation_null_p97_5": float(np.percentile(null, 97.5)),
             "n_null_delta_le_observed": k, "one_sided_permutation_p": p_perm,
             "success_components": success_components, "decision": decision,
             "interpretation_if_failure": "The observed RNA association does not translate into detected incremental predictive gain under this regularized model.",
             "caveats": ["post-hoc exploratory analysis", "same project data as D1-D4 and pooled D2/D3 findings",
                         "patient bootstrap covers evaluation sampling, not model-training variability",
                         "primary comparator uses minimally floored locked Model-C probabilities"],
             "stopping_rule": "Stop after this result; no alternative penalties, feature sets, models, subgroups, freshness variants, binary task, or endpoints."}
    _json_dump(final, FINAL_PATH)
    # Compact null distribution for audit.
    _csv_atomic(pd.DataFrame({"permutation": np.arange(B), "seed": PERM_SEED_BASE + np.arange(B), "delta_rps": null}),
                OUT_DIR / "permutation_null.csv", float_format="%.10g")
    files = [CONFIG_PATH, GATES_PATH, TIMING_PATH, OBS_PATH, OBS_SUMMARY_PATH, FINAL_PATH, OUT_DIR / "permutation_null.csv"]
    hashes = {str(p.relative_to(OUT_DIR)).replace("\\", "/"): sha256_file(str(p)) for p in files}
    combined = hashlib.sha256("\n".join(f"{k}:{hashes[k]}" for k in sorted(hashes)).encode()).hexdigest()
    _json_dump({"files": hashes, "combined_sha256": combined,
                "locked_inputs": {"oof_multiclass": LOCKED_OOF_SHA256, "fold_assignments": LOCKED_FOLD_SHA256,
                                  "multiclass_c_metrics": LOCKED_C_METRICS_SHA256}}, MANIFEST_PATH)
    print(f"\nDONE. {decision}")
    print(f"Observed delta RPS: {obs:.10g}; bootstrap 95% CI [{boot['ci_lower']:.10g}, {boot['ci_upper']:.10g}]")
    print(f"Permutation p: {p_perm:.10g} ({k}/{B} null deltas <= observed)")
    print(f"Outputs: {OUT_DIR}")
    print(f"Combined SHA256: {combined}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gates-only", action="store_true")
    g.add_argument("--timing-pilot", action="store_true")
    args = ap.parse_args()
    if args.gates_only:
        report, *_ = run_gates(write=True)
        print("GATES PASSED")
        print(json.dumps(report, indent=2))
        return
    if args.timing_pilot:
        timing_pilot(); return
    full_run()


if __name__ == "__main__":
    main()
