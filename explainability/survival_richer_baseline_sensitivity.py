"""
THE ONE FINAL richer-baseline survival sensitivity (post-hoc, exploratory, internal validation only).

QUESTION: does the locked pre-treatment-RNA incremental PFS/OS signal remain after the comparator is expanded with richer
baseline clinical-risk variables?

DESIGN = the locked survival analysis (explainability/survival_incremental_analysis.py, imported, never modified or
rerun) with ONE change: the comparator C0 is expanded. Same 674-patient pre-treatment RNA cohort (L.build_cohort), same
official PFS (primary) and OS (secondary), same C0 vs C0+RNA contrast (same 50 Hallmark pathways, same raw scores),
same 5 x 5 patient-level outer CV and seeds, same fold-local preprocessing and standardization of every column, same ridge-Cox
solver, lambda grid, 5-fold inner CV and tie rule. The only additions are the prespecified baseline columns below,
handled with the SAME pattern the locked design uses for age (fold-local median imputation + missing indicator;
everything standardized with training-fold mean/SD).

PRESPECIFIED ADDED BASELINE VARIABLES (fixed here, before any model is fitted; nothing is added or dropped on results):
  numeric (baseline = most recent harmonized value with specimen day <= 0 relative to first treatment, i.e. the same index
  date and axis as the pre-treatment RNA rule; no look-back limit; ties on the latest day -> median; negative or
  non-finite values -> missing):
    LDH         labs_deid  lab_test='serum_ldh'        (u/l)    log1p
    B2M         labs_deid  lab_test='b2m'              (mg/l)   log1p
    albumin     labs_deid  lab_test='serum_albumin'    (g/dl)   identity
    creatinine  labs_deid  lab_test='serum_creatinine' (mg/dl)  log1p
    hemoglobin  labs_deid  lab_test='hemoglobin'       (g/dl)   identity
    calcium     labs_deid  lab_test='serum_calcium'    (mg/dl)  identity
    BM plasma-cell %  sample_features_deid (+ sample_deid dates)  pct_plasma_cells_bm (flow-preferred, morphology fallback;
                valid range 0-100)  identity; baseline = most recent sample day <= 0
    (units are harmonized by the repo's own data_pipeline/build_clinical_features.py, reused unchanged)
  There is NO categorical addition: the comparator keeps the existing ISS (locked) and gains only the seven numeric
  variables above.
  ASSESSED BUT EXCLUDED: R-ISS stage (diagnosis_deid.csv r_iss_stage). Reason (fixed before any fit): it is a composite
  measure that overlaps with ISS and LDH (which are already in the comparator) and has substantial missingness (in the
  diagnosis table about 57% of rows are 'not_reported' or blank). It is never attached to the analysis frame; only its
  cohort coverage is recorded descriptively in variable_construction_audit.csv.
  EXCLUDED, recorded as UNAVAILABLE, not searched for elsewhere: cytogenetic/FISH flags (del(17p), t(4;14), 1q
  gain/amplification). No FISH/cytogenetic field exists in any table the pipeline ingests (diagnosis_deid, subject_deid
  have none; the pipeline never reads one). This script scans the raw table headers / lab-test names under data/clinical and
  RECORDS any keyword hit, but never uses an undefined field. Cytogenetic adjustment is therefore NOT possible.

REPRODUCTION GATE (before any new fit; any failure STOPS):
  G1  locked spec SHA256 fe0792db...db6e4 and needed locked files/manifest hashes (reused from
      explainability/survival_pathway_interpretation.py); cohort rebuilt with the locked asserts (n=674, 459 PFS events,
      247 OS deaths, 50 pathways).
  G2  this script's preprocessing/lambda-selection/fit machinery, run WITH NO added variables, reproduces the locked
      tuning table lambdas (exactly) and the locked OOF risks (|diff| < 1e-9) for repeat 0, all 5 outer folds, both models,
      both endpoints. This proves the added code is a pure superset of the locked design.

OUTPUT: PFS (primary) and OS (secondary) delta C-index = C(richer C0 + RNA) - C(richer C0), averaged over the 5 repeats,
with the locked patient bootstrap (2000 draws; seed 42 for PFS, 1042 for OS). Descriptive comparison with the locked
(thin-comparator) results: C(richer C0) vs C(locked C0), delta C richer vs locked, and the ratio. NO permutation test,
NO hypothesis test, NO pathway analysis, NO subgroup analysis, NO alternative model/penalty/definition.
Reading rule fixed in advance: report the delta C and its CI; if the CI still excludes 0 the signal persists after adjusting
for these variables, otherwise report the attenuation; cytogenetic adjustment was not possible.

Outputs only under artifacts/results_clean_rerun/survival_richer_baseline_sensitivity/ . Existing outputs are never
overwritten; the locked directory is only READ.

Run from the repository root:
    python -m explainability.survival_richer_baseline_sensitivity --gate-only
    python -m explainability.survival_richer_baseline_sensitivity
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import hashlib
import subprocess
import types

# The lab-harmonization and flow-cytometry sample-feature code (LAB_UNIT_CONVERSIONS / load_labs /
# load_sample_features) exists ONLY on the branches 'data-pipeline-clinical-freeze' and 'modeling-country-robustness'
# (identical blobs on both), not on the working branch of this analysis. It is loaded VERBATIM from the working tree if present,
# otherwise from git, and is accepted only if its SHA256 equals the pinned value. Nothing is copied into the repo.
_PIPELINE_SHA256 = {
    "build_clinical_features": "50aa67042edad921b1d7ab97594b6157f85fc3f26eefefe5fcb2cbe351ddeea8",
    "build_flow_cytometry_features": "f87ccc262e488cb45e3bf45a4638c66b25c7094a3dda2d1c83a3e4c8e854632f",
}
_PIPELINE_REFS = ["data-pipeline-clinical-freeze", "origin/data-pipeline-clinical-freeze",
                  "modeling-country-robustness", "origin/modeling-country-robustness"]


def _load_pipeline_module(name: str):
    rel = f"data_pipeline/{name}.py"
    candidates = []
    if Path(rel).exists():
        candidates.append((f"working tree: {rel}", Path(rel).read_bytes()))
    for ref in _PIPELINE_REFS:
        r = subprocess.run(["git", "show", f"{ref}:{rel}"], capture_output=True)
        if r.returncode == 0:
            candidates.append((f"git {ref}:{rel}", r.stdout))
    for where, src in candidates:
        if hashlib.sha256(src).hexdigest() == _PIPELINE_SHA256[name]:
            mod = types.ModuleType(name)
            mod.__file__ = where
            exec(compile(src.decode("utf-8"), where, "exec"), mod.__dict__)
            return mod
    raise SystemExit(f"[richer_baseline] could not obtain {rel} with SHA256 {_PIPELINE_SHA256[name]} from the working "
                     f"tree or from git refs {_PIPELINE_REFS} (tried: {[w for w, _ in candidates] or 'none'}). "
                     f"Fetch the branch 'modeling-country-robustness' (or 'data-pipeline-clinical-freeze') and rerun.")


BCF = _load_pipeline_module("build_clinical_features")          # repo module: lab harmonization, reused unchanged
BFC = _load_pipeline_module("build_flow_cytometry_features")    # repo module: flow/plasma-cell %, reused unchanged

from explainability import survival_incremental_analysis as L                       # noqa: E402
from explainability.survival_pathway_interpretation import (                         # noqa: E402
    EXPECTED_SPEC_SHA, Gate, now_utc, verify_locked_files, write_json,
)

LABEL = ("RICHER-BASELINE SURVIVAL SENSITIVITY (post-hoc, exploratory; internal validation only; "
         "not external validation; no causal claim)")
OUT = Path("artifacts/results_clean_rerun/survival_richer_baseline_sensitivity")
LOCKED = L.ROOT
RISK_TOL = 1e-9
ENDPOINTS = [("pfs", "primary"), ("os", "secondary")]
CLINICAL_DIR = Path("data/clinical")

# ---- prespecified added variables -------------------------------------------------------------
LAB_VARS = {            # variable -> (labs_deid lab_test, transform, unit after harmonization)
    "ldh": ("serum_ldh", "log1p", "u/l"),
    "b2m": ("b2m", "log1p", "mg/l"),
    "albumin": ("serum_albumin", "identity", "g/dl"),
    "creatinine": ("serum_creatinine", "log1p", "mg/dl"),
    "hemoglobin": ("hemoglobin", "identity", "g/dl"),
    "calcium": ("serum_calcium", "identity", "mg/dl"),
}
PLASMA_VAR = "plasma_cell_pct_bm"
EXTRA_NUMERIC = [f"{v}_base" for v in LAB_VARS] + [PLASMA_VAR]      # column names in the analysis frame
EXTRA_CAT: list = []                                                 # no categorical additions (R-ISS assessed but excluded)
RISS_EXCLUSION_REASON = ("assessed but EXCLUDED before any fit: composite measure that overlaps with ISS and LDH (already in "
                         "the comparator) and has substantial missingness (about 57% 'not_reported'/blank in the diagnosis "
                         "table)")
CYTO_PATTERN = re.compile(r"fish|cytogen|transloc|del_?17|17p|t_?4_?14|4;14|14;16|1q|amp1q|gain", re.I)

PRESPEC = {
    "label": LABEL,
    "cohort": "locked: one pre-treatment RNA unit per patient, rna_day <= 0, n=674 (L.build_cohort)",
    "endpoints": {"primary": "official PFS", "secondary": "official OS"},
    "comparator_locked": ["age_at_index (+missing indicator)", "gender", "ISS", "exact first_line regimen"],
    "comparator_richer_additions": {
        "numeric": {k: {"source": f"data/clinical/labs_deid.csv lab_test={v[0]}", "unit": v[2], "transform": v[1]}
                    for k, v in LAB_VARS.items()} | {
            "plasma_cell_pct_bm": {"source": "sample_features_deid.csv (+ sample_deid.csv dates) pct_plasma_cells_bm",
                                   "unit": "percent (0-100)", "transform": "identity"}},
        "numeric_baseline_rule": "most recent harmonized value with day <= 0 relative to first treatment; no look-back "
                                 "limit; same-day ties -> median; negative/non-finite -> missing",
        "numeric_missing_handling": "fold-local median imputation + missing indicator (the locked age pattern)",
        "categorical": "none (the comparator keeps the locked ISS and gains no categorical variable)",
        "standardization": "all resulting columns standardized with the training-fold mean/SD (locked rule)",
    },
    "excluded": {
        "R-ISS": {"status": "assessed but excluded", "source": "data/clinical/diagnosis_deid.csv r_iss_stage",
                  "reason": RISS_EXCLUSION_REASON,
                  "handling": "never attached to the analysis frame; cohort coverage recorded descriptively only"},
        "cytogenetic_FISH_flags": "UNAVAILABLE: no del(17p), t(4;14) or 1q gain/amplification field exists in any "
                                  "ingested table; not searched for elsewhere"},
    "model": "locked ridge Cox PH (Breslow), lambda grid, 5-fold inner CV by Harrell C, tie -> largest lambda",
    "outer_cv": "locked: 5 repeats x 5 folds, KFold(shuffle, seed 42 + repeat) on the sorted patient index",
    "metric": "OOF Harrell C per repeat, averaged; delta = richer C0+RNA - richer C0",
    "bootstrap": "locked: 2000 patient resamples; seed 42 (PFS), 1042 (OS)",
    "not_done": ["permutation test", "hypothesis tests", "pathway analysis", "subgroup analysis", "alternative models/penalties"],
    "locked_spec_sha256": EXPECTED_SPEC_SHA,
}


# ---------------------------------------------------------------------------
# Baseline variable construction (existing data only)
# ---------------------------------------------------------------------------
def _latest_value(long_df, id_col, day_col, val_col, ids):
    """Most recent value with day <= 0 per patient (ties on the latest day -> median). Returns (value, day, n_ties)."""
    s = long_df[long_df[id_col].isin(ids) & (long_df[day_col] <= 0)].copy()
    s = s[np.isfinite(s[val_col])]
    last = s.groupby(id_col)[day_col].transform("max")
    s = s[s[day_col] == last]
    g = s.groupby(id_col)
    return g[val_col].median(), g[day_col].max(), g.size()


def build_baseline_variables(df):
    """Attach the prespecified baseline columns to the locked cohort frame; return (frame, audit rows)."""
    ids = set(df["public_id"])
    audit = []
    out = df.copy()

    labs = BCF.load_labs()                                      # repo code: numeric, dropna, unit harmonization
    labs["days_to_specimen_collection"] = pd.to_numeric(labs["days_to_specimen_collection"], errors="coerce")
    for var, (lab_test, transform, unit) in LAB_VARS.items():
        sub = labs[labs["lab_test"] == lab_test][["public_id", "days_to_specimen_collection", "lab_test_result"]]
        val, day, ties = _latest_value(sub, "public_id", "days_to_specimen_collection", "lab_test_result", ids)
        n_neg = int((val < 0).sum())
        val = val.where(val >= 0)                               # negative values are invalid -> missing
        col = f"{var}_base"
        x = out["public_id"].map(val)
        out[col] = np.log1p(x) if transform == "log1p" else x
        audit.append({"variable": col, "source": f"labs_deid.csv lab_test={lab_test}", "unit": unit, "transform": transform,
                      "n_cohort": len(out), "n_observed": int(out[col].notna().sum()),
                      "pct_observed": float(out[col].notna().mean()), "n_negative_set_missing": n_neg,
                      "median_day_of_value": float(day.reindex(out["public_id"]).median()),
                      "min_day": float(day.min()), "n_same_day_tie_patients": int((ties > 1).sum()),
                      "decision": "included", "reason": ""})

    sf = BFC.load_sample_features(BFC.build_visit_day_lookup())       # repo code: flow cytometry, dated samples
    sf = sf.rename(columns={"days_to_sample_procurement": "sday"})
    sf["pct_plasma_cells_bm"] = pd.to_numeric(sf["pct_plasma_cells_bm"], errors="coerce")
    val, day, ties = _latest_value(sf[["public_id", "sday", "pct_plasma_cells_bm"]], "public_id", "sday",
                                   "pct_plasma_cells_bm", ids)
    n_bad = int(((val < 0) | (val > 100)).sum())
    val = val.where((val >= 0) & (val <= 100))
    out[PLASMA_VAR] = out["public_id"].map(val)
    audit.append({"variable": PLASMA_VAR, "source": "sample_features_deid.csv pct_plasma_cells_bm (dated via sample_deid.csv)",
                  "unit": "percent", "transform": "identity", "n_cohort": len(out),
                  "n_observed": int(out[PLASMA_VAR].notna().sum()), "pct_observed": float(out[PLASMA_VAR].notna().mean()),
                  "n_negative_set_missing": n_bad, "median_day_of_value": float(day.reindex(out["public_id"]).median()),
                  "min_day": float(day.min()), "n_same_day_tie_patients": int((ties > 1).sum()),
                  "decision": "included", "reason": ""})

    # R-ISS: ASSESSED BUT EXCLUDED (see RISS_EXCLUSION_REASON). Same join as the locked ISS, but the result is kept in a
    # SEPARATE frame used only to record cohort coverage; it is never attached to `out` and so can never enter a model.
    surv = pd.read_csv(L.SURV_PATH, usecols=["public_id", "dx_id"], low_memory=False)
    surv["_dx_num"] = L._numeric_dx(surv["dx_id"])
    diag = pd.read_csv(L.DIAG_PATH, usecols=["public_id", "dx_id", "r_iss_stage"], low_memory=False)
    diag["_dx_num"] = L._numeric_dx(diag["dx_id"])
    d = surv[["public_id", "_dx_num"]].merge(diag[["public_id", "_dx_num", "r_iss_stage"]], on=["public_id", "_dx_num"],
                                             how="left", validate="one_to_one")
    riss = (out[["public_id"]].merge(d[["public_id", "r_iss_stage"]], on="public_id", how="left", validate="one_to_one")
            ["r_iss_stage"].astype(str).str.strip())
    n_obs = int((~riss.isin(["not_reported", "nan", ""])).sum())
    audit.append({"variable": "r_iss_stage", "source": "diagnosis_deid.csv r_iss_stage", "unit": "categorical",
                  "transform": "not used", "n_cohort": len(out), "n_observed": n_obs,
                  "pct_observed": n_obs / len(out), "n_negative_set_missing": 0, "median_day_of_value": np.nan,
                  "min_day": np.nan, "n_same_day_tie_patients": 0, "decision": "EXCLUDED (assessed)",
                  "reason": RISS_EXCLUSION_REASON})
    if "r_iss_stage" in out.columns:                                               # guard: R-ISS must never enter a model
        raise AssertionError("[richer_baseline] r_iss_stage must not be present in the analysis frame")

    if not ((len(out) == len(df)) and out["public_id"].is_unique):                 # cohort integrity
        raise AssertionError("[richer_baseline] cohort changed while attaching baseline variables")
    return out.sort_values("public_id").reset_index(drop=True), audit


def cytogenetics_scan():
    """Record whether any FISH/cytogenetic-like field exists in the raw tables; never used if found."""
    hits, scanned = [], []
    for p in sorted(CLINICAL_DIR.glob("*_deid.csv")):
        cols = list(pd.read_csv(p, nrows=0).columns)
        scanned.append(p.name)
        for c in cols:
            if CYTO_PATTERN.search(c):
                hits.append({"table": p.name, "kind": "column", "name": c})
    labs_p = CLINICAL_DIR / "labs_deid.csv"
    if labs_p.exists():
        names = pd.read_csv(labs_p, usecols=["lab_test"])["lab_test"].dropna().astype(str).unique()
        hits += [{"table": "labs_deid.csv", "kind": "lab_test", "name": n} for n in names if CYTO_PATTERN.search(n)]
    return {"tables_scanned": scanned, "keyword_pattern": CYTO_PATTERN.pattern, "hits": hits,
            "conclusion": ("no FISH/cytogenetic field found: UNAVAILABLE (not searched for elsewhere)" if not hits else
                           "keyword hits recorded but NOT used: no validated, prespecified definition (del17p, t(4;14), 1q "
                           "gain/amplification) can be built from them without guessing"),
            "used_in_model": False}


# ---------------------------------------------------------------------------
# Extended preprocessing (locked design + prespecified extras); with no extras it is IDENTICAL to the locked one
# ---------------------------------------------------------------------------
@dataclass
class RichPP:
    age_median: float
    categories: dict
    num_medians: dict
    mean: np.ndarray
    sd: np.ndarray
    names: list
    include_rna: bool
    pathways: list
    extra_numeric: list
    extra_cat: list


def _design(df, age_median, categories, num_medians, pathways, include_rna, extra_numeric, extra_cat):
    base_cats = {k: categories[k] for k in ["gender", "iss_stage", "first_line"]}
    Xb, names = L._raw_design(df, age_median, base_cats, pathways, False)           # locked base block
    blocks, names = [Xb], list(names)
    for v in extra_numeric:                                                        # locked age pattern
        s = pd.to_numeric(df[v], errors="coerce")
        blocks += [s.fillna(num_medians[v]).to_numpy(float)[:, None], s.isna().astype(float).to_numpy()[:, None]]
        names += [v, f"{v}_missing"]
    for v in extra_cat:                                                            # locked ISS pattern
        vals, cats = df[v].astype(str).to_numpy(), categories[v]
        blocks.append(np.column_stack([(vals == c).astype(float) for c in cats]) if cats else np.zeros((len(df), 0)))
        names += [f"{v}={c}" for c in cats]
    if include_rna:                                                                # locked RNA block, appended last
        Z = df[pathways].to_numpy(float)
        if not np.isfinite(Z).all():
            raise AssertionError("[richer_baseline] non-finite RNA pathway value")
        blocks.append(Z)
        names += list(pathways)
    return np.column_stack(blocks).astype(float), names


def fit_pp(train, pathways, include_rna, extra_numeric, extra_cat):
    med = float(pd.to_numeric(train["age_at_index"], errors="coerce").median())
    cats = {c: sorted(train[c].astype(str).unique().tolist()) for c in ["gender", "iss_stage", "first_line"] + list(extra_cat)}
    meds = {}
    for v in extra_numeric:
        m = float(pd.to_numeric(train[v], errors="coerce").median())
        if not np.isfinite(m):
            raise AssertionError(f"[richer_baseline] {v} is entirely missing in a training fold")
        meds[v] = m
    X, names = _design(train, med, cats, meds, pathways, include_rna, extra_numeric, extra_cat)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)                                              # locked rule
    return RichPP(med, cats, meds, X.mean(axis=0), sd, names, include_rna, pathways, list(extra_numeric), list(extra_cat))


def transform_pp(df, pp):
    X, names = _design(df, pp.age_median, pp.categories, pp.num_medians, pp.pathways, pp.include_rna,
                       pp.extra_numeric, pp.extra_cat)
    if names != pp.names:
        raise AssertionError("[richer_baseline] design feature order changed")
    X = (X - pp.mean) / pp.sd
    if not np.isfinite(X).all():
        raise AssertionError("[richer_baseline] non-finite standardized design")
    return X


def choose_lambda_rich(train_df, pathways, include_rna, endpoint, repeat, outer_fold, extra_numeric, extra_cat):
    """Locked selection procedure (L.choose_lambda) with the extended preprocessor."""
    scores = {float(l): [] for l in L.LAMBDA_GRID}
    for tr_idx, va_idx in L.make_inner_folds(len(train_df), repeat, outer_fold):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        pp = fit_pp(tr, pathways, include_rna, extra_numeric, extra_cat)
        Xtr, Xva = transform_pp(tr, pp), transform_pp(va, pp)
        ttr, etr = L.endpoint_arrays(tr, endpoint)
        tva, eva = L.endpoint_arrays(va, endpoint)
        for lam in L.LAMBDA_GRID:
            b = L.fit_cox(Xtr, ttr, etr, float(lam))
            c = L.harrell_c(tva, eva, Xva @ b)
            if not np.isfinite(c):
                raise AssertionError("[richer_baseline] non-finite inner C-index")
            scores[float(lam)].append(c)
    means = {l: float(np.mean(v)) for l, v in scores.items()}
    best_val = max(means.values())
    tied = [l for l, v in means.items() if abs(v - best_val) <= 1e-12]
    return max(tied), means                                                         # locked tie rule: strongest ridge


def fit_outer_rich(train_df, test_df, pathways, include_rna, endpoint, repeat, outer_fold, extra_numeric, extra_cat):
    lam, inner = choose_lambda_rich(train_df, pathways, include_rna, endpoint, repeat, outer_fold, extra_numeric, extra_cat)
    pp = fit_pp(train_df, pathways, include_rna, extra_numeric, extra_cat)
    Xtr, Xte = transform_pp(train_df, pp), transform_pp(test_df, pp)
    ttr, etr = L.endpoint_arrays(train_df, endpoint)
    beta = L.fit_cox(Xtr, ttr, etr, lam)
    return Xte @ beta, lam, inner


# ---------------------------------------------------------------------------
# G2: machinery reproduction (no added variables) against the locked tuning/OOF tables
# ---------------------------------------------------------------------------
def gate_machinery(g, df, paths):
    for endpoint, label in ENDPOINTS:
        tune = pd.read_csv(LOCKED / f"{label}_{endpoint}_tuning.csv")
        oof = pd.read_csv(LOCKED / f"{label}_{endpoint}_oof_predictions.csv").set_index(["repeat", "public_id"])
        lam_map = {(int(r.repeat), int(r.fold), r.model): float(r.lambda_)
                   for r in tune.rename(columns={"lambda": "lambda_"}).itertuples(index=False)}
        fold = L.make_outer_folds(len(df), 0)
        worst, lam_ok = 0.0, True
        for f in range(L.N_FOLDS):
            te_idx, tr_idx = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            for model_name, inc in [("C0", False), ("C0_RNA", True)]:
                lp, lam, _ = fit_outer_rich(tr, te, paths, inc, endpoint, 0, f, [], [])     # NO added variables
                lam_ok &= (lam == lam_map[(0, f, model_name)])
                col = "risk_C0_RNA" if inc else "risk_C0"
                ref = oof.loc[[(0, p) for p in te["public_id"]], col].to_numpy(float)
                worst = max(worst, float(np.max(np.abs(lp - ref))))
            print(f"    [{endpoint}] gate fold {f + 1}/{L.N_FOLDS} (no added variables)", flush=True)
        g.check(f"[{endpoint}] machinery (no added variables) reproduces the locked lambdas, repeat 0, all folds, both models",
                bool(lam_ok))
        g.check(f"[{endpoint}] machinery reproduces the locked OOF risks (max |diff| < {RISK_TOL:g})", worst < RISK_TOL,
                f"max={worst:.3e}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_oof_rich(df, paths, endpoint, label):
    rows, tune_rows = [], []
    for rep in range(L.N_REPEATS):
        fold = L.make_outer_folds(len(df), rep)
        for f in range(L.N_FOLDS):
            te_idx, tr_idx = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            tte, ete = L.endpoint_arrays(te, endpoint)
            out = {}
            for model_name, inc in [("C0_richer", False), ("C0_richer_RNA", True)]:
                start = time.time()
                lp, lam, inner = fit_outer_rich(tr, te, paths, inc, endpoint, rep, f, EXTRA_NUMERIC, EXTRA_CAT)
                out[model_name] = lp
                tune_rows.append({"analysis": label, "endpoint": endpoint, "repeat": rep, "fold": f, "model": model_name,
                                  "lambda": lam, "inner_scores_json": json.dumps(inner, sort_keys=True),
                                  "seconds": time.time() - start})
                print(f"[{label} {endpoint}] repeat {rep + 1}/{L.N_REPEATS} fold {f + 1}/{L.N_FOLDS} {model_name} "
                      f"lambda={lam:g}", flush=True)
            for j, idx in enumerate(te_idx):
                rows.append({"analysis": label, "endpoint": endpoint, "repeat": rep, "fold": f,
                             "public_id": df.iloc[idx]["public_id"], "time": tte[j], "event": ete[j],
                             "risk_C0": out["C0_richer"][j], "risk_C0_RNA": out["C0_richer_RNA"][j]})
    return pd.DataFrame(rows), pd.DataFrame(tune_rows)


def summarize_c(pred):
    by_rep = []
    for rep, d in pred.groupby("repeat"):
        c0 = L.harrell_c(d["time"], d["event"], d["risk_C0"])
        cr = L.harrell_c(d["time"], d["event"], d["risk_C0_RNA"])
        by_rep.append({"repeat": int(rep), "c_C0_richer": c0, "c_C0_richer_RNA": cr, "delta_c": cr - c0})
    br = pd.DataFrame(by_rep)
    return {"c_C0_richer": float(br["c_C0_richer"].mean()), "c_C0_richer_RNA": float(br["c_C0_richer_RNA"].mean()),
            "delta_c": float(br["delta_c"].mean())}, br


def endpoint_result(df, paths, endpoint, label):
    pred, tune = run_oof_rich(df, paths, endpoint, label)
    summary, by_rep = summarize_c(pred)
    boot = L.bootstrap_delta_c(pred, L.N_BOOT, L.SEED + (0 if endpoint == "pfs" else 1000))      # locked bootstrap + seeds
    pred.to_csv(OUT / f"{label}_{endpoint}_richer_oof_predictions.csv", index=False)
    tune.to_csv(OUT / f"{label}_{endpoint}_richer_tuning.csv", index=False)
    by_rep.to_csv(OUT / f"{label}_{endpoint}_richer_repeat_metrics.csv", index=False)
    with open(LOCKED / f"{label}_{endpoint}_summary.json", encoding="utf-8") as f:
        locked = json.load(f)
    comp = {"locked_c_C0": locked["c_C0"], "locked_c_C0_RNA": locked["c_C0_RNA"], "locked_delta_c": locked["delta_c"],
            "locked_bootstrap_ci": [locked["bootstrap_delta_c"]["ci_lower"], locked["bootstrap_delta_c"]["ci_upper"]],
            "richer_minus_locked_c_C0": summary["c_C0_richer"] - locked["c_C0"],
            "delta_c_richer_minus_locked": summary["delta_c"] - locked["delta_c"],
            "delta_c_richer_over_locked": summary["delta_c"] / locked["delta_c"] if locked["delta_c"] else None,
            "note": "descriptive comparison only"}
    out = {"analysis": f"{label}_richer_baseline", "endpoint": endpoint, **summary, "bootstrap_delta_c": boot,
           "comparison_with_locked": comp}
    write_json(out, OUT / f"{label}_{endpoint}_richer_summary.json")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(LABEL)
    if (OUT / "DONE.json").exists() and not args.gate_only:
        raise SystemExit("[richer_baseline] DONE.json exists: existing outputs are never overwritten.")
    if OUT.resolve() == LOCKED.resolve():
        raise SystemExit("output directory must differ from the locked survival directory")

    g = Gate()
    print("\n=== G1: locked spec, manifest, files, cohort ===")
    man = verify_locked_files(g)
    df = paths = None
    try:
        df, paths = L.build_cohort()
        g.check("cohort rebuilt with the locked asserts (n=674, 459 PFS events, 247 OS deaths, 50 pathways)",
                len(df) == L.EXPECTED_N and len(paths) == L.EXPECTED_PATHWAYS, f"n={len(df)}")
    except Exception as e:
        g.check("cohort rebuilt with the locked asserts", False, f"{type(e).__name__}: {e}")
    if man is not None and df is not None and g.passed:
        print("\n=== G2: machinery reproduces the locked design (no added variables) ===")
        try:
            gate_machinery(g, df, paths)
        except Exception as e:
            g.check("machinery reproduction", False, f"{type(e).__name__}: {e}")
    record = {"label": LABEL, "created_utc": now_utc(), "passed": g.passed, "checks": g.rows,
              "tolerance_risk": RISK_TOL, "note": "locked analysis only read"}
    if not g.passed:
        target = OUT / ("gate_only" if args.gate_only else "") / f"gate_FAILED_{now_utc()}.json"
        write_json(record, target)
        print(f"\nGATE FAILED ({len(g.failures())} checks). Nothing was fitted. Report: {target}")
        sys.exit(1)
    if args.gate_only:
        target = OUT / "gate_only" / f"gate_{now_utc()}.json"
        write_json(record, target)
        print(f"\nGATES PASSED (gate-only). No richer-baseline model fitted. Report: {target}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(PRESPEC, OUT / "prespecification.json")                    # frozen BEFORE any richer-baseline fit
    write_json(record, OUT / "gate_report.json")
    df, audit = build_baseline_variables(df)
    pd.DataFrame(audit).to_csv(OUT / "variable_construction_audit.csv", index=False)
    cyto = cytogenetics_scan()
    write_json(cyto, OUT / "cytogenetics_availability.json")
    print("\nVariable construction (observed / cohort):")
    for a in audit:
        print(f"  {a['variable']:<22s} {a['n_observed']}/{a['n_cohort']}  ({a['pct_observed']:.1%})")
    print("Cytogenetics:", cyto["conclusion"])
    print("PRESPECIFIED SET (fixed): " + ", ".join(EXTRA_NUMERIC + EXTRA_CAT))

    results = {}
    for endpoint, label in ENDPOINTS:
        print(f"\n=== {endpoint.upper()} ({label}) richer baseline ===", flush=True)
        results[endpoint] = endpoint_result(df, paths, endpoint, label)

    final = {"label": LABEL, "interpretation": "post-hoc exploratory internal validation; not external; no causal claim; "
             "cytogenetic adjustment not possible (no FISH fields available)",
             "primary_pfs_richer": results["pfs"], "secondary_os_richer": results["os"],
             "prespecified_variables": EXTRA_NUMERIC + EXTRA_CAT, "excluded": PRESPEC["excluded"],
             "locked_spec_sha256": EXPECTED_SPEC_SHA}
    write_json(final, OUT / "final_summary.json")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "DONE.json" and "gate_only" not in p.parts]
    hashes = {str(p.relative_to(OUT)).replace("\\", "/"): L.sha256_file(p) for p in files}
    combined = L.sha256_text("\n".join(f"{k}:{hashes[k]}" for k in sorted(hashes)))
    write_json({"label": LABEL, "files": hashes, "combined_sha256": combined, "locked_spec_sha256": EXPECTED_SPEC_SHA,
                "this_script_sha256": L.sha256_file(Path(__file__)), "versions": L.versions()},
               OUT / "output_manifest.json")
    write_json({"status": "complete", "combined_sha256": combined}, OUT / "DONE.json")

    print("\n=== DONE ===")
    for ep in ("pfs", "os"):
        r = results[ep]
        c = r["comparison_with_locked"]
        print(f"{ep.upper()}: delta C (richer comparator) = {r['delta_c']:+.4f} "
              f"(95% CI {r['bootstrap_delta_c']['ci_lower']:+.4f} to {r['bootstrap_delta_c']['ci_upper']:+.4f}); "
              f"locked delta C = {c['locked_delta_c']:+.4f}; richer C0 C-index {r['c_C0_richer']:.4f} vs locked C0 {c['locked_c_C0']:.4f}")
    print(f"Output: {OUT}\nCombined SHA256: {combined}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()