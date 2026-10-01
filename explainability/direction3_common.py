"""
Direction 3 (Prof. Song) -- treatment-specific molecular response patterns.
SHARED definitions, support rule, leakage assertions, metrics and utilities.

Scripts (run from the repo root, in this order):
  python -m explainability.direction3_support      (step 0: support-rule table; fails if roles violated)
  python -m explainability.direction3_predictive   (Part A: PREDICTIVE C-vs-D within treatment context)
  python -m explainability.direction3_molecular    (Part B: RETROSPECTIVE molecular association/interaction)

INTERPRETATION (applies to every output):
  * RNA = baseline / as-of-sample tumour biology: the latest RNA sample collected at or before Vt
    (data_pipeline/build_rna_features.py). It is NOT current-state RNA and NOT causal.
  * Treatment = drug classes active AT Vt (currently_on_*), not treatment received between Vt and Vt+1.
  * Part A is predictive (out-of-fold, Vt-side inputs only). Part B is retrospective association
    (outcome = Vt+1) and supports no predictive claim.

Reuse: generic helpers come from explainability/direction2_common.py (imported, never modified).
Part A reuses the repo's frozen CV machinery by import:
  modeling/train_models.py::build_model
  modeling/repeated_cv_c_vs_d_clean_rerun.py::make_repeat_folds, fit_fold_preprocessor, transform_rows
and the LOCKED XGBoost parameters in artifacts/results_clean_rerun/{task}_{c,d}_xgboost_metrics.json.
Nothing here retrains a locked model or writes outside artifacts/results_clean_rerun/direction3/.
"""

import json
import os

import numpy as np
import pandas as pd

from explainability.direction2_common import (
    ELIGIBILITY_PATH, EXPECTED, FEATURE_SETS_PATH, MASTER_PATH, RESPONSE_RANK, SPLIT_PATH, STALENESS_COL,
    bh_qvalues, combined_hash, derive_line_group, expect, load_feature_sets, package_versions,
    pathway_columns, require, sha256_file, sha256_text, to_bool, transform_series, write_csv, write_json,
)

# ---------------------------------------------------------------------------
OUT_ROOT = "artifacts/results_clean_rerun/direction3"
LOCKED_DIR = "artifacts/results_clean_rerun"

LABEL_A = "PREDICTIVE INCREMENTAL-VALUE ANALYSIS (out-of-fold; Vt-side inputs only)"
LABEL_B = "RETROSPECTIVE MOLECULAR ASSOCIATION / INTERACTION - NOT PREDICTIVE VALIDATION"
RNA_INTERPRETATION = ("RNA = baseline / as-of-sample tumour biology (latest sample at or before Vt), "
                      "NOT current-state RNA and NOT causal. Treatment = class active AT Vt.")

RANDOM_SEED = 42
N_FOLDS = 5
N_REPEATS = 5
N_BOOT = 2000
N_PERM_NULL = 20             # Part A negative control: 20 RNA permutations, ONE 5-fold repeat (repeat 0)
N_PERM_GLOBAL = 1000         # Part B global whole-RNA-vector permutation test (parameter chosen here)
GLOBAL_PERM_SEED = RANDOM_SEED + 5000

# currently-on treatment flags (Vt-side Model-C columns built by data_pipeline/build_drug_exposure_features.py)
TREATMENT_COLS = {"pi": "currently_on_pi", "imid": "currently_on_imid", "steroid": "currently_on_steroid",
                  "cd38": "currently_on_cd38", "chemo": "currently_on_chemo",
                  "slamf7": "currently_on_slamf7", "bcma": "currently_on_bcma"}
PRIOR_EXPOSURE_COLS = {"pi": "prior_exposure_pi", "imid": "prior_exposure_imid", "steroid": "prior_exposure_steroid",
                       "cd38": "prior_exposure_cd38", "chemo": "prior_exposure_chemo",
                       "slamf7": "prior_exposure_slamf7", "bcma": "prior_exposure_bcma"}
PRIMARY_TREATMENTS = ["pi", "imid", "steroid"]
EXPLORATORY_TREATMENTS = ["cd38", "chemo"]
UNSUPPORTED_TREATMENTS = ["slamf7", "bcma"]
TREATMENT_ROLE = {**{t: "primary" for t in PRIMARY_TREATMENTS},
                  **{t: "exploratory" for t in EXPLORATORY_TREATMENTS},
                  **{t: "unsupported" for t in UNSUPPORTED_TREATMENTS}}
DECLARED_TIER = {"primary": 1, "exploratory": 2, "unsupported": 3}

# support rule (pre-specified BEFORE any RNA result was examined)
TIER1 = {"pts": 150, "events": 150, "non_events": 150}
TIER2 = {"pts": 80, "events": 75, "non_events": 75}
MULTICLASS_MIN_PER_CLASS = 20

MULTICLASS_CLASSES = list(RESPONSE_RANK)        # PD, SD, PR, VGPR, CR, sCR (same order as train_models)

# Part B frozen covariates (all Vt-side Model-C columns)
CONT_COVARIATES = {"m_protein_current": "log1p", "ldh_current": "log1p", "hemoglobin_current": "identity",
                   "albumin_current": "identity", "creatinine_current": "log1p", "b2m_current": "log1p",
                   "iss_stage": "identity", "age_at_diagnosis": "identity"}
PLASMA_COL = "pct_plasma_cells_bm"


# ---------------------------------------------------------------------------
# Directories / overwrite protection / manifests
# ---------------------------------------------------------------------------
def get_dirs3(smoke=False):
    root = f"{OUT_ROOT}/_smoke" if smoke else OUT_ROOT
    return {"root": root, "support": f"{root}/support", "partA": f"{root}/partA_predictive",
            "partB": f"{root}/partB_molecular"}


def guard_no_overwrite(marker_path, smoke, force):
    """Never overwrite existing results. --force is only permitted together with --smoke."""
    if force:
        require(smoke, "--force is only allowed together with --smoke (real outputs are never overwritten)")
    if os.path.exists(marker_path) and not force:
        raise SystemExit(f"[direction3] {marker_path} already exists: existing outputs are never overwritten. "
                         f"Delete the folder deliberately to rerun (or use --smoke --force for smoke runs).")


def write_output_manifest(folder, files, extra):
    """Hash every written output; record inputs, seeds and versions."""
    def rel(p):
        return os.path.relpath(p, folder).replace("\\", "/")
    hashes = {rel(p): sha256_file(p) for p in sorted(files)}
    manifest = {**extra, "files": hashes, "combined_sha256": combined_hash(hashes), "versions": package_versions()}
    write_json(manifest, f"{folder}/output_manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# Leakage assertions (permanent)
# ---------------------------------------------------------------------------
def outcome_leak_columns(fs):
    return (set(fs["targets"]) | set(fs["excluded_leakage"])
            | {"improved", "exact_next_response", "time_gap_days", "vt1_disease_response"})


def assert_predictors_clean(fs):
    """Model C / D raw predictors contain no outcome, no vt1_* and no excluded-leakage field."""
    leak = outcome_leak_columns(fs)
    for name in ["model_c", "model_d"]:
        cols = set(fs[name])
        bad = sorted(cols & leak)
        require(not bad, f"{name} contains outcome/leakage fields: {bad}")
        require(not any(c.startswith("vt1_") for c in cols), f"{name} contains a vt1_* field")
    require(set(fs["model_c"]) <= set(fs["model_d"]), "Model C is not a subset of Model D")


def assert_treatment_fields_clean(fs):
    """Treatment-group and covariate source fields are Vt-side Model-C inputs, never outcome fields."""
    mc, leak = set(fs["model_c"]), outcome_leak_columns(fs)
    fields = (list(TREATMENT_COLS.values()) + list(PRIOR_EXPOSURE_COLS.values())
              + list(CONT_COVARIATES) + [PLASMA_COL, "vt_disease_response", "current_line_number"])
    missing = sorted(set(fields) - mc)
    require(not missing, f"treatment/covariate fields not in Model C (not Vt-side inputs): {missing}")
    require(not (set(fields) & leak), "treatment/covariate fields overlap outcome/leakage columns")
    require(not any(c.startswith("vt1_") for c in fields), "a treatment/covariate field is a vt1_* column")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_master():
    """Frozen master table + split + eligibility, with asserted cohort counts. Adds `rna_avail`,
    boolean treatment flags and `currently_on_other` (SLAMF7 or BCMA)."""
    m = pd.read_csv(MASTER_PATH, low_memory=False)
    expect("raw master column count", m.shape[1], EXPECTED["master_columns"])
    sp = pd.read_csv(SPLIT_PATH, usecols=["pair_id", "split"])
    el = pd.read_csv(ELIGIBILITY_PATH, usecols=["pair_id", "eligible_for_binary"])
    m = m.merge(sp, on="pair_id", how="left", validate="one_to_one")
    m = m.merge(el, on="pair_id", how="left", validate="one_to_one")
    require(m["split"].notna().all() and m["eligible_for_binary"].notna().all(), "split/eligibility merge gaps")
    m["eligible_for_binary"] = to_bool(m["eligible_for_binary"])
    for c in list(TREATMENT_COLS.values()) + list(PRIOR_EXPOSURE_COLS.values()):
        require(m[c].notna().all(), f"{c} has missing values (unexpected)")
        m[c] = to_bool(m[c])
    m["currently_on_other"] = m["currently_on_slamf7"] | m["currently_on_bcma"]
    m["prior_exposure_other"] = m["prior_exposure_slamf7"] | m["prior_exposure_bcma"]
    m["rna_avail"] = m[STALENESS_COL].notna()

    expect("master rows", len(m), EXPECTED["master_rows"])
    expect("master patients", int(m["public_id"].nunique()), EXPECTED["master_patients"])
    expect("RNA-available rows", int(m["rna_avail"].sum()), EXPECTED["rna_rows"])
    expect("RNA-available patients", int(m.loc[m["rna_avail"], "public_id"].nunique()), EXPECTED["rna_patients"])
    expect("binary-eligible rows", int(m["eligible_for_binary"].sum()), EXPECTED["binary_eligible_rows"])
    sp_pat = m.drop_duplicates("public_id").groupby("split").size().to_dict()
    for s, n in EXPECTED["split_patients"].items():
        expect(f"patients in split {s}", int(sp_pat.get(s, 0)), n)
    # patient-disjoint splits
    sets = {s: set(m.loc[m["split"] == s, "public_id"]) for s in ["train", "val", "test"]}
    require(not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
            and not (sets["val"] & sets["test"]), "patients appear in more than one split")
    return m


def build_pathway_z(m, path_cols):
    """z-score each pathway across DISTINCT (patient, sample) vectors (expected 751)."""
    r = m[m["rna_avail"]]
    require(not r[path_cols].isna().any().any(), "RNA-available rows with missing pathway values")
    require(m.loc[~m["rna_avail"], path_cols].isna().all().all(), "pathway values present on non-RNA rows")
    h = pd.util.hash_pandas_object(r[path_cols].round(8), index=False).to_numpy()
    pid = pd.factorize(r["public_id"])[0]
    first = ~pd.DataFrame({"pid": pid, "h": h}).duplicated().to_numpy()
    expect("distinct (patient, RNA sample) vectors", int(first.sum()), EXPECTED["distinct_rna_samples"])
    distinct = r.loc[first, path_cols]
    mu, sd = distinct.mean(axis=0), distinct.std(axis=0, ddof=1)
    require((sd > 0).all(), "a pathway has zero variance across distinct samples")
    Z = (r[path_cols] - mu) / sd
    Z.index = r["pair_id"].to_numpy()
    return Z


# ---------------------------------------------------------------------------
# Support rule
# ---------------------------------------------------------------------------
def tier_of(pts, events, non_events):
    if pts >= TIER1["pts"] and events >= TIER1["events"] and non_events >= TIER1["non_events"]:
        return 1
    if pts >= TIER2["pts"] and events >= TIER2["events"] and non_events >= TIER2["non_events"]:
        return 2
    return 3


def _counts(d):
    return {"rows": int(len(d)), "pts": int(d["public_id"].nunique()),
            "events": int(d["improved"].sum()), "non_events": int((1 - d["improved"]).sum())}


def _mc_min(d):
    vc = d["exact_next_response"].value_counts().reindex(MULTICLASS_CLASSES, fill_value=0)
    return int(vc.min()), {k: int(v) for k, v in vc.items()}


def support_table(m):
    """Support by treatment for Part A (train+val RNA-available) and Part B (pooled RNA-available, binary)."""
    rna_bin = m[m["rna_avail"] & m["eligible_for_binary"]]
    a_bin = rna_bin[rna_bin["split"] != "test"]
    a_mc = m[m["rna_avail"] & (m["split"] != "test")]
    b_bin = rna_bin[rna_bin["current_line_number"].notna()]
    rows = []
    for t, col in TREATMENT_COLS.items():
        for uni, dfb in [("partA_trainval", a_bin), ("partB_pooled", b_bin)]:
            g, c = dfb[dfb[col]], dfb[~dfb[col]]
            cg, cc = _counts(g), _counts(c)
            tg, tc = tier_of(cg["pts"], cg["events"], cg["non_events"]), tier_of(cc["pts"], cc["events"], cc["non_events"])
            row = {"treatment": t, "role_declared": TREATMENT_ROLE[t], "universe": uni,
                   **{f"group_{k}": v for k, v in cg.items()}, **{f"complement_{k}": v for k, v in cc.items()},
                   "tier_group": tg, "tier_complement": tc, "tier": max(tg, tc),
                   "tier_declared": DECLARED_TIER[TREATMENT_ROLE[t]]}
            row["tier_matches_declared_role"] = bool(row["tier"] == row["tier_declared"])
            if uni == "partA_trainval":
                mg, pg = _mc_min(a_mc[a_mc[col]])
                mc_, pc_ = _mc_min(a_mc[~a_mc[col]])
                row.update({"multiclass_min_class_group": mg, "multiclass_min_class_complement": mc_,
                            "multiclass_class_counts_group": json.dumps(pg),
                            "multiclass_ok": bool(mg >= MULTICLASS_MIN_PER_CLASS and mc_ >= MULTICLASS_MIN_PER_CLASS)})
            rows.append(row)
    return pd.DataFrame(rows)


def assert_support_matches_declared_roles(tbl):
    """The frozen role assignment (PI/IMiD/steroid primary; CD38/chemo exploratory; SLAMF7/BCMA
    unsupported) must be what the objective support rule yields. Otherwise STOP."""
    bad = tbl[~tbl["tier_matches_declared_role"]]
    require(bad.empty, "support rule does NOT reproduce the declared treatment roles:\n"
            + bad[["treatment", "universe", "role_declared", "tier", "tier_declared"]].to_string(index=False))


def inference_treatments(tbl, task):
    """Treatments with Part A inference for a task: primary + exploratory (tier <= 2 in the train+val
    universe); multiclass additionally needs >= 20 rows per class in group and complement."""
    a = tbl[tbl["universe"] == "partA_trainval"].set_index("treatment")
    out = []
    for t in PRIMARY_TREATMENTS + EXPLORATORY_TREATMENTS:
        if int(a.loc[t, "tier"]) > 2:
            continue
        if task == "multiclass" and not bool(a.loc[t, "multiclass_ok"]):
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Metrics (weighted versions are used by the patient bootstrap)
# ---------------------------------------------------------------------------
def rank_structure(score):
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    distinct = np.flatnonzero(np.r_[np.diff(s) != 0, True])
    return order, distinct


def weighted_ap_auroc(y_sorted, w_sorted, distinct):
    """Weighted average precision (step-sum, as scikit-learn) and ROC-AUC (trapezoid), ties grouped."""
    tps = np.cumsum(w_sorted * y_sorted)[distinct]
    fps = np.cumsum(w_sorted * (1.0 - y_sorted))[distinct]
    P, N = tps[-1], fps[-1]
    if not (P > 0 and N > 0):
        return np.nan, np.nan
    denom = tps + fps
    precision = np.where(denom > 0, tps / np.maximum(denom, 1e-300), 0.0)
    recall = tps / P
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    tpr = np.r_[0.0, recall]
    fpr = np.r_[0.0, fps / N]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    return ap, auc


def binary_logloss_rows(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def multiclass_logloss_rows(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1.0))


def rps_rows(P, y):
    """Ranked probability score for ordered classes 0..K-1 (lower is better)."""
    K = P.shape[1]
    cum = np.cumsum(P, axis=1)[:, :-1]
    obs = (np.arange(K - 1)[None, :] >= y[:, None]).astype(float)
    return ((cum - obs) ** 2).sum(axis=1) / (K - 1)


def selfcheck_metrics():
    """Runtime self-check of the custom weighted metrics against scikit-learn / a naive loop."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.RandomState(0)
    n = 600
    y = (rng.rand(n) < 0.3).astype(float)
    s = np.round(rng.rand(n), 2)                       # deliberate ties
    w = rng.randint(0, 4, size=n).astype(float)
    order, distinct = rank_structure(s)
    ap, auc = weighted_ap_auroc(y[order], w[order], distinct)
    require(abs(ap - average_precision_score(y, s, sample_weight=w)) < 1e-9, "weighted AP self-check failed")
    require(abs(auc - roc_auc_score(y, s, sample_weight=w)) < 1e-9, "weighted AUROC self-check failed")
    P = rng.dirichlet(np.ones(6), size=50)
    yy = rng.randint(0, 6, size=50)
    naive = []
    for i in range(50):
        cp = np.cumsum(P[i])
        naive.append(sum((cp[k] - (1.0 if yy[i] <= k else 0.0)) ** 2 for k in range(5)) / 5)
    require(np.allclose(rps_rows(P, yy), naive), "RPS self-check failed")


def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    require(np.isfinite(p).all(), "Holm received a non-finite p-value")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj
