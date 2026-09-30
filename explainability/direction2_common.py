"""
Direction 2 (Prof. Song) -- shared FROZEN definitions for the
"clinically similar at Vt, different at Vt+1" analysis.

*** RETROSPECTIVE MOLECULAR ASSOCIATION -- NOT PREDICTIVE VALIDATION ***
Discordance between matched visits is defined by the Vt+1 response, so nothing
produced by the Direction 2 scripts can be read as evidence that RNA predicts
response. Predictive claims belong to the locked C-vs-D analyses only.

Files in this package (all run from the repo root):
  explainability/direction2_common.py       (this file: frozen spec, assertions, utils)
  explainability/direction2_matching.py     (STEP 1: outcome- and RNA-blind matching,
                                             writes + hashes the frozen pair tables)
  explainability/direction2_association.py  (STEP 2: verifies the hash, THEN merges
                                             outcomes and pathway scores, runs the
                                             frozen association analysis + S1-S7)

RNA-BLINDNESS RULE (enforced by assertions, see assert_matching_frame_columns /
assert_frame_blind / assert_loaded_columns_clean):
  RNA availability (days_since_rna_sample notna) is used for exactly ONE purpose:
  to define the pre-specified RNA-available universe (row membership). No RNA
  VALUE and no RNA AGE ever enters a distance, a stratum, the caliper, the
  candidate ordering or the pair selection. The timing column is reduced to a
  boolean mask at load time and dropped, and the frame handed to the distance
  code must contain exactly the whitelisted Vt-side Model-C columns.

Nothing in this package reads from or writes to any existing frozen output.
All outputs go under artifacts/results_clean_rerun/direction2/ only.

Reuse of repo conventions (not imports, on purpose): data paths, the
model_feature_sets.json forbidden-column pattern from
explainability/subgroup_analysis_common.py::assert_no_leakage_in_subgroup_definitions,
and RESPONSE_RANK from data_pipeline/build_visit_pairs.py. The matching step
must stay RNA-blind, so it deliberately does NOT import the modelling stack
(train_models / build_preprocessing), which loads RNA-bearing tables.
"""

import hashlib
import json
import os
import platform
import sys
from importlib import metadata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (repo-relative; run every script from the repo root)
# ---------------------------------------------------------------------------
CLINICAL_DIR = "data/clinical"
SPLITS_DIR = "data/splits"
MASTER_PATH = f"{CLINICAL_DIR}/visit_pairs_with_rna.csv"
SPLIT_PATH = f"{SPLITS_DIR}/visit_pair_splits.csv"
ELIGIBILITY_PATH = f"{SPLITS_DIR}/task_eligibility.csv"
FEATURE_SETS_PATH = "data_pipeline/model_feature_sets.json"
OUT_ROOT = "artifacts/results_clean_rerun/direction2"

ANALYSIS_LABEL = "RETROSPECTIVE MOLECULAR ASSOCIATION - NOT PREDICTIVE VALIDATION"

RANDOM_SEED = 42
S4_SEED_BASE = RANDOM_SEED + 1000               # S4 draw k uses seed S4_SEED_BASE + k
FAMILIES = ["ordinal", "binary"]                # ordinal = multiclass universe
FAMILY_ROLE = {"ordinal": "PRIMARY", "binary": "SECONDARY"}
STALENESS_COL = "days_since_rna_sample"

# IMWG response scale (same as data_pipeline/build_visit_pairs.py::RESPONSE_RANK)
RESPONSE_RANK = {
    "progressive_disease": 0,
    "stable_disease": 1,
    "partial_response": 2,
    "very_good_partial_response": 3,
    "complete_response": 4,
    "stringent_complete_response": 5,
}

# ---------------------------------------------------------------------------
# FROZEN similarity specification (Model C, Vt-side inputs only)
# ---------------------------------------------------------------------------
LINE_GROUP_BINS = [0, 1, 2, 100]
LINE_GROUP_LABELS = ["1", "2", "3+"]

# exact strata: Vt response x line group {1,2,3+} x currently on PI / IMiD / CD38
EXACT_KEYS = ["vt_disease_response", "line_group",
              "currently_on_pi", "currently_on_imid", "currently_on_cd38"]
EXACT_SOURCE_COLS = ["vt_disease_response", "current_line_number",
                     "currently_on_pi", "currently_on_imid", "currently_on_cd38"]

# the 17 continuous Model-C features and their FROZEN transforms
CONTINUOUS_FEATURES = {
    "m_protein_current": "log1p",
    "kappa_flc_current": "log1p",
    "lambda_flc_current": "log1p",
    "calcium_current": "identity",
    "creatinine_current": "log1p",
    "hemoglobin_current": "identity",
    "b2m_current": "log1p",
    "albumin_current": "identity",
    "ldh_current": "log1p",
    "m_protein_change": "slog",
    "hemoglobin_change": "identity",
    "creatinine_change": "slog",
    "albumin_change": "identity",
    "line_duration_so_far": "log1p",
    "n_prior_lines_completed": "identity",
    "iss_stage": "identity",
    "age_at_diagnosis": "identity",
}
SIMILARITY_SOURCE_COLS = EXACT_SOURCE_COLS + list(CONTINUOUS_FEATURES)

# The ONLY columns the frame handed to the distance code may contain.
MATCHING_FRAME_COLUMNS = set(SIMILARITY_SOURCE_COLS) | {"pair_id", "public_id", "eligible_for_binary", "line_group"}

# S3/S5 adjustment covariates = the 17 matching features MINUS line_duration_so_far.
# line_duration_so_far is STRUCTURALLY missing whenever no treatment line is active at Vt
# (it reflects treatment state, not ordinary missing data), so keeping it under complete-case
# handling would silently restrict S3/S5 to actively treated visits. The treatment context is still
# adjusted for through the exact strata (line group x currently-on PI / IMiD / CD38 fixed effects)
# and n_prior_lines_completed. This does NOT touch the matching distance, which keeps all 17 features
# (partial distances tolerate missing values there).
S3_EXCLUDED_FEATURES = ["line_duration_so_far"]
S3_CONTINUOUS_FEATURES = [c for c in CONTINUOUS_FEATURES if c not in S3_EXCLUDED_FEATURES]

WINSOR_PERCENTILES = (1, 99)
MIN_JOINT_FEATURES = 12          # >= 12 of 17 features jointly observed
CALIPER_PERCENTILE_PRIMARY = 75  # FROZEN primary caliper (p75 of NN distances)
CALIPER_PERCENTILES_S1 = (50, 90)
N_RANDOM_DRAWS_DEFAULT = 200     # S4

# ---------------------------------------------------------------------------
# Expected cohort numbers (verified during the design audit). Violations FAIL.
# ---------------------------------------------------------------------------
EXPECTED = {
    "master_rows": 13451,
    "master_patients": 1025,
    "master_columns": 203,
    "rna_rows": 9173,
    "rna_patients": 707,
    "binary_eligible_rows": 12298,
    "universe_rows": {"ordinal": 9172, "binary": 8261},
    "universe_patients": {"ordinal": 707, "binary": 703},
    "n_pathways": 50,
    "distinct_rna_samples": 751,
    "split_patients": {"train": 716, "val": 154, "test": 155},
    "n_similarity_inputs": 22,
    "n_continuous_features": 17,
}
# Counts established by the earlier scratch audit. They are recorded and
# COMPARED EXACTLY (no tolerance). A difference never fails the run by itself
# (explicit pair_id tie-breaking may legitimately change which tied pairs are
# chosen); it is reported together with a tie-breaking diagnosis.
AUDIT_REFERENCE = {
    "ordinal": {"pairs": 259, "informative_pairs": 120},
    "binary": {"pairs": 261, "informative_pairs": 63},
}

# S2 (frozen): keep informative pairs whose two members' RNA ages differ by at
# most this many days, i.e. |days_since_rna_sample(a) - days_since_rna_sample(b)| <= cap.
S2_RNA_AGE_GAP_CAP_DAYS = 365
# S3/S5 (frozen handling rules, see ANALYSIS_SPEC): strata with fewer complete-case
# rows than this are EXCLUDED (never pooled) and counted.
S3_MIN_STRATUM_ROWS = 20

FROZEN_SPEC = {
    "direction": 2,
    "label": ANALYSIS_LABEL,
    "primary_family": "ordinal (multiclass universe; pairs whose Vt+1 classes differ)",
    "secondary_family": "binary (binary universe; pairs discordant on 'improved')",
    "universe": ("RNA-available rows (days_since_rna_sample notna; used ONLY to define row membership) "
                 "with non-missing current_line_number; binary additionally eligible_for_binary"),
    "rna_usage_in_matching": ("RNA availability defines the universe only. No RNA value and no RNA age is "
                              "used for distance, strata, caliper, candidate ordering or pair selection."),
    "pooled_across_splits": True,
    "exact_keys": EXACT_KEYS,
    "line_group_bins": LINE_GROUP_BINS,
    "line_group_labels": LINE_GROUP_LABELS,
    "continuous_features": CONTINUOUS_FEATURES,
    "winsor_percentiles": list(WINSOR_PERCENTILES),
    "standardisation": "z-score per feature on the matching universe after winsorisation",
    "distance": "root-mean-square z-difference over jointly observed features",
    "min_joint_features": MIN_JOINT_FEATURES,
    "same_patient_excluded": True,
    "caliper_primary": f"p{CALIPER_PERCENTILE_PRIMARY} of each row's nearest cross-patient same-stratum distance",
    "caliper_S1": [f"p{p}" for p in CALIPER_PERCENTILES_S1],
    "M1": ("global greedy pairing in ascending distance; ties broken by "
           "(min pair_id, max pair_id); each PATIENT used at most once"),
    "S4": (f"randomised greedy nearest-neighbour (seeded random VISITING ORDER): draw k uses seed "
           f"{S4_SEED_BASE}+k; rows are visited in that order; each row whose patient is unused is paired "
           f"with its nearest in-caliper partner from an unused patient (ties by pair_id); patient-once"),
    "seed": RANDOM_SEED,
    "s4_seed_base": S4_SEED_BASE,
}

# Analysis-stage rules (recorded by direction2_association.py in analysis_spec.json)
ANALYSIS_SPEC = {
    "label": ANALYSIS_LABEL,
    "primary_test": ("exact sign-flip permutation if n_pairs <= 20, else 10,000 Monte Carlo flips with "
                     "add-one correction; studentised mean paired difference; pair-level percentile "
                     "bootstrap (10,000); BH within family x analysis over 50 pathways; single-step max-T; "
                     "global test = sum of squared d_z under the same flips"),
    "seeds": ("per-analysis seed = 42 + crc32('<family>|<analysis>') % 100000 for the sign-flip test; "
              "bootstrap seed = that + 1; S4 matching draw k seed = 1042 + k. All recorded in the outputs."),
    "S2": (f"RNA-age GAP between the two matched members: |days_since_rna_sample(a) - days_since_rna_sample(b)| "
           f"<= {S2_RNA_AGE_GAP_CAP_DAYS} days. It is NOT a cap on either member's own RNA age."),
    "S3": {
        "model_binary": "GEE binomial-logit, independence working correlation, patient-clustered robust SEs",
        "model_ordinal": ("OrdinalGEE cumulative-logit, independence working correlation, patient-clustered "
                          "robust SEs. NO fallback model: a failed fit is reported as failed."),
        "covariates": ("16 frozen continuous Model-C features = the 17 matching features MINUS line_duration_so_far "
                       "(transform, winsorise 1/99, z-score on the analysis rows) + log1p(days_since_rna_sample) z-scored"),
        "line_duration_so_far": ("REMOVED from the S3/S5 covariate set only: it is structurally missing when no treatment "
                                 "line is active at Vt (treatment state, not ordinary missingness), so complete-case "
                                 "handling would restrict the regression to actively treated visits. Treatment context "
                                 "is retained through the exact strata (line group x PI/IMiD/CD38) and "
                                 "n_prior_lines_completed. The primary matching (17-feature distance) is unchanged."),
        "exact_strata_fixed_effects": "one dummy per exact stratum (Vt response x line group x PI/IMiD/CD38); no pooling",
        "missing_covariates": ("COMPLETE-CASE on the remaining genuinely-missing covariates: rows with any missing "
                               "retained covariate are excluded (no imputation, no zero-filling); counts per covariate "
                               "and rows lost are reported; rows with structurally missing line_duration_so_far are "
                               "NOT excluded and their number is reported"),
        "sparse_strata": (f"strata with < {S3_MIN_STRATUM_ROWS} complete-case rows are EXCLUDED with their rows "
                          f"(not pooled); counts reported"),
        "no_variation_strata": ("strata whose outcome has no variation among the remaining rows (binary: all events "
                                "or all non-events; ordinal: a single outcome class) are excluded because their "
                                "fixed-effect dummy is not estimable; counts reported"),
        "fit_failures": "exception, non-convergence or rank-deficient design => that pathway/analysis is marked failed",
    },
    "S5": "S3 + pct_plasma_cells_bm (z-scored); additionally complete-case on pct_plasma_cells_bm",
    "S7": "DESCRIPTIVE ONLY (no confirmatory claim)",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def require(condition, message):
    """Fail loudly (not stripped by python -O, unlike assert)."""
    if not condition:
        raise AssertionError(f"[direction2] {message}")


def expect(name, actual, expected):
    require(actual == expected, f"{name}: expected {expected!r}, got {actual!r}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")


def write_csv(df, path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", **kwargs)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combined_hash(file_hashes):
    """One hash over {relative_path: sha256}, independent of dict order."""
    lines = "\n".join(f"{k}:{file_hashes[k]}" for k in sorted(file_hashes))
    return sha256_text(lines)


def package_versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ["numpy", "pandas", "scipy", "statsmodels"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def get_dirs(smoke=False):
    root = f"{OUT_ROOT}/_smoke" if smoke else OUT_ROOT
    return {"root": root, "frozen": f"{root}/frozen_matching", "assoc": f"{root}/association"}


def load_feature_sets():
    with open(FEATURE_SETS_PATH, encoding="utf-8") as f:
        return json.load(f)


def pathway_columns(fs):
    """The 50 Hallmark columns = Model D raw columns not in Model C, minus the timing column."""
    c = set(fs["model_c"])
    cols = [x for x in fs["model_d"] if x not in c and x != STALENESS_COL]
    expect("number of Hallmark pathway columns", len(cols), EXPECTED["n_pathways"])
    return cols


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def derive_line_group(line_number):
    g = pd.cut(line_number.astype(float), bins=LINE_GROUP_BINS, labels=LINE_GROUP_LABELS)
    require(g.notna().all(), "current_line_number could not be mapped to {1,2,3+}")
    return g.astype(str)


# ---------------------------------------------------------------------------
# PERMANENT leakage / RNA-blindness assertions
# ---------------------------------------------------------------------------
def forbidden_columns(fs):
    model_c, model_d = set(fs["model_c"]), set(fs["model_d"])
    rna_cols = model_d - model_c                       # 50 pathways + days_since_rna_sample
    return (set(fs["targets"]) | set(fs["excluded_leakage"]) | rna_cols
            | {"improved", "exact_next_response", "time_gap_days", "vt1_disease_response"})


def assert_similarity_inputs_clean(fs):
    """Every similarity input is a Model-C (Vt-side) column and NONE is an
    outcome, vt1_*, excluded-leakage or RNA field (values or RNA age)."""
    model_c = set(fs["model_c"])
    forbidden = forbidden_columns(fs)
    sim = set(SIMILARITY_SOURCE_COLS)
    expect("number of similarity inputs", len(SIMILARITY_SOURCE_COLS), EXPECTED["n_similarity_inputs"])
    expect("number of continuous features", len(CONTINUOUS_FEATURES), EXPECTED["n_continuous_features"])
    require(len(sim) == len(SIMILARITY_SOURCE_COLS), "duplicate similarity input columns")
    not_c = sorted(sim - model_c)
    require(not not_c, f"similarity inputs not in Model C: {not_c}")
    bad = sorted(sim & forbidden)
    require(not bad, f"similarity inputs overlap forbidden outcome/RNA/leakage fields: {bad}")
    require(not any(c.startswith("vt1_") for c in sim), "a vt1_* column is a similarity input")
    require(STALENESS_COL not in sim, "RNA age (days_since_rna_sample) must never be a similarity input")
    require(set(EXACT_SOURCE_COLS) <= sim and set(CONTINUOUS_FEATURES) <= sim,
            "exact/continuous inputs are not a subset of the similarity inputs")
    return True


def assert_loaded_columns_clean(loaded_cols):
    """Matching step: only the similarity inputs + identifiers + the RNA
    AVAILABILITY MASK source (days_since_rna_sample, reduced to a boolean and
    dropped immediately) + the binary eligibility flag may be loaded. No pathway,
    target or vt1_* column may be read."""
    allowed = set(SIMILARITY_SOURCE_COLS) | {"pair_id", "public_id", STALENESS_COL, "eligible_for_binary"}
    extra = sorted(set(loaded_cols) - allowed)
    require(not extra, f"matching step attempted to load non-whitelisted columns: {extra}")
    require(not any(c.startswith("vt1_") for c in loaded_cols), "matching step loaded a vt1_* column")


def assert_frame_blind(frame, fs):
    """Runtime guard called immediately before any distance is computed: the
    frame must hold NO outcome, vt1_*, excluded-leakage or RNA column
    (including the timing column)."""
    forbidden = forbidden_columns(fs)
    present = sorted(set(frame.columns) & forbidden)
    require(not present, f"matching frame contains forbidden outcome/RNA/leakage columns: {present}")
    require(not any(str(c).startswith("vt1_") for c in frame.columns), "matching frame contains a vt1_* column")


def assert_matching_frame_columns(frame):
    """RNA-blindness guard: the frame passed to distance/strata/caliper/pairing code must
    contain EXACTLY the whitelisted Vt-side columns -- in particular no RNA availability
    flag, RNA age or RNA value. RNA availability only defined which rows are in the frame."""
    cols = set(frame.columns)
    extra = sorted(cols - MATCHING_FRAME_COLUMNS)
    missing = sorted(MATCHING_FRAME_COLUMNS - cols)
    require(not extra, f"matching frame has non-whitelisted columns (possible RNA/outcome leak): {extra}")
    require(not missing, f"matching frame is missing whitelisted columns: {missing}")


# ---------------------------------------------------------------------------
# Universe + similarity matrix (shared by matching and by S3/S5 covariates)
# ---------------------------------------------------------------------------
def select_universe(df, family):
    """Matching universe. `df` must carry: pair_id, public_id, rna_available (bool),
    current_line_number, eligible_for_binary, vt_disease_response, currently_on_*.
    rna_available is used ONLY here, to define row membership. Sorted by pair_id so
    results do not depend on file row order."""
    require(family in FAMILIES, f"unknown family {family!r}")
    d = df.copy()
    for c in ["currently_on_pi", "currently_on_imid", "currently_on_cd38"]:
        d[c] = to_bool(d[c])
    d["eligible_for_binary"] = to_bool(d["eligible_for_binary"])
    mask = d["rna_available"].astype(bool) & d["current_line_number"].notna()
    if family == "binary":
        mask = mask & d["eligible_for_binary"]
    u = d[mask].sort_values("pair_id").reset_index(drop=True)
    u["line_group"] = derive_line_group(u["current_line_number"])
    expect(f"universe rows [{family}]", len(u), EXPECTED["universe_rows"][family])
    expect(f"universe patients [{family}]", int(u["public_id"].nunique()),
           EXPECTED["universe_patients"][family])
    require(u["pair_id"].is_unique, "duplicate pair_id in universe")
    return u


def transform_series(kind, s):
    x = s.astype(float)
    if kind == "log1p":
        return np.log1p(x.clip(lower=0))
    if kind == "slog":
        return np.sign(x) * np.log1p(np.abs(x))
    if kind == "identity":
        return x
    raise ValueError(f"unknown transform {kind!r}")


def continuous_matrix(u, features=None):
    """z-scored matrix of the frozen continuous features, NaN kept. `features` defaults to ALL 17
    (matching); S3/S5 pass S3_CONTINUOUS_FEATURES (16). Transform -> winsorise at 1st/99th
    percentile -> z-score, all computed on `u` (covariates only; no outcome, no RNA)."""
    names = list(CONTINUOUS_FEATURES) if features is None else list(features)
    require(set(names) <= set(CONTINUOUS_FEATURES), "unknown continuous feature requested")
    cols = [transform_series(CONTINUOUS_FEATURES[name], u[name]).to_numpy(dtype=float)
            for name in names]
    X = np.column_stack(cols)
    lo = np.nanpercentile(X, WINSOR_PERCENTILES[0], axis=0)
    hi = np.nanpercentile(X, WINSOR_PERCENTILES[1], axis=0)
    X = np.clip(X, lo, hi)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------
def bh_qvalues(p):
    """Benjamini-Hochberg q-values; NaN p-values stay NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = out
    return q
