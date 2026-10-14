"""
EXPLORATORY, POST HOC, WITHIN-COMMPASS ROBUSTNESS TEST.
NOT external validation. NOT a pristine untouched test set (this is a new,
purpose-built split invented after seeing the main results, specifically to
probe country generalization).

ORIGINAL-TRAIN-ONLY REVISION. Derived from modeling/us_to_spain_canada_model_c_ustuned.py (which is derived from
modeling/us_to_spain_canada_model_c.py on origin/modeling-country-robustness); both are unchanged.

Design principle (Professor Song's feedback): patients in the main analysis's validation and test partitions, and the
Spain/Canada evaluation patients, must not contribute to ANY model-development step. Here every learned quantity is
derived only from US patients in the ORIGINAL TRAINING PARTITION (data/splits/patient_split_assignments.csv,
split == "train"; 535 US patients):

    * hyperparameter tuning                -> fit set (5-fold patient-grouped CV, main-analysis grid and metrics)
    * preprocessing (imputation, one-hot)  -> fit set
    * final model                          -> fit set
    * calibration method + binary threshold-> calibration set (cross-fitted within it; then frozen)

The 535 patients are divided with the SAME US 82/18 random split as the earlier experiments (seed 42; sorted US ids;
train_test_split), restricted to original-train patients: fit set = original-train patients in US-train (439);
calibration set = original-train patients in US-val (96). No new random draw is introduced. The 236 US patients in the
original validation/test partitions are excluded from every step (not used for fitting, tuning, preprocessing,
calibration, thresholds or evaluation). Spain and Canada (116 patients) are used once, for evaluation, and Italy is
excluded entirely.

Compared with the US-tuned version: the fit set is 439 instead of 632 patients (193 original-validation/test patients
removed), the calibration set 96 instead of 139 (43 removed), so preprocessing, the model, the calibration choice and the
threshold are all re-derived. The evaluation cohort (116 Spain/Canada patients; 1,588 binary / 1,694 six-class pairs) is
unchanged.

Calibration source (documentation correction): this script does NOT read calibration_summary.json or any other
calibration file. The candidate list (binary: uncalibrated/platt/isotonic; six-class: uncalibrated/temperature) and the
binary selection rule (5-fold patient-grouped cross-fitting; lowest cross-fitted log loss, preferring the simpler method
if its cross-fitted Brier is within 0.001) were COPIED from the main calibration design (calibration_summary.json, on
origin/subgroup-analysis) and are re-implemented below. The six-class rule is UNCHANGED from the earlier experiment
(cross-fitted log loss only); as a diagnostic, the audit JSON records whether the main-analysis rule (with Brier
tie-break) would select the same method.

Status: EXPLORATORY, POST HOC, within-CoMMpass robustness test; not external validation. The earlier country results had
been seen before this redesign, so the redesign is not blind. Italy was excluded after inspecting its outcome
distribution (post hoc).

Outputs: artifacts/results_clean_rerun/geographic_origtrain/ (metrics and audit JSON; prediction CSVs contain patient
IDs and stay gitignored) and artifacts/models_clean_rerun/geographic_origtrain/ (gitignored). Nothing existing is
modified. Run from the repository root:

    python modeling/us_to_spain_canada_model_c_origtrain.py
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss,
    confusion_matrix, precision_recall_fscore_support, accuracy_score,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, "data_pipeline")
sys.path.insert(0, "modeling")
import build_preprocessing as bp  # noqa: E402
import train_models as tm  # noqa: E402  (import is side-effect free; its outputs are under __main__)

CLINICAL_DIR = "data/clinical"
SPLITS_DIR = "data/splits"
RESULTS_DIR = "artifacts/results_clean_rerun/geographic_origtrain"
MODELS_DIR = "artifacts/models_clean_rerun/geographic_origtrain"
USTUNED_RESULTS_DIR = "artifacts/results_clean_rerun/geographic_us_tuned"  # comparison only (read-only)
PREVIOUS_RESULTS_GIT = ("origin/modeling-country-robustness", "results/robustness_country")
PREVIOUS_SCRIPT_GIT = ("origin/modeling-country-robustness", "modeling/us_to_spain_canada_model_c.py")

# Expected cohort sizes (definition of the experiment, independent of any tuning).
# fit/calibration sets = original-train patients within the earlier 632/139 US split (independently recounted in verification)
EXPECTED = {"n_us_train_patients": 439, "n_us_val_patients": 96, "n_us_excluded_not_original_train": 236,
            "n_spain_canada_patients": 116,
            "n_spain_patients": 80, "n_canada_patients": 36,
            "pairs_spain_canada": {"binary": 1588, "multiclass": 1694}}

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
US_TRAIN_FRAC = 0.82  # matches original ~70/15 relative train:val ratio (70/85=0.824)

MULTICLASS_TARGET = "exact_next_response"
BINARY_TARGET = "improved"
MULTICLASS_CLASSES = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]

# ---- MODEL C DEFINITION ----
LOCKED_ALGORITHM = "xgboost"
# Hyperparameters of the EARLIER country experiment (reused from the main analysis, whose tuning split contained
# 80 of the 116 evaluated patients). Recorded ONLY to report whether re-tuning changed the selection; NOT used here.
PREVIOUS_PARAMS = {
    "binary": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    "multiclass": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
}
LOCKED_CALIBRATION_CANDIDATES = {
    "binary": ["uncalibrated", "platt", "isotonic"],
    "multiclass": ["uncalibrated", "temperature"],
}


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 1. Load data, attach country, define cohorts
# --------------------------------------------------------------------------

def load_data():
    vp = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    subj = pd.read_csv(f"{CLINICAL_DIR}/subject_deid.csv")
    country = subj.set_index("public_id")["country_of_residence_at_enrollment"]
    vp["country"] = vp["public_id"].map(country)
    assert vp["country"].notna().all(), "Some patients in visit_pairs have no country mapping."
    # Original (main-analysis) patient partition; used ONLY to define the tuning pool and for audit.
    ps = pd.read_csv(f"{SPLITS_DIR}/patient_split_assignments.csv")
    assert ps["public_id"].is_unique and set(ps["split"]) == {"train", "val", "test"}
    vp["orig_split"] = vp["public_id"].map(ps.set_index("public_id")["split"])
    assert vp["orig_split"].notna().all(), "Some patients have no original split assignment."
    pair_split = pd.read_csv(f"{SPLITS_DIR}/visit_pair_splits.csv").set_index("pair_id")["split"]
    assert (vp["pair_id"].map(pair_split) == vp["orig_split"]).all(), \
        "Pair-level and patient-level original splits disagree."
    return vp


def define_eligibility(df):
    df = df.copy()
    df["eligible_for_binary"] = df["vt_disease_response"] != "stringent_complete_response"
    df["eligible_for_multiclass"] = df[MULTICLASS_TARGET].notna()
    assert df["eligible_for_multiclass"].all(), "Unexpected missing exact_next_response."
    return df


def load_model_c_feature_list():
    with open("data_pipeline/model_feature_sets.json") as f:
        fs = json.load(f)
    return fs["model_c"]


# --------------------------------------------------------------------------
# 2. US train/val split (patient-level, US only)
# --------------------------------------------------------------------------

def make_us_split(df):
    us_patients = sorted(df.loc[df["country"] == "united_states", "public_id"].unique())
    train_ids, val_ids = train_test_split(
        us_patients, train_size=US_TRAIN_FRAC, random_state=RANDOM_SEED
    )
    train_ids, val_ids = set(train_ids), set(val_ids)
    assert train_ids.isdisjoint(val_ids)
    split_map = {}
    for pid in train_ids:
        split_map[pid] = "us_train"
    for pid in val_ids:
        split_map[pid] = "us_val"
    return split_map


def id_digest(ids):
    """SHA-256 of the sorted patient-id list: lets an independent tally confirm membership
    without writing any identifier to a results file."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def git_text(ref, path):
    try:
        return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
    except Exception:  # branch or file unavailable: recorded as such, never fatal
        return None


def environment_record():
    import importlib.metadata as md
    pk = {}
    for name in ("xgboost", "scikit-learn", "numpy", "pandas", "scipy", "joblib"):
        try:
            pk[name] = md.version(name)
        except Exception:
            pk[name] = None
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        head = branch = None
    prev = git_text(*PREVIOUS_SCRIPT_GIT)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": pk,
            "git_head": head, "git_branch": branch,
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "previous_script_source": "%s:%s" % PREVIOUS_SCRIPT_GIT,
            "previous_script_sha256": hashlib.sha256(prev).hexdigest() if prev else None,
            "previous_script_note": ("the earlier experiment's package versions were not recorded; "
                                     "metric differences could in principle arise from library versions")}


def define_tuning_pool(df, us_train_mask, elig_mask):
    """Rows eligible for hyperparameter tuning: US patients in BOTH the original training
    partition AND this experiment's US-train set, task-eligible pairs only."""
    return us_train_mask & elig_mask & (df["orig_split"] == "train") & (df["country"] == "united_states")


# --------------------------------------------------------------------------
# 3. Preprocessing - refit from scratch on US-train ONLY
#    (reuses build_preprocessing.py's classify_columns / build_preprocessor
#    so column-type handling is IDENTICAL to the locked pipeline; only the
#    fitting population changes)
# --------------------------------------------------------------------------

def fit_preprocessor_on_us_train(df, feature_cols, us_train_mask):
    buckets = bp.classify_columns(df, feature_cols, us_train_mask)
    log("Column classification (fit on US-train only):")
    for k, v in buckets.items():
        log(f"  {k:18s}: {len(v)} columns")
    preprocessor = bp.build_preprocessor(buckets)
    X_train_raw = df.loc[us_train_mask, feature_cols]
    preprocessor.fit(X_train_raw)
    return preprocessor, buckets


def audit_unseen_categories(preprocessor, buckets, df, eval_mask, feature_cols):
    """For every one-hot-encoded column, report categories present in the
    evaluation rows that were never seen in US-train fitting (these get
    handle_unknown='ignore' -> all-zero encoding, not an error, but worth
    counting)."""
    cat_cols = buckets["categorical"] + buckets["object_boolean"]
    report = {}
    named = dict(preprocessor.named_transformers_.items())
    # map each categorical column to its fitted OneHotEncoder categories
    for name, cols_group in [
        ("categorical", [c for c in buckets["categorical"] if c not in bp.HIGH_CARDINALITY_COLS]),
        ("categorical_high_card", [c for c in buckets["categorical"] if c in bp.HIGH_CARDINALITY_COLS]),
        ("object_boolean", buckets["object_boolean"]),
    ]:
        if not cols_group or name not in named:
            continue
        ohe = named[name].named_steps["onehot"]
        for i, col in enumerate(cols_group):
            seen = set(ohe.categories_[i])
            eval_vals = df.loc[eval_mask, col]
            if name == "object_boolean":
                eval_vals = bp._stringify_preserving_na(eval_vals)
            eval_unique = set(eval_vals.dropna().unique())
            unseen = eval_unique - seen
            if unseen:
                n_rows_affected = eval_vals.isin(unseen).sum()
                report[col] = {
                    "unseen_categories": sorted(str(u) for u in unseen),
                    "n_eval_rows_affected": int(n_rows_affected),
                }
    return report


def transform(preprocessor, df, feature_cols, mask):
    return np.asarray(preprocessor.transform(df.loc[mask, feature_cols]))


# --------------------------------------------------------------------------
# 4. Model
# --------------------------------------------------------------------------

def build_model(task, params):
    import xgboost as xgb
    objective = "binary:logistic" if task == "binary" else "multi:softprob"
    metric = "logloss" if task == "binary" else "mlogloss"
    return xgb.XGBClassifier(objective=objective, random_state=RANDOM_SEED,
                              eval_metric=metric, **params)


# --------------------------------------------------------------------------
# 5. Calibration selection (fit candidates via patient-grouped cross-fitting
#    on US-val ONLY, pick per the selection rule below, freeze).
#    Candidate list and binary rule are COPIED from the main calibration design
#    (calibration_summary.json on origin/subgroup-analysis); this script never reads it.
# --------------------------------------------------------------------------

def select_threshold(y_true, proba):
    candidates = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in candidates:
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def cross_fit_calibration_binary(y_val, proba_val, groups_val, n_folds=5, seed=RANDOM_SEED):
    """Reproduces the locked selection rule: 5-fold patient-grouped
    cross-fitting on validation; lowest cross-fitted log-loss, but prefer
    a simpler method if its cross-fitted Brier is within 0.001."""
    gkf = GroupKFold(n_splits=min(n_folds, len(set(groups_val))))
    folds = list(gkf.split(proba_val, y_val, groups=groups_val))
    methods = {"uncalibrated": None, "platt": "platt", "isotonic": "isotonic"}
    scored = {}
    for name, kind in methods.items():
        cf_logloss, cf_brier, oof = [], [], np.zeros_like(proba_val, dtype=float)
        for tr_idx, va_idx in folds:
            p_tr, y_tr = proba_val[tr_idx], y_val[tr_idx]
            p_va = proba_val[va_idx]
            if kind is None:
                p_va_cal = p_va
            elif kind == "platt":
                lr = LogisticRegression()
                lr.fit(p_tr.reshape(-1, 1), y_tr)
                p_va_cal = lr.predict_proba(p_va.reshape(-1, 1))[:, 1]
            else:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_tr, y_tr)
                p_va_cal = iso.predict(p_va)
            p_va_cal = np.clip(p_va_cal, 1e-6, 1 - 1e-6)
            oof[va_idx] = p_va_cal
        cf_logloss = log_loss(y_val, oof, labels=[0, 1])
        cf_brier = brier_score_loss(y_val, oof)
        scored[name] = {"cross_fit_log_loss": float(cf_logloss), "cross_fit_brier": float(cf_brier)}
    # selection rule: lowest log-loss, but prefer simpler if within 0.001 Brier
    order = ["uncalibrated", "platt", "isotonic"]  # simplicity order
    best_name = min(scored, key=lambda k: scored[k]["cross_fit_log_loss"])
    best_brier = scored[best_name]["cross_fit_brier"]
    for name in order:
        if scored[name]["cross_fit_brier"] <= best_brier + 0.001:
            best_name = name
            break
    return best_name, scored


def fit_final_calibrator_binary(method, proba_val, y_val):
    if method == "uncalibrated":
        return None
    if method == "platt":
        lr = LogisticRegression()
        lr.fit(proba_val.reshape(-1, 1), y_val)
        return lr
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba_val, y_val)
        return iso
    raise ValueError(method)


def apply_calibrator_binary(method, calibrator, proba):
    if method == "uncalibrated":
        return proba
    if method == "platt":
        return calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return np.clip(calibrator.predict(proba), 1e-6, 1 - 1e-6)
    raise ValueError(method)


def cross_fit_calibration_multiclass(y_val, proba_val, groups_val, n_folds=5, seed=RANDOM_SEED):
    """Temperature scaling vs uncalibrated, cross-fit on US-val."""
    from scipy.optimize import minimize_scalar

    def nll_at_T(logT, logits, y):
        T = np.exp(logT)
        scaled = logits / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        p = np.clip(p, 1e-9, 1)
        return -np.mean(np.log(p[np.arange(len(y)), y]))

    log_proba = np.log(np.clip(proba_val, 1e-9, 1))  # treat as pseudo-logits
    gkf = GroupKFold(n_splits=min(n_folds, len(set(groups_val))))
    folds = list(gkf.split(proba_val, y_val, groups=groups_val))
    scored = {}
    oof_uncal = proba_val.copy()
    oof_temp = np.zeros_like(proba_val)
    for tr_idx, va_idx in folds:
        res = minimize_scalar(
            lambda logT: nll_at_T(logT, log_proba[tr_idx], y_val[tr_idx]),
            bounds=(-3, 3), method="bounded",
        )
        T = np.exp(res.x)
        scaled = log_proba[va_idx] / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        oof_temp[va_idx] = p
    onehot = np.eye(6)[y_val]
    for name, oof in (("uncalibrated", oof_uncal), ("temperature", oof_temp)):
        scored[name] = {"cross_fit_log_loss": float(log_loss(y_val, oof, labels=list(range(6)))),
                        # diagnostic only (not used by the rule applied below): multiclass Brier
                        "cross_fit_brier_multiclass": float(np.mean(np.sum((oof - onehot) ** 2, axis=1)))}
    # Rule APPLIED here: lowest cross-fitted log loss only (as in the earlier country experiment).
    best_name = min(scored, key=lambda k: scored[k]["cross_fit_log_loss"])
    # Diagnostic: the main-analysis rule (simpler method preferred if cross-fitted Brier within 0.001).
    best_brier = scored[best_name]["cross_fit_brier_multiclass"]
    main_rule_name = next(n for n in ("uncalibrated", "temperature")
                          if scored[n]["cross_fit_brier_multiclass"] <= best_brier + 0.001)
    scored["_main_rule_check"] = {"selected_by_log_loss_only": best_name,
                                  "selected_by_main_rule_with_brier_tiebreak": main_rule_name,
                                  "same_selection": best_name == main_rule_name}
    return best_name, scored


def fit_final_temperature(proba_val, y_val):
    from scipy.optimize import minimize_scalar
    log_proba = np.log(np.clip(proba_val, 1e-9, 1))

    def nll_at_T(logT):
        T = np.exp(logT)
        scaled = log_proba / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        p = np.clip(p, 1e-9, 1)
        return -np.mean(np.log(p[np.arange(len(y_val)), y_val]))

    res = minimize_scalar(nll_at_T, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(T, proba):
    log_proba = np.log(np.clip(proba, 1e-9, 1))
    scaled = log_proba / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    p = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
    return p


# --------------------------------------------------------------------------
# 6. Metrics (identical definitions to modeling/train_models.py)
# --------------------------------------------------------------------------

def binary_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "auprc": float(average_precision_score(y_true, proba)),
        "auroc": float(roc_auc_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-9, 1 - 1e-9), labels=[0, 1])),
        "sensitivity": float(sens), "specificity": float(spec),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold_used": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def multiclass_metrics(y_true, y_pred):
    labels = list(range(len(MULTICLASS_CLASSES)))
    p, r, f1s, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy_secondary": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {MULTICLASS_CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                      "f1": float(f1s[i]), "support": int(sup[i])} for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def patient_bootstrap_ci_multi(y_true, pred_or_proba, public_ids, metric_fns, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    """metric_fns: dict of name -> fn(y_true_sub, pred_sub) -> float"""
    rng = np.random.RandomState(seed)
    uniq = np.unique(public_ids)
    patient_to_rows = {p: np.where(public_ids == p)[0] for p in uniq}
    out = {name: [] for name in metric_fns}
    for _ in range(n):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        row_idx = np.concatenate([patient_to_rows[p] for p in sampled])
        if len(row_idx) < 2:
            continue
        for name, fn in metric_fns.items():
            try:
                out[name].append(fn(y_true[row_idx], pred_or_proba[row_idx]))
            except (ValueError, ZeroDivisionError):
                continue
    result = {}
    for name, scores in out.items():
        scores = np.array(scores)
        result[name] = {
            "n_resamples_used": int(len(scores)),
            "mean": float(np.mean(scores)) if len(scores) else None,
            "ci_lower_2.5": float(np.percentile(scores, 2.5)) if len(scores) else None,
            "ci_upper_97.5": float(np.percentile(scores, 97.5)) if len(scores) else None,
        }
    return result


def _compare(all_results, loader):
    out = {}
    for task, key in (("binary", "auprc"), ("multiclass", "macro_f1")):
        prev = loader(task)
        if prev is None:
            out[task] = {"available": False}
            continue
        new = all_results["tasks"][task]
        diffs = {}
        for g in ("us_val", "spain_canada_pooled", "spain_only", "canada_only"):
            pm, nm = prev[f"{g}_metrics"], new[f"{g}_metrics"]
            diffs[g] = {"previous": pm[key], "new": nm[key], "abs_diff": abs(pm[key] - nm[key])}
            if g != "us_val":
                pc, nc = prev[f"{g}_bootstrap_ci"][key], new[f"{g}_bootstrap_ci"][key]
                diffs[g].update({"previous_ci": [pc["ci_lower_2.5"], pc["ci_upper_97.5"]],
                                 "new_ci": [nc["ci_lower_2.5"], nc["ci_upper_97.5"]]})
        out[task] = {"available": True, "primary_metric": key,
                     "params_previous": prev["params"], "params_new": new["params"],
                     "params_equal": prev["params"] == new["params"],
                     "calibration_previous": prev["calibration_method_selected_on_us_val"],
                     "calibration_new": new["calibration_method_selected_on_us_val"],
                     "threshold_previous": prev.get("threshold_selected_on_us_val"),
                     "threshold_new": new.get("threshold_selected_on_us_val"),
                     "n_train_previous": prev["n_train"], "n_train_new": new["n_train"],
                     "n_val_previous": prev["n_val"], "n_val_new": new["n_val"],
                     "metrics": diffs, "max_abs_diff_primary_metric": max(v["abs_diff"] for v in diffs.values())}
    return out


def compare_with_previous(all_results):
    """Compare with (1) the earlier country experiment (hyperparameters reused from the main analysis; read via
    `git show`) and (2) the US-tuned version (tuning restricted, fit on all of US-train). Read-only."""
    def earlier(task):
        raw = git_text(PREVIOUS_RESULTS_GIT[0], f"{PREVIOUS_RESULTS_GIT[1]}/{task}_c_xgboost_us_to_spain_canada_metrics.json")
        return json.loads(raw) if raw else None

    def ustuned(task):
        pth = f"{USTUNED_RESULTS_DIR}/{task}_c_xgboost_us_to_spain_canada_metrics.json"
        return json.load(open(pth, encoding="utf-8")) if os.path.exists(pth) else None

    return {"vs_earlier_experiment_hyperparameters_reused": {"source": "%s:%s" % PREVIOUS_RESULTS_GIT,
                                                             **_compare(all_results, earlier)},
            "vs_us_tuned_version_fit_on_all_us_train": {"source": USTUNED_RESULTS_DIR, **_compare(all_results, ustuned)}}


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    log("=" * 70)
    log("EXPLORATORY POST HOC WITHIN-COMMPASS ROBUSTNESS TEST (ORIGINAL-TRAIN-ONLY REVISION)")
    log("NOT external validation. NOT a pristine untouched test set.")
    log("=" * 70)

    df = load_data()
    df = define_eligibility(df)
    feature_cols = load_model_c_feature_list()

    split_map = make_us_split(df)
    df["rob_split"] = df["public_id"].map(split_map)
    # Spain+Canada: evaluation only, never touches fitting
    df.loc[df["country"].isin(["spain", "canada"]), "rob_split"] = "spain_canada_test"
    # ORIGINAL-TRAIN-ONLY: US patients in the original validation/test partitions take no part in any step.
    us_not_orig_train = df["rob_split"].isin(["us_train", "us_val"]) & (df["orig_split"] != "train")
    n_us_before = int(df.loc[df["rob_split"].isin(["us_train", "us_val"]), "public_id"].nunique())
    excluded_by_split = {sp: int(df.loc[us_not_orig_train & (df["orig_split"] == sp), "public_id"].nunique())
                         for sp in ("val", "test")}
    split_before = {k: int(df.loc[df["rob_split"] == k, "public_id"].nunique()) for k in ("us_train", "us_val")}
    excluded_from_fit_by_old_partition = {
        k: int(df.loc[us_not_orig_train & (df["rob_split"] == k), "public_id"].nunique()) for k in ("us_train", "us_val")}
    df.loc[us_not_orig_train, "rob_split"] = "us_excluded_not_original_train"
    # Italy: excluded entirely from rob_split (stays NaN -> never selected by any mask below)

    us_train_mask = df["rob_split"] == "us_train"
    us_val_mask = df["rob_split"] == "us_val"
    sc_mask = df["rob_split"] == "spain_canada_test"
    spain_mask = sc_mask & (df["country"] == "spain")
    canada_mask = sc_mask & (df["country"] == "canada")

    # ---- AUDIT: patient disjointness ----
    us_train_ids = set(df.loc[us_train_mask, "public_id"])
    us_val_ids = set(df.loc[us_val_mask, "public_id"])
    sc_ids = set(df.loc[sc_mask, "public_id"])
    italy_ids = set(df.loc[df["country"] == "italy", "public_id"])
    # Patients outside the original training partition (original validation + test):
    # never allowed into hyperparameter tuning.
    non_train_ids = set(df.loc[df["orig_split"] != "train", "public_id"])
    # Everything that must never reach tuning (also enforced inside select_best_params).
    banned_for_tuning = sc_ids | italy_ids | us_val_ids | non_train_ids
    excluded_us_ids = set(df.loc[df["rob_split"] == "us_excluded_not_original_train", "public_id"])
    audit = {
        # counts only: no patient identifier is written to any results file
        "development_population": "US patients in the original training partition only",
        "n_us_patients_total": n_us_before,
        "earlier_us_split_patients": split_before,
        "n_us_excluded_not_original_train": len(excluded_us_ids),
        "us_excluded_by_original_partition": excluded_by_split,
        "us_excluded_by_earlier_us_split": excluded_from_fit_by_old_partition,
        "n_fit_or_calibration_overlap_with_original_val_or_test": len((us_train_ids | us_val_ids) & non_train_ids),
        "n_excluded_us_overlap_with_any_step": len(excluded_us_ids & (us_train_ids | us_val_ids | sc_ids)),
        "n_us_train_val_overlap": len(us_train_ids & us_val_ids),
        "n_us_train_sc_overlap": len(us_train_ids & sc_ids),
        "n_us_val_sc_overlap": len(us_val_ids & sc_ids),
        "n_us_train_patients": len(us_train_ids),
        "n_us_val_patients": len(us_val_ids),
        "n_spain_canada_patients": len(sc_ids),
        "n_spain_patients": int(df.loc[spain_mask, "public_id"].nunique()),
        "n_canada_patients": int(df.loc[canada_mask, "public_id"].nunique()),
        "n_italy_patients_excluded": len(italy_ids),
        "id_set_sha256": {"us_train": id_digest(us_train_ids), "us_val": id_digest(us_val_ids),
                          "spain_canada_evaluation": id_digest(sc_ids)},
        "mode": "primary_original_train_only",
    }
    assert audit["n_fit_or_calibration_overlap_with_original_val_or_test"] == 0, "fit/calibration set has non-train patients!"
    assert audit["n_excluded_us_overlap_with_any_step"] == 0
    assert audit["n_us_patients_total"] == 771
    assert audit["n_us_train_val_overlap"] == 0, "US train/val patient overlap!"
    assert audit["n_us_train_sc_overlap"] == 0, "US train/Spain+Canada patient overlap!"
    assert audit["n_us_val_sc_overlap"] == 0, "US val/Spain+Canada patient overlap!"
    for k, v in (("n_us_excluded_not_original_train", EXPECTED["n_us_excluded_not_original_train"]),
                 ("n_us_train_patients", EXPECTED["n_us_train_patients"]),
                 ("n_us_val_patients", EXPECTED["n_us_val_patients"]),
                 ("n_spain_canada_patients", EXPECTED["n_spain_canada_patients"]),
                 ("n_spain_patients", EXPECTED["n_spain_patients"]),
                 ("n_canada_patients", EXPECTED["n_canada_patients"])):
        assert audit[k] == v, f"cohort definition changed: {k} = {audit[k]}, expected {v}"
    log(f"\nPatient overlap audit: PASS (0/0/0) - "
        f"{audit['n_us_train_patients']} US-train, {audit['n_us_val_patients']} US-val, "
        f"{audit['n_spain_canada_patients']} Spain+Canada patients")

    all_results = {"audit": audit, "environment": environment_record(), "tasks": {}}

    for task in ["binary", "multiclass"]:
        log(f"\n{'='*70}\nTASK: {task}\n{'='*70}")
        elig_col = "eligible_for_binary" if task == "binary" else "eligible_for_multiclass"
        elig_mask = df[elig_col]

        tr_mask = us_train_mask & elig_mask
        va_mask = us_val_mask & elig_mask
        sc_eval_mask = sc_mask & elig_mask
        spain_eval_mask = spain_mask & elig_mask
        canada_eval_mask = canada_mask & elig_mask

        # ---- preprocessing: fit on US-train ONLY ----
        preprocessor, buckets = fit_preprocessor_on_us_train(df, feature_cols, tr_mask)

        X_tr = transform(preprocessor, df, feature_cols, tr_mask)
        X_va = transform(preprocessor, df, feature_cols, va_mask)
        X_sc = transform(preprocessor, df, feature_cols, sc_eval_mask)
        X_spain = transform(preprocessor, df, feature_cols, spain_eval_mask)
        X_canada = transform(preprocessor, df, feature_cols, canada_eval_mask)

        unseen_report = audit_unseen_categories(preprocessor, buckets, df, sc_eval_mask, feature_cols)
        log(f"Unseen-category audit (Spain+Canada vs US-train categories): "
            f"{len(unseen_report)} column(s) with unseen values")
        for col, info in unseen_report.items():
            log(f"  {col}: {info['unseen_categories']} ({info['n_eval_rows_affected']} rows)")

        g_tr = df.loc[tr_mask, "public_id"].to_numpy()
        g_va = df.loc[va_mask, "public_id"].to_numpy()
        g_sc = df.loc[sc_eval_mask, "public_id"].to_numpy()
        g_spain = df.loc[spain_eval_mask, "public_id"].to_numpy()
        g_canada = df.loc[canada_eval_mask, "public_id"].to_numpy()
        pid_sc = df.loc[sc_eval_mask, "pair_id"].to_numpy()

        if task == "binary":
            y_tr = df.loc[tr_mask, BINARY_TARGET].to_numpy()
            y_va = df.loc[va_mask, BINARY_TARGET].to_numpy()
            y_sc = df.loc[sc_eval_mask, BINARY_TARGET].to_numpy()
            y_spain = df.loc[spain_eval_mask, BINARY_TARGET].to_numpy()
            y_canada = df.loc[canada_eval_mask, BINARY_TARGET].to_numpy()
        else:
            cls_index = {c: i for i, c in enumerate(MULTICLASS_CLASSES)}
            y_tr = df.loc[tr_mask, MULTICLASS_TARGET].map(cls_index).to_numpy()
            y_va = df.loc[va_mask, MULTICLASS_TARGET].map(cls_index).to_numpy()
            y_sc = df.loc[sc_eval_mask, MULTICLASS_TARGET].map(cls_index).to_numpy()
            y_spain = df.loc[spain_eval_mask, MULTICLASS_TARGET].map(cls_index).to_numpy()
            y_canada = df.loc[canada_eval_mask, MULTICLASS_TARGET].map(cls_index).to_numpy()

        # ---- hyperparameters: tuned on the tuning pool ONLY (US patients, original train partition,
        #      inside US-train). Spain/Canada/Italy/US-val/original-val/original-test never enter. ----
        tune_mask = define_tuning_pool(df, us_train_mask, elig_mask)
        tune_ids = set(df.loc[tune_mask, "public_id"])
        assert tune_ids <= us_train_ids, "tuning pool is not inside US-train"
        assert not (tune_ids & banned_for_tuning), "tuning pool contains a banned patient"
        assert set(df.loc[tune_mask, "country"]) == {"united_states"}
        assert set(df.loc[tune_mask, "orig_split"]) == {"train"}
        assert not (tune_ids & sc_ids) and not (tune_ids & us_val_ids) and not (tune_ids & italy_ids)
        in_tune = tune_mask[tr_mask].to_numpy()  # tuning rows within the US-train rows (tune_mask <= tr_mask)
        assert in_tune.sum() == tune_mask.sum()
        X_tune, y_tune, g_tune = X_tr[in_tune], y_tr[in_tune], g_tr[in_tune]
        assert set(g_tune) == tune_ids
        tuning = {
            "n_patients": len(tune_ids), "n_pairs": int(in_tune.sum()),
            "n_improvement_events": int(y_tune.sum()) if task == "binary" else None,
            "definition": ("fit set = US patients in the original training partition and in the earlier US-train; "
                           "task-eligible visit pairs; identical to the set used for preprocessing and the final fit"),
            "selection_metric": "AUPRC (binary) / macro-F1 (six-class), mean over 5 patient-grouped folds",
            "n_cv_folds": tm.N_CV_FOLDS, "grid": tm.get_param_grid(LOCKED_ALGORITHM),
            "id_set_sha256": id_digest(tune_ids),
            "n_overlap_with_spain_canada_evaluation": len(tune_ids & sc_ids),
            "n_overlap_with_us_val": len(tune_ids & us_val_ids),
            "n_overlap_with_italy": len(tune_ids & italy_ids),
            "n_overlap_with_original_val_or_test": len(tune_ids & non_train_ids),
            "us_train_patients_not_in_tuning_pool": len(us_train_ids - tune_ids),
            "us_train_composition_by_original_split": {
                s: int(df.loc[us_train_mask & (df["orig_split"] == s), "public_id"].nunique())
                for s in ("train", "val", "test")},
        }
        params, cv_results = tm.select_best_params(
            X_tune, y_tune, g_tune, LOCKED_ALGORITHM, task, banned_groups=banned_for_tuning)
        tuning["performed"] = True
        assert tune_ids == set(g_tr), "tuning pool must equal the fit set (task-eligible patients)"
        tuning["cv_results"] = cv_results
        tuning["selected_params"] = params
        tuning["previous_params_reused_from_main_analysis"] = PREVIOUS_PARAMS[task]
        tuning["selected_params_equal_previous"] = bool(params == PREVIOUS_PARAMS[task])
        log(f"Tuning pool: {tuning['n_patients']} patients, {tuning['n_pairs']} pairs; "
            f"selected {params} (equal to previous: {tuning['selected_params_equal_previous']})")

        # ---- final fit on the fit set (original-train US patients only) ----
        model = build_model(task, params)
        model.fit(X_tr, y_tr)
        joblib.dump(model, f"{MODELS_DIR}/model_c_{task}_xgboost_origtrain_only.joblib")
        joblib.dump(preprocessor, f"{MODELS_DIR}/model_c_{task}_preprocessor_origtrain_only.joblib")

        # ---- calibration: select on US-val ONLY via cross-fitting, freeze ----
        if task == "binary":
            proba_va = model.predict_proba(X_va)[:, 1]
            method, cf_scores = cross_fit_calibration_binary(y_va, proba_va, g_va)
            calibrator = fit_final_calibrator_binary(method, proba_va, y_va)
            log(f"Calibration selected on US-val (binary): {method}  "
                f"(cross-fit scores: {cf_scores})")

            proba_va_cal = apply_calibrator_binary(method, calibrator, proba_va)
            threshold = select_threshold(y_va, proba_va_cal)

            proba_sc = apply_calibrator_binary(method, calibrator, model.predict_proba(X_sc)[:, 1])
            proba_spain = apply_calibrator_binary(method, calibrator, model.predict_proba(X_spain)[:, 1])
            proba_canada = apply_calibrator_binary(method, calibrator, model.predict_proba(X_canada)[:, 1])

            val_metrics = binary_metrics(y_va, proba_va_cal, threshold)
            pooled_metrics = binary_metrics(y_sc, proba_sc, threshold)
            spain_metrics = binary_metrics(y_spain, proba_spain, threshold) if len(y_spain) else None
            canada_metrics = binary_metrics(y_canada, proba_canada, threshold) if len(y_canada) else None

            metric_fns = {
                "auprc": lambda a, b: average_precision_score(a, b),
                "auroc": lambda a, b: roc_auc_score(a, b),
                "brier_score": lambda a, b: brier_score_loss(a, b),
                "sensitivity": lambda a, b: (
                    confusion_matrix(a, (b >= threshold).astype(int), labels=[0, 1])[1, 1]
                    / max(confusion_matrix(a, (b >= threshold).astype(int), labels=[0, 1])[1, :].sum(), 1)
                ),
                "specificity": lambda a, b: (
                    confusion_matrix(a, (b >= threshold).astype(int), labels=[0, 1])[0, 0]
                    / max(confusion_matrix(a, (b >= threshold).astype(int), labels=[0, 1])[0, :].sum(), 1)
                ),
            }
            ci_pooled = patient_bootstrap_ci_multi(y_sc, proba_sc, g_sc, metric_fns, n=N_BOOTSTRAP)
            ci_spain = patient_bootstrap_ci_multi(y_spain, proba_spain, g_spain, metric_fns, n=N_BOOTSTRAP) if len(y_spain) else None
            ci_canada = patient_bootstrap_ci_multi(y_canada, proba_canada, g_canada, metric_fns, n=N_BOOTSTRAP) if len(y_canada) else None

            preds_df = pd.DataFrame({
                "pair_id": pid_sc, "public_id": g_sc, "country": df.loc[sc_eval_mask, "country"].to_numpy(),
                "y_true": y_sc, "proba_calibrated": proba_sc, "pred": (proba_sc >= threshold).astype(int),
            })

            task_result = {
                "algorithm": LOCKED_ALGORITHM, "params": params, "tuning": tuning,
                "params_source": "tuned on the fit set: US patients in the original training partition (see tuning)",
                "calibration_method_selected_on_us_val": method,
                "calibration_cross_fit_scores": cf_scores,
                "threshold_selected_on_us_val": threshold,
                "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                "n_patients_train": int(len(set(g_tr))), "n_patients_val": int(len(set(g_va))),
                "n_test_pooled": int(len(y_sc)), "n_patients_test_pooled": int(len(set(g_sc))),
                "n_test_spain": int(len(y_spain)), "n_patients_test_spain": int(len(set(g_spain))),
                "n_test_canada": int(len(y_canada)), "n_patients_test_canada": int(len(set(g_canada))),
                "improvement_events_train": int(y_tr.sum()), "improvement_events_val": int(y_va.sum()),
                "improvement_events_pooled": int(y_sc.sum()),
                "improvement_events_spain": int(y_spain.sum()) if len(y_spain) else None,
                "improvement_events_canada": int(y_canada.sum()) if len(y_canada) else None,
                "us_val_metrics": val_metrics,
                "spain_canada_pooled_metrics": pooled_metrics,
                "spain_canada_pooled_bootstrap_ci": ci_pooled,
                "spain_only_metrics": spain_metrics,
                "spain_only_bootstrap_ci": ci_spain,
                "canada_only_metrics": canada_metrics,
                "canada_only_bootstrap_ci": ci_canada,
                "unseen_category_audit": unseen_report,
            }
            preds_df.to_csv(f"{results_dir}/binary_c_xgboost_us_to_spain_canada_predictions.csv", index=False)

        else:
            proba_va = model.predict_proba(X_va)
            method, cf_scores = cross_fit_calibration_multiclass(y_va, proba_va, g_va)
            log(f"Calibration selected on US-val (multiclass): {method}  "
                f"(cross-fit scores: {cf_scores})")
            T = fit_final_temperature(proba_va, y_va) if method == "temperature" else None

            def apply_mc(proba):
                return apply_temperature(T, proba) if method == "temperature" else proba

            proba_va_cal = apply_mc(proba_va)
            pred_va = proba_va_cal.argmax(axis=1)
            val_metrics = multiclass_metrics(y_va, pred_va)

            proba_sc = apply_mc(model.predict_proba(X_sc))
            proba_spain = apply_mc(model.predict_proba(X_spain))
            proba_canada = apply_mc(model.predict_proba(X_canada))
            pred_sc, pred_spain, pred_canada = (
                proba_sc.argmax(axis=1),
                proba_spain.argmax(axis=1) if len(y_spain) else np.array([]),
                proba_canada.argmax(axis=1) if len(y_canada) else np.array([]),
            )

            pooled_metrics = multiclass_metrics(y_sc, pred_sc)
            spain_metrics = multiclass_metrics(y_spain, pred_spain) if len(y_spain) else None
            canada_metrics = multiclass_metrics(y_canada, pred_canada) if len(y_canada) else None

            metric_fns = {
                "macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
                "weighted_f1": lambda a, b: f1_score(a, b, average="weighted", zero_division=0),
                "accuracy_secondary": lambda a, b: accuracy_score(a, b),
            }
            ci_pooled = patient_bootstrap_ci_multi(y_sc, pred_sc, g_sc, metric_fns, n=N_BOOTSTRAP)
            ci_spain = patient_bootstrap_ci_multi(y_spain, pred_spain, g_spain, metric_fns, n=N_BOOTSTRAP) if len(y_spain) else None
            ci_canada = patient_bootstrap_ci_multi(y_canada, pred_canada, g_canada, metric_fns, n=N_BOOTSTRAP) if len(y_canada) else None

            preds_df = pd.DataFrame({
                "pair_id": pid_sc, "public_id": g_sc, "country": df.loc[sc_eval_mask, "country"].to_numpy(),
                "y_true": [MULTICLASS_CLASSES[i] for i in y_sc],
                "pred": [MULTICLASS_CLASSES[i] for i in pred_sc],
            })

            task_result = {
                "algorithm": LOCKED_ALGORITHM, "params": params, "tuning": tuning,
                "params_source": "tuned on the fit set: US patients in the original training partition (see tuning)",
                "calibration_method_selected_on_us_val": method,
                "calibration_cross_fit_scores": cf_scores,
                "temperature_fitted_on_us_val": T,
                "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                "n_patients_train": int(len(set(g_tr))), "n_patients_val": int(len(set(g_va))),
                "n_test_pooled": int(len(y_sc)), "n_patients_test_pooled": int(len(set(g_sc))),
                "n_test_spain": int(len(y_spain)), "n_patients_test_spain": int(len(set(g_spain))),
                "n_test_canada": int(len(y_canada)), "n_patients_test_canada": int(len(set(g_canada))),
                "class_order": MULTICLASS_CLASSES,
                "us_val_metrics": val_metrics,
                "spain_canada_pooled_metrics": pooled_metrics,
                "spain_canada_pooled_bootstrap_ci": ci_pooled,
                "spain_only_metrics": spain_metrics,
                "spain_only_bootstrap_ci": ci_spain,
                "canada_only_metrics": canada_metrics,
                "canada_only_bootstrap_ci": ci_canada,
                "unseen_category_audit": unseen_report,
            }
            preds_df.to_csv(f"{results_dir}/multiclass_c_xgboost_us_to_spain_canada_predictions.csv", index=False)

        # evaluation cohort definition must not depend on tuning
        assert len(y_sc) == EXPECTED["pairs_spain_canada"][task],             f"{task}: Spain+Canada pairs {len(y_sc)} != {EXPECTED['pairs_spain_canada'][task]}"
        assert set(g_sc) == sc_ids and not (set(g_sc) & tune_ids)
        task_result["evaluation_cohort_id_sha256"] = id_digest(set(g_sc))
        task_result["tuning_evaluation_patient_overlap"] = len(set(g_sc) & tune_ids)

        all_results["tasks"][task] = task_result
        with open(f"{results_dir}/{task}_c_xgboost_us_to_spain_canada_metrics.json", "w") as f:
            json.dump(task_result, f, indent=2, default=str)
        log(f"\nSaved: {results_dir}/{task}_c_xgboost_us_to_spain_canada_metrics.json")

    all_results["comparison_to_previous_run"] = compare_with_previous(all_results)
    with open(f"{results_dir}/full_run_audit_and_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved: {results_dir}/full_run_audit_and_summary.json")
    log("\nDONE. This is an exploratory, post hoc, within-CoMMpass robustness test.")
    log("It is NOT external validation and NOT a pristine untouched test set.")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Original-train-only geographic robustness analysis (see module docstring)").parse_args()
    main()
