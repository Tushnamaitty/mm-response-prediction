"""
EXPLORATORY, POST HOC, WITHIN-COMMPASS ROBUSTNESS TEST.
NOT external validation. NOT a pristine untouched test set (this is a new,
purpose-built split invented after seeing the main results, specifically to
probe country generalization).

Trains the LOCKED Model C (XGBoost, both tasks; hyperparameters and
calibration method taken from artifacts/results_clean_rerun and
artifacts/results_clean_rerun/calibration/calibration_summary.json)
on US patients ONLY, and evaluates it ONCE on Spain+Canada (pooled and
per-country). Italy is excluded from fitting and evaluation entirely
(see audit in conversation: single site, 2015-16 only, non-standard
followup_1-4 visit labels, ~3x longer visit gaps, and a documented
same-day-duplicate pattern that drops 20/218 pairs via the existing
zero/negative-gap filter - none of this is a coding bug, but it makes
Italy unusable as a like-for-like country comparison).

All preprocessing (imputation, one-hot encoding) and the model itself
are fit ONLY on US-train. Calibration candidates are selected via
patient-grouped cross-fitting on US-val ONLY, then frozen. Spain+Canada
data touches the pipeline exactly once, at final evaluation.

Outputs go to artifacts_robustness_country/ - a directory the locked
clean-rerun pipeline (artifacts/results_clean_rerun/, etc.) never reads
from or writes to. Nothing under artifacts/ or artifacts/results_clean_rerun/
is modified by this script.
"""

import json
import os
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
import build_preprocessing as bp  # noqa: E402

CLINICAL_DIR = "data/clinical"
OUT_DIR = "artifacts_robustness_country"
RESULTS_DIR = f"{OUT_DIR}/results"
MODELS_DIR = f"{OUT_DIR}/models"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
US_TRAIN_FRAC = 0.82  # matches original ~70/15 relative train:val ratio (70/85=0.824)

MULTICLASS_TARGET = "exact_next_response"
BINARY_TARGET = "improved"
MULTICLASS_CLASSES = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]

# ---- LOCKED MODEL C DEFINITION (identified from repo, see accompanying report) ----
LOCKED_ALGORITHM = "xgboost"
LOCKED_PARAMS = {
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
#    on US-val ONLY, pick per the locked selection rule, freeze)
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
    scored["uncalibrated"] = {"cross_fit_log_loss": float(log_loss(y_val, oof_uncal, labels=list(range(6))))}
    scored["temperature"] = {"cross_fit_log_loss": float(log_loss(y_val, oof_temp, labels=list(range(6))))}
    best_name = min(scored, key=lambda k: scored[k]["cross_fit_log_loss"])
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


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    log("=" * 70)
    log("EXPLORATORY POST HOC WITHIN-COMMPASS ROBUSTNESS TEST")
    log("NOT external validation. NOT a pristine untouched test set.")
    log("=" * 70)

    df = load_data()
    df = define_eligibility(df)
    feature_cols = load_model_c_feature_list()

    split_map = make_us_split(df)
    df["rob_split"] = df["public_id"].map(split_map)
    # Spain+Canada: evaluation only, never touches fitting
    df.loc[df["country"].isin(["spain", "canada"]), "rob_split"] = "spain_canada_test"
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
    audit = {
        "us_train_val_overlap": sorted(us_train_ids & us_val_ids),
        "us_train_sc_overlap": sorted(us_train_ids & sc_ids),
        "us_val_sc_overlap": sorted(us_val_ids & sc_ids),
        "n_us_train_patients": len(us_train_ids),
        "n_us_val_patients": len(us_val_ids),
        "n_spain_canada_patients": len(sc_ids),
    }
    assert not audit["us_train_val_overlap"], "US train/val patient overlap!"
    assert not audit["us_train_sc_overlap"], "US train/Spain+Canada patient overlap!"
    assert not audit["us_val_sc_overlap"], "US val/Spain+Canada patient overlap!"
    log(f"\nPatient overlap audit: PASS (0/0/0) - "
        f"{audit['n_us_train_patients']} US-train, {audit['n_us_val_patients']} US-val, "
        f"{audit['n_spain_canada_patients']} Spain+Canada patients")

    all_results = {"audit": audit, "tasks": {}}

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

        # ---- fit model on US-train ONLY, locked hyperparameters, no re-tuning ----
        model = build_model(task, LOCKED_PARAMS[task])
        model.fit(X_tr, y_tr)
        joblib.dump(model, f"{MODELS_DIR}/model_c_{task}_xgboost_us_only.joblib")
        joblib.dump(preprocessor, f"{MODELS_DIR}/model_c_{task}_preprocessor_us_only.joblib")

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
                "algorithm": LOCKED_ALGORITHM, "params": LOCKED_PARAMS[task],
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
            preds_df.to_csv(f"{RESULTS_DIR}/binary_c_xgboost_us_to_spain_canada_predictions.csv", index=False)

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
                "algorithm": LOCKED_ALGORITHM, "params": LOCKED_PARAMS[task],
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
            preds_df.to_csv(f"{RESULTS_DIR}/multiclass_c_xgboost_us_to_spain_canada_predictions.csv", index=False)

        all_results["tasks"][task] = task_result
        with open(f"{RESULTS_DIR}/{task}_c_xgboost_us_to_spain_canada_metrics.json", "w") as f:
            json.dump(task_result, f, indent=2, default=str)
        log(f"\nSaved: {RESULTS_DIR}/{task}_c_xgboost_us_to_spain_canada_metrics.json")

    with open(f"{RESULTS_DIR}/full_run_audit_and_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved: {RESULTS_DIR}/full_run_audit_and_summary.json")
    log("\nDONE. This is an exploratory, post hoc, within-CoMMpass robustness test.")
    log("It is NOT external validation and NOT a pristine untouched test set.")


if __name__ == "__main__":
    main()
