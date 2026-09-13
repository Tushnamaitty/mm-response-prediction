"""
Step 13 (PRELIMINARY): Train/evaluate Models A-D, binary + multiclass.
Results are preliminary, not final.

Frozen inputs read, never modified: Steps 1-12 outputs, split, eligibility,
preprocessors.

This run only replaces the 8 Logistic Regression combinations. The
RNA-specific XGBoost analyses (Model D without rna_available, freshness
sensitivity, RNA-subset A-D comparison) and the imbalance/heterogeneity
reports are NOT rerun here - their existing files are untouched.
"""

import json, os, sys, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    confusion_matrix, precision_recall_fscore_support, accuracy_score,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", category=ConvergenceWarning)  # keep these visible

# preprocessors were pickled under __main__, so patch it before loading
sys.path.insert(0, "data_pipeline")
import build_preprocessing as _bp
sys.modules["__main__"].OrdinalResponseEncoder = _bp.OrdinalResponseEncoder
sys.modules["__main__"]._cast_to_int = _bp._cast_to_int
sys.modules["__main__"]._stringify_preserving_na = _bp._stringify_preserving_na

CLINICAL_DIR = "data/clinical"
SPLITS_DIR = "data/splits"
PREPROC_DIR = "artifacts/preprocessing"
MODELS_DIR = "artifacts/models"
RESULTS_DIR = "artifacts/results"

RANDOM_SEED = 42
N_CV_FOLDS = 5
N_BOOTSTRAP = 200
RNA_INDICATOR_COL = "rna_available"
STALENESS_COL = "days_since_rna_sample"
MODEL_STAGES = ["a", "b", "c", "d"]
ALGORITHMS = ["logistic", "lightgbm", "xgboost"]

BINARY_TARGET = "improved"
MULTICLASS_TARGET = "exact_next_response"
MULTICLASS_CLASSES = [
    "progressive_disease", "stable_disease", "partial_response",
    "very_good_partial_response", "complete_response", "stringent_complete_response",
]

FRESHNESS_CUTOFFS_DAYS = [365, 545, 730]  # 1yr, 1.5yr, 2yr - not used this run


def already_done(key):
    return (os.path.exists(f"{MODELS_DIR}/{key}.joblib")
            and os.path.exists(f"{RESULTS_DIR}/{key}_metrics.json")
            and os.path.exists(f"{RESULTS_DIR}/{key}_predictions.csv"))


def load_master_data() -> pd.DataFrame:
    df = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    splits = pd.read_csv(f"{SPLITS_DIR}/visit_pair_splits.csv")
    elig = pd.read_csv(f"{SPLITS_DIR}/task_eligibility.csv")
    df = df.merge(splits[["pair_id", "split"]], on="pair_id", how="left")
    df = df.merge(elig[["pair_id", "eligible_for_binary", "eligible_for_multiclass"]],
                  on="pair_id", how="left")
    assert df["split"].notna().all()
    assert df["eligible_for_binary"].notna().all()
    return df


def load_preprocessor_and_meta(stage: str):
    prep = joblib.load(f"{PREPROC_DIR}/model_{stage}_preprocessor.joblib")
    with open(f"{PREPROC_DIR}/model_{stage}_metadata.json") as f:
        meta = json.load(f)
    return prep, meta


def build_feature_matrix(df, stage, preprocessor, meta):
    work_df = df.copy()
    cols = list(meta["input_columns"])
    if stage == "d":
        rna_cols = meta["rna_columns_imputed"]
        work_df[RNA_INDICATOR_COL] = work_df[rna_cols[0]].notna().astype(int)
    X = preprocessor.transform(work_df[cols])
    return np.asarray(X), list(preprocessor.get_feature_names_out())


def make_group_folds(public_ids, n_folds=N_CV_FOLDS):
    gkf = GroupKFold(n_splits=n_folds)
    dummy_y = np.zeros(len(public_ids))
    return list(gkf.split(dummy_y, dummy_y, groups=public_ids))


def get_param_grid(algorithm):
    if algorithm == "logistic":
        return [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]
    if algorithm == "lightgbm":
        return [
            {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.05},
            {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05},
            {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.1},
        ]
    if algorithm == "xgboost":
        return [
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        ]
    raise ValueError(algorithm)


def build_model(algorithm, task, params):
    if algorithm == "logistic":
        # scaling (applied at call sites) fixes the real convergence
        # issue, so standard iter/tol are fine now
        return LogisticRegression(max_iter=1000, tol=1e-4, random_state=RANDOM_SEED, **params)
    if algorithm == "lightgbm":
        import lightgbm as lgb
        objective = "binary" if task == "binary" else "multiclass"
        return lgb.LGBMClassifier(objective=objective, random_state=RANDOM_SEED,
                                   verbosity=-1, **params)
    if algorithm == "xgboost":
        import xgboost as xgb
        objective = "binary:logistic" if task == "binary" else "multi:softprob"
        metric = "logloss" if task == "binary" else "mlogloss"
        return xgb.XGBClassifier(objective=objective, random_state=RANDOM_SEED,
                                  eval_metric=metric, **params)
    raise ValueError(algorithm)


def select_best_params(X, y, groups, algorithm, task):
    grid = get_param_grid(algorithm)
    folds = make_group_folds(groups)
    scored = []
    for params in grid:
        fold_scores = []
        for tr_idx, va_idx in folds:
            X_tr_fold, X_va_fold = X[tr_idx], X[va_idx]
            if algorithm == "logistic":
                scaler = StandardScaler().fit(X_tr_fold)
                X_tr_fold = scaler.transform(X_tr_fold)
                X_va_fold = scaler.transform(X_va_fold)
            m = build_model(algorithm, task, params)
            m.fit(X_tr_fold, y[tr_idx])
            if task == "binary":
                proba = m.predict_proba(X_va_fold)[:, 1]
                score = average_precision_score(y[va_idx], proba)
            else:
                pred = m.predict(X_va_fold)
                score = f1_score(y[va_idx], pred, average="macro", zero_division=0)
            fold_scores.append(score)
        scored.append({"params": params, "mean": float(np.mean(fold_scores)),
                        "std": float(np.std(fold_scores)),
                        "fold_scores": [float(s) for s in fold_scores]})
    best = max(scored, key=lambda r: r["mean"])
    return best["params"], scored


def select_threshold(y_true, proba):
    candidates = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in candidates:
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def binary_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "auprc": float(average_precision_score(y_true, proba)),
        "auroc": float(roc_auc_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
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


def patient_bootstrap_ci(y_true, pred_or_proba, public_ids, metric_fn, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    uniq = np.unique(public_ids)
    # precompute each patient's row indices once
    patient_to_rows = {p: np.where(public_ids == p)[0] for p in uniq}
    scores = []
    for _ in range(n):
        sampled_patients = rng.choice(uniq, size=len(uniq), replace=True)
        # concatenate with repeats - if a patient is drawn twice, their
        # rows appear twice in this replicate, preserving true multiplicity
        row_idx = np.concatenate([patient_to_rows[p] for p in sampled_patients])
        if len(row_idx) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[row_idx], pred_or_proba[row_idx]))
        except (ValueError, ZeroDivisionError):
            continue
    scores = np.array(scores)
    return {"n_resamples_used": int(len(scores)),
            "mean": float(np.mean(scores)) if len(scores) else None,
            "ci_lower_2.5": float(np.percentile(scores, 2.5)) if len(scores) else None,
            "ci_upper_97.5": float(np.percentile(scores, 97.5)) if len(scores) else None}


def get_feature_importance(model, algorithm, feature_names):
    if algorithm == "logistic":
        coefs = model.coef_[0] if model.coef_.shape[0] == 1 else np.abs(model.coef_).mean(axis=0)
        vals = np.abs(coefs)
    elif hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        return {}
    order = np.argsort(vals)[::-1]
    return {feature_names[i]: float(vals[i]) for i in order[:30]}


def get_binary_xyz(df, stage, preprocessor, meta, split):
    eligible = df[df["eligible_for_binary"]]
    sub = eligible[eligible["split"] == split]
    X, feat_names = build_feature_matrix(sub, stage, preprocessor, meta)
    y = sub[BINARY_TARGET].to_numpy()
    groups = sub["public_id"].to_numpy()
    pair_ids = sub["pair_id"].to_numpy()
    return X, y, groups, pair_ids, feat_names


def get_multiclass_xyz(df, stage, preprocessor, meta, split):
    sub = df[df["split"] == split]
    X, feat_names = build_feature_matrix(sub, stage, preprocessor, meta)
    y = np.array([MULTICLASS_CLASSES.index(v) for v in sub[MULTICLASS_TARGET]])
    groups = sub["public_id"].to_numpy()
    pair_ids = sub["pair_id"].to_numpy()
    return X, y, groups, pair_ids, feat_names


def run_one(df, task, stage, algorithm, row_filter=None, tag=""):
    preprocessor, meta = load_preprocessor_and_meta(stage)
    work_df = df if row_filter is None else df[row_filter]

    getter = get_binary_xyz if task == "binary" else get_multiclass_xyz
    X_tr, y_tr, g_tr, pid_tr, feat_names = getter(work_df, stage, preprocessor, meta, "train")
    X_va, y_va, g_va, pid_va, _ = getter(work_df, stage, preprocessor, meta, "val")
    X_te, y_te, g_te, pid_te, _ = getter(work_df, stage, preprocessor, meta, "test")

    X_trv = np.vstack([X_tr, X_va]); y_trv = np.concatenate([y_tr, y_va]); g_trv = np.concatenate([g_tr, g_va])
    best_params, cv_results = select_best_params(X_trv, y_trv, g_trv, algorithm, task)

    # X_va/X_te stay UNSCALED here - if logistic, model is a Pipeline that
    # scales internally on each predict call, exactly once.
    if algorithm == "logistic":
        scaler = StandardScaler().fit(X_tr)
        clf = build_model(algorithm, task, best_params)
        clf.fit(scaler.transform(X_tr), y_tr)
        model = Pipeline([("scale", scaler), ("clf", clf)])
    else:
        model = build_model(algorithm, task, best_params)
        model.fit(X_tr, y_tr)

    result = {
        "task": task, "model_stage": stage, "algorithm": algorithm, "tag": tag,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
        "n_patients_train": int(len(set(g_tr))), "n_patients_val": int(len(set(g_va))),
        "n_patients_test": int(len(set(g_te))),
        "best_params": best_params, "cv_results": cv_results, "seed": RANDOM_SEED,
    }

    if task == "binary":
        val_proba = model.predict_proba(X_va)[:, 1]
        threshold = select_threshold(y_va, val_proba)
        result["val_metrics"] = binary_metrics(y_va, val_proba, threshold)
        test_proba = model.predict_proba(X_te)[:, 1]
        test_m = binary_metrics(y_te, test_proba, threshold); test_m["PRELIMINARY"] = True
        result["test_metrics_PRELIMINARY"] = test_m
        result["test_auprc_bootstrap_ci_PRELIMINARY"] = patient_bootstrap_ci(
            y_te, test_proba, g_te, lambda a, b: average_precision_score(a, b))
        pt, pp = calibration_curve(y_te, test_proba, n_bins=10, strategy="quantile")
        result["calibration_curve"] = {"prob_true": pt.tolist(), "prob_pred": pp.tolist()}
        preds_df = pd.DataFrame({"pair_id": pid_te, "public_id": g_te, "y_true": y_te,
                                  "proba": test_proba, "pred": (test_proba >= threshold).astype(int)})
    else:
        val_pred = model.predict(X_va)
        result["val_metrics"] = multiclass_metrics(y_va, val_pred)
        test_pred = model.predict(X_te)
        test_m = multiclass_metrics(y_te, test_pred); test_m["PRELIMINARY"] = True
        result["test_metrics_PRELIMINARY"] = test_m
        result["test_macro_f1_bootstrap_ci_PRELIMINARY"] = patient_bootstrap_ci(
            y_te, test_pred, g_te, lambda a, b: f1_score(a, b, average="macro", zero_division=0))
        result["class_order"] = MULTICLASS_CLASSES
        preds_df = pd.DataFrame({"pair_id": pid_te, "public_id": g_te,
                                  "y_true": [MULTICLASS_CLASSES[i] for i in y_te],
                                  "pred": [MULTICLASS_CLASSES[i] for i in test_pred]})

    importance_source = model.named_steps["clf"] if algorithm == "logistic" else model
    result["feature_importance_top30"] = get_feature_importance(importance_source, algorithm, feat_names)
    if stage == "d" and STALENESS_COL in result["feature_importance_top30"]:
        result["days_since_rna_sample_importance"] = result["feature_importance_top30"][STALENESS_COL]

    return model, result, preds_df


def rebuild_summary():
    summary_rows = []
    for task in ["binary", "multiclass"]:
        for stage in MODEL_STAGES:
            for algo in ALGORITHMS:
                key = f"{task}_{stage}_{algo}"
                path = f"{RESULTS_DIR}/{key}_metrics.json"
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    result = json.load(f)
                main_val = result["val_metrics"]["auprc"] if task == "binary" else result["val_metrics"]["macro_f1"]
                main_test = result["test_metrics_PRELIMINARY"]["auprc"] if task == "binary" else result["test_metrics_PRELIMINARY"]["macro_f1"]
                summary_rows.append({
                    "task": task, "model_stage": stage, "algorithm": algo,
                    "n_train": result["n_train"], "n_val": result["n_val"], "n_test": result["n_test"],
                    "n_patients_train": result["n_patients_train"], "n_patients_val": result["n_patients_val"],
                    "n_patients_test": result["n_patients_test"],
                    "val_main_metric": main_val, "test_main_metric_PRELIMINARY": main_test,
                    "best_params": json.dumps(result["best_params"]), "seed": RANDOM_SEED,
                })
    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_master_data()

    print("STEP 13 - Logistic Regression rerun only (scaling fix)")

    for task in ["binary", "multiclass"]:
        for stage in MODEL_STAGES:
            for algo in ALGORITHMS:
                key = f"{task}_{stage}_{algo}"
                if already_done(key):
                    print(f"SKIP (already done): {key}")
                    continue
                print(f"{task} | {stage} | {algo}")
                model, result, preds_df = run_one(df, task, stage, algo)
                joblib.dump(model, f"{MODELS_DIR}/{key}.joblib")
                with open(f"{RESULTS_DIR}/{key}_metrics.json", "w") as f:
                    json.dump(result, f, indent=2)
                preds_df.to_csv(f"{RESULTS_DIR}/{key}_predictions.csv", index=False)

    rebuild_summary().to_csv(f"{RESULTS_DIR}/summary_PRELIMINARY.csv", index=False)

    print("DONE. RNA-specific analyses and imbalance/heterogeneity reports were NOT touched this run.")