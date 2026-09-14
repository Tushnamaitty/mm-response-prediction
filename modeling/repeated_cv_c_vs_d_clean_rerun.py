"""
Repeated patient-grouped CV robustness check: does Model D (C + RNA)
outperform Model C, beyond what the single locked test-set split shows?

CV UNIVERSE IS RESTRICTED TO train+val PATIENTS ONLY. The original
held-out test split is loaded from data/splits/visit_pair_splits.csv
and explicitly excluded before any fold is built - test patients never
enter CV training or evaluation. This keeps the locked primary test
set fully untouched and reserved for the primary A/B/C/D result, while
this repeated-CV analysis draws its own, separate evidence from the
train+val pool only.

5-fold x 5-repeat deterministic patient-grouped CV, pooled over
train+val patients. This is a separate robustness analysis for the RNA
sub-question - it does NOT re-report or alter the locked primary
results.

Preprocessing is refit INSIDE each fold's training rows only (reusing
classify_columns/build_preprocessor from build_preprocessing.py - same
logic as the frozen Step 12 preprocessors, just fold-scoped). No new
hyperparameter tuning: reuses the locked clean-rerun C/D XGBoost
best_params.

Each fold trains C and D on all of that fold's training rows, but the
C-vs-D comparison is evaluated only on that fold's RNA-available
patients (same convention as rna_subset_paired_analysis.py).

Treated as a ROBUSTNESS ANALYSIS, not a significance test: no p-values,
no formal power calculation. An APPROXIMATE, EXPLICITLY-LABELED
empirical detectable-effect scale is reported instead, derived from the
observed fold-level D-C variability - this is a descriptive heuristic
about dispersion, not a formal MDE or statistical power calculation,
and the 25 replicates are not fully independent (shared patient pool
across repeats).
"""

import json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, "modeling")
sys.path.insert(0, "data_pipeline")
from train_models import build_model, RANDOM_SEED, MULTICLASS_CLASSES, STALENESS_COL
from build_preprocessing import classify_columns, build_preprocessor

CLINICAL_DIR = "data/clinical"
SPLITS_DIR = "data/splits"
CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
FEATURE_SETS_PATH = "data_pipeline/model_feature_sets.json"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/rna_ablation"

N_FOLDS = 5
N_REPEATS = 5
RNA_INDICATOR_COL = "rna_available"
BINARY_TARGET = "improved"
MULTICLASS_TARGET = "exact_next_response"
N_FOLD_BOOTSTRAP = 1000  # bootstrap over the 25 fold-level differences, not patients


def load_data_and_eligibility():
    df = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    elig = pd.read_csv(f"{SPLITS_DIR}/task_eligibility.csv")
    splits = pd.read_csv(f"{SPLITS_DIR}/visit_pair_splits.csv")
    df = df.merge(elig[["pair_id", "eligible_for_binary"]], on="pair_id", how="left")
    df = df.merge(splits[["pair_id", "split"]], on="pair_id", how="left")
    assert df["eligible_for_binary"].notna().all()
    assert df["split"].notna().all(), "Some rows have no split assignment."

    # --- Exclude the original held-out test split from the CV universe ---
    n_before = len(df)
    df = df[df["split"] != "test"].reset_index(drop=True)
    n_after = len(df)
    print(f"Excluded {n_before - n_after} test-split rows from the repeated-CV universe "
          f"({n_after} train+val rows remain).")
    assert (df["split"] != "test").all(), "Test-split rows leaked into the repeated-CV universe."
    assert set(df["split"].unique()) <= {"train", "val"}, \
        f"Unexpected split value(s) in repeated-CV universe: {set(df['split'].unique())}"
    return df


def make_repeat_folds(unique_patients, n_folds, seed):
    """Deterministic shuffled partition into n_folds disjoint patient groups."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_patients)
    folds = np.array_split(shuffled, n_folds)
    # --- Assert: full coverage, no overlap, no duplicates ---
    all_ids = np.concatenate(folds)
    assert len(all_ids) == len(unique_patients), "Fold split lost or duplicated patients."
    assert len(set(all_ids)) == len(unique_patients), "Duplicate patient across folds."
    return folds


def fit_fold_preprocessor(df, predictor_cols, train_mask, is_model_d, rna_cols=None):
    work_df = df.copy()
    cols_to_use = list(predictor_cols)
    if is_model_d:
        assert rna_cols, "rna_cols required for Model D fold preprocessing."
        work_df[RNA_INDICATOR_COL] = work_df[rna_cols[0]].notna().astype(int)
        cols_to_use = cols_to_use + [RNA_INDICATOR_COL]

    buckets = classify_columns(work_df, cols_to_use, train_mask)
    preprocessor = build_preprocessor(buckets)
    X_train_raw = work_df.loc[train_mask, cols_to_use]
    preprocessor.fit(X_train_raw)
    return preprocessor, work_df, cols_to_use


def transform_rows(preprocessor, work_df, cols_to_use, row_mask):
    return np.asarray(preprocessor.transform(work_df.loc[row_mask, cols_to_use]))


def run_repeated_cv(df, fs, task, best_params_c, best_params_d):
    model_c_cols = fs["model_c"]
    model_d_cols = fs["model_d"]
    rna_cols = [c for c in model_d_cols if c not in model_c_cols]
    assert STALENESS_COL in rna_cols, "days_since_rna_sample missing from RNA column set - check model_feature_sets.json"

    if task == "binary":
        task_df = df[df["eligible_for_binary"]].reset_index(drop=True)
        target_col = BINARY_TARGET
    else:
        task_df = df.reset_index(drop=True)
        target_col = MULTICLASS_TARGET

    # --- Re-assert test exclusion at the point of use, not just at load time ---
    assert (task_df["split"] != "test").all(), \
        f"[{task}] test-split rows present in the repeated-CV task frame."

    unique_patients = task_df["public_id"].unique()
    rna_available_mask_all = task_df[STALENESS_COL].notna()

    per_fold_rows = []

    for repeat in range(N_REPEATS):
        seed = RANDOM_SEED + repeat
        folds = make_repeat_folds(unique_patients, N_FOLDS, seed=seed)

        for fold_idx, test_patients in enumerate(folds):
            train_patients = np.setdiff1d(unique_patients, test_patients)
            assert set(train_patients).isdisjoint(set(test_patients)), \
                f"repeat {repeat} fold {fold_idx}: train/test patient overlap"

            train_mask = task_df["public_id"].isin(train_patients)
            # Evaluation restricted to RNA-available patients in the fold's held-out patients
            eval_mask = task_df["public_id"].isin(test_patients) & rna_available_mask_all

            # --- Explicit guard: no split=="test" row enters CV training or evaluation ---
            assert not (task_df.loc[train_mask, "split"] == "test").any(), \
                f"repeat {repeat} fold {fold_idx}: original test-split row entered CV training"
            assert not (task_df.loc[eval_mask, "split"] == "test").any(), \
                f"repeat {repeat} fold {fold_idx}: original test-split row entered CV evaluation"

            if eval_mask.sum() < 2 or train_mask.sum() < 2:
                print(f"SKIP repeat {repeat} fold {fold_idx} ({task}): insufficient rows")
                continue

            # --- Fold-local preprocessing, fit on this fold's TRAIN rows only ---
            prep_c, wdf_c, cols_c = fit_fold_preprocessor(task_df, model_c_cols, train_mask, is_model_d=False)
            prep_d, wdf_d, cols_d = fit_fold_preprocessor(task_df, model_d_cols, train_mask, is_model_d=True,
                                                           rna_cols=rna_cols)
            assert len(cols_c) == len(model_c_cols), "Model C fold column count drifted from model_feature_sets.json"
            assert len(cols_d) == len(model_d_cols) + 1, "Model D fold column count drifted (expected +rna_available)"

            X_tr_c = transform_rows(prep_c, wdf_c, cols_c, train_mask)
            X_ev_c = transform_rows(prep_c, wdf_c, cols_c, eval_mask)
            X_tr_d = transform_rows(prep_d, wdf_d, cols_d, train_mask)
            X_ev_d = transform_rows(prep_d, wdf_d, cols_d, eval_mask)

            pair_ids_c = task_df.loc[eval_mask, "pair_id"].to_numpy()
            pair_ids_d = task_df.loc[eval_mask, "pair_id"].to_numpy()
            assert np.array_equal(pair_ids_c, pair_ids_d), \
                f"repeat {repeat} fold {fold_idx}: C/D eval pair_id mismatch"

            if task == "binary":
                y_tr = task_df.loc[train_mask, target_col].to_numpy()
                y_ev = task_df.loc[eval_mask, target_col].to_numpy()
            else:
                y_tr = np.array([MULTICLASS_CLASSES.index(v) for v in task_df.loc[train_mask, target_col]])
                y_ev = np.array([MULTICLASS_CLASSES.index(v) for v in task_df.loc[eval_mask, target_col]])

            model_c = build_model("xgboost", task, best_params_c)
            model_c.fit(X_tr_c, y_tr)
            model_d = build_model("xgboost", task, best_params_d)
            model_d.fit(X_tr_d, y_tr)

            n_rows_eval = int(eval_mask.sum())
            n_patients_eval = int(task_df.loc[eval_mask, "public_id"].nunique())
            n_rows_train = int(train_mask.sum())
            n_patients_train = int(len(train_patients))

            if task == "binary":
                proba_c = model_c.predict_proba(X_ev_c)[:, 1]
                proba_d = model_d.predict_proba(X_ev_d)[:, 1]
                for metric_name, metric_fn in [("auprc", average_precision_score), ("auroc", roc_auc_score)]:
                    if len(np.unique(y_ev)) < 2:
                        continue
                    c_val = float(metric_fn(y_ev, proba_c))
                    d_val = float(metric_fn(y_ev, proba_d))
                    per_fold_rows.append({
                        "task": task, "repeat": repeat, "fold": fold_idx, "metric": metric_name,
                        "C": c_val, "D": d_val, "D_minus_C": d_val - c_val,
                        "n_rows_eval": n_rows_eval, "n_patients_eval": n_patients_eval,
                        "n_rows_train": n_rows_train, "n_patients_train": n_patients_train,
                    })
            else:
                pred_c = model_c.predict(X_ev_c)
                pred_d = model_d.predict(X_ev_d)
                c_val = float(f1_score(y_ev, pred_c, average="macro", zero_division=0))
                d_val = float(f1_score(y_ev, pred_d, average="macro", zero_division=0))
                per_fold_rows.append({
                    "task": task, "repeat": repeat, "fold": fold_idx, "metric": "macro_f1",
                    "C": c_val, "D": d_val, "D_minus_C": d_val - c_val,
                    "n_rows_eval": n_rows_eval, "n_patients_eval": n_patients_eval,
                    "n_rows_train": n_rows_train, "n_patients_train": n_patients_train,
                })

            print(f"repeat {repeat} fold {fold_idx} ({task}): "
                  f"n_eval={n_rows_eval} rows / {n_patients_eval} patients - done")

    return per_fold_rows


def aggregate(per_fold_df):
    agg_rows = []
    for (task, metric), grp in per_fold_df.groupby(["task", "metric"]):
        diffs = grp["D_minus_C"].to_numpy()
        n = len(diffs)
        rng = np.random.RandomState(RANDOM_SEED)
        boot_means = [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(N_FOLD_BOOTSTRAP)]

        # --- Approximate, explicitly-labeled empirical detectable-effect scale ---
        # Heuristic: roughly 2x the standard error of the mean fold-level
        # difference, using the OBSERVED std across the n fold-level
        # replicates. This describes "how big would the mean D-C difference
        # need to be, relative to the noise actually observed here, to stand
        # out from that noise" - it is NOT a formal minimum-detectable-effect
        # calculation and NOT a statistical power analysis (no assumed
        # effect size, no target power, no independence assumption beyond
        # what's stated below).
        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
        approx_scale = float(2 * std_diff / np.sqrt(n)) if n > 1 else None

        agg_rows.append({
            "task": task, "metric": metric,
            "n_fold_replicates": n,
            "mean_D_minus_C": float(np.mean(diffs)),
            "std_D_minus_C": std_diff,
            "min_D_minus_C": float(np.min(diffs)),
            "max_D_minus_C": float(np.max(diffs)),
            "fold_bootstrap_ci_lower": float(np.percentile(boot_means, 2.5)),
            "fold_bootstrap_ci_upper": float(np.percentile(boot_means, 97.5)),
            "approx_empirical_detectable_effect_scale": approx_scale,
            "note": ("APPROXIMATE/EMPIRICAL heuristic only - roughly 2x the standard error of "
                     "the mean fold-level D-C difference, computed from the OBSERVED dispersion "
                     "across the n fold-level replicates above. This is a descriptive sensitivity "
                     "diagnostic, NOT a formal minimum-detectable-effect (MDE) calculation and NOT "
                     "a statistical power analysis. The 25 replicates are not fully independent "
                     "(shared patient pool across the 5 repeats), so this likely UNDERSTATES true "
                     "uncertainty. Treat this analysis as a robustness check, not a significance test."),
        })
    return pd.DataFrame(agg_rows)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(FEATURE_SETS_PATH) as f:
        fs = json.load(f)

    df = load_data_and_eligibility()

    all_rows = []
    for task in ["binary", "multiclass"]:
        with open(f"{CLEAN_RESULTS_DIR}/{task}_c_xgboost_metrics.json") as f:
            best_params_c = json.load(f)["best_params"]
        with open(f"{CLEAN_RESULTS_DIR}/{task}_d_xgboost_metrics.json") as f:
            best_params_d = json.load(f)["best_params"]
        print(f"\n=== {task}: repeated CV, C params={best_params_c}, D params={best_params_d} ===")
        all_rows.extend(run_repeated_cv(df, fs, task, best_params_c, best_params_d))

    per_fold_df = pd.DataFrame(all_rows)
    per_fold_df.to_csv(f"{OUT_DIR}/repeated_cv_c_vs_d_per_fold.csv", index=False)

    agg_df = aggregate(per_fold_df)
    agg_df.to_csv(f"{OUT_DIR}/repeated_cv_c_vs_d_aggregate.csv", index=False)

    with open(f"{OUT_DIR}/repeated_cv_c_vs_d_summary.json", "w") as f:
        json.dump({"per_fold": all_rows, "aggregate": agg_df.to_dict(orient="records")}, f, indent=2)

    print(f"\nDONE. Saved: {OUT_DIR}/repeated_cv_c_vs_d_per_fold.csv")
    print(f"      Saved: {OUT_DIR}/repeated_cv_c_vs_d_aggregate.csv")
    print(f"      Saved: {OUT_DIR}/repeated_cv_c_vs_d_summary.json")