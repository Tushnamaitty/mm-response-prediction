"""
Subgroup analysis - File 4 of 5 (SECONDARY sensitivity analysis only):
repeated patient-grouped CV, C-XGBoost vs D-XGBoost, restricted to each
subgroup's train+val rows (original test split untouched, same
discipline as modeling/repeated_cv_c_vs_d_clean_rerun.py).

IMPORTANT - different estimand from the primary analysis (File 3):
File 3 evaluates the LOCKED whole-cohort models on subgroup test rows.
This file RETRAINS C and D from scratch within each subgroup (models see
only that subgroup's rows), so it asks "does RNA help a model built for
this subgroup?", not "does the deployed model's RNA benefit differ in
this subgroup?". Results are supporting evidence only.

5-fold x 5-repeat deterministic patient-grouped CV. Fold-local
preprocessing (fit on that fold's training rows only). Reuses the LOCKED
clean-rerun C/D XGBoost best_params - no new tuning. Each fold trains on
all of that fold's training rows and evaluates D-vs-C only on that
fold's RNA-available held-out patients. Multiclass macro-F1 always
averages over the fixed 6 MULTICLASS_CLASSES.

Reports per-fold results plus mean / SD / min / max / fraction of folds
with D > C. Deliberately NO confidence interval: the 25 fold replicates
reuse the same patient pool across repeats and are not independent, so
bootstrapping them gives misleadingly narrow intervals.

Does NOT retrain or modify any canonical/locked model. Writes ONLY
under artifacts/results_clean_rerun/subgroup_analysis/.
"""

import json
import os

import numpy as np
import pandas as pd

from explainability.subgroup_analysis_common import (
    load_master_data, build_subgroup_frame, assert_no_leakage_in_subgroup_definitions,
    make_repeat_folds, fit_fold_preprocessor, transform_rows, build_model,
    TASK_METRICS, SUBGROUPS, TASKS, PRIMARY_ALGORITHM, COMPARATOR_STAGE, RNA_STAGE,
    SUBGROUP_OUT_DIR, CLEAN_RESULTS_DIR, FEATURE_SETS_PATH, BINARY_TARGET, MULTICLASS_TARGET,
    STALENESS_COL, MULTICLASS_CLASSES, RANDOM_SEED, N_FOLDS, N_REPEATS,
)

PER_FOLD_PATH = f"{SUBGROUP_OUT_DIR}/subgroup_repeated_cv_per_fold.csv"
AGGREGATE_PATH = f"{SUBGROUP_OUT_DIR}/subgroup_repeated_cv_aggregate.csv"
ESTIMAND_NOTE = ("SECONDARY sensitivity analysis: C/D retrained within subgroup train+val rows "
                 "(unlike the primary locked-model analysis). Mean/SD over 25 non-independent "
                 "fold replicates - descriptive dispersion only, not a CI or significance test.")


def load_best_params(task, stage):
    with open(f"{CLEAN_RESULTS_DIR}/{task}_{stage}_{PRIMARY_ALGORITHM}_metrics.json") as f:
        return json.load(f)["best_params"]


def run_subgroup_repeated_cv(df, subgroup_name, task, fs):
    model_c_cols = fs[f"model_{COMPARATOR_STAGE}"]
    model_d_cols = fs[f"model_{RNA_STAGE}"]
    rna_cols = [c for c in model_d_cols if c not in model_c_cols]
    assert STALENESS_COL in rna_cols
    best_params_c = load_best_params(task, COMPARATOR_STAGE)
    best_params_d = load_best_params(task, RNA_STAGE)

    sub = build_subgroup_frame(df, subgroup_name, task)
    sub = sub[sub["split"] != "test"].reset_index(drop=True)  # train+val only
    assert (sub["split"] != "test").all(), f"{subgroup_name}/{task}: test-split row leaked into CV universe"

    target_col = BINARY_TARGET if task == "binary" else MULTICLASS_TARGET
    if task == "binary":
        y_all = sub[target_col].astype(int).to_numpy()
    else:
        y_all = np.array([MULTICLASS_CLASSES.index(v) for v in sub[target_col]])
    unique_patients = sub["public_id"].unique()
    rna_available_mask_all = sub[STALENESS_COL].notna()

    print(f"\n--- {subgroup_name} | {task} | repeated CV ---")
    print(f"  train+val universe: {len(sub)} rows / {len(unique_patients)} patients")

    per_fold_rows = []
    for repeat in range(N_REPEATS):
        folds = make_repeat_folds(unique_patients, N_FOLDS, seed=RANDOM_SEED + repeat)

        for fold_idx, test_patients in enumerate(folds):
            train_mask = ~sub["public_id"].isin(test_patients)
            eval_mask = sub["public_id"].isin(test_patients) & rna_available_mask_all
            if eval_mask.sum() < 2 or train_mask.sum() < 2:
                print(f"  repeat {repeat} fold {fold_idx}: SKIPPED (too few rows)")
                continue

            prep_c, wdf_c, cols_c = fit_fold_preprocessor(sub, model_c_cols, train_mask, is_model_d=False)
            prep_d, wdf_d, cols_d = fit_fold_preprocessor(sub, model_d_cols, train_mask, is_model_d=True,
                                                          rna_cols=rna_cols)
            y_tr, y_ev = y_all[train_mask.to_numpy()], y_all[eval_mask.to_numpy()]

            model_c = build_model(PRIMARY_ALGORITHM, task, best_params_c)
            model_c.fit(transform_rows(prep_c, wdf_c, cols_c, train_mask), y_tr)
            model_d = build_model(PRIMARY_ALGORITHM, task, best_params_d)
            model_d.fit(transform_rows(prep_d, wdf_d, cols_d, train_mask), y_tr)

            X_ev_c = transform_rows(prep_c, wdf_c, cols_c, eval_mask)
            X_ev_d = transform_rows(prep_d, wdf_d, cols_d, eval_mask)
            if task == "binary":
                score_c, score_d = model_c.predict_proba(X_ev_c)[:, 1], model_d.predict_proba(X_ev_d)[:, 1]
            else:
                # map encoded predictions back to class names so the shared
                # fixed-label macro-F1 sees the same label space everywhere
                classes = np.array(MULTICLASS_CLASSES)
                y_ev = classes[y_ev]
                score_c, score_d = classes[model_c.predict(X_ev_c)], classes[model_d.predict(X_ev_d)]

            n_rows_eval = int(eval_mask.sum())
            n_patients_eval = int(sub.loc[eval_mask, "public_id"].nunique())
            for metric_name, metric_fn in TASK_METRICS[task].items():
                c_val, d_val = metric_fn(y_ev, score_c), metric_fn(y_ev, score_d)
                if not (np.isfinite(c_val) and np.isfinite(d_val)):
                    print(f"  repeat {repeat} fold {fold_idx}: {metric_name} undefined - skipped")
                    continue
                per_fold_rows.append({"subgroup": subgroup_name, "task": task, "repeat": repeat,
                                      "fold": fold_idx, "metric": metric_name, "C": c_val, "D": d_val,
                                      "D_minus_C": d_val - c_val, "n_rows_eval": n_rows_eval,
                                      "n_patients_eval": n_patients_eval})

            print(f"  repeat {repeat} fold {fold_idx}: n_eval={n_rows_eval} rows / {n_patients_eval} patients")

    return per_fold_rows


def aggregate(per_fold_df):
    agg_rows = []
    for (subgroup, task, metric), grp in per_fold_df.groupby(["subgroup", "task", "metric"], sort=False):
        diffs = grp["D_minus_C"].to_numpy()
        agg_rows.append({
            "subgroup": subgroup, "task": task, "metric": metric,
            "comparison": f"D - C ({PRIMARY_ALGORITHM}, retrained within subgroup)",
            "n_fold_replicates": int(len(diffs)),
            "mean_C": float(grp["C"].mean()), "mean_D": float(grp["D"].mean()),
            "mean_D_minus_C": float(np.mean(diffs)),
            "sd_D_minus_C": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan"),
            "min_D_minus_C": float(np.min(diffs)), "max_D_minus_C": float(np.max(diffs)),
            "frac_folds_D_better": float(np.mean(diffs > 0)),
            "note": ESTIMAND_NOTE,
        })
    return pd.DataFrame(agg_rows)


if __name__ == "__main__":
    os.makedirs(SUBGROUP_OUT_DIR, exist_ok=True)
    df = load_master_data()
    assert_no_leakage_in_subgroup_definitions(df)
    with open(FEATURE_SETS_PATH) as f:
        fs = json.load(f)

    all_rows = []
    for subgroup_name in SUBGROUPS:
        for task in TASKS:
            all_rows.extend(run_subgroup_repeated_cv(df, subgroup_name, task, fs))

    per_fold_df = pd.DataFrame(all_rows)
    per_fold_df.to_csv(PER_FOLD_PATH, index=False)
    agg_df = aggregate(per_fold_df)
    agg_df.to_csv(AGGREGATE_PATH, index=False)
    print(agg_df.drop(columns="note").to_string(index=False))
    print(f"\nSaved: {PER_FOLD_PATH}\nSaved: {AGGREGATE_PATH}")
