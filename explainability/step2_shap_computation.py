"""
Step 2 (XAI): SHAP value computation for the 4 canonical models.

Refuses to run unless Step 1's verification JSON confirms
all_checks_passed is true - SHAP is never computed against an
unverified model.

METHOD NOTE (important for the methods section): this uses
shap.TreeExplainer with feature_perturbation="tree_path_dependent",
NOT "interventional". Interventional SHAP was tried first (preferred
in principle, since it computes true marginal expectations rather than
walking tree decision paths, and this feature set has real correlation
- e.g. M-protein / FLC ratios, related longitudinal features - that
path-dependent attribution can split unevenly between correlated
features). It was abandoned because it empirically failed the raw-
margin additivity check by 0.0645 for the frozen C-LightGBM model - a
real failure, not a threshold-tuning issue (the 1e-3 threshold was
kept, not loosened). tree_path_dependent is exact by construction and
passed additivity cleanly (~1e-6) for all four models.

CONSEQUENCE TO CARRY FORWARD: because of this switch, Step 10's
reliability check on correlated-feature effects is now load-bearing,
not optional - any individually-attributed feature importance here
should be read with the possibility that correlated features shared
or split credit in a way that doesn't reflect true marginal
contribution.

Before saving anything, verifies sum(shap_values) + expected_value
matches the model's raw margin output (log-odds / pre-softmax score)
for a sample of test rows - this catches computation or link-function
mistakes independently of shap's own internal additivity check (which
is skipped via check_additivity=False, since our own check is stricter
and applied uniformly across binary/multiclass).

Handles shap library version differences: some versions return
shap_values() as a list of per-class arrays, others return a single
ndarray with the class axis embedded (rows, features, classes).
normalize_shap_output() detects and handles both, and always
normalizes to a fixed output shape (asserted immediately after).

Saved artifact shapes:
- binary {key}_shap_values.npy:     (n_test_rows, n_features)
- multiclass {key}_shap_values.npy: (n_classes, n_test_rows, n_features),
  class order = MULTICLASS_CLASSES from train_models.py

Does NOT retrain or modify any model. Writes ONLY under
artifacts/results_clean_rerun/xai/.
"""

import json, os, sys
import numpy as np
import pandas as pd
import joblib
import shap

sys.path.insert(0, "modeling")
from modeling.train_models import (
    load_master_data, load_preprocessor_and_meta,
    get_binary_xyz, get_multiclass_xyz, MULTICLASS_CLASSES, RANDOM_SEED,
)

CLEAN_MODELS_DIR = "artifacts/models_clean_rerun"
CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
OUT_DIR = f"{CLEAN_RESULTS_DIR}/xai"
SHAP_DIR = f"{OUT_DIR}/shap_artifacts"

SHAP_METHOD = "tree_path_dependent"
N_SANITY_CHECK_ROWS = 20

SELECTED_MODELS = [
    {"task": "binary", "stage": "c", "algorithm": "lightgbm"},
    {"task": "binary", "stage": "d", "algorithm": "lightgbm"},
    {"task": "multiclass", "stage": "a", "algorithm": "xgboost"},
    {"task": "multiclass", "stage": "d", "algorithm": "xgboost"},
]


def raw_margin(model, algorithm, X):
    """Raw model output before sigmoid/softmax - the space SHAP values
    are additive in with model_output='raw'."""
    if algorithm == "xgboost":
        return model.predict(X, output_margin=True)
    if algorithm == "lightgbm":
        return model.predict(X, raw_score=True)
    raise ValueError(algorithm)


def normalize_shap_output(task, raw_shap_values, raw_expected_value):
    """
    Normalize SHAP outputs to one fixed format, regardless of which
    shap version produced them (list-of-arrays vs. single ndarray with
    an embedded class axis):

    Binary:
        (n_rows, n_features) - positive class only

    Multiclass:
        (n_classes, n_rows, n_features)
    """
    if task == "binary":
        if isinstance(raw_shap_values, list):
            shap_values = np.asarray(raw_shap_values[1])
            expected_value = (
                raw_expected_value[1]
                if isinstance(raw_expected_value, (list, np.ndarray))
                else raw_expected_value
            )
        else:
            arr = np.asarray(raw_shap_values)
            if arr.ndim == 3:
                # (rows, features, classes)
                shap_values = arr[:, :, 1]
                ev = np.asarray(raw_expected_value)
                expected_value = ev[1] if ev.ndim > 0 else float(ev)
            else:
                shap_values = arr
                ev = np.asarray(raw_expected_value)
                expected_value = float(ev.ravel()[0]) if ev.ndim > 0 else float(ev)

        return np.asarray(shap_values), float(expected_value)

    # Multiclass
    if isinstance(raw_shap_values, list):
        shap_values = np.stack(raw_shap_values, axis=0)
    else:
        arr = np.asarray(raw_shap_values)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected multiclass SHAP shape: {arr.shape}")

        if arr.shape[2] == len(MULTICLASS_CLASSES):
            # Modern SHAP: (rows, features, classes)
            shap_values = np.moveaxis(arr, 2, 0)
        elif arr.shape[0] == len(MULTICLASS_CLASSES):
            # Already (classes, rows, features)
            shap_values = arr
        else:
            raise ValueError(f"Cannot identify class axis in SHAP shape {arr.shape}")

    expected_value = np.asarray(raw_expected_value).reshape(-1)
    if len(expected_value) != len(MULTICLASS_CLASSES):
        raise ValueError(
            f"Expected {len(MULTICLASS_CLASSES)} base values, got {len(expected_value)}"
        )

    return shap_values, expected_value


def compute_and_verify(task, stage, algorithm):
    key = f"{task}_{stage}_{algorithm}"
    print(f"\n{'='*70}\n{key}\n{'='*70}")

    model_path = f"{CLEAN_MODELS_DIR}/{key}.joblib"
    assert os.path.exists(model_path), f"Missing model: {model_path}"
    model = joblib.load(model_path)
    print(f"Loaded frozen model: {model_path}")

    df = load_master_data()
    preprocessor, meta = load_preprocessor_and_meta(stage)
    getter = get_binary_xyz if task == "binary" else get_multiclass_xyz
    X_tr, y_tr, g_tr, pid_tr, feat_names = getter(df, stage, preprocessor, meta, "train")
    X_te, y_te, g_te, pid_te, _ = getter(df, stage, preprocessor, meta, "test")
    print(f"Training matrix: {X_tr.shape}   Test matrix: {X_te.shape}")

    explainer = shap.TreeExplainer(model, feature_perturbation=SHAP_METHOD, model_output="raw")
    print(f"TreeExplainer created (feature_perturbation='{SHAP_METHOD}', model_output='raw')")
    print(f"Explainer expected_value: {explainer.expected_value}")

    print(f"Computing SHAP values for {X_te.shape[0]} test rows x {X_te.shape[1]} features "
          f"(this may take a few minutes, longer for multiclass)...")
    raw_shap_values = explainer.shap_values(X_te, check_additivity=False)
    raw_expected_value = explainer.expected_value
    shap_values, expected_value = normalize_shap_output(task, raw_shap_values, raw_expected_value)

    print(f"Normalized SHAP shape: {np.asarray(shap_values).shape}")
    print(f"Normalized expected value(s): {expected_value}")

    if task == "binary":
        assert shap_values.shape == (X_te.shape[0], X_te.shape[1]), (
            f"{key}: unexpected binary SHAP shape {shap_values.shape}"
        )
    else:
        assert shap_values.shape == (len(MULTICLASS_CLASSES), X_te.shape[0], X_te.shape[1]), (
            f"{key}: unexpected multiclass SHAP shape {shap_values.shape}"
        )

    # --- Our own independent additivity check (stricter than, and separate
    # from, shap's internal check.py — always run, threshold never relaxed) ---
    rng = np.random.RandomState(RANDOM_SEED)
    check_idx = rng.choice(X_te.shape[0], size=min(N_SANITY_CHECK_ROWS, X_te.shape[0]), replace=False)
    margins = raw_margin(model, algorithm, X_te[check_idx])

    if task == "binary":
        shap_sum = np.array(shap_values)[check_idx].sum(axis=1) + expected_value
        max_diff = float(np.max(np.abs(shap_sum - margins)))
        mean_diff = float(np.mean(np.abs(shap_sum - margins)))
        print(f"Sanity check (binary, raw-margin space): max abs diff = {max_diff:.2e}, mean = {mean_diff:.2e}")
    else:
        n_classes = len(shap_values)
        assert n_classes == len(MULTICLASS_CLASSES), (
            f"{key}: expected {len(MULTICLASS_CLASSES)} SHAP class arrays, got {n_classes}"
        )
        max_diff = 0.0
        print(f"Sanity check (multiclass, raw-margin space, {n_classes} classes):")
        for c in range(n_classes):
            shap_sum_c = shap_values[c][check_idx].sum(axis=1) + expected_value[c]
            diff_c = float(np.max(np.abs(shap_sum_c - margins[:, c])))
            print(f"  {MULTICLASS_CLASSES[c]}: max abs diff = {diff_c:.2e}")
            max_diff = max(max_diff, diff_c)
        print(f"  overall max abs diff = {max_diff:.2e}")

    sanity_passed = max_diff < 1e-3
    if not sanity_passed:
        raise AssertionError(
            f"{key}: SHAP sanity check FAILED (max abs diff {max_diff:.2e}) - "
            f"sum(shap_values) + expected_value does not match the model's raw margin output. "
            f"Nothing was saved for this model - do not proceed until this is resolved."
        )
    print(f"{key}: sanity check PASSED.")

    # --- Save artifacts ---
    np.save(f"{SHAP_DIR}/{key}_shap_values.npy", np.array(shap_values))
    np.save(f"{SHAP_DIR}/{key}_X_test.npy", X_te)
    pd.DataFrame({"pair_id": pid_te, "public_id": g_te}).to_csv(f"{SHAP_DIR}/{key}_row_ids.csv", index=False)
    with open(f"{SHAP_DIR}/{key}_feature_names.json", "w") as f:
        json.dump(feat_names, f, indent=2)

    expected_value_out = float(expected_value) if task == "binary" else [float(v) for v in expected_value]
    with open(f"{SHAP_DIR}/{key}_expected_value.json", "w") as f:
        json.dump({
            "expected_value": expected_value_out, "task": task,
            "feature_perturbation": SHAP_METHOD,
            "shap_values_shape": ("n_test_rows, n_features" if task == "binary"
                                   else "n_classes, n_test_rows, n_features (class order = MULTICLASS_CLASSES)"),
        }, f, indent=2)

    print("Artifacts saved successfully.")
    return {
        "key": key, "task": task, "model_stage": stage, "algorithm": algorithm,
        "feature_perturbation": SHAP_METHOD,
        "n_test_rows": int(X_te.shape[0]), "n_features": int(X_te.shape[1]),
        "sanity_check_max_abs_diff": max_diff, "sanity_check_passed": bool(sanity_passed),
    }


if __name__ == "__main__":
    os.makedirs(SHAP_DIR, exist_ok=True)

    verification_path = f"{OUT_DIR}/xai_model_verification.json"
    assert os.path.exists(verification_path), f"Missing {verification_path} - run Step 1 first."
    with open(verification_path) as f:
        step1_result = json.load(f)
    assert step1_result["all_checks_passed"], (
        "Step 1 verification did not pass for all models - fix that before computing SHAP values."
    )
    print(f"Step 1 gate confirmed passed. Proceeding with {SHAP_METHOD} SHAP computation.")

    summary_rows = []
    for spec in SELECTED_MODELS:
        result = compute_and_verify(spec["task"], spec["stage"], spec["algorithm"])
        summary_rows.append(result)

    with open(f"{OUT_DIR}/step2_shap_computation_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n{'='*70}\nSTEP 2 COMPLETE\n{'='*70}")
    print(f"SHAP artifacts saved under: {SHAP_DIR}/")
    print(f"Summary saved: {OUT_DIR}/step2_shap_computation_summary.json")
    print("All four models used feature_perturbation='tree_path_dependent' "
          "(see docstring for why interventional was abandoned).")
    print("Safe to proceed to Step 3.")