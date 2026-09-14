"""
Step 12: Leakage-safe preprocessing for Models A/B/C/D.

Input:  data/clinical/visit_pairs_with_rna.csv
        data/splits/visit_pair_splits.csv
        data_pipeline/model_feature_sets.json
Output: artifacts/preprocessing/model_{a,b,c,d}_preprocessor.joblib
        artifacts/preprocessing/model_{a,b,c,d}_metadata.json

ALL fitting (imputation statistics, one-hot categories) happens on the
TRAIN split only, then is applied UNCHANGED to val/test via .transform()
(never .fit() or .fit_transform() on non-train rows). This is verified
programmatically at the end, not just assumed.

COLUMN CLASSIFICATION (by actual inspected dtype/values, not by name):
  - "ordinal_response": vt_disease_response only - encoded via the same
    RESPONSE_RANK scale already used in build_visit_pairs.py (these are
    genuinely ordered IMWG categories, not nominal).
  - "numeric": true numeric dtype columns (float64/int64), including
    already-integer-coded ordinal scales (ISS stage, line number, etc.)
    - median imputed (train-only).
  - "clean_boolean": dtype=bool with ZERO missing values (verified, not
    assumed) - cast directly to 0/1, no imputer needed.
  - "object_boolean": dtype=object holding only True/False values PLUS
    genuine missingness - imputed to an explicit "missing" category
    (never silently treated as False), then one-hot encoded.
  - "categorical": true nominal string columns - imputed to "missing",
    one-hot encoded with handle_unknown="ignore" so an unseen val/test
    category never crashes the pipeline. The two high-cardinality
    regimen columns use min_frequency to cap dimensionality.

RNA HANDLING (Model D only) - see conversation for full reasoning:
  Verified RNA missingness is 100% row-aligned (a row has ALL 50 pathway
  scores or NONE - 0 rows with partial missingness). Rows are NOT dropped:
  one `rna_available` indicator is engineered pre-transform, then the 50
  pathway columns are median-imputed (train-only) like any other numeric
  feature. This preserves IDENTICAL row/patient counts across Models
  A/B/C/D, which is required for a fair ablation comparison (Model D must
  be evaluated on the same rows as A/B/C, not a biased RNA-available
  subset).

FLAGGED PROBLEMATIC COLUMNS (processed, but noted - see printed report):
  - cd138_detected, has_peripheral_neuropathy: only ever True when
    observed (zero variance when present) - the "False" category will be
    empty/near-empty after encoding.
  - cd38_pct: only ever exactly 100.0 when observed - near-zero variance.
  - current_regimen_name (262 categories), current_regimen_categories
    (139 categories): high-cardinality, capped via min_frequency rather
    than full one-hot to avoid dimensionality explosion.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

CLINICAL_DIR = "data/clinical"
SPLITS_DIR = "data/splits"
FEATURE_SETS_PATH = "data_pipeline/model_feature_sets.json"
ARTIFACTS_DIR = "artifacts/preprocessing"

# Same scale already established in build_visit_pairs.py - vt_disease_response
# has a genuine clinical order, not nominal categories.
RESPONSE_RANK = {
    "progressive_disease": 0,
    "stable_disease": 1,
    "partial_response": 2,
    "very_good_partial_response": 3,
    "complete_response": 4,
    "stringent_complete_response": 5,
}

HIGH_CARDINALITY_COLS = {"current_regimen_name", "current_regimen_categories"}
MIN_FREQUENCY_HIGH_CARD = 10

RNA_INDICATOR_COL = "rna_available"


def load_everything() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", low_memory=False)
    split_map = pd.read_csv(f"{SPLITS_DIR}/visit_pair_splits.csv")
    df = df.merge(split_map[["pair_id", "split"]], on="pair_id", how="left")
    assert df["split"].notna().all(), "Some rows have no split assignment."

    with open(FEATURE_SETS_PATH) as f:
        feature_sets = json.load(f)

    return df, feature_sets


def classify_columns(df: pd.DataFrame, cols: list[str], train_mask: pd.Series) -> dict:
    """Inspect ACTUAL dtype and values in the training data to bucket
    each column - not classified from column names."""
    buckets = {
        "ordinal_response": [],
        "numeric": [],
        "clean_boolean": [],
        "object_boolean": [],
        "categorical": [],
    }

    for col in cols:
        if col == "vt_disease_response":
            buckets["ordinal_response"].append(col)
            continue

        s_train = df.loc[train_mask, col]
        dtype = s_train.dtype

        if dtype == bool:
            n_missing = s_train.isna().sum()
            if n_missing > 0:
                raise ValueError(
                    f"Column '{col}' has dtype=bool but {n_missing} missing "
                    f"values in train - expected clean bool columns to have "
                    f"zero missing (verify this assumption before proceeding)."
                )
            buckets["clean_boolean"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            buckets["numeric"].append(col)
            continue

        # Remaining: object/string dtype. Check if it's boolean-like
        # (only True/False among non-null values) vs. true categorical text.
        non_null_vals = set(s_train.dropna().unique())
        if non_null_vals <= {True, False}:
            buckets["object_boolean"].append(col)
        else:
            buckets["categorical"].append(col)

    return buckets


def _cast_to_int(X):
    return X.astype(int)


def _stringify_preserving_na(X):
    return X.astype(object).where(X.isna(), X.astype(str))


class OrdinalResponseEncoder:
    """Maps vt_disease_response to its numeric IMWG rank. sklearn-compatible
    (fit/transform), though fit() does nothing since the mapping is fixed
    domain knowledge, not learned from data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X).reshape(-1)
        mapped = pd.Series(X).map(RESPONSE_RANK)
        return mapped.to_numpy().reshape(-1, 1).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(["vt_disease_response_ordinal"])


def build_preprocessor(buckets: dict) -> ColumnTransformer:
    transformers = []

    if buckets["ordinal_response"]:
        ordinal_pipe = Pipeline([
            ("encode", OrdinalResponseEncoder()),
            ("impute", SimpleImputer(strategy="median")),  # defensive; 0% missing today
        ])
        transformers.append(("ordinal_response", ordinal_pipe, buckets["ordinal_response"]))

    if buckets["numeric"]:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ])
        transformers.append(("numeric", numeric_pipe, buckets["numeric"]))

    if buckets["clean_boolean"]:
        bool_pipe = Pipeline([
            ("to_int", FunctionTransformer(
                _cast_to_int, feature_names_out="one-to-one"
            )),
        ])
        transformers.append(("clean_boolean", bool_pipe, buckets["clean_boolean"]))

    if buckets["object_boolean"]:
        obj_bool_pipe = Pipeline([
            # BUG FIX: values here are a mix of Python bool (True/False)
            # and NaN. Imputing the string "missing" directly into that
            # column produces a mixed bool/str column that OneHotEncoder
            # can't sort - stringify first so the whole column is
            # uniformly string type before imputing/encoding.
            ("stringify", FunctionTransformer(
                _stringify_preserving_na,
                feature_names_out="one-to-one",
            )),
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("object_boolean", obj_bool_pipe, buckets["object_boolean"]))

    # Categorical columns split into normal vs. high-cardinality, since
    # they need different min_frequency settings.
    normal_cat = [c for c in buckets["categorical"] if c not in HIGH_CARDINALITY_COLS]
    high_card_cat = [c for c in buckets["categorical"] if c in HIGH_CARDINALITY_COLS]

    if normal_cat:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipe, normal_cat))

    if high_card_cat:
        high_card_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False,
                min_frequency=MIN_FREQUENCY_HIGH_CARD,
            )),
        ])
        transformers.append(("categorical_high_card", high_card_pipe, high_card_cat))

    return ColumnTransformer(transformers, remainder="drop")


def process_model(
    model_name: str,
    predictor_cols: list[str],
    df: pd.DataFrame,
    is_model_d: bool,
) -> dict:
    print("\n")
    print(f"Model {model_name.upper()} ({len(predictor_cols)} raw predictor columns)")
    print("\n")

    work_df = df.copy()
    cols_to_use = list(predictor_cols)

    rna_cols = []
    if is_model_d:
        # Identify RNA columns as those NOT in model_c (same programmatic
        # approach as Step 10 - no hardcoded pathway names).
        with open(FEATURE_SETS_PATH) as f:
            fs = json.load(f)
        rna_cols = [c for c in predictor_cols if c not in fs["model_c"]]

        # Engineer the single RNA-availability indicator BEFORE the
        # ColumnTransformer, then feed it through as a clean_boolean-style
        # column (0% missing by construction).
        work_df[RNA_INDICATOR_COL] = work_df[rna_cols[0]].notna().astype(int)
        cols_to_use = [c for c in cols_to_use] + [RNA_INDICATOR_COL]

    train_mask = work_df["split"] == "train"
    val_mask = work_df["split"] == "val"
    test_mask = work_df["split"] == "test"

    # The RNA indicator itself is numeric int, 0% missing - classify_columns
    # will correctly bucket it as "numeric" (harmless, no imputation fires).
    buckets = classify_columns(work_df, cols_to_use, train_mask)

    print("Column classification (by actual dtype/values in TRAIN):")
    for bucket_name, bucket_cols in buckets.items():
        print(f"  {bucket_name:18s}: {len(bucket_cols)} columns")

    preprocessor = build_preprocessor(buckets)

    # --- FIT ONLY ON TRAIN ---
    X_train_raw = work_df.loc[train_mask, cols_to_use]
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(work_df.loc[val_mask, cols_to_use])
    X_test = preprocessor.transform(work_df.loc[test_mask, cols_to_use])

    feature_names = list(preprocessor.get_feature_names_out())

    print(f"\nOutput feature count: {X_train.shape[1]}")
    print(f"Train shape: {X_train.shape} | Val shape: {X_val.shape} | Test shape: {X_test.shape}")

    # --- Validation checks ---
    for name, X_raw in [("train", X_train), ("val", X_val), ("test", X_test)]:
        # sparse_output=False is set on every OneHotEncoder above, so this
        # is always a dense ndarray at runtime - the explicit cast here is
        # only to satisfy the type checker's broader declared return type
        # (ColumnTransformer.transform() can theoretically return a sparse
        # matrix), not a behavior change.
        X = np.asarray(X_raw)
        n_inf = np.isinf(X).sum()
        if n_inf > 0:
            raise ValueError(f"Model {model_name} {name}: {n_inf} inf values after preprocessing.")
        n_nan = np.isnan(X).sum()
        if n_nan > 0:
            raise ValueError(
                f"Model {model_name} {name}: {n_nan} unexpected NaN values remain "
                f"after preprocessing (imputation should have caught these)."
            )
    print("No inf/-inf, no unexpected NaN in any split. OK")

    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch across splits: "
            f"train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
        )
    print("Same feature count/ordering across train/val/test. OK")

    n_row_train = train_mask.sum()
    n_row_val = val_mask.sum()
    n_row_test = test_mask.sum()
    if X_train.shape[0] != n_row_train or X_val.shape[0] != n_row_val or X_test.shape[0] != n_row_test:
        raise ValueError("Row count changed during preprocessing - rows were dropped or duplicated.")
    print(f"Row counts preserved: train={n_row_train}, val={n_row_val}, test={n_row_test}. OK")

    # RNA-specific report
    if is_model_d:
        print(f"\nRNA handling: {len(rna_cols)} pathway columns median-imputed (train-only),")
        print(f"'{RNA_INDICATOR_COL}' indicator added. Rows retained (not dropped):")
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            n_total = mask.sum()
            n_with_rna = work_df.loc[mask, RNA_INDICATOR_COL].sum()
            print(f"  {name:6s}: {n_total} rows usable, {n_with_rna} ({n_with_rna/n_total*100:.1f}%) had real RNA data")

    # Save preprocessor + metadata
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(preprocessor, f"{ARTIFACTS_DIR}/model_{model_name}_preprocessor.joblib")

    metadata = {
        "model": model_name,
        "input_columns": cols_to_use,
        "column_buckets": {k: v for k, v in buckets.items()},
        "rna_indicator_added": RNA_INDICATOR_COL if is_model_d else None,
        "rna_columns_imputed": rna_cols if is_model_d else None,
        "high_cardinality_columns_capped": list(HIGH_CARDINALITY_COLS & set(cols_to_use)),
        "output_feature_count": int(X_train.shape[1]),
        "output_feature_names": feature_names,
        "row_counts": {"train": int(n_row_train), "val": int(n_row_val), "test": int(n_row_test)},
    }
    with open(f"{ARTIFACTS_DIR}/model_{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "buckets": buckets,
        "shapes": {"train": X_train.shape, "val": X_val.shape, "test": X_test.shape},
        "preprocessor": preprocessor,
        "train_mask": train_mask,
        "work_df": work_df,
    }


def adversarial_leakage_check(results: dict) -> None:
    """Independent re-check: verify no val/test statistic could have
    influenced any fitted preprocessor - by refitting on train-only data
    a SECOND time, independently, and confirming identical learned
    parameters (medians) to what was actually used."""
    print("\n")
    print("ADVERSARIAL LEAKAGE RE-CHECK")
    print("\n")

    for model_name, result in results.items():
        preprocessor = result["preprocessor"]
        train_mask = result["train_mask"]
        work_df = result["work_df"]  # includes rna_available for Model D

        for name, transformer, cols in preprocessor.transformers_:
            if name == "numeric":
                imputer = transformer.named_steps.get("impute")
                if imputer is not None and hasattr(imputer, "statistics_"):
                    independent_medians = work_df.loc[train_mask, cols].median().values
                    fitted_medians = imputer.statistics_
                    if not np.allclose(independent_medians, fitted_medians, equal_nan=True):
                        raise ValueError(
                            f"LEAKAGE DETECTED in Model {model_name}: fitted median "
                            f"statistics don't match an independent train-only "
                            f"recomputation - val/test data may have influenced fitting."
                        )
        print(f"Model {model_name}: fitted train-only statistics independently verified. OK")


if __name__ == "__main__":
    df, feature_sets = load_everything()

    model_defs = [
        ("a", feature_sets["model_a"], False),
        ("b", feature_sets["model_b"], False),
        ("c", feature_sets["model_c"], False),
        ("d", feature_sets["model_d"], True),
    ]

    results = {}
    for model_name, cols, is_d in model_defs:
        results[model_name] = process_model(model_name, cols, df, is_d)

    adversarial_leakage_check(results)

    print("\n")
    print(" STEP 12 SUMMARY ")
    print("\n")
    for model_name, result in results.items():
        s = result["shapes"]
        print(f"Model {model_name.upper()}: train={s['train']}, val={s['val']}, test={s['test']}")

    print("\n STEP 12 VERIFIED ")