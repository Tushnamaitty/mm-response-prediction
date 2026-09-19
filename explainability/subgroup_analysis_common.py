"""
Subgroup analysis - File 1 of 5: shared utilities for the two a priori,
hypothesis-driven subgroup analyses (deep responders; frontline vs
later-line). Subgroup row-filtering is applied upstream of the SAME
methodology already verified in the clean-rerun robustness scripts -
the fold/bootstrap helpers are imported from those scripts, not copied:
- modeling/rna_subset_paired_analysis_clean_rerun.py (patient bootstrap)
- modeling/repeated_cv_c_vs_d_clean_rerun.py (fold-local preprocessing)

Does NOT modify or retrain any canonical model. All functions here are
read-only with respect to data/, artifacts/models_clean_rerun/, and
the canonical files in artifacts/results_clean_rerun/.

Run every subgroup script from the repo root as a module, e.g.
    python -m explainability.subgroup_summary
"""

import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# canonical helpers below are re-exported for the other subgroup scripts
sys.path.insert(0, "modeling")
sys.path.insert(0, "data_pipeline")
from train_models import (  # noqa: E402,F401
    load_master_data, build_model, RANDOM_SEED, MULTICLASS_CLASSES, STALENESS_COL,
)
from rna_subset_paired_analysis_clean_rerun import patient_bootstrap_indices  # noqa: E402,F401
from repeated_cv_c_vs_d_clean_rerun import (  # noqa: E402,F401
    make_repeat_folds, fit_fold_preprocessor, transform_rows,
)

CLEAN_RESULTS_DIR = "artifacts/results_clean_rerun"
SUBGROUP_OUT_DIR = f"{CLEAN_RESULTS_DIR}/subgroup_analysis"
FEATURE_SETS_PATH = "data_pipeline/model_feature_sets.json"

BINARY_TARGET = "improved"
MULTICLASS_TARGET = "exact_next_response"
TASKS = ["binary", "multiclass"]

# Primary comparison for BOTH tasks: the locked clean-rerun XGBoost C vs D
# (D = C + RNA), the same pair as the whole-cohort
# rna_subset_paired_analysis_clean_rerun.py "D - C" rows.
PRIMARY_ALGORITHM = "xgboost"
COMPARATOR_STAGE = "c"
RNA_STAGE = "d"

N_BOOTSTRAP = 1000
N_FOLDS = 5
N_REPEATS = 5

DEEP_RESPONSE_CATEGORIES = ["very_good_partial_response", "complete_response", "stringent_complete_response"]

# subgroup name -> the ONE Vt-side field its membership may read
SUBGROUP_FIELDS = {
    "deep_responders": "vt_disease_response",
    "non_deep_responders": "vt_disease_response",
    "frontline": "current_line_number",
    "later_line": "current_line_number",
}
SUBGROUPS = list(SUBGROUP_FIELDS)
ALLOWED_MEMBERSHIP_FIELDS = {"vt_disease_response", "current_line_number", "eligible_for_binary"}

# (group_a, group_b): interaction = (D-C)_a - (D-C)_b
INTERACTIONS = [
    ("deep_responders", "non_deep_responders"),
    ("frontline", "later_line"),
]


def get_subgroup_mask(df, subgroup_name):
    """Boolean mask over df's rows for the named subgroup. Reads ONLY
    df[SUBGROUP_FIELDS[subgroup_name]] - a Vt-side model input - never a
    Vt+1, target, or outcome field. Rows with a missing membership value
    belong to neither side of the split."""
    col = df[SUBGROUP_FIELDS[subgroup_name]]
    if subgroup_name == "deep_responders":
        return col.isin(DEEP_RESPONSE_CATEGORIES)
    if subgroup_name == "non_deep_responders":
        return col.notna() & ~col.isin(DEEP_RESPONSE_CATEGORIES)
    if subgroup_name == "frontline":
        return col == 1
    if subgroup_name == "later_line":
        return col >= 2
    raise ValueError(f"Unknown subgroup_name: {subgroup_name}")


def apply_task_filter(df, task):
    """Binary applies the existing eligible_for_binary flag (excludes
    Vt=sCR pairs) - the SAME rule the canonical models use. Multiclass
    applies no additional filter."""
    if task == "binary":
        return df[df["eligible_for_binary"].astype(bool)]
    return df


def build_subgroup_frame(df, subgroup_name, task):
    """Subgroup mask, then task eligibility."""
    return apply_task_filter(df[get_subgroup_mask(df, subgroup_name)], task)


def assert_no_leakage_in_subgroup_definitions(df):
    """Runtime (not just static) leakage check:
    1. every membership field is an allowed Vt-side field, is a Model C
       input, and is not a target / excluded-leakage / vt1_* column;
    2. every subgroup mask + task filter recomputed on a frame holding
       ONLY the allowed fields equals the mask on the full frame - so a
       future edit that reads any other column fails here with KeyError
       or a mismatch."""
    with open(FEATURE_SETS_PATH) as f:
        fs = json.load(f)
    forbidden = set(fs["targets"]) | set(fs["excluded_leakage"]) | {
        BINARY_TARGET, MULTICLASS_TARGET, "time_gap_days"}
    forbidden |= {c for c in df.columns if c.startswith("vt1_")}

    for name, field in SUBGROUP_FIELDS.items():
        assert field in ALLOWED_MEMBERSHIP_FIELDS, f"{name}: {field} is not an allowed membership field"
        assert not field.startswith("vt1_"), f"{name}: {field} looks like a Vt+1 field"
        assert field not in forbidden, f"{name}: {field} is a target/excluded-leakage field"
        assert field in fs["model_c"], f"{name}: {field} is not a validated Model C (Vt) input"
    assert not (ALLOWED_MEMBERSHIP_FIELDS & forbidden), "Allowed membership field overlaps forbidden set"

    restricted = df[sorted(ALLOWED_MEMBERSHIP_FIELDS)]
    for name in SUBGROUPS:
        for task in TASKS:
            full_idx = build_subgroup_frame(df, name, task).index
            restricted_idx = build_subgroup_frame(restricted, name, task).index
            assert full_idx.equals(restricted_idx), f"{name}/{task}: membership depends on a non-allowed field"
    print(f"Leakage check passed: subgroup membership reads only {sorted(ALLOWED_MEMBERSHIP_FIELDS)}; "
          f"no vt1_*/target/excluded-leakage field ({len(forbidden)} forbidden columns checked).")


# ---------------------------------------------------------------------
# Metrics: NaN (not an exception) when undefined on a resample
# ---------------------------------------------------------------------

def _binary_metric(fn):
    def metric(y_true, score):
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            return float("nan")
        return float(fn(y_true, score))
    return metric


def macro_f1_fixed_labels(y_true, y_pred, labels=MULTICLASS_CLASSES):
    """Macro-F1 over the FIXED class list, so every group/fold/resample
    averages over the same 6 classes (a class absent from both y_true and
    y_pred contributes 0)."""
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


TASK_METRICS = {
    "binary": {"auroc": _binary_metric(roc_auc_score), "auprc": _binary_metric(average_precision_score)},
    "multiclass": {"macro_f1": macro_f1_fixed_labels},
}
# prediction column each metric scores
TASK_SCORE_COL = {"binary": "proba", "multiclass": "pred"}


def percentile_ci(values):
    """NaN-safe 95% percentile CI. Returns (lo, hi, n_valid)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None, None, 0
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), int(len(arr))


def outcome_summary(y_true, task):
    """Binary: prevalence of improved=1. Multiclass: counts per fixed class."""
    y_true = np.asarray(y_true)
    if task == "binary":
        return {"outcome_prevalence": float(np.mean(y_true)) if len(y_true) else None}
    counts = {c: int((y_true == c).sum()) for c in MULTICLASS_CLASSES}
    return {"outcome_distribution": json.dumps(counts)}
