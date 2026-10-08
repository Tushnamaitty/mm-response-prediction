"""
Calibration analysis - Step 1 of 2: regenerate VALIDATION and TEST
predicted probabilities for the locked clean-rerun C-XGBoost and
D-XGBoost models (binary and multiclass). INFERENCE ONLY - no training,
no tuning, no refitting of any preprocessor.

Why this is needed: train_models.py saved only test probabilities for
the binary task and only predicted labels for the multiclass task, and
no validation predictions at all. Calibration needs validation
probabilities (to fit/select a calibrator) and test probabilities (to
evaluate it once).

Loads:
- artifacts/models_clean_rerun/{task}_{c,d}_xgboost.joblib (fit on TRAIN only)
- artifacts/preprocessing/model_{c,d}_preprocessor.joblib (fit on TRAIN only)
using the SAME feature-matrix code path as train_models.py.

Reproduction checks (fail loudly on any mismatch):
- regenerated TEST output reproduces the locked
  artifacts/results_clean_rerun/{task}_{stage}_xgboost_predictions.csv
  (binary: probabilities; multiclass: argmax labels) on identical pair_ids;
- regenerated VALIDATION metrics reproduce the locked val_metrics in
  {task}_{stage}_xgboost_metrics.json;
- val/test patient sets are disjoint.

Writes ONLY under artifacts/results_clean_rerun/calibration/:
- probabilities/{task}_{stage}_xgboost_{val,test}.csv  (row-level; *.csv is gitignored)
- calibration_predictions_manifest.json

Run from the repo root:
    python modeling/calibration_predictions_clean_rerun.py
"""

import hashlib
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

sys.path.insert(0, "modeling")
from train_models import (  # noqa: E402  (also registers the preprocessing classes needed to unpickle)
    load_master_data, load_preprocessor_and_meta, get_binary_xyz, get_multiclass_xyz,
    MODELS_DIR, RESULTS_DIR, MULTICLASS_CLASSES, STALENESS_COL,
)

ALGORITHM = "xgboost"
STAGES = ["c", "d"]
TASKS = ["binary", "multiclass"]
SPLITS = ["val", "test"]
OUT_DIR = f"{RESULTS_DIR}/calibration"
PROBA_DIR = f"{OUT_DIR}/probabilities"
MANIFEST_PATH = f"{OUT_DIR}/calibration_predictions_manifest.json"
PROBA_COLS = [f"p_{c}" for c in MULTICLASS_CLASSES]
TEST_PROBA_ATOL = 1e-6
VAL_METRIC_ATOL = 1e-9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def required_inputs():
    paths = []
    for task in TASKS:
        for stage in STAGES:
            key = f"{task}_{stage}_{ALGORITHM}"
            paths += [f"{MODELS_DIR}/{key}.joblib", f"{RESULTS_DIR}/{key}_predictions.csv",
                      f"{RESULTS_DIR}/{key}_metrics.json"]
    for stage in STAGES:
        paths += [f"artifacts/preprocessing/model_{stage}_preprocessor.joblib",
                  f"artifacts/preprocessing/model_{stage}_metadata.json"]
    return paths


def predict_split(df, task, stage, model, preprocessor, meta, split):
    getter = get_binary_xyz if task == "binary" else get_multiclass_xyz
    X, y, groups, pair_ids, _ = getter(df, stage, preprocessor, meta, split)
    proba = model.predict_proba(X)
    assert proba.shape[1] == (2 if task == "binary" else len(MULTICLASS_CLASSES)), \
        f"{task}/{stage}/{split}: unexpected predict_proba shape {proba.shape}"
    assert np.all(np.isfinite(proba)) and proba.min() >= 0 and proba.max() <= 1, \
        f"{task}/{stage}/{split}: probabilities outside [0, 1] or non-finite"

    out = pd.DataFrame({"pair_id": pair_ids, "public_id": groups, "split": split})
    if task == "binary":
        out["y_true"] = y.astype(int)
        out["proba"] = proba[:, 1]
    else:
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), f"{task}/{stage}/{split}: rows do not sum to 1"
        out["y_true"] = [MULTICLASS_CLASSES[i] for i in y]
        for j, col in enumerate(PROBA_COLS):
            out[col] = proba[:, j]
    rna = df.set_index("pair_id").loc[out["pair_id"], STALENESS_COL]
    out["rna_available"] = rna.notna().to_numpy().astype(int)
    return out


def check_test_reproduction(task, stage, test_df):
    locked = pd.read_csv(f"{RESULTS_DIR}/{task}_{stage}_{ALGORITHM}_predictions.csv")
    assert set(locked["pair_id"]) == set(test_df["pair_id"]) and len(locked) == len(test_df), \
        f"{task}/{stage}: regenerated test pair_ids differ from locked predictions"
    merged = test_df.merge(locked, on="pair_id", suffixes=("", "_locked"), validate="one_to_one")
    assert (merged["public_id"] == merged["public_id_locked"]).all(), f"{task}/{stage}: public_id mismatch"
    if task == "binary":
        assert (merged["y_true"] == merged["y_true_locked"].astype(int)).all(), f"{task}/{stage}: y_true mismatch"
        max_diff = float(np.max(np.abs(merged["proba"] - merged["proba_locked"])))
        assert max_diff <= TEST_PROBA_ATOL, \
            f"{task}/{stage}: regenerated test probabilities differ from locked (max |diff| {max_diff:.2e}) - " \
            f"check package versions match the clean-rerun training environment"
        return {"max_abs_proba_diff_vs_locked": max_diff}
    assert (merged["y_true"] == merged["y_true_locked"]).all(), f"{task}/{stage}: y_true mismatch"
    argmax = np.array(MULTICLASS_CLASSES)[merged[PROBA_COLS].to_numpy().argmax(axis=1)]
    # "pred" exists only in the locked file, so merge leaves it unsuffixed
    n_label_mismatch = int((argmax != merged["pred"].to_numpy()).sum())
    assert n_label_mismatch == 0, \
        f"{task}/{stage}: {n_label_mismatch} regenerated test labels differ from locked predictions"
    return {"n_argmax_label_mismatches_vs_locked": 0}


def check_val_reproduction(task, stage, val_df):
    with open(f"{RESULTS_DIR}/{task}_{stage}_{ALGORITHM}_metrics.json") as f:
        locked = json.load(f)["val_metrics"]
    if task == "binary":
        recomputed = {"auprc": average_precision_score(val_df["y_true"], val_df["proba"]),
                      "brier_score": brier_score_loss(val_df["y_true"], val_df["proba"])}
    else:
        pred = np.array(MULTICLASS_CLASSES)[val_df[PROBA_COLS].to_numpy().argmax(axis=1)]
        # train_models.py computes macro-F1 on encoded labels over classes present
        y_idx = [MULTICLASS_CLASSES.index(v) for v in val_df["y_true"]]
        p_idx = [MULTICLASS_CLASSES.index(v) for v in pred]
        recomputed = {"macro_f1": f1_score(y_idx, p_idx, average="macro", zero_division=0)}
    for name, value in recomputed.items():
        assert abs(value - locked[name]) <= VAL_METRIC_ATOL, \
            f"{task}/{stage}: regenerated val {name} {value} != locked {locked[name]}"
    assert len(val_df) == locked["n"], f"{task}/{stage}: val row count differs from locked val_metrics"
    return {f"val_{k}_matches_locked": True for k in recomputed}


if __name__ == "__main__":
    missing = [p for p in required_inputs() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))
    hashes_before = {p: sha256(p) for p in required_inputs()}
    os.makedirs(PROBA_DIR, exist_ok=True)

    df = load_master_data()
    manifest = {"description": "Inference-only val/test probabilities for locked C/D XGBoost models.",
                "models": {}, "outputs": []}

    for task in TASKS:
        for stage in STAGES:
            key = f"{task}_{stage}_{ALGORITHM}"
            model = joblib.load(f"{MODELS_DIR}/{key}.joblib")
            preprocessor, meta = load_preprocessor_and_meta(stage)
            frames = {split: predict_split(df, task, stage, model, preprocessor, meta, split) for split in SPLITS}

            val_ids, test_ids = set(frames["val"]["public_id"]), set(frames["test"]["public_id"])
            assert val_ids.isdisjoint(test_ids), f"{key}: val/test patient overlap"

            checks = {**check_test_reproduction(task, stage, frames["test"]),
                      **check_val_reproduction(task, stage, frames["val"]),
                      "val_test_patients_disjoint": True}
            info = {"checks": checks}
            for split, frame in frames.items():
                path = f"{PROBA_DIR}/{key}_{split}.csv"
                frame.sort_values("pair_id").to_csv(path, index=False)
                manifest["outputs"].append(path)
                info[f"n_rows_{split}"] = int(len(frame))
                info[f"n_patients_{split}"] = int(frame["public_id"].nunique())
                info[f"n_rna_available_{split}"] = int(frame["rna_available"].sum())
            manifest["models"][key] = info
            print(f"{key}: val {info['n_rows_val']} rows / {info['n_patients_val']} pts, "
                  f"test {info['n_rows_test']} rows / {info['n_patients_test']} pts - reproduction checks PASS "
                  f"{checks}")

    hashes_after = {p: sha256(p) for p in required_inputs()}
    assert hashes_before == hashes_after, "A locked input changed during the run"
    manifest["locked_inputs_unchanged"] = True
    manifest["locked_input_sha256"] = hashes_after
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAll reproduction checks passed. Saved {len(manifest['outputs'])} probability files and {MANIFEST_PATH}")
