"""
Supplementary Figure S1 (calibration) data extraction and verification (read-only).

Run from repo root:
    python figures/figS_calibration_extract_data.py

Sources
    * Locked test/validation probabilities of XGBoost Layer C and Layer D (current branch):
        artifacts/results_clean_rerun/calibration/probabilities/{binary,multiclass}_{c,d}_xgboost_{val,test}.csv
    * Locked next-visit predictions (cross-check of y_true and probabilities):
        artifacts/results_clean_rerun/binary_{c,d}_xgboost_predictions.csv
    * Calibration summary and analysis script, read from branch origin/subgroup-analysis with
      `git show` (no checkout, working tree untouched):
        artifacts/results_clean_rerun/calibration/calibration_summary.json
        modeling/calibration_analysis_clean_rerun.py

What is derived here (no model is fitted or refitted)
    * Reliability-curve bins: 10 quantile bins of the stored, UNCALIBRATED test probabilities
      (the method selected on validation for both binary models), using the same binning rule as the
      calibration script (rank(method="first") then qcut into 10 groups). Bin counts, events,
      mean predicted probability and observed improvement rate.
    * Prediction-distribution histograms of the same probabilities.
    * Point metrics recomputed from stored probabilities with the script's formulas, solely to verify
      the summary (Brier, log loss, calibration intercept/slope, mean predicted, observed rate).
    Layer D - Layer C paired differences and their 95% CIs are taken verbatim from the summary
    (1,000 patient-level bootstrap resamples of the test set); no interval is computed here.

The script exits non-zero if any check fails.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

BRANCH = "origin/subgroup-analysis"
SUMMARY_GIT = "artifacts/results_clean_rerun/calibration/calibration_summary.json"
SCRIPT_GIT = "modeling/calibration_analysis_clean_rerun.py"
PROB = Path("artifacts/results_clean_rerun/calibration/probabilities")
MANIFEST = Path("artifacts/results_clean_rerun/calibration/calibration_predictions_manifest.json")
LOCKED_PRED = Path("artifacts/results_clean_rerun")
OUT = Path("figures/data/FigS_calibration")
N_BINS, EPS = 10, 1e-6
LAYER = {"c": "Layer C (clinical + history + treatment)", "d": "Layer D (Layer C + RNA)"}
MODEL_KEY = {"c": "C-XGBoost", "d": "D-XGBoost"}

checks = []


def check(name, observed, expected, source, tol=None):
    ok = abs(float(observed) - float(expected)) <= tol if tol is not None else observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(ok),
                   "source": str(source)})


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_show(path):
    blob = subprocess.run(["git", "rev-parse", f"{BRANCH}:{path}"], capture_output=True, text=True, check=True).stdout.strip()
    data = subprocess.run(["git", "show", f"{BRANCH}:{path}"], capture_output=True, check=True).stdout
    return data, blob


# --- metric formulas copied from modeling/calibration_analysis_clean_rerun.py (verification only) ---
def _logistic_fit(x, y, offset=None, fit_slope=True, max_iter=100, tol=1e-10):
    if len(y) < 2 or np.all(y == y[0]):
        return None
    X = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((len(x), 1))
    off = np.zeros_like(x) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = expit(X @ beta + off)
        w = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return beta
    return None


def point_metrics(y, p):
    lp = logit(np.clip(p, EPS, 1 - EPS))
    icpt = _logistic_fit(lp, y, offset=lp, fit_slope=False)
    slope = _logistic_fit(lp, y)
    pc = np.clip(p, EPS, 1 - EPS)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "calibration_intercept": float(icpt[0]), "calibration_slope": float(slope[1]),
            "mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y))}


def reliability_bins(y, p):
    bins = pd.qcut(pd.Series(p).rank(method="first"), q=N_BINS, labels=False)
    f = pd.DataFrame({"bin": bins, "p": p, "y": y})
    g = f.groupby("bin")
    return pd.DataFrame({"bin": g.size().index + 1, "n_rows": g.size().values, "n_events": g.y.sum().values.astype(int),
                         "mean_predicted": g.p.mean().values, "observed_rate": g.y.mean().values,
                         "p_min": g.p.min().values, "p_max": g.p.max().values})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, blob = git_show(SUMMARY_GIT)
    summ = json.loads(raw)
    (OUT / "calibration_summary_from_subgroup_branch.json").write_bytes(raw)
    script_raw, script_blob = git_show(SCRIPT_GIT)
    check("summary checks all passed", all(summ["checks"].values()), True, f"{BRANCH}:{SUMMARY_GIT}")
    check("calibrators fit on validation only", summ["checks"]["binary_calibrators_fit_on_validation_only"], True,
          f"{BRANCH}:{SUMMARY_GIT}")
    check("script uses 10 bins and eps 1e-6", ("N_BINS = 10" in script_raw.decode()) and ("EPS = 1e-6" in script_raw.decode()),
          True, f"{BRANCH}:{SCRIPT_GIT}")
    check("script binning rule = rank(first) + qcut", 'qcut(pd.Series(p).rank(method="first")' in script_raw.decode(), True,
          f"{BRANCH}:{SCRIPT_GIT}")

    # inputs used by the summary must be the files on disk now
    for path, h in summ["input_sha256"].items():
        check(f"input hash {Path(path).name}", sha_bytes(Path(path).read_bytes()), h, path)
    man = json.load(open(MANIFEST))
    sel = summ["results"]["binary"]["selected_method_on_validation"]
    check("binary method selected on validation: uncalibrated for C and D", sel,
          {"C-XGBoost": "uncalibrated", "D-XGBoost": "uncalibrated"}, f"{BRANCH}:{SUMMARY_GIT}")
    check("six-class selected methods", summ["results"]["multiclass"]["selected_method_on_validation"],
          {"C-XGBoost": "uncalibrated", "D-XGBoost": "temperature"}, f"{BRANCH}:{SUMMARY_GIT}")

    tm = pd.DataFrame(summ["results"]["binary"]["test_metrics"])
    bins_all, hist_all, metric_rows = [], [], []
    frames = {}
    for m in ("c", "d"):
        f = pd.read_csv(PROB / f"binary_{m}_xgboost_test.csv")
        v = pd.read_csv(PROB / f"binary_{m}_xgboost_val.csv")
        frames[m] = f
        mm = man["models"][f"binary_{m}_xgboost"]
        check(f"{m.upper()} test rows / patients", (len(f), f.public_id.nunique()), (mm["n_rows_test"], mm["n_patients_test"]), MANIFEST)
        check(f"{m.upper()} test rows = 1930, patients = 155", (len(f), f.public_id.nunique()), (1930, 155), PROB)
        check(f"{m.upper()} split column = test", set(f.split), {"test"}, PROB)
        check(f"{m.upper()} val/test patients disjoint", len(set(f.public_id) & set(v.public_id)), 0, PROB)
        check(f"{m.upper()} probabilities in [0, 1]", bool(f.proba.between(0, 1).all()), True, PROB)
        check(f"{m.upper()} labels binary 0/1", set(f.y_true.unique()), {0, 1}, PROB)
        lk = pd.read_csv(LOCKED_PRED / f"binary_{m}_xgboost_predictions.csv")
        j = f.merge(lk, on="pair_id", suffixes=("", "_locked"), validate="one_to_one")
        check(f"{m.upper()} pair_ids identical to locked predictions", len(j), len(f), LOCKED_PRED)
        check(f"{m.upper()} patient ids identical to locked", bool((j.public_id == j.public_id_locked).all()), True, LOCKED_PRED)
        check(f"{m.upper()} y_true identical to locked", bool((j.y_true == j.y_true_locked).all()), True, LOCKED_PRED)
        check(f"{m.upper()} proba vs locked max abs diff", float((j.proba - j.proba_locked).abs().max()), 0.0, LOCKED_PRED, 1e-6)

        y, p = f.y_true.to_numpy().astype(float), f.proba.to_numpy()
        pm = point_metrics(y, p)
        for k, val in pm.items():
            s = tm[(tm.model == MODEL_KEY[m]) & (tm.method == "uncalibrated") & (tm.metric == k)].iloc[0]
            check(f"{m.upper()} recomputed {k} = summary", round(val, 6), round(float(s.estimate), 6), f"{BRANCH}:{SUMMARY_GIT}", 2e-6)
            metric_rows.append({"layer": m.upper(), "layer_label": LAYER[m], "algorithm": "XGBoost",
                                "calibration": "uncalibrated (selected on validation)", "metric": k,
                                "estimate": float(s.estimate), "ci_lower": float(s.ci_lower), "ci_upper": float(s.ci_upper),
                                "recomputed_from_probabilities": val})
        b = reliability_bins(y, p)
        b.insert(0, "layer", m.upper())
        b["layer_label"] = LAYER[m]
        b["probabilities"] = "uncalibrated test probabilities (no transformation)"
        bins_all.append(b)
        check(f"{m.upper()} bins sum to rows", int(b.n_rows.sum()), 1930, PROB)
        check(f"{m.upper()} bins sum to events", int(b.n_events.sum()), int(y.sum()), PROB)
        check(f"{m.upper()} bins of equal size (193)", sorted(b.n_rows.unique().tolist()), [193], PROB)
        edges = np.round(np.arange(0, 1.0001, 0.05), 2)
        cnt, _ = np.histogram(p, bins=edges)
        hist_all.append(pd.DataFrame({"layer": m.upper(), "bin_lower": edges[:-1], "bin_upper": edges[1:], "n_rows": cnt}))
    check("C and D on identical test pairs", bool((frames["c"].pair_id.values == frames["d"].pair_id.values).all()), True, PROB)
    check("observed improvement rate 353/1930", int(frames["c"].y_true.sum()), 353, PROB)

    pd.concat(bins_all).to_csv(OUT / "figS1A_reliability_bins.csv", index=False)
    pd.concat(hist_all).to_csv(OUT / "figS1A_prediction_histogram.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "figS1_binary_calibration_metrics.csv", index=False)

    # ---- S1B: Layer D - Layer C paired differences (verbatim) ----
    dmc = pd.DataFrame(summ["results"]["binary"]["d_minus_c"])
    keep = ["calibration_slope", "calibration_intercept", "brier", "log_loss", "scaled_brier", "mean_predicted"]
    info = {"calibration_slope": ("Calibration slope", "closer to 1 is better (1 = ideal)"),
            "calibration_intercept": ("Calibration intercept (calibration-in-the-large)", "closer to 0 is better"),
            "brier": ("Brier score", "lower is better"), "log_loss": ("Log loss", "lower is better"),
            "scaled_brier": ("Scaled Brier score", "higher is better"),
            "mean_predicted": ("Mean predicted probability", "compare with observed rate 0.183")}
    rows = []
    for aset in ("all_test_rows", "rna_available_test_rows"):
        for k in keep:
            s = dmc[(dmc.analysis_set == aset) & (dmc.method == "uncalibrated") & (dmc.metric == k)]
            if s.empty:
                continue
            s = s.iloc[0]
            rows.append({"analysis_set": aset, "metric": k, "metric_label": info[k][0], "interpretation": info[k][1],
                         "comparison": "Layer D - Layer C (XGBoost, uncalibrated, paired on identical test pairs)",
                         "delta": s.estimate, "ci_lower": s.ci_lower, "ci_upper": s.ci_upper,
                         "ci_method": "95% patient-level paired bootstrap (1,000 resamples of the test set; calibrators frozen)",
                         "ci_includes_0": bool(s.ci_lower <= 0 <= s.ci_upper)})
    b_df = pd.DataFrame(rows)
    b_df.to_csv(OUT / "figS1B_d_minus_c_calibration.csv", index=False)
    a = b_df[b_df.analysis_set == "all_test_rows"].set_index("metric")
    tmi = tm[tm.method == "uncalibrated"].set_index(["model", "metric"]).estimate
    for k in ("calibration_slope", "calibration_intercept", "brier", "log_loss"):
        check(f"B delta {k} = D - C point estimates", float(a.loc[k, "delta"]),
              float(tmi[("D-XGBoost", k)] - tmi[("C-XGBoost", k)]), f"{BRANCH}:{SUMMARY_GIT}", 2e-6)
    check("B all-test-row D-C CIs all include 0", bool(a.ci_includes_0.all()), True, f"{BRANCH}:{SUMMARY_GIT}")
    check("B rows available (all test rows)", sorted(a.index), sorted(keep), f"{BRANCH}:{SUMMARY_GIT}")

    defs = {
        "evidence_level": "held-out test set (155 patients, 1,930 binary visit pairs); descriptive calibration assessment",
        "task": "binary improvement at the next visit; test improvement rate 18.3% (353/1,930)",
        "models": "XGBoost Layer C and Layer D (locked clean-rerun models; reporting reference algorithm)",
        "calibration_method": ("Candidates (uncalibrated, Platt, isotonic) compared by 5-fold patient-grouped cross-fitting "
                               "on the validation set; uncalibrated selected for both C and D, so S1 shows raw model "
                               "probabilities with no validation-fitted transformation"),
        "reliability_bins": "10 equal-count quantile bins (193 visit pairs each) of the test probabilities; same rule as the original script",
        "bin_uncertainty": "none computed (no bin-level intervals exist in the locked outputs)",
        "slope_intercept": "logistic recalibration of the outcome on logit(p): slope (ideal 1) and intercept with logit(p) as offset (ideal 0)",
        "d_minus_c_ci": "95% patient-level paired bootstrap, 1,000 resamples, from the locked summary (not recomputed)",
        "scale_note": ("Slope, intercept, Brier and log loss have different units and magnitudes; plot each D - C "
                       "difference on its own axis (small multiples), never on one shared axis"),
        "excluded_by_design": summ["design"]["excluded"],
        "six_class_note": "six-class calibration (temperature scaling selected for Layer D) is not part of S1A/S1B",
    }
    json.dump(defs, open(OUT / "figS1_definitions.json", "w"), indent=2)

    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "figS1_verification_checks.csv", index=False)
    inputs = {str(PROB / f"binary_{m}_xgboost_{s}.csv"): sha_bytes((PROB / f"binary_{m}_xgboost_{s}.csv").read_bytes())
              for m in "cd" for s in ("test", "val")}
    inputs.update({str(LOCKED_PRED / f"binary_{m}_xgboost_predictions.csv"):
                   sha_bytes((LOCKED_PRED / f"binary_{m}_xgboost_predictions.csv").read_bytes()) for m in "cd"})
    inputs[str(MANIFEST)] = sha_bytes(MANIFEST.read_bytes())
    json.dump({"script": "figures/figS_calibration_extract_data.py",
               "git_sources": {f"{BRANCH}:{SUMMARY_GIT}": {"git_blob": blob, "sha256": sha_bytes(raw)},
                               f"{BRANCH}:{SCRIPT_GIT}": {"git_blob": script_blob, "sha256": sha_bytes(script_raw)}},
               "inputs_sha256": inputs,
               "outputs": sorted(p.name for p in OUT.iterdir() if p.name != "figS1_provenance.json") + ["figS1_provenance.json"]},
              open(OUT / "figS1_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
