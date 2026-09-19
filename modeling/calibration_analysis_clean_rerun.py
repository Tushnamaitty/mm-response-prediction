"""
Calibration analysis - Step 2 of 2: can the predicted probabilities of
the locked clean-rerun C-XGBoost and D-XGBoost models be trusted?

Reads ONLY the validation/test probabilities written by
calibration_predictions_clean_rerun.py (Step 1). No model is retrained.

Design (pre-specified; the TEST split is used exactly once, for evaluation):
1. Models were fit on TRAIN only (train_models.py). VALIDATION was never
   used for training or tuning, so it is the calibration set.
2. Candidate methods, ordered simplest -> most complex:
   - binary:     uncalibrated, Platt (logistic on logit p), isotonic
   - multiclass: uncalibrated, temperature scaling (1 parameter; argmax,
                 and therefore macro-F1, unchanged)
3. Method selection uses VALIDATION ONLY, by 5-fold patient-grouped
   cross-fitting within validation (each calibrator is fit on 4/5 of
   validation patients and scored on the held-out 1/5). Rule: take the
   method with the lowest cross-fitted log-loss; if a SIMPLER method's
   cross-fitted Brier is within BRIER_TIE_MARGIN of it, take the simpler
   method instead.
4. Every calibrator is then refit on ALL validation rows and frozen.
5. TEST: every pre-specified method is evaluated once and reported; the
   validation-selected method is flagged. Test results never feed back
   into selection.
6. Uncertainty: 1000 patient-level bootstrap resamples of TEST
   (seed RANDOM_SEED), shared across C/D and methods so differences are
   paired. Calibrators stay frozen inside the bootstrap, so the CIs
   reflect test sampling only, not calibrator-fit uncertainty.

Metrics:
- binary: AUROC, AUPRC, Brier, scaled Brier (1 - Brier / prev(1-prev)),
  log-loss, calibration intercept (logistic, logit p as offset),
  calibration slope (logistic on logit p), mean predicted vs observed rate.
- multiclass: macro-F1 (fixed 6-class label set), multiclass Brier
  (mean of per-row summed squared error, range 0-2), scaled Brier
  (vs the prevalence-only forecast), log-loss; per class one-vs-rest
  Brier, calibration intercept and slope.
Threshold-based metrics are deliberately excluded (the existing 0.25
threshold was chosen on uncalibrated validation probabilities).

Writes ONLY under artifacts/results_clean_rerun/calibration/.

Run from the repo root, after Step 1:
    python modeling/calibration_analysis_clean_rerun.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import expit, logit, softmax  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "modeling")
from calibration_predictions_clean_rerun import (  # noqa: E402
    OUT_DIR, PROBA_DIR, MANIFEST_PATH, PROBA_COLS, STAGES, TASKS, SPLITS, ALGORITHM, sha256,
)
from rna_subset_paired_analysis_clean_rerun import patient_bootstrap_indices  # noqa: E402
from train_models import MULTICLASS_CLASSES, RANDOM_SEED  # noqa: E402

FIG_DIR = f"{OUT_DIR}/figures"
SUMMARY_PATH = f"{OUT_DIR}/calibration_summary.json"
N_BOOTSTRAP = 1000
N_CV_FOLDS = 5
N_BINS = 10
EPS = 1e-6
BRIER_TIE_MARGIN = 0.001
MODEL_LABEL = {"c": "C-XGBoost", "d": "D-XGBoost"}
CLASS_SHORT = dict(zip(MULTICLASS_CLASSES, ["PD", "SD", "PR", "VGPR", "CR", "sCR"]))
ANALYSIS_SETS = {"all_test_rows": None, "rna_available_test_rows": "rna_available"}

# Figure tokens: categorical slots 1-3 in fixed order (validated palette),
# marker shape as secondary encoding; text in ink, never series colour.
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#262624", "#6b6a64", "#e6e5e0"
METHOD_STYLE = {"uncalibrated": ("#2a78d6", "o"), "platt": ("#eb6834", "s"),
                "temperature": ("#eb6834", "s"), "isotonic": ("#1baf7a", "^")}


# ---------------------------------------------------------------------
# Metrics (return NaN, never raise, when undefined on a resample)
# ---------------------------------------------------------------------

def _logistic_fit(x, y, offset=None, fit_slope=True, max_iter=100, tol=1e-10):
    """Unpenalised logistic regression by Newton-Raphson. Returns the
    coefficient vector, or None if y is single-class or the fit fails."""
    if len(y) < 2 or np.all(y == y[0]):
        return None
    X = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((len(x), 1))
    off = np.zeros_like(x) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = expit(X @ beta + off)
        w = mu * (1 - mu)
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return None
        if np.max(np.abs(step)) < tol:
            return beta
    return None


def calibration_intercept_slope(y01, p):
    lp = logit(np.clip(p, EPS, 1 - EPS))
    icpt = _logistic_fit(lp, y01, offset=lp, fit_slope=False)
    slope = _logistic_fit(lp, y01)
    return (float(icpt[0]) if icpt is not None else np.nan,
            float(slope[1]) if slope is not None else np.nan)


def binary_scores(y, p):
    single_class = len(np.unique(y)) < 2
    prev = float(np.mean(y))
    brier = float(np.mean((p - y) ** 2))
    icpt, slope = calibration_intercept_slope(y, p)
    pc = np.clip(p, EPS, 1 - EPS)
    return {
        "auroc": np.nan if single_class else float(roc_auc_score(y, p)),
        "auprc": np.nan if single_class else float(average_precision_score(y, p)),
        "brier": brier,
        "scaled_brier": np.nan if single_class else 1 - brier / (prev * (1 - prev)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "calibration_intercept": icpt,
        "calibration_slope": slope,
        "mean_predicted": float(np.mean(p)),
        "observed_rate": prev,
    }


def multiclass_scores(y, P):
    Y = np.eye(len(MULTICLASS_CLASSES))[y]
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    q = Y.mean(axis=0)
    ref = 1 - float(np.sum(q ** 2))
    out = {
        "macro_f1": float(f1_score(y, P.argmax(axis=1), labels=list(range(len(MULTICLASS_CLASSES))),
                                   average="macro", zero_division=0)),
        "brier": brier,
        "scaled_brier": 1 - brier / ref if ref > 0 else np.nan,
        "log_loss": float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)))),
    }
    for k, cls in enumerate(MULTICLASS_CLASSES):
        icpt, slope = calibration_intercept_slope(Y[:, k], P[:, k])
        out[f"{cls}|brier"] = float(np.mean((P[:, k] - Y[:, k]) ** 2))
        out[f"{cls}|calibration_intercept"] = icpt
        out[f"{cls}|calibration_slope"] = slope
        out[f"{cls}|mean_predicted"] = float(np.mean(P[:, k]))
        out[f"{cls}|observed_rate"] = float(q[k])
    return out


# ---------------------------------------------------------------------
# Calibrators: fit(scores, y) on VALIDATION only; predict(scores)
# ---------------------------------------------------------------------

class Uncalibrated:
    params = {}

    def fit(self, s, y):
        return self

    def predict(self, s):
        return s


class Platt:
    def fit(self, p, y):
        beta = _logistic_fit(logit(np.clip(p, EPS, 1 - EPS)), y)
        assert beta is not None, "Platt fit failed"
        self.a, self.b = float(beta[0]), float(beta[1])
        self.params = {"platt_a": self.a, "platt_b": self.b}
        return self

    def predict(self, p):
        return expit(self.a + self.b * logit(np.clip(p, EPS, 1 - EPS)))


class Isotonic:
    params = {}

    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p):
        return self.iso.predict(p)


class Temperature:
    def fit(self, P, y):
        logP = np.log(np.clip(P, EPS, 1))

        def nll(t):
            Q = softmax(logP / t, axis=1)
            return -np.mean(np.log(np.clip(Q[np.arange(len(y)), y], EPS, 1)))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        assert res.success, "Temperature fit failed"
        self.T = float(res.x)
        self.params = {"temperature": self.T}
        return self

    def predict(self, P):
        return softmax(np.log(np.clip(P, EPS, 1)) / self.T, axis=1)


TASK_SPECS = {
    "binary": {"methods": {"uncalibrated": Uncalibrated, "platt": Platt, "isotonic": Isotonic},
               "score_fn": binary_scores},
    "multiclass": {"methods": {"uncalibrated": Uncalibrated, "temperature": Temperature},
                   "score_fn": multiclass_scores},
}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def check_step1_manifest():
    assert os.path.exists(MANIFEST_PATH), f"Missing {MANIFEST_PATH} - run calibration_predictions_clean_rerun.py first"
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert manifest.get("locked_inputs_unchanged") is True, "Step 1 did not confirm locked inputs unchanged"
    for key, info in manifest["models"].items():
        assert all(v is True or k.startswith(("max_abs", "n_argmax")) for k, v in info["checks"].items()), \
            f"Step 1 reproduction check failed for {key}"
    missing = [p for p in manifest["outputs"] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing Step 1 output(s):\n  " + "\n  ".join(missing))
    return manifest


def load_task(task):
    """{(stage, split): frame}, rows sorted by pair_id; C and D aligned."""
    frames = {(stage, split): pd.read_csv(f"{PROBA_DIR}/{task}_{stage}_{ALGORITHM}_{split}.csv")
              .sort_values("pair_id").reset_index(drop=True)
              for stage in STAGES for split in SPLITS}
    for split in SPLITS:
        c, d = frames[("c", split)], frames[("d", split)]
        for col in ["pair_id", "public_id", "y_true", "rna_available"]:
            assert c[col].equals(d[col]), f"{task}/{split}: C and D rows not aligned on {col}"
    assert set(frames[("c", "val")]["public_id"]).isdisjoint(frames[("c", "test")]["public_id"]), \
        f"{task}: val/test patient overlap"
    return frames


def xy(frame, task):
    if task == "binary":
        return frame["y_true"].to_numpy(dtype=int), frame["proba"].to_numpy(dtype=float)
    y = np.array([MULTICLASS_CLASSES.index(v) for v in frame["y_true"]])
    return y, frame[PROBA_COLS].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Selection on VALIDATION only
# ---------------------------------------------------------------------

def cross_fit_validation(task, y_val, s_val, groups_val):
    """Out-of-fold (patient-grouped) calibrated validation scores per method."""
    spec = TASK_SPECS[task]
    folds = list(GroupKFold(n_splits=N_CV_FOLDS).split(s_val, y_val, groups_val))
    results = {}
    for name, cls in spec["methods"].items():
        oof = np.zeros_like(s_val)
        for tr, te in folds:
            oof[te] = cls().fit(s_val[tr], y_val[tr]).predict(s_val[te])
        results[name] = spec["score_fn"](y_val, oof)
    return results


def select_method(cv_results, order):
    best = min(order, key=lambda m: cv_results[m]["log_loss"])
    for m in order[:order.index(best)]:
        if cv_results[m]["brier"] <= cv_results[best]["brier"] + BRIER_TIE_MARGIN:
            return m, best
    return best, best


# ---------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------

def bootstrap_table(score_fn, y, s, resamples):
    return pd.DataFrame([score_fn(y[idx], s[idx]) for idx in resamples])


def ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, 0
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), int(len(arr))


def reliability_bins(y01, p, n_bins=N_BINS):
    bins = pd.qcut(pd.Series(p).rank(method="first"), q=n_bins, labels=False)
    frame = pd.DataFrame({"bin": bins, "p": p, "y": y01})
    g = frame.groupby("bin")
    return pd.DataFrame({"bin": g.size().index, "mean_predicted": g["p"].mean().values,
                         "observed_rate": g["y"].mean().values, "n_rows": g.size().values,
                         "n_events": g["y"].sum().values.astype(int)})


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_binary(curves, test_scores, selected):
    fig = plt.figure(figsize=(10, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.08, wspace=0.18)
    for col, stage in enumerate(STAGES):
        ax, axh = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        for a in (ax, axh):
            _style_axis(a)
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1, label="Perfect calibration")
        for method in TASK_SPECS["binary"]["methods"]:
            c = curves[(curves["model"] == MODEL_LABEL[stage]) & (curves["method"] == method)]
            color, marker = METHOD_STYLE[method]
            is_sel = method == selected[stage]
            ax.plot(c["mean_predicted"], c["observed_rate"], color=color, marker=marker,
                    markersize=6, linewidth=2 if is_sel else 1.2, alpha=1 if is_sel else 0.75,
                    markeredgecolor=SURFACE, markeredgewidth=1,
                    label=f"{method}{' (selected on validation)' if is_sel else ''}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{MODEL_LABEL[stage]} - binary (improved)", color=INK, fontsize=10, loc="left")
        ax.set_ylabel("Observed proportion improved" if col == 0 else "", color=INK, fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="upper left")
        u = test_scores[stage]["uncalibrated"]
        ax.text(0.98, 0.04, f"Uncalibrated: Brier {u['brier']:.3f}\nintercept {u['calibration_intercept']:+.2f}, "
                f"slope {u['calibration_slope']:.2f}", ha="right", va="bottom", fontsize=7.5, color=INK_MUTED,
                transform=ax.transAxes)
        axh.hist(test_scores[stage]["_uncal_p"], bins=40, range=(0, 1), color=INK_MUTED, alpha=0.6)
        axh.set_xlim(0, 1)
        axh.set_yticks([])
        axh.set_xlabel("Predicted probability (test, 10 equal-count bins)", color=INK, fontsize=9)
    path = f"{FIG_DIR}/binary_reliability_c_vs_d.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_multiclass(curves, stage, selected):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, cls in zip(axes.ravel(), MULTICLASS_CLASSES):
        _style_axis(ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1, label="Perfect calibration")
        for method in TASK_SPECS["multiclass"]["methods"]:
            c = curves[(curves["model"] == MODEL_LABEL[stage]) & (curves["method"] == method)
                       & (curves["class"] == cls)]
            color, marker = METHOD_STYLE[method]
            is_sel = method == selected
            ax.plot(c["mean_predicted"], c["observed_rate"], color=color, marker=marker, markersize=5,
                    linewidth=2 if is_sel else 1.2, alpha=1 if is_sel else 0.75,
                    markeredgecolor=SURFACE, markeredgewidth=1,
                    label=f"{method}{' (selected)' if is_sel else ''}")
        n_events = int(c["n_events"].sum()) if len(c) else 0
        ax.set_title(f"{CLASS_SHORT[cls]} (n events = {n_events})", color=INK, fontsize=9, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    for ax in axes[1, :]:
        ax.set_xlabel("Predicted probability (one-vs-rest)", color=INK, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed proportion", color=INK, fontsize=8.5)
    fig.suptitle(f"{MODEL_LABEL[stage]} - multiclass next IMWG response, test set", color=INK,
                 fontsize=10.5, x=0.02, ha="left")
    path = f"{FIG_DIR}/multiclass_ovr_reliability_{stage}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------
# Main per-task analysis
# ---------------------------------------------------------------------

def analyse_task(task, checks):
    spec = TASK_SPECS[task]
    order = list(spec["methods"])
    frames = load_task(task)
    selection_rows, metric_rows, curve_rows, diff_rows = [], [], [], []
    selected, test_calibrated, test_scores = {}, {}, {}

    test_frame = frames[("c", "test")]
    groups_test = test_frame["public_id"].to_numpy()
    subsets = {name: (np.ones(len(test_frame), bool) if col is None else test_frame[col].to_numpy() == 1)
               for name, col in ANALYSIS_SETS.items()}

    for stage in STAGES:
        y_val, s_val = xy(frames[(stage, "val")], task)
        g_val = frames[(stage, "val")]["public_id"].to_numpy()
        y_te, s_te = xy(frames[(stage, "test")], task)

        # --- selection: VALIDATION ONLY ---
        cv_results = cross_fit_validation(task, y_val, s_val, g_val)
        sel, lowest = select_method(cv_results, order)
        selected[stage] = sel

        # --- freeze calibrators fit on ALL validation rows, apply to TEST once ---
        test_calibrated[stage], test_scores[stage] = {}, {"_uncal_p": s_te}
        for method, cls in spec["methods"].items():
            cal = cls().fit(s_val, y_val)
            selection_rows.append({
                "task": task, "model": MODEL_LABEL[stage], "method": method,
                "val_crossfit_log_loss": cv_results[method]["log_loss"],
                "val_crossfit_brier": cv_results[method]["brier"],
                "lowest_crossfit_log_loss": method == lowest, "selected": method == sel,
                "n_val_rows_fit": int(len(y_val)), "n_val_patients_fit": int(len(set(g_val))),
                **cal.params})
            if method == "platt":
                assert cal.b > 0, f"{task}/{stage}: Platt slope <= 0 would reverse the ranking"
            test_calibrated[stage][method] = cal.predict(s_te)
            test_scores[stage][method] = spec["score_fn"](y_te, test_calibrated[stage][method])

        # uncalibrated validation reference row (no CI; not used for selection)
        for metric, value in spec["score_fn"](y_val, s_val).items():
            metric_rows.append({"task": task, "model": MODEL_LABEL[stage], "method": "uncalibrated",
                                "split": "val", "metric": metric, "estimate": value})

        # reliability curves (test)
        for method, s in test_calibrated[stage].items():
            if task == "binary":
                b = reliability_bins(y_te, s)
                curve_rows.append(b.assign(task=task, model=MODEL_LABEL[stage], method=method))
            else:
                for k, cls in enumerate(MULTICLASS_CLASSES):
                    b = reliability_bins((y_te == k).astype(int), s[:, k])
                    curve_rows.append(b.assign(task=task, model=MODEL_LABEL[stage], method=method, **{"class": cls}))

    # --- structural checks on the frozen calibrators ---
    for stage in STAGES:
        if task == "binary":
            gap = abs(test_scores[stage]["platt"]["auroc"] - test_scores[stage]["uncalibrated"]["auroc"])
            assert gap < 1e-6, f"{stage}: Platt changed AUROC by {gap}"
        else:
            assert np.array_equal(test_calibrated[stage]["temperature"].argmax(axis=1),
                                  test_calibrated[stage]["uncalibrated"].argmax(axis=1)), \
                f"{stage}: temperature scaling changed predicted labels"
        for s in test_calibrated[stage].values():
            assert np.all((s >= 0) & (s <= 1)), "calibrated probability outside [0, 1]"
            if task == "multiclass":
                assert np.allclose(s.sum(axis=1), 1.0, atol=1e-6), "calibrated rows do not sum to 1"
    checks[f"{task}_monotone_recalibration_preserves_discrimination"] = True
    checks[f"{task}_calibrators_fit_on_validation_only"] = True

    # --- TEST evaluation with shared patient-bootstrap resamples ---
    y_te, _ = xy(test_frame, task)
    for set_name, mask in subsets.items():
        idx_rows = np.where(mask)[0]
        resamples = patient_bootstrap_indices(groups_test[idx_rows], n=N_BOOTSTRAP, seed=RANDOM_SEED)
        resamples = [idx_rows[r] for r in resamples]
        boot = {(stage, method): bootstrap_table(spec["score_fn"], y_te, s, resamples)
                for stage in STAGES for method, s in test_calibrated[stage].items()}
        point = {(stage, method): spec["score_fn"](y_te[idx_rows], s[idx_rows])
                 for stage in STAGES for method, s in test_calibrated[stage].items()}

        if set_name == "all_test_rows":
            for (stage, method), pt in point.items():
                for metric, value in pt.items():
                    lo, hi, n_valid = ci(boot[(stage, method)][metric])
                    metric_rows.append({"task": task, "model": MODEL_LABEL[stage], "method": method,
                                        "split": "test", "selected": method == selected[stage],
                                        "metric": metric, "estimate": value, "ci_lower": lo, "ci_upper": hi,
                                        "n_boot_valid": n_valid, "n_rows": int(len(idx_rows)),
                                        "n_patients": int(len(set(groups_test[idx_rows])))})

        headline = [m for m in point[("c", "uncalibrated")] if "|" not in m]
        for method in order:  # D - C, same method, paired
            for metric in headline:
                d = boot[("d", method)][metric] - boot[("c", method)][metric]
                lo, hi, n_valid = ci(d)
                diff_rows.append({"task": task, "analysis_set": set_name, "comparison": "D - C",
                                  "method": method, "metric": metric,
                                  "estimate": point[("d", method)][metric] - point[("c", method)][metric],
                                  "ci_lower": lo, "ci_upper": hi, "n_boot_valid": n_valid,
                                  "n_rows": int(len(idx_rows))})
        if set_name == "all_test_rows":  # recalibrated - uncalibrated, within model
            for stage in STAGES:
                for method in order[1:]:
                    for metric in ["brier", "log_loss", "calibration_intercept", "calibration_slope"]:
                        if metric not in point[(stage, method)]:
                            continue
                        d = boot[(stage, method)][metric] - boot[(stage, "uncalibrated")][metric]
                        lo, hi, n_valid = ci(d)
                        diff_rows.append({"task": task, "analysis_set": set_name,
                                          "comparison": f"{MODEL_LABEL[stage]}: {method} - uncalibrated",
                                          "method": method, "metric": metric,
                                          "estimate": point[(stage, method)][metric]
                                          - point[(stage, "uncalibrated")][metric],
                                          "ci_lower": lo, "ci_upper": hi, "n_boot_valid": n_valid,
                                          "n_rows": int(len(idx_rows))})

    curves = pd.concat(curve_rows, ignore_index=True)
    return (pd.DataFrame(selection_rows), pd.DataFrame(metric_rows), curves,
            pd.DataFrame(diff_rows), selected, test_scores)


def split_multiclass_metrics(metrics_df):
    per_class = metrics_df[metrics_df["metric"].str.contains("|", regex=False)].copy()
    per_class[["class", "metric"]] = per_class["metric"].str.split("|", expand=True)
    overall = metrics_df[~metrics_df["metric"].str.contains("|", regex=False)]
    return overall, per_class


if __name__ == "__main__":
    manifest = check_step1_manifest()
    os.makedirs(FIG_DIR, exist_ok=True)
    checks = {"step1_reproduction_checks_passed": True}
    inputs_before = {p: sha256(p) for p in manifest["outputs"] + [MANIFEST_PATH]}

    outputs, summary_results = [], {}
    all_selection, all_diffs = [], []
    for task in TASKS:
        print(f"\n=== {task} ===")
        sel_df, metrics_df, curves, diff_df, selected, test_scores = analyse_task(task, checks)
        all_selection.append(sel_df)
        all_diffs.append(diff_df)
        print(sel_df[["model", "method", "val_crossfit_log_loss", "val_crossfit_brier", "selected"]]
              .to_string(index=False))

        if task == "binary":
            files = {"calibration_binary_metrics.csv": metrics_df, "calibration_binary_curves.csv": curves}
            outputs.append(plot_binary(curves, test_scores, selected))
        else:
            overall, per_class = split_multiclass_metrics(metrics_df)
            files = {"calibration_multiclass_metrics.csv": overall,
                     "calibration_multiclass_per_class.csv": per_class,
                     "calibration_multiclass_ovr_curves.csv": curves}
            for stage in STAGES:
                outputs.append(plot_multiclass(curves, stage, selected[stage]))
        for name, frame in files.items():
            frame.to_csv(f"{OUT_DIR}/{name}", index=False)
            outputs.append(f"{OUT_DIR}/{name}")

        test_rows = metrics_df[(metrics_df["split"] == "test") & ~metrics_df["metric"].str.contains("|", regex=False)]
        summary_results[task] = {
            "selected_method_on_validation": {MODEL_LABEL[s]: m for s, m in selected.items()},
            "test_metrics": json.loads(test_rows[["model", "method", "selected", "metric", "estimate",
                                                  "ci_lower", "ci_upper"]].to_json(orient="records",
                                                                                   double_precision=6)),
            "d_minus_c": json.loads(diff_df[diff_df["comparison"] == "D - C"][
                ["analysis_set", "method", "metric", "estimate", "ci_lower", "ci_upper"]]
                .to_json(orient="records", double_precision=6)),
        }
        for _, r in test_rows[test_rows["metric"].isin(["brier", "calibration_intercept", "calibration_slope",
                                                        "macro_f1", "auroc"])].iterrows():
            print(f"  test {r['model']} {r['method']:12s} {r['metric']:22s} {r['estimate']:.4f} "
                  f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]{'  <- selected' if r['selected'] else ''}")

    pd.concat(all_selection).to_csv(f"{OUT_DIR}/calibration_val_selection.csv", index=False)
    pd.concat(all_diffs).to_csv(f"{OUT_DIR}/calibration_paired_differences.csv", index=False)
    outputs += [f"{OUT_DIR}/calibration_val_selection.csv", f"{OUT_DIR}/calibration_paired_differences.csv"]

    inputs_after = {p: sha256(p) for p in manifest["outputs"] + [MANIFEST_PATH]}
    assert inputs_before == inputs_after, "A Step 1 input changed during the run"
    checks["step1_inputs_unchanged_during_run"] = True

    summary = {
        "description": "Calibration of locked clean-rerun C-XGBoost and D-XGBoost (binary and multiclass).",
        "design": {
            "calibration_set": "validation split (never used for training or tuning)",
            "evaluation_set": "test split, evaluated once; all task-eligible rows (D - C also on RNA-available rows)",
            "methods": {t: list(TASK_SPECS[t]["methods"]) for t in TASKS},
            "selection_rule": (f"{N_CV_FOLDS}-fold patient-grouped cross-fitting within validation; lowest "
                               f"cross-fitted log-loss, but a simpler method is preferred if its cross-fitted "
                               f"Brier is within {BRIER_TIE_MARGIN}. Final calibrators refit on all validation "
                               f"rows and frozen."),
            "uncertainty": (f"{N_BOOTSTRAP} patient-level bootstrap resamples of test, seed {RANDOM_SEED}, "
                            f"shared across models/methods (paired); calibrators frozen (test-sampling "
                            f"uncertainty only)."),
            "excluded": "threshold-based metrics; ECE; per-class isotonic/Dirichlet multiclass calibration",
        },
        "results": summary_results,
        "checks": checks,
        "input_sha256": inputs_after,
        "outputs": outputs,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    for p in outputs + [SUMMARY_PATH]:
        print(f"Saved: {p}")
