"""
Independent recomputation of the Figure 3C (earlier working label: Figure 2C) paired-difference bootstrap intervals (read-only).

Run from the repository root:
    python figures/verify_fig3c_paired_cis.py

The plotted intervals are in figures/data/Fig2/fig2_paired_deltas_full_test.csv (24 rows: 3 algorithms x 2 tasks x B-A, C-B, D-C, D-A; 1,000 resamples). Their
extraction script is not in the repository, so this script recomputes them from the locked test predictions
(artifacts/results_clean_rerun/{task}_{layer}_{algorithm}_predictions.csv) with the repository's own patient-bootstrap procedure
(modeling/regenerate_main_model_cis_clean_rerun.py: patient_bootstrap_indices, seed 42, 1,000 resamples, all rows of a sampled patient, percentile 2.5/97.5), using the SAME
resample for the two layers being compared, and compares the result row by row. Three further seeds show how much the intervals move with the random seed.
It trains and refits nothing and changes no result file. Writes figures/data/Fig2/fig2_paired_deltas_independent_check.csv (aggregates only).
Exit code 1 if the point differences do not match the locked estimates.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

RC = Path("artifacts/results_clean_rerun")
PLOTTED = Path("figures/data/Fig2/fig2_paired_deltas_full_test.csv")
LOCKED = RC / "uncertainty" / "main_model_cis.csv"
OUT = Path("figures/data/Fig2/fig2_paired_deltas_independent_check.csv")
LAYERS = ["a", "b", "c", "d"]
ALGOS = ["logistic", "lightgbm", "xgboost"]
COMPARISONS = {"B - A": ("b", "a"), "C - B": ("c", "b"), "D - C": ("d", "c"), "D - A": ("d", "a")}
EXTRA_SEEDS = (7, 123, 2024)
CLASSES = ["progressive_disease", "stable_disease", "partial_response", "very_good_partial_response", "complete_response", "stringent_complete_response"]

spec = importlib.util.spec_from_file_location("regen_cis", "modeling/regenerate_main_model_cis_clean_rerun.py")
regen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regen)          # the __main__ block is guarded; only the helper functions are used

fails = []


def metric(task, y, p, pred, idx):
    if task == "binary":
        yy = y[idx]
        return np.nan if len(np.unique(yy)) < 2 else average_precision_score(yy, p[idx])
    return f1_score(y[idx], pred[idx], labels=CLASSES, average="macro", zero_division=0)


def load(task, layer, algo):
    d = pd.read_csv(RC / f"{task}_{layer}_{algo}_predictions.csv")
    return d.sort_values("pair_id").reset_index(drop=True)


def boot_diffs(task, algo, seed):
    data = {l: load(task, l, algo) for l in LAYERS}
    ref = data["a"]
    for l in LAYERS[1:]:
        if not (data[l].pair_id.to_numpy() == ref.pair_id.to_numpy()).all() or not (data[l].public_id.to_numpy() == ref.public_id.to_numpy()).all():
            raise SystemExit(f"{task} {algo}: layers {l} and a do not cover the same visit pairs; paired comparison impossible")
    pid = ref.public_id.to_numpy()
    resamples = regen.patient_bootstrap_indices(pid, seed=seed)
    y = {l: data[l].y_true.to_numpy() for l in LAYERS}
    p = {l: (data[l].proba.to_numpy() if task == "binary" else None) for l in LAYERS}
    pr = {l: (data[l].pred.to_numpy() if task != "binary" else None) for l in LAYERS}
    full = {l: metric(task, y[l], p[l], pr[l], np.arange(len(ref))) for l in LAYERS}
    boots = {l: np.array([metric(task, y[l], p[l], pr[l], idx) for idx in resamples]) for l in LAYERS}
    rows = {}
    for name, (hi, lo) in COMPARISONS.items():
        diffs = boots[hi] - boots[lo]
        diffs = diffs[~np.isnan(diffs)]
        rows[name] = (full[hi] - full[lo], float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), len(diffs), len(ref), len(np.unique(pid)))
    return rows


plotted = pd.read_csv(PLOTTED)
locked = pd.read_csv(LOCKED)
out = []
for task in ("binary", "multiclass"):
    for algo in ALGOS:
        main = boot_diffs(task, algo, seed=42)
        extra = [boot_diffs(task, algo, seed=s) for s in EXTRA_SEEDS]
        for name, (obs, lo, hi, nres, nrows, npat) in main.items():
            r = plotted[(plotted.task == task) & (plotted.algorithm == algo) & (plotted.comparison == name)]
            assert len(r) == 1, (task, algo, name)
            r = r.iloc[0]
            hi_l, lo_l = COMPARISONS[name]
            m = "auprc" if task == "binary" else "macro_f1"
            lk = locked[(locked.task == task) & (locked.algorithm == algo) & (locked.metric == m)].set_index("model_stage").point_estimate
            locked_diff = float(lk[hi_l]) - float(lk[lo_l])
            seeds = [e[name] for e in extra]
            out.append({
                "task": task, "algorithm": algo, "comparison": name, "metric": m,
                "plotted_diff": r.observed_diff, "recomputed_diff": obs, "locked_point_estimate_diff": locked_diff,
                "plotted_ci_lower": r.ci_lower, "plotted_ci_upper": r.ci_upper, "recomputed_seed42_ci_lower": lo, "recomputed_seed42_ci_upper": hi,
                "abs_diff_lower": abs(lo - r.ci_lower), "abs_diff_upper": abs(hi - r.ci_upper), "n_resamples_recomputed": nres, "n_resamples_plotted": int(r.n_resamples_used),
                "other_seed_ci_lower_min": min(s[1] for s in seeds), "other_seed_ci_lower_max": max(s[1] for s in seeds),
                "other_seed_ci_upper_min": min(s[2] for s in seeds), "other_seed_ci_upper_max": max(s[2] for s in seeds),
                "ci_includes_zero_plotted": bool(r.ci_lower <= 0 <= r.ci_upper), "ci_includes_zero_recomputed": bool(lo <= 0 <= hi),
                "n_rows": nrows, "n_patients": npat})
        print(f"done {task} {algo}", flush=True)

df = pd.DataFrame(out)
df.to_csv(OUT, index=False)
pt = np.abs(df.plotted_diff - df.recomputed_diff).max()
pt2 = np.abs(df.recomputed_diff - df.locked_point_estimate_diff).max()
ci_max = max(df.abs_diff_lower.max(), df.abs_diff_upper.max())
n_exact = int(((df.abs_diff_lower < 1e-9) & (df.abs_diff_upper < 1e-9)).sum())
span = ((df.other_seed_ci_upper_max - df.other_seed_ci_lower_min) - (df.recomputed_seed42_ci_upper - df.recomputed_seed42_ci_lower)).max()
zero_agree = int((df.ci_includes_zero_plotted == df.ci_includes_zero_recomputed).sum())
print(f"rows: {len(df)}")
print(f"max |plotted diff - recomputed diff| = {pt:.2e}; max |recomputed diff - locked point-estimate diff| = {pt2:.2e}")
print(f"seed-42 CIs identical to the plotted CIs (|difference| < 1e-9) in {n_exact} of {len(df)} rows; max |CI difference| = {ci_max:.2e}")
print(f"plotted vs recomputed 'CI includes zero' agree in {zero_agree} of {len(df)} rows")
print(f"other seeds: largest widening of the interval relative to seed 42 = {span:.4f}")
if pt > 1e-9 or pt2 > 1e-9:
    print("FAIL: point differences do not match", file=sys.stderr)
    sys.exit(1)
