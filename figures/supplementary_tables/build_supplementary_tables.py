"""
Build Supplementary Tables S1-S5 from verified figure data and locked results (read-only).

Run from repo root:
    python figures/supplementary_tables/build_supplementary_tables.py

Writes, for each table, a CSV and a formatted Markdown file to figures/supplementary_tables/, plus
supplementary_tables_verification.csv and supplementary_tables_provenance.json. No model is trained or
refit and no patient-level data are written. The only recomputation is the S2 per-layer point estimates on
RNA-available test pairs (stored test probabilities, same formulas as the locked calibration script); their
Layer D - Layer C differences are checked against the locked paired differences.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

OUT = Path("figures/supplementary_tables")
FD = Path("figures/data")
RC = Path("artifacts/results_clean_rerun")
SRC = {
    "s1_strata": FD / "Fig3/fig3C_treatment_and_trajectory_strata.csv",
    "s1_inter": FD / "Fig3/fig3C_interactions_secondary.csv",
    "s1_lock_d3": RC / "direction3/partA_predictive/partA_delta_within_group.csv",
    "s1_lock_d4": RC / "direction4/partA_predictive/partA_state_delta.csv",
    "s2_metrics": FD / "FigS_calibration/figS1_binary_calibration_metrics.csv",
    "s2_dmc": FD / "FigS_calibration/figS1B_d_minus_c_calibration.csv",
    "s2_summary": FD / "FigS_calibration/calibration_summary_from_subgroup_branch.json",
    "s2_prob_c": RC / "calibration/probabilities/binary_c_xgboost_test.csv",
    "s2_prob_d": RC / "calibration/probabilities/binary_d_xgboost_test.csv",
    "s2_rna_pd": RC / "uncertainty/rna_subset_paired_differences.csv",
    "s3_shares": FD / "FigS_shap/figS3A_ext_component_shares.csv",
    "s3_perclass": FD / "FigS_shap/figS3A_ext_component_shares_per_class.csv",
    "s3_orig": FD / "FigS_shap/figS3A_family_shares.csv",
    "s4_auprc": FD / "FigS_country/figS2A_binary_auprc.csv",
    "s4_f1": FD / "FigS_country/figS2B_sixclass_macro_f1.csv",
    "s4_auroc": FD / "FigS_country/figS2_auroc_caption_only.csv",
    "s4_overlap": FD / "FigS_country/figS2_tuning_overlap.json",
    "s5_ibs": FD / "Fig4/fig4_ibs_caption_only.csv",
    "s5_forest": FD / "Fig4/fig4B_delta_c_forest.csv",
    "s5_summary": FD / "Fig4/fig4A_cindex_summary.csv",
    "s5_lock": RC / "survival_incremental/final_summary.json",
    "s5_lock_richer": RC / "survival_richer_baseline_sensitivity/final_summary.json",
}
COUNTRY_GIT = ("origin/modeling-country-robustness",
               "results/robustness_country/binary_c_xgboost_us_to_spain_canada_metrics.json")
EPS = 1e-6
checks = []


def check(table, name, ok, observed="", expected=""):
    checks.append({"table": table, "check": name, "passed": bool(ok), "observed": str(observed)[:160],
                   "expected": str(expected)[:160]})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def ci(lo, hi, nd=4):
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return "not available"
    return f"{f(lo, nd)} to {f(hi, nd)}"


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def md_table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write(name, csv_df, md_text):
    csv_df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(md_text, encoding="utf-8")


# ------------------------------------------------------------------------------------------------
def _sec_sig(df):
    s = df[(df.row_type == "stratum") & df.significant_after_adjustment & ~df.metric.isin(["auprc", "rps"])]
    lab = {"auroc": "ΔAUROC", "logloss": "Δlog loss"}
    return "; ".join(f"{r.stratum} ({r.task}) {lab[r.metric]} {f(r.estimate_d_minus_c, 4, True)} "
                     f"[{ci(r.ci_lower, r.ci_upper)}], BH p = {pfmt(r.p_adjusted)}" for _, r in s.iterrows())


def table_s1():
    T = "S1"
    s = pd.read_csv(SRC["s1_strata"])
    it = pd.read_csv(SRC["s1_inter"])
    d3, d4 = pd.read_csv(SRC["s1_lock_d3"]), pd.read_csv(SRC["s1_lock_d4"])
    d4 = d4[d4.analysis == "primary"]
    # cross-check every stratum row against the locked Direction 3 / 4 files
    mx = 0.0
    for _, r in s.iterrows():
        if r.source_analysis.startswith("Direction 3"):
            l = d3[(d3.task == r.task) & (d3.group == r.stratum) & (d3.metric == r.metric)]
        else:
            l = d4[(d4.task == r.task) & (d4.state == r.stratum) & (d4.metric == r.metric)]
        check(T, f"locked row exists {r.task}/{r.stratum}/{r.metric}", len(l) == 1, len(l), 1)
        l = l.iloc[0]
        for a, b in (("delta", "delta_D_minus_C"), ("ci_lower", "ci_lo"), ("ci_upper", "ci_hi"),
                     ("p_value", "p_boot_two_sided")):
            mx = max(mx, abs(r[a] - l[b]))
        same_adj = (pd.isna(r.adjusted_p) and pd.isna(l.adjusted_p)) or abs(r.adjusted_p - l.adjusted_p) < 1e-12
        check(T, f"adjusted p = locked {r.task}/{r.stratum}/{r.metric}", same_adj, r.adjusted_p, l.adjusted_p)
        check(T, f"n = locked {r.task}/{r.stratum}/{r.metric}",
              (int(r.n_visit_pairs), int(r.n_patients)) == (int(l.n_rows), int(l.n_patients)))
    check(T, "estimates/CIs/raw p = locked (max abs diff)", mx < 1e-12, mx, "< 1e-12")
    sig = s[s.significant_after_adjustment.astype(bool) & s.metric.isin(["auprc", "rps"])]
    check(T, "significant primary-metric rows = six-class RPS: PI, IMiD, steroid, CD38, stable",
          sorted(zip(sig.stratum, sig.metric)) == sorted([("on_pi", "rps"), ("on_imid", "rps"), ("on_steroid", "rps"),
                                                          ("on_cd38", "rps"), ("stable", "rps")]), list(zip(sig.stratum, sig.metric)))
    rows = []
    for _, r in s.iterrows():
        rows.append({"row_type": "stratum", "analysis": "Treatment class active at V(t)" if r.stratum_group != "trajectory"
                     else "Response trajectory at V(t)", "stratum": r.stratum_label, "role": r.role,
                     "task": r.task_label, "metric": r.metric, "better_when": r.better_when,
                     "n_visit_pairs": int(r.n_visit_pairs), "n_patients": int(r.n_patients),
                     "layer_c": r.layer_c_value, "layer_d": r.layer_d_value, "estimate_d_minus_c": r.delta,
                     "ci_lower": r.ci_lower, "ci_upper": r.ci_upper, "p_raw": r.p_value, "p_adjusted": r.adjusted_p,
                     "adjustment": r.adjustment, "significant_after_adjustment": bool(r.significant_after_adjustment),
                     "pct_change_vs_layer_c": r.pct_change_vs_layer_c})
    for _, r in it.iterrows():
        rows.append({"row_type": "interaction", "analysis": r.source_analysis, "stratum": r.contrast_label,
                     "role": r.role, "task": r.task_label, "metric": r.metric, "better_when": r.better_when,
                     "estimate_d_minus_c": r.interaction, "ci_lower": r.ci_lower, "ci_upper": r.ci_upper,
                     "p_raw": r.p_value, "p_adjusted": r.adjusted_p, "adjustment": r.adjustment,
                     "significant_after_adjustment": bool(pd.notna(r.adjusted_p) and r.adjusted_p < 0.05)})
    df = pd.DataFrame(rows)
    df["note"] = ("Out-of-fold, 5x5 patient-grouped CV on training+validation patients; XGBoost Layer D - Layer C; "
                  "95% patient-cluster bootstrap CI (2,000 resamples); exploratory")
    # markdown: primary metrics only
    p = df[(df.row_type == "stratum") & (df.metric.isin(["auprc", "rps"]))].copy()
    p["metric_lbl"] = p.metric.map({"auprc": "ΔAUPRC", "rps": "ΔRPS"})
    p["n"] = p.apply(lambda r: f"{int(r.n_visit_pairs):,} / {int(r.n_patients)}", axis=1)
    p["nd"] = p.metric.map({"auprc": 4, "rps": 5})
    p["est"] = p.apply(lambda r: f(r.estimate_d_minus_c, r.nd, True), axis=1)
    p["ci95"] = p.apply(lambda r: ci(r.ci_lower, r.ci_upper, r.nd), axis=1)
    p["praw"] = p.p_raw.map(pfmt)
    p["padj"] = p.apply(lambda r: (pfmt(r.p_adjusted) + (" (Holm)" if str(r.adjustment).startswith("Holm") else " (BH)"))
                        if pd.notna(r.p_adjusted) else "unadjusted", axis=1)
    p["sig"] = p.apply(lambda r: "n/a" if pd.isna(r.p_adjusted) else ("**yes**" if r.significant_after_adjustment else "no"), axis=1)
    md = ["# Supplementary Table S1. RNA contribution by treatment class and response trajectory (Figure 3C)", "",
          "XGBoost information Layer D − Layer C, out-of-fold predictions from 5 repeats × 5-fold patient-grouped "
          "cross-validation on training and validation patients (not the held-out test set). AUPRC: higher is better. "
          "RPS (ranked probability score): lower is better, so ΔRPS > 0 means RNA worsened predictions. 95% "
          "patient-cluster bootstrap CIs (2,000 resamples). Multiplicity: Holm across the three primary treatments (PI, "
          "IMiD, corticosteroid) and, separately, the two primary trajectory states (stable, worsening), per task on its "
          "primary metric; Benjamini–Hochberg (BH) for exploratory treatments and secondary states; complement rows and "
          "the reference row are unadjusted. Treatment strata overlap. Exploratory, post hoc.", ""]
    for task in ("Binary improvement", "Six-class IMWG response"):
        sub = p[p.task == task]
        md += [f"## {task}", "", md_table(sub, ["analysis", "stratum", "role", "metric_lbl", "n", "est", "ci95", "praw", "padj", "sig"],
                                          ["Analysis", "Stratum", "Role", "Metric", "Visit pairs / patients", "D − C",
                                           "95% CI", "Raw p", "Adjusted p", "Significant after adjustment"]), ""]
    ip = df[df.row_type == "interaction"].copy()
    ip["nd"] = ip.metric.map(lambda m: 5 if m == "rps" else 4)
    ip["est"] = ip.apply(lambda r: f(r.estimate_d_minus_c, r.nd, True), axis=1)
    ip["ci95"] = ip.apply(lambda r: ci(r.ci_lower, r.ci_upper, r.nd), axis=1)
    ip["praw"] = ip.p_raw.map(pfmt)
    ip["padj"] = ip.p_adjusted.map(pfmt)
    md += ["## Secondary interaction contrasts (difference in D − C between strata)", "",
           md_table(ip, ["analysis", "stratum", "task", "metric", "est", "ci95", "praw", "padj", "adjustment"],
                    ["Analysis", "Contrast", "Task", "Metric", "Interaction", "95% CI", "Raw p", "Adjusted p", "Adjustment"]), "",
           "Six-class chemotherapy stratum not evaluated (stratum support rule: too few CR/sCR outcomes). AUROC and log-loss "
           "rows (secondary metrics) are in the CSV; two of them were significant after BH adjustment, both in the direction "
           f"of RNA worsening: {_sec_sig(df)}. Source: `figures/data/Fig3/fig3C_*.csv`, cross-checked against "
           "`artifacts/results_clean_rerun/direction3|direction4/partA_predictive/`."]
    write("Table_S1_fig3C_subgroups", df, "\n".join(md))


# ------------------------------------------------------------------------------------------------
def _logistic_fit(x, y, offset=None, fit_slope=True):
    X = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((len(x), 1))
    off = np.zeros_like(x) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        mu = expit(X @ beta + off)
        step = np.linalg.solve(X.T @ (X * (mu * (1 - mu))[:, None]), X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    return None


def cal_metrics(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    lp = logit(np.clip(p, EPS, 1 - EPS))
    pc = np.clip(p, EPS, 1 - EPS)
    prev = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "brier": brier,
            "scaled_brier": 1 - brier / (prev * (1 - prev)),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "calibration_intercept": float(_logistic_fit(lp, y, offset=lp, fit_slope=False)[0]),
            "calibration_slope": float(_logistic_fit(lp, y)[1]), "mean_predicted": float(p.mean()), "observed_rate": float(prev)}


def table_s2():
    T = "S2"
    m = pd.read_csv(SRC["s2_metrics"])
    dmc = pd.read_csv(SRC["s2_dmc"])
    summ = json.load(open(SRC["s2_summary"], encoding="utf-8"))
    locked = pd.DataFrame(summ["results"]["binary"]["d_minus_c"])
    locked = locked[locked.method == "uncalibrated"]
    pc, pd_ = pd.read_csv(SRC["s2_prob_c"]), pd.read_csv(SRC["s2_prob_d"])
    check(T, "C and D test pairs identical", bool((pc.pair_id.values == pd_.pair_id.values).all()))
    mask = pc.rna_available.to_numpy() == 1
    check(T, "RNA-available test pairs / patients", (int(mask.sum()), pc[mask].public_id.nunique()), (1323, 107))
    y = pc.y_true.to_numpy().astype(float)
    sub = {L: cal_metrics(y[mask], fr.proba.to_numpy()[mask]) for L, fr in (("C", pc), ("D", pd_))}
    allr = {L: cal_metrics(y, fr.proba.to_numpy()) for L, fr in (("C", pc), ("D", pd_))}
    metrics = ["calibration_slope", "calibration_intercept", "brier", "scaled_brier", "log_loss", "mean_predicted",
               "observed_rate", "auroc", "auprc"]
    labels = {"calibration_slope": "Calibration slope (ideal 1)", "calibration_intercept": "Calibration intercept (ideal 0)",
              "brier": "Brier score (lower better)", "scaled_brier": "Scaled Brier (higher better)",
              "log_loss": "Log loss (lower better)", "mean_predicted": "Mean predicted probability",
              "observed_rate": "Observed improvement rate", "auroc": "AUROC (higher better)", "auprc": "AUPRC (higher better)"}
    rows = []
    for aset, vals in (("all_test_rows", allr), ("rna_available_test_rows", sub)):
        for k in metrics:
            lk = locked[(locked.analysis_set == aset) & (locked.metric == k)].iloc[0]
            rec = vals["D"][k] - vals["C"][k]
            check(T, f"{aset} {k}: recomputed D-C = locked D-C", abs(rec - lk.estimate) < 2e-6, round(rec, 7), round(lk.estimate, 7))
            row = {"analysis_set": aset, "metric": k, "metric_label": labels[k], "d_minus_c": lk.estimate,
                   "d_minus_c_ci_lower": lk.ci_lower, "d_minus_c_ci_upper": lk.ci_upper,
                   "ci_includes_0": bool(lk.ci_lower <= 0 <= lk.ci_upper) if k != "observed_rate" else None}
            if aset == "all_test_rows":
                for L in "CD":
                    r = m[(m.layer == L) & (m.metric == k)]
                    if len(r):
                        r = r.iloc[0]
                        check(T, f"all rows {L} {k}: recomputed = locked", abs(vals[L][k] - r.estimate) < 2e-6)
                        row.update({f"layer_{L}": r.estimate, f"layer_{L}_ci_lower": r.ci_lower, f"layer_{L}_ci_upper": r.ci_upper})
                    else:
                        tm = pd.DataFrame(summ["results"]["binary"]["test_metrics"])
                        r = tm[(tm.model == f"{L}-XGBoost") & (tm.method == "uncalibrated") & (tm.metric == k)].iloc[0]
                        row.update({f"layer_{L}": r.estimate, f"layer_{L}_ci_lower": r.ci_lower, f"layer_{L}_ci_upper": r.ci_upper})
                row["layer_values_source"] = "locked calibration summary (patient-bootstrap CIs)"
            else:
                row.update({"layer_C": vals["C"][k], "layer_D": vals["D"][k], "layer_C_ci_lower": np.nan,
                            "layer_C_ci_upper": np.nan, "layer_D_ci_lower": np.nan, "layer_D_ci_upper": np.nan,
                            "layer_values_source": "recomputed point estimate from stored test probabilities; per-layer CI not available"})
            rows.append(row)
    rpd = pd.read_csv(SRC["s2_rna_pd"])
    for k in ("auprc", "auroc"):
        a = rpd[(rpd.task == "binary") & (rpd.comparison == "D - C") & (rpd.metric == k)].iloc[0]
        lk = locked[(locked.analysis_set == "rna_available_test_rows") & (locked.metric == k)].iloc[0]
        check(T, f"RNA-subset {k} D-C = rna_subset_paired_differences.csv", abs(a.observed_diff - lk.estimate) < 1e-6)
    df = pd.DataFrame(rows)
    df["n_visit_pairs"] = df.analysis_set.map({"all_test_rows": 1930, "rna_available_test_rows": 1323})
    df["n_patients"] = df.analysis_set.map({"all_test_rows": 155, "rna_available_test_rows": 107})
    sel = summ["results"]["binary"]["selected_method_on_validation"]
    check(T, "uncalibrated selected for C and D", sel == {"C-XGBoost": "uncalibrated", "D-XGBoost": "uncalibrated"})
    md = ["# Supplementary Table S2. Calibration of binary next-visit improvement predictions, all test pairs and RNA-available test pairs", "",
          "XGBoost information Layer C and Layer D on the held-out test set; uncalibrated probabilities (selected on the "
          "validation set for both layers). Layer D − Layer C differences are paired on identical visit pairs with 95% paired "
          "patient-level bootstrap CIs (1,000 resamples; calibrators fixed). For RNA-available pairs, per-layer values are point "
          "estimates recomputed from stored test probabilities with the locked formulas (their D − C equals the locked paired "
          "difference); per-layer CIs are not available. Intervals are unadjusted. Descriptive.", ""]
    for aset, title in (("all_test_rows", "All test pairs (1,930 pairs; 155 patients)"),
                        ("rna_available_test_rows", "RNA-available test pairs (1,323 pairs; 107 patients)")):
        sub_ = df[(df.analysis_set == aset) & (df.metric != "observed_rate")].copy()
        nd = sub_.metric.map(lambda k: 5 if k in ("brier", "log_loss") else (4 if k == "mean_predicted" else 3))
        sub_["C"] = [f(v, n) + (f" ({ci(lo, hi, n)})" if pd.notna(lo) else "") for v, lo, hi, n in
                     zip(sub_.layer_C, sub_.layer_C_ci_lower, sub_.layer_C_ci_upper, nd)]
        sub_["D"] = [f(v, n) + (f" ({ci(lo, hi, n)})" if pd.notna(lo) else "") for v, lo, hi, n in
                     zip(sub_.layer_D, sub_.layer_D_ci_lower, sub_.layer_D_ci_upper, nd)]
        sub_["dmc"] = [f(v, n, True) for v, n in zip(sub_.d_minus_c, nd)]
        sub_["dci"] = [ci(lo, hi, n) for lo, hi, n in zip(sub_.d_minus_c_ci_lower, sub_.d_minus_c_ci_upper, nd)]
        sub_["inc0"] = sub_.ci_includes_0.map({True: "yes", False: "**no**"})
        md += [f"## {title}", "", md_table(sub_, ["metric_label", "C", "D", "dmc", "dci", "inc0"],
                                          ["Metric", "Layer C", "Layer D", "D − C", "95% CI (D − C)", "CI includes 0"]), ""]
    md += ["On RNA-available pairs, the intercept difference (−0.026) and mean-predicted difference (+0.003) have CIs "
           "excluding 0; these are small, unadjusted and descriptive. Observed improvement rate: 0.183 (all pairs), "
           f"{sub['C']['observed_rate']:.3f} (RNA-available pairs). Source: `figures/data/FigS_calibration/`, "
           "`artifacts/results_clean_rerun/calibration/probabilities/`."]
    write("Table_S2_rna_available_calibration", df, "\n".join(md))


# ------------------------------------------------------------------------------------------------
def table_s3():
    T = "S3"
    sh = pd.read_csv(SRC["s3_shares"])
    pcl = pd.read_csv(SRC["s3_perclass"])
    orig = pd.read_csv(SRC["s3_orig"])
    for _, r in orig.iterrows():
        meas = "net_family_magnitude" if r.measure == "net_family_magnitude" else "summed_source_feature_importance"
        x = sh[(sh.task == r.task) & (sh.analysis == "primary") & (sh.measure == meas) & (sh.component == r.family)].iloc[0]
        check(T, f"{r.task}/{meas}/{r.family} share = original extraction", abs(x.share_pct - 100 * r.share) < 1e-6)
    for (t, a, c, mm), g in pd.concat([sh, pcl]).groupby(["task", "analysis", "class", "measure"]):
        check(T, f"{t}/{a}/{c}/{mm} shares sum to 100", abs(g.share_pct.sum() - 100) < 1e-9)
    df = pd.concat([sh, pcl], ignore_index=True)
    df["measure_label"] = df.measure.map({"net_family_magnitude": "Net family magnitude (Figure S3A)",
                                          "summed_source_feature_importance": "Total source-feature importance (not Figure S3A)"})
    df["units"] = df.task.map({"binary": "log-odds SHAP", "multiclass": "raw class-score SHAP"})
    md = ["# Supplementary Table S3. SHAP attribution shares for XGBoost information Layer D", "",
          "Held-out test set (155 patients; 1,930 binary and 2,079 six-class visit pairs). Two different measures, not "
          "interchangeable: **net family magnitude** (Figure S3A) = mean over pairs of |sum of signed source SHAP within the "
          "family|; **total source-feature importance** = sum over the family's features of mean |source SHAP|. Total "
          "importance is larger for groups whose features push in opposite directions (notably RNA pathways). Six-class "
          "values are raw class scores averaged equally over classes (class average) or shown per class. Shares are of model "
          "attribution, not accuracy, variance explained or causation. Shares use largest-remainder rounding (each column "
          "sums to 100.0%, as in Figure S3A); CIs are from the exact shares. 95% patient-level bootstrap CIs (1,000 resamples).", ""]
    for meas in ("net_family_magnitude", "summed_source_feature_importance"):
        for analysis, alab in (("primary", "all test pairs"), ("patient_equal_weighted", "patients weighted equally"),
                               ("rna_available_rows", "RNA-available pairs only")):
            g = sh[(sh.measure == meas) & (sh.analysis == analysis)]
            piv = []
            for comp in g.component.drop_duplicates():
                row = {"Component": comp, "Layer": g[g.component == comp].information_layer.iloc[0]}
                for t, tl in (("binary", "Binary"), ("multiclass", "Six-class")):
                    r = g[(g.component == comp) & (g.task == t)].iloc[0]
                    row[tl] = f"{r.share_pct_display:.1f}% ({r.share_pct_ci_lower:.1f}–{r.share_pct_ci_upper:.1f})"
                piv.append(row)
            pv = pd.DataFrame(piv)
            n = g.groupby("task").first()
            md += [f"## {df[df.measure == meas].measure_label.iloc[0]} — {alab}",
                   f"Binary {int(n.loc['binary', 'n_rows_used']):,} pairs / {int(n.loc['binary', 'n_patients_used'])} patients; "
                   f"six-class {int(n.loc['multiclass', 'n_rows_used']):,} / {int(n.loc['multiclass', 'n_patients_used'])}.", "",
                   md_table(pv, ["Component", "Layer", "Binary", "Six-class"],
                            ["Component", "Layer", "Binary share (95% CI)", "Six-class share (95% CI)"]), ""]
    g = pcl[(pcl.measure == "net_family_magnitude") & (pcl.analysis == "primary")]
    piv = []
    for comp in g.component.drop_duplicates():
        row = {"Component": comp}
        for c in g["class"].drop_duplicates():
            r = g[(g.component == comp) & (g["class"] == c)].iloc[0]
            row[c] = f"{r.share_pct_display:.1f}%"
        piv.append(row)
    pv = pd.DataFrame(piv)
    cls = list(g["class"].drop_duplicates())
    short = {"progressive_disease": "PD", "stable_disease": "SD", "partial_response": "PR",
             "very_good_partial_response": "VGPR", "complete_response": "CR", "stringent_complete_response": "sCR"}
    md += ["## Six-class net family magnitude share by class (all test pairs)", "",
           md_table(pv, ["Component"] + cls, ["Component"] + [short[c] for c in cls]), "",
           "Per-class CIs and the total-importance measure per class are in the CSV. Source: "
           "`figures/data/FigS_shap/figS3A_ext_component_shares*.csv` (cross-checked against `figS3A_family_shares.csv`)."]
    write("Table_S3_shap_attribution", df, "\n".join(md))


# ------------------------------------------------------------------------------------------------
def table_s4():
    T = "S4"
    a, b, r = pd.read_csv(SRC["s4_auprc"]), pd.read_csv(SRC["s4_f1"]), pd.read_csv(SRC["s4_auroc"])
    ov = json.load(open(SRC["s4_overlap"]))
    raw = subprocess.run(["git", "show", f"{COUNTRY_GIT[0]}:{COUNTRY_GIT[1]}"], capture_output=True, check=True).stdout
    lk = json.loads(raw)
    key = {"spain_canada_pooled": "spain_canada_pooled", "spain_only": "spain_only", "canada_only": "canada_only"}
    for k, g in key.items():
        x = r[r.cohort_key == k].iloc[0]
        check(T, f"{k} AUROC = locked", abs(x.auroc - lk[f"{g}_metrics"]["auroc"]) < 1e-12)
        check(T, f"{k} AUROC CI = locked", abs(x.ci_lower - lk[f"{g}_bootstrap_ci"]["auroc"]["ci_lower_2.5"]) < 1e-12 and
              abs(x.ci_upper - lk[f"{g}_bootstrap_ci"]["auroc"]["ci_upper_97.5"]) < 1e-12)
        y = a[a.cohort_key == k].iloc[0]
        check(T, f"{k} AUPRC = locked", abs(y.auprc - lk[f"{g}_metrics"]["auprc"]) < 1e-12)
    check(T, "US val AUROC = locked", abs(r[r.cohort_key == "us_val"].auroc.iloc[0] - lk["us_val_metrics"]["auroc"]) < 1e-12)
    check(T, "tuning overlap 80/116", ov["spain_canada_patients_in_original_split"]["total_in_original_train_tuning_population"] == 80)
    df = a.merge(b[["cohort_key", "n_visit_pairs", "macro_f1", "ci_lower", "ci_upper"]], on="cohort_key", suffixes=("", "_sixclass"))
    df = df.merge(r[["cohort_key", "auroc", "ci_lower", "ci_upper"]], on="cohort_key", suffixes=("", "_auroc"))
    df = df.rename(columns={"n_visit_pairs": "binary_visit_pairs", "ci_lower": "auprc_ci_lower", "ci_upper": "auprc_ci_upper",
                            "n_visit_pairs_sixclass": "sixclass_visit_pairs", "ci_lower_sixclass": "macro_f1_ci_lower",
                            "ci_upper_sixclass": "macro_f1_ci_upper", "ci_lower_auroc": "auroc_ci_lower",
                            "ci_upper_auroc": "auroc_ci_upper"})
    keep = ["cohort", "cohort_key", "role", "n_patients", "binary_visit_pairs", "n_improvement_events", "event_rate", "auprc",
            "auprc_ci_lower", "auprc_ci_upper", "auroc", "auroc_ci_lower", "auroc_ci_upper", "sixclass_visit_pairs", "macro_f1",
            "macro_f1_ci_lower", "macro_f1_ci_upper"]
    df = df[keep]
    df["original_tuning_split_patients"] = df.cohort_key.map({"spain_canada_pooled": 80, "spain_only": 53, "canada_only": 27})
    df["note"] = df.cohort_key.map(lambda k: "descriptive reference: informed calibration and threshold choices; no CI computed"
                                   if k == "us_val" else "evaluated once; 95% patient-bootstrap CI (1,000 resamples)")
    m = df.copy()
    m["pairs"] = m.apply(lambda r: f"{int(r.binary_visit_pairs):,} / {int(r.sixclass_visit_pairs):,}", axis=1)
    m["ev"] = m.apply(lambda r: f"{int(r.n_improvement_events)} ({r.event_rate:.1%})", axis=1)
    for met in ("auprc", "auroc", "macro_f1"):
        m[met + "_s"] = m.apply(lambda r: f(r[met], 3) + (f" ({ci(r[met + '_ci_lower'], r[met + '_ci_upper'], 3)})"
                                                         if pd.notna(r[met + "_ci_lower"]) else " (no CI)"), axis=1)
    m["tune"] = m.original_tuning_split_patients.map(lambda v: "—" if pd.isna(v) else str(int(v)))
    md = ["# Supplementary Table S4. Internal geographic hold-out within MMRF CoMMpass (Figure S2)", "",
          "XGBoost information Layer C refit on US patients only (new random patient-level 82/18 split: 632 training, 139 "
          "validation); preprocessing and fitting on US training, calibration method and threshold on US validation; evaluated "
          "once on patients enrolled in Spain and Canada. Not independent external validation: hyperparameters were reused from "
          "the main analysis, whose tuning population included 80 of the 116 evaluated patients. No statistical comparison "
          "between cohorts was performed. Italy (138 patients) was excluded post hoc.", "",
          md_table(m, ["cohort", "n_patients", "pairs", "ev", "auprc_s", "auroc_s", "macro_f1_s", "tune"],
                   ["Cohort", "Patients", "Visit pairs (binary / six-class)", "Improvement events (rate)", "AUPRC (95% CI)",
                    "AUROC (95% CI)", "Macro-F1 (95% CI)", "Patients in original tuning split"]), "",
          "The event rate is the AUPRC of a non-informative model in each cohort. Source: `figures/data/FigS_country/`, "
          f"cross-checked against `{COUNTRY_GIT[0]}:{COUNTRY_GIT[1]}`."]
    write("Table_S4_geographic_robustness", df, "\n".join(md))
    return raw


# ------------------------------------------------------------------------------------------------
def table_s5():
    T = "S5"
    ib, fo, su = pd.read_csv(SRC["s5_ibs"]), pd.read_csv(SRC["s5_forest"]), pd.read_csv(SRC["s5_summary"])
    lk, lr = json.load(open(SRC["s5_lock"])), json.load(open(SRC["s5_lock_richer"]))
    global REP
    REP = {}
    rep_files = {("PFS", "primary"): RC / "survival_incremental/primary_pfs_repeat_metrics.csv",
                 ("OS", "primary"): RC / "survival_incremental/secondary_os_repeat_metrics.csv",
                 ("PFS", "rna_timing_m30_0"): RC / "survival_incremental/sensitivity_m30_0_pfs_repeat_metrics.csv",
                 ("PFS", "expanded_baseline"): RC / "survival_richer_baseline_sensitivity/primary_pfs_richer_repeat_metrics.csv",
                 ("OS", "expanded_baseline"): RC / "survival_richer_baseline_sensitivity/secondary_os_richer_repeat_metrics.csv"}
    for k, pth in rep_files.items():
        rp = pd.read_csv(pth)
        REP[k] = (rp.delta_c.min(), rp.delta_c.max())
        SRC[f"s5_rep_{k[0]}_{k[1]}"] = pth
        y = fo[(fo.endpoint == k[0]) & (fo.analysis == k[1])].iloc[0]
        check(T, f"{k} mean of 5 repeat deltas = plotted delta C", len(rp) == 5 and abs(rp.delta_c.mean() - y.delta_c) < 1e-12)
    key = {("PFS", "primary"): lk["primary_pfs"], ("OS", "primary"): lk["secondary_os"],
           ("PFS", "rna_timing_m30_0"): lk["sensitivity_pfs"]}
    for (ep, an), s in key.items():
        x = ib[(ib.endpoint == ep) & (ib.analysis == an)].iloc[0]
        check(T, f"{ep}/{an} IBS = locked", abs(x.ibs_clinical_baseline - s["ibs_C0"]) < 1e-12 and
              abs(x.ibs_baseline_plus_rna - s["ibs_C0_RNA"]) < 1e-12 and abs(x.delta_ibs - s["delta_ibs"]) < 1e-12)
        y = fo[(fo.endpoint == ep) & (fo.analysis == an)].iloc[0]
        check(T, f"{ep}/{an} delta C = locked", abs(y.delta_c - s["delta_c"]) < 1e-12)
    for ep, s in (("PFS", lr["primary_pfs_richer"]), ("OS", lr["secondary_os_richer"])):
        y = fo[(fo.endpoint == ep) & (fo.analysis == "expanded_baseline")].iloc[0]
        check(T, f"{ep} expanded delta C = locked", abs(y.delta_c - s["delta_c"]) < 1e-12)
    check(T, "PFS expanded = +0.035 [0.014, 0.056]", (round(fo.query("endpoint=='PFS' and analysis=='expanded_baseline'").delta_c.iloc[0], 3),
          round(fo.query("endpoint=='PFS' and analysis=='expanded_baseline'").ci_lower.iloc[0], 3)) == (0.035, 0.014))
    check(T, "OS expanded = +0.049 [0.022, 0.076]", round(fo.query("endpoint=='OS' and analysis=='expanded_baseline'").delta_c.iloc[0], 3) == 0.049)
    rows = []
    label = {"primary": "Primary", "expanded_baseline": "Expanded clinical baseline", "rna_timing_m30_0": "RNA day −30 to 0 only"}
    for _, y in fo.iterrows():
        x = ib[(ib.endpoint == y.endpoint) & (ib.analysis == y.analysis)]
        x = x.iloc[0] if len(x) else None
        sm = su[su.endpoint == y.endpoint].iloc[0] if y.analysis == "primary" else None
        rows.append({"endpoint": y.endpoint, "analysis": y.analysis, "analysis_label": label[y.analysis],
                     "evaluated": bool(y.evaluated), "n_patients": y.n_patients, "n_events": y.n_events,
                     "c_index_baseline": y.c_index_baseline, "c_index_plus_rna": y.c_index_plus_rna, "delta_c": y.delta_c,
                     "delta_c_ci_lower": y.ci_lower, "delta_c_ci_upper": y.ci_upper,
                     "delta_c_repeat_min": REP.get((y.endpoint, y.analysis), (np.nan, np.nan))[0],
                     "delta_c_repeat_max": REP.get((y.endpoint, y.analysis), (np.nan, np.nan))[1],
                     "permutation_p": y.permutation_p_one_sided,
                     "ibs_baseline": x.ibs_clinical_baseline if x is not None and pd.notna(x.ibs_clinical_baseline) else np.nan,
                     "ibs_plus_rna": x.ibs_baseline_plus_rna if x is not None and pd.notna(x.ibs_baseline_plus_rna) else np.nan,
                     "delta_ibs": x.delta_ibs if x is not None and pd.notna(x.delta_ibs) else np.nan,
                     "c_index_ci": "not available", "ibs_ci": "not available (point estimates only)" if x is not None and
                     pd.notna(x.ibs_clinical_baseline) else "IBS not computed",
                     "not_evaluated_reason": y.get("not_evaluated_reason", "")})
    df = pd.DataFrame(rows)
    m = df.copy()
    m["n"] = m.apply(lambda r: "—" if not r.evaluated else f"{int(r.n_patients)} / {int(r.n_events)}", axis=1)
    m["c"] = m.apply(lambda r: "not evaluated" if not r.evaluated else f"{r.c_index_baseline:.3f} → {r.c_index_plus_rna:.3f}", axis=1)
    m["dc"] = m.apply(lambda r: "—" if not r.evaluated else f"{f(r.delta_c, 3, True)} ({ci(r.delta_c_ci_lower, r.delta_c_ci_upper, 3)})", axis=1)
    m["rep"] = m.apply(lambda r: "—" if pd.isna(r.delta_c_repeat_min) else f"{r.delta_c_repeat_min:.3f}–{r.delta_c_repeat_max:.3f}", axis=1)
    m["perm"] = m.permutation_p.map(lambda v: "not performed" if pd.isna(v) else f"{v:.3f} (0/200)")
    m["ibs"] = m.apply(lambda r: ("not computed" if r.evaluated else "—") if pd.isna(r.ibs_baseline) else
                       f"{r.ibs_baseline:.3f} → {r.ibs_plus_rna:.3f} (Δ {f(r.delta_ibs, 3, True)}; no CI)", axis=1)
    md = ["# Supplementary Table S5. Survival discrimination and overall prediction error with and without pre-treatment RNA (Figure 4)", "",
          "Survival cohort: patients with one RNA sample collected on or before the start of first treatment (CoMMpass index "
          "date); official MMRF PFS and OS. Ridge-penalized Cox models; clinical baseline = age at index, gender, ISS, exact "
          "first-line regimen; expanded baseline adds LDH, β2-microglobulin, albumin, creatinine, haemoglobin, calcium and "
          "bone-marrow plasma-cell %; + RNA = 50 Hallmark pathway scores. Harrell's C-index (higher is better) on pooled "
          "out-of-fold predictions per repeat of 5×5 patient-level cross-validation, averaged over repeats. ΔC 95% CIs: "
          "patient bootstrap (2,000 resamples) of fixed out-of-fold predictions. Integrated Brier score (IBS; IPCW, 365–1,095 "
          "days; lower is better) point estimates only. Exploratory internal cross-validation, not external validation; the "
          "cohort includes 100 of the 155 Figure 2 test patients.", "",
          md_table(m, ["endpoint", "analysis_label", "n", "c", "dc", "rep", "perm", "ibs"],
                   ["Endpoint", "Analysis", "Patients / events", "C-index: baseline → + RNA", "ΔC (95% CI)",
                    "ΔC range over 5 repeats", "Permutation p (one-sided)", "IBS: baseline → + RNA"]), "",
          "No CIs exist for absolute C-index or IBS values. The OS RNA-timing analysis was specified for PFS only and was not "
          "run. Rows differ in cohort or comparator and were not compared statistically. Source: `figures/data/Fig4/`, "
          "cross-checked against `artifacts/results_clean_rerun/survival_incremental/` and "
          "`survival_richer_baseline_sensitivity/final_summary.json`."]
    write("Table_S5_survival", df, "\n".join(md))


# ------------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table_s1()
    table_s2()
    table_s3()
    raw = table_s4()
    table_s5()
    # no patient identifiers in any output
    for p in OUT.glob("Table_*"):
        txt = p.read_text(encoding="utf-8")
        check("all", f"no patient identifiers in {p.name}", ("mmrf_" not in txt) and ("public_id" not in txt))
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "supplementary_tables_verification.csv", index=False)
    prov = {"script": "figures/supplementary_tables/build_supplementary_tables.py",
            "inputs_sha256": {str(p): sha(p) for p in SRC.values()},
            "git_sources": {f"{COUNTRY_GIT[0]}:{COUNTRY_GIT[1]}": hashlib.sha256(raw).hexdigest()},
            "outputs": sorted(p.name for p in OUT.glob("Table_*"))}
    json.dump(prov, open(OUT / "supplementary_tables_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
