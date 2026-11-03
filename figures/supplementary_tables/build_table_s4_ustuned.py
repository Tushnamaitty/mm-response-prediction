"""
Build Supplementary Table S4 (geographic hold-out), US-TUNED revision (read-only).

Run from repo root, after figures/figS_country_ustuned_extract_data.py:
    python figures/supplementary_tables/build_table_s4_ustuned.py

Writes Table_S4_geographic_robustness_ustuned.{csv,md} and table_s4_ustuned_verification.csv next to this script. The
original Table_S4_geographic_robustness.md (hyperparameters reused from the main analysis) and its builder are left
unchanged. Values come from figures/data/FigS_country_ustuned/ and are cross-checked against the result JSONs in
artifacts/results_clean_rerun/geographic_us_tuned/. No model is trained or refit; no patient identifiers are written.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("figures/supplementary_tables")
FD = Path("figures/data/FigS_country_ustuned")
RES = Path("artifacts/results_clean_rerun/geographic_us_tuned")
checks = []


def check(name, ok, observed="", expected=""):
    checks.append({"check": name, "passed": bool(ok), "observed": str(observed)[:160], "expected": str(expected)[:160]})


def f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(lo, hi, nd=3):
    return f"{f(lo, nd)} to {f(hi, nd)}"


def md_table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    a, b, r = (pd.read_csv(FD / "figS2A_binary_auprc.csv"), pd.read_csv(FD / "figS2B_sixclass_macro_f1.csv"),
               pd.read_csv(FD / "figS2_auroc_caption_only.csv"))
    jb = json.load(open(RES / "binary_c_xgboost_us_to_spain_canada_metrics.json", encoding="utf-8"))
    au = json.load(open(RES / "full_run_audit_and_summary.json", encoding="utf-8"))
    jm = au["tasks"]["multiclass"]
    tun = json.load(open(FD / "figS2_tuning_audit.json", encoding="utf-8"))["tuning"]
    for k in ("spain_canada_pooled", "spain_only", "canada_only"):
        x, y, z = (t[t.cohort_key == k].iloc[0] for t in (r, a, b))
        check(f"{k} AUROC = result file", abs(x.auroc - jb[f"{k}_metrics"]["auroc"]) < 1e-12)
        check(f"{k} AUPRC = result file", abs(y.auprc - jb[f"{k}_metrics"]["auprc"]) < 1e-12)
        check(f"{k} macro-F1 = result file", abs(z.macro_f1 - jm[f"{k}_metrics"]["macro_f1"]) < 1e-12)
    check("US validation AUROC = result file", abs(r[r.cohort_key == "us_val"].auroc.iloc[0] - jb["us_val_metrics"]["auroc"]) < 1e-12)
    for t in ("binary", "multiclass"):
        check(f"{t}: no tuning overlap with evaluation patients", au["tasks"][t]["tuning_evaluation_patient_overlap"] == 0)

    df = a.merge(b[["cohort_key", "n_visit_pairs", "macro_f1", "ci_lower", "ci_upper"]], on="cohort_key", suffixes=("", "_sixclass"))
    df = df.merge(r[["cohort_key", "auroc", "ci_lower", "ci_upper"]], on="cohort_key", suffixes=("", "_auroc"))
    df = df.rename(columns={"n_visit_pairs": "binary_visit_pairs", "ci_lower": "auprc_ci_lower", "ci_upper": "auprc_ci_upper",
                            "n_visit_pairs_sixclass": "sixclass_visit_pairs", "ci_lower_sixclass": "macro_f1_ci_lower",
                            "ci_upper_sixclass": "macro_f1_ci_upper", "ci_lower_auroc": "auroc_ci_lower",
                            "ci_upper_auroc": "auroc_ci_upper"})
    keep = ["cohort", "cohort_key", "role", "n_patients", "binary_visit_pairs", "n_improvement_events", "event_rate", "auprc",
            "auprc_ci_lower", "auprc_ci_upper", "auroc", "auroc_ci_lower", "auroc_ci_upper", "sixclass_visit_pairs", "macro_f1",
            "macro_f1_ci_lower", "macro_f1_ci_upper"]
    df = df[keep].copy()
    df["note"] = df.cohort_key.map(lambda k: "descriptive reference: informed calibration and threshold choices; no CI computed"
                                   if k == "us_val" else "evaluated once; 95% patient-bootstrap CI (1,000 resamples)")
    m = df.copy()
    m["pairs"] = m.apply(lambda x: f"{int(x.binary_visit_pairs):,} / {int(x.sixclass_visit_pairs):,}", axis=1)
    m["ev"] = m.apply(lambda x: f"{int(x.n_improvement_events)} ({x.event_rate:.1%})", axis=1)
    for met in ("auprc", "auroc", "macro_f1"):
        m[met + "_s"] = m.apply(lambda x: f(x[met]) + (f" ({ci(x[met + '_ci_lower'], x[met + '_ci_upper'])})"
                                                      if pd.notna(x[met + "_ci_lower"]) else " (no CI)"), axis=1)
    tb, tm_ = tun["binary"], tun["multiclass"]
    pb, pm = tb["selected_params"], tm_["selected_params"]
    fmt = lambda p: f"{p['n_estimators']} trees, depth {p['max_depth']}, learning rate {p['learning_rate']}"  # noqa: E731
    md = ["# Supplementary Table S4. Internal geographic hold-out within MMRF CoMMpass (Figure S2), US-tuned revision", "",
          "XGBoost information Layer C refit on US patients only (new random patient-level 82/18 split: 632 training, 139 "
          "validation). Hyperparameters were re-tuned by 5-fold patient-grouped cross-validation over the main analysis's "
          f"three-configuration grid, using only US patients in the original training partition and in the US training set "
          f"({tb['n_patients']} patients, {tb['n_pairs']:,} pairs for the binary task; {tm_['n_patients']} patients, "
          f"{tm_['n_pairs']:,} pairs for six-class). No Spain, Canada, Italy, US-validation, original-validation or "
          "original-test patient entered tuning. Preprocessing and fitting used US training patients; calibration method and "
          "binary threshold were chosen on US validation; the model was evaluated once on patients enrolled in Spain and "
          "Canada. Selected hyperparameters: binary " + fmt(pb) + " (same as the earlier version); six-class " + fmt(pm) +
          " (earlier version: " + fmt(tm_["previous_params_reused_from_main_analysis"]) + "). "
          "This is a post hoc, exploratory internal robustness check, not independent external validation: the earlier "
          "version of the experiment had been seen before this redesign. No statistical comparison between cohorts was "
          "performed. Italy (138 patients) was excluded post hoc, after inspecting its outcome distribution.", "",
          md_table(m, ["cohort", "n_patients", "pairs", "ev", "auprc_s", "auroc_s", "macro_f1_s"],
                   ["Cohort", "Patients", "Visit pairs (binary / six-class)", "Improvement events (rate)", "AUPRC (95% CI)",
                    "AUROC (95% CI)", "Macro-F1 (95% CI)"]), "",
          "AUROC is for the binary task. The event rate is the AUPRC of a non-informative model in each cohort. Binary results "
          "are identical to the earlier version because re-tuning selected the same hyperparameters. Source: "
          "`figures/data/FigS_country_ustuned/`, cross-checked against "
          "`artifacts/results_clean_rerun/geographic_us_tuned/`."]
    df.to_csv(OUT / "Table_S4_geographic_robustness_ustuned.csv", index=False)
    (OUT / "Table_S4_geographic_robustness_ustuned.md").write_text("\n".join(md), encoding="utf-8")
    txt = "\n".join(md)
    check("no patient identifiers in output", "mmrf_" not in txt and "public_id" not in txt)
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "table_s4_ustuned_verification.csv", index=False)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
