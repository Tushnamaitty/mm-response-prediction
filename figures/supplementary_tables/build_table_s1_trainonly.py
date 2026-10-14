"""
Build Supplementary Table S1 (RNA contribution by treatment class and response trajectory), TRAIN-ONLY redesign (read-only).

Run from repo root, after figures/fig3ABC_trainonly_extract_data.py:
    python figures/supplementary_tables/build_table_s1_trainonly.py

Writes Table_S1_rna_subgroups_trainonly.{csv,md} and table_s1_trainonly_verification.csv next to this script. The earlier
Table S1 (pooled train+validation CV; Table_S1_fig3C_subgroups.md) and its builder are left unchanged. Values come from
figures/data/Fig3ABC_trainonly/ and are cross-checked against artifacts/results_clean_rerun/rna_contribution_trainonly/.
Held-out = validation + test patients with RNA (213 patients); locked train-only models; no model is fitted here and no patient
identifier is written.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("figures/supplementary_tables")
FD = Path("figures/data/Fig3ABC_trainonly")
R = Path("artifacts/results_clean_rerun/rna_contribution_trainonly")
checks = []


def check(name, ok, observed="", expected=""):
    checks.append({"check": name, "passed": bool(ok), "observed": str(observed)[:160], "expected": str(expected)[:160]})


def f(x, nd=4, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = 0.0 if round(x, nd) == 0 else x
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")


def cell(r, nd, scale=1.0):
    if r is None:
        return "not evaluable"
    s = f"{f(r.delta * scale, nd)} ({f(r.ci_lower * scale, nd, False)} to {f(r.ci_upper * scale, nd, False)})"
    if bool(r.significant_after_adjustment):
        s += f"; {r.adjustment} p = {r.adjusted_p:.3f}"
    elif isinstance(r.adjustment, str) and r.adjustment in ("Holm", "BH"):
        s += f"; {r.adjustment} p = {r.adjusted_p:.2f}"
    return s


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    S = pd.read_csv(FD / "fig3C_trainonly_strata.csv")
    comp = pd.read_csv(FD / "tableS1_trainonly_complements.csv")
    inter = pd.read_csv(FD / "tableS1_trainonly_interactions.csv")
    dev = pd.read_csv(FD / "tableS1_trainonly_development_reference.csv")
    dinter = pd.read_csv(FD / "tableS1_trainonly_development_interactions.csv")
    src_w = pd.read_csv(R / "heldout_within_groups.csv")
    src_i = pd.read_csv(R / "heldout_interactions.csv")
    S["adjustment"] = S["adjustment"].fillna("")

    # ---- cross-checks against the locked held-out files ----
    n_match = 0
    for _, r in S.iterrows():
        g = "all" if r.stratum == "all" else (f"on_{r.stratum}" if r.stratum in ("pi", "imid", "steroid", "cd38", "chemo") else f"state_{r.stratum}")
        s = src_w[(src_w.task == r.task) & (src_w.group == g) & (src_w.metric == r.metric)]
        ok = len(s) == 1 and abs(s.iloc[0].delta_D_minus_C - r.delta) < 1e-12 and abs(s.iloc[0].ci_lo - r.ci_lower) < 1e-12 and int(s.iloc[0].n_patients) == int(r.n_patients)
        n_match += int(ok)
    check("every strata row equals the locked held-out within-group file (delta, CI, patients)", n_match == len(S), n_match, len(S))
    check("interaction rows equal the locked held-out interaction file", len(inter) == len(src_i[src_i.family.isin(["treatment", "state"])]) and
          np.allclose(np.sort(inter.interaction_a_minus_b.to_numpy()), np.sort(src_i[src_i.family.isin(["treatment", "state"])].interaction_a_minus_b.to_numpy()), atol=1e-12))
    check("held-out denominators: 213 patients in the reference row", int(S[(S.stratum == "all")].n_patients.iloc[0]) == 213)
    sig = S[S.significant_after_adjustment & S.metric.isin(["auprc", "rps"])]
    check("no within-stratum primary-metric test is significant after adjustment", len(sig) == 0, len(sig), 0)
    sigi = inter[(inter.adjusted_p < 0.05)]
    check("only the two six-class trajectory interactions have BH p < 0.05 among treatment/trajectory interactions",
          sorted(zip(sigi.task, sigi.contrast, sigi.metric)) == sorted([("multiclass", "improving_minus_stable", "rps"), ("multiclass", "worsening_minus_stable", "rps"),
                                                                        *[(t, c, m) for t, c, m in zip(sigi.task, sigi.contrast, sigi.metric) if m != "rps"]]), len(sigi))

    order = [("all", "All RNA-available visit pairs (reference, unadjusted)", "reference"), ("pi", "Proteasome inhibitor", "primary"),
             ("imid", "Immunomodulatory drug", "primary"), ("steroid", "Corticosteroid", "primary"), ("cd38", "Anti-CD38 antibody", "exploratory"),
             ("chemo", "Chemotherapy", "exploratory"), ("stable", "Stable trajectory", "primary"), ("worsening", "Worsening trajectory", "primary"),
             ("improving", "Improving trajectory", "secondary"), ("no_history", "No previous visit", "secondary")]

    def get(st, task, met):
        s = S[(S.stratum == st) & (S.task == task) & (S.metric == met)]
        return None if s.empty else s.iloc[0]

    rows = []
    for st, lab, role in order:
        pb, pm = get(st, "binary", "auprc"), get(st, "multiclass", "rps")
        rows.append([lab, role, f"{int(pb.n_patients)} / {int(pm.n_patients) if pm is not None else '—'}",
                     cell(pb, 4), cell(get(st, "binary", "auroc"), 4), cell(get(st, "binary", "logloss"), 4),
                     cell(pm, 2, 1e3) if pm is not None else "not evaluable (frozen support rule)",
                     cell(get(st, "multiclass", "logloss"), 4)])
    main_tbl = md_table(rows, ["Stratum", "Role", "Patients (binary / six-class)", "Binary ΔAUPRC (95% CI)", "Binary ΔAUROC (95% CI)",
                               "Binary Δlog loss (95% CI)", "Six-class ΔRPS ×10⁻³ (95% CI)", "Six-class Δlog loss (95% CI)"])

    drows = []
    for st, lab, role in order:
        def dv(task, met):
            g = "all" if st == "all" else (f"on_{st}" if st in ("pi", "imid", "steroid", "cd38", "chemo") else f"state_{st}")
            s = dev[(dev.task == task) & (dev.group == g) & (dev.metric == met)]
            if s.empty:
                return "not evaluable"
            r = s.iloc[0]
            ap = f"; {r.adjustment_method} p = {r.adjusted_p:.3f}" if isinstance(r.adjustment_method, str) and r.adjustment_method in ("Holm", "BH") and r.adjusted_p < 0.05 else ""
            sc = 1e3 if met == "rps" else 1.0
            return f"{f(r.delta_D_minus_C * sc, 2 if met == 'rps' else 4)} ({f(r.ci_lo * sc, 2 if met == 'rps' else 4, False)} to {f(r.ci_hi * sc, 2 if met == 'rps' else 4, False)}){ap}"
        s_all = dev[(dev.group == ("all" if st == "all" else (f"on_{st}" if st in ("pi", "imid", "steroid", "cd38", "chemo") else f"state_{st}"))) & (dev.metric == "auprc")]
        npat = int(s_all.iloc[0].n_patients) if len(s_all) else "—"
        drows.append([lab, role, npat, dv("binary", "auprc"), dv("multiclass", "rps")])
    dev_tbl = md_table(drows, ["Stratum", "Role", "Patients (binary)", "Binary ΔAUPRC (95% CI)", "Six-class ΔRPS ×10⁻³ (95% CI)"])

    irows = []
    for _, r in inter.sort_values(["task", "family", "contrast", "metric"]).iterrows():
        irows.append([r.task, r.contrast.replace("_", " "), r.metric, f"{f(r.interaction_a_minus_b * (1e3 if r.metric == 'rps' else 1), 2 if r.metric == 'rps' else 4)} "
                      f"({f(r.ci_lo * (1e3 if r.metric == 'rps' else 1), 2 if r.metric == 'rps' else 4, False)} to {f(r.ci_hi * (1e3 if r.metric == 'rps' else 1), 2 if r.metric == 'rps' else 4, False)})",
                      f"{r.p_boot_two_sided:.3f}", f"{r.adjusted_p:.3f}" if pd.notna(r.adjusted_p) else "—", r.adjustment_family if isinstance(r.adjustment_family, str) else ""])
    inter_tbl = md_table(irows, ["Task", "Contrast (on − not on / state − stable)", "Metric", "Interaction (95% CI; RPS ×10⁻³)", "Unadjusted p", "BH p", "Family"])

    crows = []
    for _, r in comp.sort_values(["task", "group", "metric"]).iterrows():
        sc = 1e3 if r.metric == "rps" else 1.0
        crows.append([r.task, r.group.replace("not_", "not on "), r.metric, int(r.n_patients), f"{f(r.delta_D_minus_C * sc, 2 if r.metric == 'rps' else 4)} "
                      f"({f(r.ci_lo * sc, 2 if r.metric == 'rps' else 4, False)} to {f(r.ci_hi * sc, 2 if r.metric == 'rps' else 4, False)})"])
    comp_tbl = md_table(crows, ["Task", "Complement stratum", "Metric", "Patients", "Δ (95% CI; RPS ×10⁻³); descriptive, unadjusted"])

    md = ["# Supplementary Table S1. RNA contribution (Layer D − Layer C) by treatment class and response trajectory: held-out evaluation (Figure 4C), train-only redesign", "",
          "XGBoost Layer C and Layer D, fitted and tuned on train patients only (locked models; no refitting), were evaluated once on the 213 held-out validation + test patients with RNA "
          "(2,405 binary visit pairs; 2,689 six-class pairs). This analysis is specific to XGBoost, which was used alone without a recorded a priori rationale; it does not show that XGBoost is superior to Logistic Regression or LightGBM. Treatment strata are the class active at V(t) and overlap; trajectory strata compare the IMWG category at V(t) "
          "with the previous visit (six audited continuity rows excluded). Which treatments receive inference was fixed from train-patient counts before the held-out data were read "
          "(PI, IMiD, steroid primary; CD38, chemotherapy exploratory; six-class chemotherapy not evaluable: too few CR/sCR). Δ = Layer D − Layer C; for AUPRC and AUROC positive = RNA better; "
          "for RPS and log loss negative = RNA better. 95% patient-cluster bootstrap CIs (2,000 resamples). Adjustment: Holm over the three primary treatments and, separately, over stable + "
          "worsening, per task on the primary metric; BH for exploratory treatments, secondary trajectory states and interaction contrasts; secondary metrics and complements are "
          "descriptive. Post hoc and exploratory; not independent validation.", "", main_tbl, "",
          "No within-stratum test on the primary metric is significant after adjustment. The patients column counts held-out patients contributing rows to the stratum.", "",
          "## Interaction contrasts (held-out)", "", inter_tbl, "",
          "Two six-class trajectory contrasts (worsening − stable and improving − stable, ΔRPS) have BH p < 0.05. They are secondary, exploratory and unreplicated; no primary test is significant.", "",
          "## Complement strata (held-out; descriptive)", "", comp_tbl, "",
          "## Development-only cross-validation reference (train patients; not an evaluation)", "",
          "5 × 5 patient-grouped CV within the 716 train patients (locked hyperparameters, tuned in the same patients, so absolute performance is optimistic), with the same strata, "
          "families and bootstrap. Shown for context only; it cannot replace the held-out evaluation.", "", dev_tbl, "",
          "Sources: `figures/data/Fig3ABC_trainonly/` (built by `figures/fig3ABC_trainonly_extract_data.py`), cross-checked against `artifacts/results_clean_rerun/rna_contribution_trainonly/` "
          "and the frozen protocol `explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md`. The earlier Table S1 (pooled train+validation CV, in which validation patients trained the fold models) "
          "is superseded."]
    txt = "\n".join(md)
    check("no patient identifiers in the output", "mmrf_" not in txt and "public_id" not in txt)

    long_rows = [S.assign(table="held-out strata"),
                 comp.rename(columns={"group": "stratum", "delta_D_minus_C": "delta", "ci_lo": "ci_lower", "ci_hi": "ci_upper"}).assign(table="held-out complements"),
                 inter.assign(table="held-out interactions"),
                 dev.assign(table="development-only CV reference strata"), dinter.assign(table="development-only CV reference interactions")]
    pd.concat(long_rows, ignore_index=True, sort=False).to_csv(OUT / "Table_S1_rna_subgroups_trainonly.csv", index=False)
    (OUT / "Table_S1_rna_subgroups_trainonly.md").write_text(txt, encoding="utf-8")
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "table_s1_trainonly_verification.csv", index=False)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
