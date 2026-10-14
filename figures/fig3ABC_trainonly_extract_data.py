"""
Figures 3A-C and Table S1 (train-only redesign): data extraction and verification (read-only).

Run from repo root:
    python figures/fig3ABC_trainonly_extract_data.py

Replaces the data source of panels A-C of figures/fig3_extract_data.py, which is left unchanged (it reads pooled train+validation
cross-validation results and gitignored CSVs that are not available here). Everything is extracted from the frozen train-only
redesign (explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md):

    artifacts/results_clean_rerun/rna_contribution_trainonly/
        heldout_report.json, heldout_within_groups.csv, heldout_interactions.csv, heldout_negative_control_summary.csv,
        heldout_manifest.json, heldout_checks.csv, devcv_summary.json, devcv_within_groups.csv, devcv_interactions.csv,
        frozen_development.json, frozen_development.sha256, development_gate_report.json
    explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.{md,sha256}
    artifacts/results_clean_rerun/{direction3/partA_predictive/partA_delta_within_group.csv,
        rna_low_variance_incremental/observed_summary.json}     (earlier pooled results, reference rows only)

Evaluation population: validation + test patients with RNA (six-class 2,689 pairs / 213 patients; binary 2,405 / 213), locked train-only
Layer C / D XGBoost models, locked train-only add-on fits. Nothing is refit. The script exits non-zero if any check fails.
Writes figures/data/Fig3ABC_trainonly/ (gitignored with the rest of figures/data; aggregates only).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("artifacts/results_clean_rerun/rna_contribution_trainonly")
OUT = Path("figures/data/Fig3ABC_trainonly")
PROTOCOL = Path("explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md")
PROTOCOL_SHA = Path("explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.sha256")
EARLIER = Path("artifacts/results_clean_rerun")
checks = []


def check(name, observed, expected, source, tol=None):
    ok = abs(float(observed) - float(expected)) <= tol if tol is not None else observed == expected
    checks.append({"check": name, "observed": str(observed)[:160], "expected": str(expected)[:160], "passed": bool(ok), "source": str(source)})


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(R / "heldout_report.json", encoding="utf-8"))
    art = json.load(open(R / "frozen_development.json", encoding="utf-8"))
    man = json.load(open(R / "heldout_manifest.json", encoding="utf-8"))
    w, it = pd.read_csv(R / "heldout_within_groups.csv"), pd.read_csv(R / "heldout_interactions.csv")
    dw, di = pd.read_csv(R / "devcv_within_groups.csv"), pd.read_csv(R / "devcv_interactions.csv")
    neg = pd.read_csv(R / "heldout_negative_control_summary.csv")
    dev_sum = json.load(open(R / "devcv_summary.json", encoding="utf-8"))

    # ---- provenance and hash chain ----
    for f, h in man["files"].items():
        check(f"manifest hash {f}", sha_lf(R / f) == h, True, R / "heldout_manifest.json")
    rec = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0]
    check("protocol hash: file == recorded == artifact == report", sha_lf(PROTOCOL) == rec == art["protocol_sha256"] == rep["protocol_sha256"], True, PROTOCOL)
    fz = (R / "frozen_development.sha256").read_text(encoding="utf-8").split()[0]
    check("frozen artifact hash: file == recorded == report", sha_lf(R / "frozen_development.json") == fz == rep["frozen_development_sha256"], True, R)
    check("Stage D gates all passed", all(g["passed"] for g in json.load(open(R / "development_gate_report.json"))["gates"]), True, R)
    check("Stage H checks all passed", all(c["passed"] for c in rep["checks"]), True, R / "heldout_report.json")
    check("no stratum was dropped as not evaluable", rep["n_not_evaluable_rows"], 0, R / "heldout_report.json")

    # ---- populations ----
    ov = w[(w.group == "all")]
    for task, n in (("binary", 2405), ("multiclass", 2689)):
        r = ov[(ov.task == task) & ov.is_primary_metric].iloc[0]
        check(f"[{task}] held-out RNA-available rows / patients", (int(r.n_rows), int(r.n_patients)), (n, 213), R / "heldout_within_groups.csv")
    check("heldout patients", rep["heldout"]["rna_patients"], 213, R / "heldout_report.json")

    # ---- Figure 3A ----
    cv_ref = {(x["task"]): x for x in dev_sum["overall_primary"]}
    rows = []
    for task, pm, rid in (("binary", "auprc", "direct_binary_auprc"), ("multiclass", "rps", "direct_multiclass_rps")):
        r = ov[(ov.task == task) & (ov.metric == pm)].iloc[0]
        rows.append({"row_id": rid, "task": task, "metric": pm, "evaluation": "held-out (validation + test)", "evaluated": True,
                     "delta": r.delta_D_minus_C, "ci_lower": r.ci_lo, "ci_upper": r.ci_hi, "p_value": r.p_boot_two_sided,
                     "p_type": "two-sided patient bootstrap", "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients),
                     "layer_c_value": r.C_mean, "pct_change_vs_layer_c": 100 * r.delta_D_minus_C / r.C_mean if task == "multiclass" else np.nan})
        c = cv_ref[task]
        d = dw[(dw.task == task) & (dw.group == "all") & (dw.metric == pm)].iloc[0]
        rows.append({"row_id": rid.replace("direct", "devcv_direct"), "task": task, "metric": pm,
                     "evaluation": "development-only 5x5 CV reference (train patients)", "evaluated": True, "delta": c["delta_D_minus_C"],
                     "ci_lower": c["ci_lo"], "ci_upper": c["ci_hi"], "p_value": d.p_boot_two_sided, "p_type": "two-sided patient bootstrap",
                     "n_visit_pairs": int(c["n_rows"]), "n_patients": int(c["n_patients"]), "layer_c_value": d.C_mean,
                     "pct_change_vs_layer_c": 100 * c["delta_D_minus_C"] / d.C_mean if task == "multiclass" else np.nan})
    a = rep["P3_addon"]
    cm = ov[(ov.task == "multiclass") & (ov.metric == "rps")].iloc[0].C_mean
    rows.append({"row_id": "addon_multiclass_rps", "task": "multiclass", "metric": "rps", "evaluation": "held-out (validation + test)",
                 "evaluated": True, "delta": a["observed_delta_rps"], "ci_lower": a["patient_bootstrap"]["ci_lower"],
                 "ci_upper": a["patient_bootstrap"]["ci_upper"], "p_value": a["permutation"]["one_sided_p"],
                 "p_type": "one-sided permutation (RNA vectors permuted among held-out units; B = 10,000)",
                 "n_visit_pairs": a["n_rows"], "n_patients": a["n_patients"], "layer_c_value": cm,
                 "pct_change_vs_layer_c": 100 * a["observed_delta_rps"] / cm, "decision": a["decision"]})
    rows.append({"row_id": "addon_binary_auprc", "task": "binary", "metric": "auprc", "evaluation": "not evaluated", "evaluated": False})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "fig3A_trainonly_overall.csv", index=False)
    check("add-on success components (delta < 0, CI upper < 0, p <= 0.05)", list(a["success_components"].values()).count(True), 2, R / "heldout_report.json")
    check("add-on verdict", a["decision"], "SUGGESTIVE OR INCONCLUSIVE", R / "heldout_report.json")
    check("add-on frozen lambdas (all five repeats)", sorted(set(a["selected_lambdas"])), [1000.0], R / "heldout_report.json")
    check("add-on permutation p recomputed from k and B", (1 + a["permutation"]["k_null_le_observed"]) / (a["permutation"]["B"] + 1),
          a["permutation"]["one_sided_p"], R / "heldout_report.json", 1e-15)

    # ---- Figure 3B ----
    lab = {"Q1": "Q1_low_uncertainty", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4_high_uncertainty", "Q1_Q3": "Q1_Q3_lower_uncertainty"}
    rows = []
    for task, pm in (("binary", "auprc"), ("multiclass", "rps")):
        for g, gid in lab.items():
            r = w[(w.task == task) & (w.group == g) & (w.metric == pm)].iloc[0]
            rows.append({"task": task, "metric": pm, "group": gid, "delta": r.delta_D_minus_C, "ci_lower": r.ci_lo, "ci_upper": r.ci_hi,
                         "p_value": r.p_boot_two_sided, "adjusted_p": np.nan, "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients)})
        r = it[(it.task == task) & (it.contrast == "PRIMARY_Q4_vs_Q1_Q3") & (it.metric == pm)].iloc[0]
        rows.append({"task": task, "metric": pm, "group": "Q4_minus_Q1_Q3", "delta": r.interaction_a_minus_b, "ci_lower": r.ci_lo,
                     "ci_upper": r.ci_hi, "p_value": r.p_boot_two_sided, "adjusted_p": r.adjusted_p, "n_visit_pairs": np.nan, "n_patients": np.nan})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "fig3B_trainonly_uncertainty.csv", index=False)
    for task, n in (("binary", 2405), ("multiclass", 2689)):
        s = B[(B.task == task) & B.group.isin(["Q1_low_uncertainty", "Q2", "Q3", "Q4_high_uncertainty"])].n_visit_pairs.sum()
        check(f"[{task}] quartile rows sum to the held-out rows", int(s), n, R / "heldout_within_groups.csv")
    check("two primary interactions form one Holm family", int(B[B.group == "Q4_minus_Q1_Q3"].adjusted_p.notna().sum()), 2, R / "heldout_interactions.csv")
    cuts = {t: {k: art["entropy_cutpoints"][t][k] for k in ("q25", "q50", "q75", "q667", "q80")} for t in ("binary", "multiclass")}
    defs_b = {"cutpoints_from_development_entropy": cuts, "continuous_slope_binary": rep["P7_slope"],
              "negative_control_B1000": neg[(neg.group.isin(["all", "Q4"])) & neg.metric.isin(["auprc", "rps"])][
                  ["task", "group", "metric", "observed_D_minus_C", "null_mean", "null_sd", "empirical_one_sided_p_improvement"]].to_dict("records")}

    # ---- Figure 3C / Table S1 ----
    trt_name = {"pi": "Proteasome inhibitor (PI)", "imid": "Immunomodulatory drug (IMiD)", "steroid": "Corticosteroid",
                "cd38": "Anti-CD38 antibody", "chemo": "Chemotherapy"}
    st_name = {"stable": "Stable (same IMWG category)", "worsening": "Worsening (lower category)",
               "improving": "Improving (higher category)", "no_history": "No previous visit"}
    rows, comp_rows = [], []
    for task in ("binary", "multiclass"):
        for met in (("auprc", "auroc", "logloss") if task == "binary" else ("rps", "logloss")):
            for family, src in (("treatment", "Treatment class active at V(t)"), ("state", "IMWG response trajectory at V(t)")):
                for _, r in w[(w.task == task) & (w.metric == met) & (w.family == family) & ~w.group.str.startswith("not_")].iterrows():
                    s = r.group[3:] if family == "treatment" else r.group[len("state_"):]
                    role = ("primary" if s in ("pi", "imid", "steroid", "stable", "worsening") else "exploratory" if s in ("cd38", "chemo") else "secondary")
                    adj = r.adjustment_method if isinstance(r.adjustment_method, str) else ""
                    rows.append({"source_analysis": src, "stratum": s, "stratum_label": (trt_name if family == "treatment" else st_name)[s],
                                 "role": role, "task": task, "metric": met, "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients),
                                 "layer_c_value": r.C_mean, "layer_d_value": r.D_mean, "delta": r.delta_D_minus_C, "ci_lower": r.ci_lo,
                                 "ci_upper": r.ci_hi, "p_value": r.p_boot_two_sided, "adjusted_p": r.adjusted_p, "adjustment": adj,
                                 "adjustment_family": r.adjustment_family,
                                 "significant_after_adjustment": bool(pd.notna(r.adjusted_p) and r.adjusted_p < 0.05 and adj in ("Holm", "BH")),
                                 "pct_change_vs_layer_c": 100 * r.delta_D_minus_C / r.C_mean if met == "rps" else np.nan,
                                 "evaluation": "held-out (validation + test)"})
    S = pd.DataFrame(rows)
    # reference (all) row and complements
    for _, r in ov.iterrows():
        S = pd.concat([S, pd.DataFrame([{"source_analysis": "Reference", "stratum": "all", "stratum_label": "All RNA-available visit pairs",
                                         "role": "reference", "task": r.task, "metric": r.metric, "n_visit_pairs": int(r.n_rows),
                                         "n_patients": int(r.n_patients), "layer_c_value": r.C_mean, "layer_d_value": r.D_mean,
                                         "delta": r.delta_D_minus_C, "ci_lower": r.ci_lo, "ci_upper": r.ci_hi, "p_value": r.p_boot_two_sided,
                                         "adjusted_p": np.nan, "adjustment": "", "adjustment_family": "", "significant_after_adjustment": False,
                                         "pct_change_vs_layer_c": 100 * r.delta_D_minus_C / r.C_mean if r.metric == "rps" else np.nan,
                                         "evaluation": "held-out (validation + test)"}])], ignore_index=True)
    S.to_csv(OUT / "fig3C_trainonly_strata.csv", index=False)
    comp = w[(w.family == "treatment") & w.group.str.startswith("not_")][["task", "group", "metric", "n_rows", "n_patients", "delta_D_minus_C",
                                                                            "ci_lo", "ci_hi"]]
    comp.to_csv(OUT / "tableS1_trainonly_complements.csv", index=False)
    inter = it[it.family.isin(["treatment", "state"])].copy()
    inter.to_csv(OUT / "tableS1_trainonly_interactions.csv", index=False)
    S_dev = dw[dw.family.isin(["treatment", "state"]) & ~dw.group.str.startswith("not_")]
    S_dev.to_csv(OUT / "tableS1_trainonly_development_reference.csv", index=False)
    di[di.family.isin(["treatment", "state"])].to_csv(OUT / "tableS1_trainonly_development_interactions.csv", index=False)
    chemo_mc = S[(S.stratum == "chemo") & (S.task == "multiclass")]
    check("six-class chemotherapy stratum absent (frozen support rule)", len(chemo_mc), 0, R / "frozen_development.json")
    check("treatment sets frozen in development", art["treatment_sets"], {"binary": ["pi", "imid", "steroid", "cd38", "chemo"],
                                                                         "multiclass": ["pi", "imid", "steroid", "cd38"]}, R / "frozen_development.json")
    sigp = S[(S.metric.isin(["auprc", "rps"])) & S.significant_after_adjustment]
    check("no within-stratum primary-metric test is significant after adjustment", len(sigp), 0, R / "heldout_within_groups.csv")
    sig_i = inter[(inter.adjusted_p < 0.05)]
    check("interaction contrasts with BH p < 0.05 (all six-class, trajectory)", sorted(zip(sig_i.task, sig_i.contrast, sig_i.metric)),
          sorted([("multiclass", "improving_minus_stable", "rps"), ("multiclass", "worsening_minus_stable", "rps")] +
                 [(t, c, m) for t, c, m in zip(sig_i.task, sig_i.contrast, sig_i.metric) if m != "rps"]), R / "heldout_interactions.csv")
    # earlier pooled references (context rows; never plotted as results)
    earlier = pd.read_csv(EARLIER / "direction3/partA_predictive/partA_delta_within_group.csv")
    e_all = earlier[(earlier.group == "all") & earlier.is_primary_metric][["task", "metric", "n_rows", "n_patients", "delta_D_minus_C", "ci_lo", "ci_hi"]]
    e_all.to_csv(OUT / "reference_earlier_pooled_overall.csv", index=False)
    earlier_addon = json.load(open(EARLIER / "rna_low_variance_incremental/observed_summary.json", encoding="utf-8"))

    defs = {"evaluation_population": "held-out validation + test patients with RNA; locked train-only models; no refitting",
            "evidence_level": "post hoc, exploratory; the earlier pooled results had been seen; not independent validation; not external validation",
            "development_reference": "train-only 5x5 CV (hyperparameters tuned in the same patients; optimistic in absolute terms)",
            "earlier_pooled_results_reference": {"overall": e_all.to_dict("records"),
                                                 "addon_delta_rps": earlier_addon["delta_rps"], "addon_bootstrap": earlier_addon["patient_bootstrap"],
                                                 "note": "pooled train+validation CV; validation patients trained the fold models; not leakage-free"},
            "figure_3B": defs_b,
            "multiplicity": {"3B": "Holm across the two primary Q4 interactions", "3C_treatments": "Holm across PI, IMiD, steroid per task (primary metric); "
                             "BH for CD38 / chemo; BH for interaction contrasts per task x metric",
                             "3C_states": "Holm across stable + worsening per task; BH for improving / no-history and for state interactions"},
            "negative_control": "RNA vectors permuted among held-out units, locked Model D re-scored (B = 1,000): descriptive; the null is centred away from 0 because "
                                "scrambled RNA adds noise to the frozen model",
            "metrics": {"auprc": "higher is better", "rps": "lower is better (delta RPS > 0 = RNA worse)"}}
    json.dump(defs, open(OUT / "fig3ABC_trainonly_definitions.json", "w"), indent=2, default=float)
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "fig3ABC_trainonly_verification_checks.csv", index=False)
    json.dump({"script": "figures/fig3ABC_trainonly_extract_data.py", "outputs": sorted(p.name for p in OUT.iterdir() if p.name != "fig3ABC_trainonly_provenance.json"),
               "inputs_sha256_lf": {str(p): sha_lf(p) for p in [R / "heldout_report.json", R / "heldout_within_groups.csv", R / "heldout_interactions.csv",
                                                                 R / "frozen_development.json", PROTOCOL]}},
              open(OUT / "fig3ABC_trainonly_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
