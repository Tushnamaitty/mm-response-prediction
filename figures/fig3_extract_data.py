"""
Figure 3 data extraction and verification (read-only).

Run from repo root:
    python figures/fig3_extract_data.py

Reads locked clean-rerun result files only. Never retrains, rescores, or modifies them.
Writes one tidy data file per panel plus metadata to figures/data/Fig3/.

Panels
    A  Overall RNA contribution: direct XGBoost Layer D - Layer C, and the regularized RNA
       add-on over Layer C (six-class only; binary add-on not evaluated).
    B  RNA contribution (Layer D - Layer C) by Layer-C prediction-uncertainty quartile, with the
       prespecified Q4 vs Q1-Q3 interaction and its Holm adjustment; post-hoc sensitivity checks.
    C  RNA contribution (Layer D - Layer C) by treatment class active at V(t) and by response
       trajectory at V(t), with Holm / Benjamini-Hochberg adjusted p-values.
    D  Proliferation/MYC composite: pathway-selection evidence (pooled, included test patients),
       discovery estimate (train+validation) and post-hoc internal split-sample test estimate.

Evaluation population for A-C: out-of-fold predictions from 5 repeats x 5-fold patient-grouped
cross-validation on TRAIN + VALIDATION patients only; XGBoost Layer C and Layer D refitted in each
fold with the locked clean-rerun hyperparameters; evaluated on RNA-available visit pairs.
The held-out test patients are NOT used in A-C.

Every extracted value is checked against the value verified in the Figure 3 audit; any locked
source file listed in an output_manifest.json is SHA-256 checked. The script exits non-zero if a
check fails.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RC = Path("artifacts/results_clean_rerun")
OUT = Path("figures/data/Fig3")

SRC = {
    "d1_within": RC / "direction1/uncertainty/direction1_within_uncertainty_groups.csv",
    "d1_inter": RC / "direction1/uncertainty/direction1_uncertainty_interactions.csv",
    "d1_run": RC / "direction1/uncertainty/run_summary.json",
    "d1_perm": RC / "direction1_sensitivity/q4_permutation_null/permutation_summary.json",
    "d1_perm_gate": RC / "direction1_sensitivity/q4_permutation_null/gate_decision.json",
    "d1_slope": RC / "direction1_sensitivity/entropy_slope/entropy_slope_primary_and_secondary.csv",
    "lv_run": RC / "rna_low_variance_incremental/run_summary.json",
    "lv_obs": RC / "rna_low_variance_incremental/observed_summary.json",
    "lv_cfg": RC / "rna_low_variance_incremental/frozen_config.json",
    "lv_gates": RC / "rna_low_variance_incremental/gates_report.json",
    "lv_null": RC / "rna_low_variance_incremental/permutation_null.csv",
    "d3_delta": RC / "direction3/partA_predictive/partA_delta_within_group.csv",
    "d3_inter": RC / "direction3/partA_predictive/partA_interaction.csv",
    "d3_run": RC / "direction3/partA_predictive/run_summary.json",
    "d4_delta": RC / "direction4/partA_predictive/partA_state_delta.csv",
    "d4_inter": RC / "direction4/partA_predictive/partA_state_interactions.csv",
    "d4_run": RC / "direction4/partA_predictive/run_summary.json",
    "d2_s3": RC / "direction2/association/S3_S5_regression_results.csv",
    "d2_global": RC / "direction2/association/global_tests.csv",
    "d2_attr": RC / "direction2/association/attrition_summary.csv",
    "px_disc": RC / "direction2_replication/proliferation_axis/discovery_report.json",
    "px_test": RC / "direction2_replication/proliferation_axis/test_report.json",
    "px_load": RC / "direction2_replication/proliferation_axis/discovery_composite_loadings.csv",
    "px_frozen": RC / "direction2_replication/proliferation_axis/frozen_composite_artifact.json",
}

MODEL_CV = ("XGBoost, locked clean-rerun hyperparameters, refitted in each fold of 5 repeats x 5-fold "
            "patient-grouped cross-validation on train+validation patients (not the locked test-set models)")
POP = {"binary": "train+validation patients, RNA-available binary-eligible visit pairs (out-of-fold)",
       "multiclass": "train+validation patients, RNA-available six-class visit pairs (out-of-fold)"}
METRIC_INFO = {
    "auprc": ("AUPRC (area under the precision-recall curve)", "higher", "Layer D - Layer C; positive = RNA improves"),
    "auroc": ("AUROC (area under the ROC curve)", "higher", "Layer D - Layer C; positive = RNA improves"),
    "logloss": ("log loss", "lower", "Layer D - Layer C; negative = RNA improves"),
    "rps": ("RPS (ranked probability score, six ordered IMWG categories)", "lower",
            "Layer D - Layer C; negative = RNA improves, positive = RNA worsens"),
}
TASK_NAME = {"binary": "Binary improvement", "multiclass": "Six-class IMWG response"}

checks = []


def check(name, observed, expected, source, tol=1e-9):
    if isinstance(expected, float):
        ok = observed is not None and abs(float(observed) - expected) <= tol
    else:
        ok = observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(ok),
                   "source": str(source)})
    return ok


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jload(k):
    return json.load(open(SRC[k], encoding="utf-8"))


def verify_manifests():
    """SHA-256 check every source file that its folder's output_manifest.json lists."""
    for key, p in SRC.items():
        man = p.parent / "output_manifest.json"
        if not man.exists():
            checks.append({"check": f"manifest hash {p.name}", "observed": "no manifest in folder",
                           "expected": None, "passed": None, "source": str(p)})
            continue
        files = json.load(open(man, encoding="utf-8")).get("files", {})
        if p.name in files:
            check(f"manifest hash {p.name}", sha(p), files[p.name], man)
        else:
            checks.append({"check": f"manifest hash {p.name}", "observed": "not listed in manifest",
                           "expected": None, "passed": None, "source": str(man)})


def metric_cols(task, metric):
    label, better, direction = METRIC_INFO[metric]
    return {"task": task, "task_label": TASK_NAME[task], "metric": metric, "metric_label": label,
            "better_when": better, "comparison": direction}


# ------------------------------------------------------------------------------------------------
# Panel A
# ------------------------------------------------------------------------------------------------
def panel_a():
    w = pd.read_csv(SRC["d1_within"])
    lv, obs, cfg, gates = jload("lv_run"), jload("lv_obs"), jload("lv_cfg"), jload("lv_gates")
    rows = []
    for task, metric in [("binary", "auprc"), ("binary", "auroc"), ("binary", "logloss"),
                         ("multiclass", "rps"), ("multiclass", "logloss")]:
        r = w[(w.task == task) & (w.group == "all") & (w.metric == metric)].iloc[0]
        primary = metric in ("auprc", "rps")
        rows.append({
            "panel": "A", "row_id": f"direct_{task}_{metric}",
            "estimator": "Direct: XGBoost Layer D - XGBoost Layer C",
            "model_definition": MODEL_CV + "; Layer D = Layer C + 50 Hallmark RNA pathway scores + RNA age "
                                "(+ RNA-availability indicator)",
            **metric_cols(task, metric), "evaluated": True,
            "population": POP[task], "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients),
            "layer_c_value": r.C_mean, "comparator_value": r.D_mean,
            "delta": r.delta_D_minus_C, "ci_lower": r.ci_lo, "ci_upper": r.ci_hi,
            "ci_method": "95% patient-cluster bootstrap (2,000 resamples), unadjusted",
            "p_value": r.p_boot_two_sided, "p_method": "two-sided bootstrap",
            "adjusted_p": np.nan, "adjustment": "none (descriptive overall reference)",
            "pct_change_vs_layer_c": 100 * r.delta_D_minus_C / r.C_mean,
            "is_primary_metric": primary, "role": "descriptive overall reference (exploratory, post hoc)",
            "interpretation_flag": _flag(r.delta_D_minus_C, r.ci_lo, r.ci_hi, METRIC_INFO[metric][1]),
        })
    # regularized add-on, six-class only
    pb = obs["patient_bootstrap"]
    c_rps = float(gates["floored_rps_mean"])
    rows.append({
        "panel": "A", "row_id": "addon_multiclass_rps",
        "estimator": "Regularized RNA add-on - Layer C",
        "model_definition": ("Ridge-penalized multinomial correction of locked XGBoost Layer C out-of-fold "
                             "probabilities using the 50 Hallmark RNA pathway scores; penalty chosen by "
                             "nested cross-validation (lambda grid " + str(cfg["lambda_grid"]) + "); same "
                             "5 x 5 folds as Layer C; comparator = Layer C probabilities floored at 1e-6"),
        **metric_cols("multiclass", "rps"), "evaluated": True,
        "population": POP["multiclass"], "n_visit_pairs": int(gates["n_rna_rows"]),
        "n_patients": int(gates["n_rna_patients"]),
        "layer_c_value": c_rps, "comparator_value": c_rps + float(obs["delta_rps"]),
        "delta": float(obs["delta_rps"]), "ci_lower": float(pb["ci_lower"]), "ci_upper": float(pb["ci_upper"]),
        "ci_method": "95% patient-cluster bootstrap (2,000 resamples)",
        "p_value": float(lv["one_sided_permutation_p"]),
        "p_method": (f"one-sided whole-RNA-vector permutation, {lv['n_permutations']} permutations, "
                     f"{lv['n_null_delta_le_observed']} null values <= observed; minimum attainable p = "
                     f"1/{lv['n_permutations'] + 1}"),
        "adjusted_p": np.nan, "adjustment": "none (single prespecified test within this analysis)",
        "pct_change_vs_layer_c": 100 * float(obs["delta_rps"]) / c_rps,
        "is_primary_metric": True, "role": "post-hoc exploratory; prespecified success rule met",
        "interpretation_flag": _flag(float(obs["delta_rps"]), float(pb["ci_lower"]), float(pb["ci_upper"]), "lower"),
    })
    rows.append({
        "panel": "A", "row_id": "addon_binary_auprc",
        "estimator": "Regularized RNA add-on - Layer C",
        "model_definition": "Not evaluated: the analysis's prespecified stopping rule excluded the binary task",
        **metric_cols("binary", "auprc"), "evaluated": False, "population": POP["binary"],
        "role": "not evaluated", "interpretation_flag": "Not evaluated",
    })
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "fig3A_overall_rna_contribution.csv", index=False)
    pd.read_csv(SRC["lv_null"]).assign(observed_delta_rps=obs["delta_rps"]).to_csv(
        OUT / "fig3A_addon_permutation_null.csv", index=False)

    g = a.set_index("row_id")
    check("A direct binary dAUPRC", g.loc["direct_binary_auprc", "delta"], 0.004764462117, SRC["d1_within"], 1e-11)
    check("A direct binary n pairs", int(g.loc["direct_binary_auprc", "n_visit_pairs"]), 6939, SRC["d1_within"])
    check("A direct binary n patients", int(g.loc["direct_binary_auprc", "n_patients"]), 596, SRC["d1_within"])
    check("A direct six-class dRPS", g.loc["direct_multiclass_rps", "delta"], 0.0007515795024, SRC["d1_within"], 1e-12)
    check("A direct six-class dRPS CI lower", g.loc["direct_multiclass_rps", "ci_lower"], 0.0002310248564, SRC["d1_within"], 1e-12)
    check("A direct six-class n pairs", int(g.loc["direct_multiclass_rps", "n_visit_pairs"]), 7724, SRC["d1_within"])
    check("A add-on dRPS", g.loc["addon_multiclass_rps", "delta"], -0.00033976481221190634, SRC["lv_obs"], 1e-15)
    check("A add-on CI upper", g.loc["addon_multiclass_rps", "ci_upper"], -0.00010691985731233264, SRC["lv_obs"], 1e-15)
    check("A add-on permutation p", g.loc["addon_multiclass_rps", "p_value"], 1 / 201, SRC["lv_run"], 1e-12)
    check("A add-on n pairs / patients", (int(gates["n_rna_rows"]), int(gates["n_rna_patients"])), (7724, 600), SRC["lv_gates"])
    check("A same Layer C baseline in add-on and direct analyses (raw RPS)",
          float(gates["raw_rps_mean"]), float(g.loc["direct_multiclass_rps", "layer_c_value"]), SRC["lv_gates"], 1e-9)
    check("A add-on decision", lv["decision"], "SUPPORTED (EXPLORATORY)", SRC["lv_run"])
    return a


def _flag(delta, lo, hi, better):
    if lo > 0 or hi < 0:
        improves = (delta > 0) if better == "higher" else (delta < 0)
        return "CI excludes 0: RNA improves" if improves else "CI excludes 0: RNA worsens"
    return "CI includes 0: no detectable difference"


# ------------------------------------------------------------------------------------------------
# Panel B
# ------------------------------------------------------------------------------------------------
QUART = {"Q1_low_uncertainty": ("Q1 (lowest uncertainty)", 1), "Q2": ("Q2", 2), "Q3": ("Q3", 3),
         "Q4_high_uncertainty": ("Q4 (highest uncertainty)", 4),
         "Q1_Q3_lower_uncertainty": ("Q1-Q3 combined (comparator)", 5), "all": ("All RNA-available pairs", 0)}


def panel_b():
    w = pd.read_csv(SRC["d1_within"])
    it = pd.read_csv(SRC["d1_inter"])
    run = jload("d1_run")
    rows = []
    for (task, metric) in [("binary", "auprc"), ("binary", "auroc"), ("binary", "logloss"),
                           ("multiclass", "rps"), ("multiclass", "logloss")]:
        for g, (lab, order) in QUART.items():
            r = w[(w.task == task) & (w.group == g) & (w.metric == metric)].iloc[0]
            rows.append({"panel": "B", "row_type": "within_group", "group": g, "group_label": lab,
                         "group_order": order, **metric_cols(task, metric),
                         "is_primary_metric": metric in ("auprc", "rps"),
                         "model_definition": MODEL_CV, "population": POP[task],
                         "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients),
                         "layer_c_value": r.C_mean, "layer_d_value": r.D_mean,
                         "delta": r.delta_D_minus_C, "ci_lower": r.ci_lo, "ci_upper": r.ci_hi,
                         "p_value": r.p_boot_two_sided, "adjusted_p": np.nan,
                         "adjustment": "none (within-group estimate; descriptive)",
                         "pct_change_vs_layer_c": 100 * r.delta_D_minus_C / r.C_mean,
                         "role": r.role,
                         "interpretation_flag": _flag(r.delta_D_minus_C, r.ci_lo, r.ci_hi, METRIC_INFO[metric][1])})
        x = it[(it.task == task) & (it.contrast == "PRIMARY_Q4_vs_Q1_Q3") & (it.metric == metric)].iloc[0]
        primary = metric in ("auprc", "rps")
        rows.append({"panel": "B", "row_type": "interaction", "group": "Q4_minus_Q1_Q3",
                     "group_label": "Interaction: (D - C in Q4) - (D - C in Q1-Q3)", "group_order": 6,
                     **metric_cols(task, metric), "is_primary_metric": primary,
                     "comparison": x.interpret_sign, "model_definition": MODEL_CV, "population": POP[task],
                     "delta": x.interaction_uncertain_minus_comparator, "ci_lower": x.ci_lo, "ci_upper": x.ci_hi,
                     "p_value": x.p_boot_two_sided, "adjusted_p": x.adjusted_p if primary else np.nan,
                     "adjustment": (f"Holm across the 2 primary interaction tests (binary AUPRC, six-class RPS)"
                                    if primary else "none (secondary/descriptive)"),
                     "role": "prespecified primary interaction" if primary else "secondary/descriptive",
                     "interpretation_flag": _inter_flag(x, primary)})
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "fig3B_uncertainty_quartiles.csv", index=False)

    perm, gate = jload("d1_perm"), jload("d1_perm_gate")
    sl = pd.read_csv(SRC["d1_slope"]).set_index("analysis")
    sens = {
        "uncertainty_definition": run["uncertainty_definition"],
        "primary_estimand": run["primary_estimand"],
        "multiplicity": run["multiplicity"],
        "interpretation_limit": run["interpretation_limit"],
        "post_hoc_sensitivity_checks": {
            "label": "post-hoc exploratory robustness checks of the binary AUPRC interaction; outside the Holm family",
            "scrambled_rna_permutation": {
                "observed_interaction_auprc": perm["summary"]["auprc"]["observed_interaction"],
                "n_permutations": perm["summary"]["auprc"]["n_permutations"],
                "n_null_ge_observed": perm["summary"]["auprc"]["n_exceeding_or_equal"],
                "empirical_one_sided_p": perm["summary"]["auprc"]["empirical_one_sided_p"],
                "p_monte_carlo_95_interval": perm["summary"]["auprc"]["p_monte_carlo_95_interval"],
                "note": gate["note"]},
            "continuous_entropy_slope_logloss_benefit": {
                "slope": float(sl.loc["PRIMARY_rank_slope", "slope_per_full_rank_range"]),
                "ci": [float(sl.loc["PRIMARY_rank_slope", "ci_lo"]), float(sl.loc["PRIMARY_rank_slope", "ci_hi"])],
                "p_boot_two_sided": float(sl.loc["PRIMARY_rank_slope", "p_boot_two_sided"]),
                "unit": sl.loc["PRIMARY_rank_slope", "unit_note"],
                "n_visit_pairs": int(sl.loc["PRIMARY_rank_slope", "n_pairs"]),
                "n_patients": int(sl.loc["PRIMARY_rank_slope", "n_patients"])}},
        "plotting_note": ("Raw AUPRC is not comparable across quartiles because the improvement rate differs "
                          "between quartiles; plot only Layer D - Layer C differences."),
    }
    json.dump(sens, open(OUT / "fig3B_definitions_and_sensitivity.json", "w"), indent=2)

    bi = b[(b.row_type == "interaction")].set_index(["task", "metric"])
    check("B binary AUPRC interaction", bi.loc[("binary", "auprc"), "delta"], 0.01394803061, SRC["d1_inter"], 1e-10)
    check("B binary AUPRC interaction Holm p", bi.loc[("binary", "auprc"), "adjusted_p"], 0.05997001499, SRC["d1_inter"], 1e-10)
    check("B six-class RPS interaction Holm p", bi.loc[("multiclass", "rps"), "adjusted_p"], 0.1419290355, SRC["d1_inter"], 1e-10)
    q4 = b[(b.group == "Q4_high_uncertainty") & (b.metric == "rps")].iloc[0]
    check("B six-class Q4 dRPS", q4.delta, 0.001597655192, SRC["d1_within"], 1e-12)
    check("B permutation p", sens["post_hoc_sensitivity_checks"]["scrambled_rna_permutation"]["empirical_one_sided_p"],
          4 / 201, SRC["d1_perm"], 1e-12)
    check("B entropy slope p", sens["post_hoc_sensitivity_checks"]["continuous_entropy_slope_logloss_benefit"]["p_boot_two_sided"],
          0.1209395302, SRC["d1_slope"], 1e-9)
    for task, metric, n in [("binary", "auprc", 6939), ("multiclass", "rps", 7724)]:
        qs = b[(b.task == task) & (b.metric == metric) & b.group.isin(["Q1_low_uncertainty", "Q2", "Q3", "Q4_high_uncertainty"])]
        check(f"B {task} quartile rows sum to all", int(qs.n_visit_pairs.sum()), n, SRC["d1_within"])
    return b


def _inter_flag(x, primary):
    sig = x.ci_lo > 0 or x.ci_hi < 0
    if primary:
        adj = x.adjusted_p
        return (f"unadjusted CI excludes 0; Holm-adjusted p = {adj:.3f}"
                + (" (not significant after adjustment)" if adj >= 0.05 else " (significant after adjustment)"))
    return "CI excludes 0 (unadjusted, descriptive)" if sig else "CI includes 0"


# ------------------------------------------------------------------------------------------------
# Panel C
# ------------------------------------------------------------------------------------------------
TREAT_LABEL = {"pi": "Proteasome inhibitor (PI)", "imid": "Immunomodulatory drug (IMiD)",
               "steroid": "Corticosteroid", "cd38": "Anti-CD38 antibody", "chemo": "Chemotherapy"}
STATE_LABEL = {"stable": "Stable (same IMWG category as previous visit)",
               "worsening": "Worsening (lower category than previous visit)",
               "improving": "Improving (higher category than previous visit)",
               "no_history": "No previous visit (first visit pair)"}


def panel_c():
    d3, d3i = pd.read_csv(SRC["d3_delta"]), pd.read_csv(SRC["d3_inter"])
    d4, d4i = pd.read_csv(SRC["d4_delta"]), pd.read_csv(SRC["d4_inter"])
    rows = []
    for _, r in d3.iterrows():
        on = r.group.startswith("on_")
        if r.group == "all":
            lab, grp, stratum = "All RNA-available visit pairs", "reference", "all"
        else:
            lab = ("On " if on else "Not on ") + TREAT_LABEL[r.treatment] + " at V(t)"
            grp, stratum = "treatment", r.group
        rows.append(_c_row(r, "Direction 3: treatment class active at V(t)", grp, stratum, lab,
                           r.treatment if r.group != "all" else "all",
                           {"reference": "reference", "primary": "primary" if on else "complement (descriptive)",
                            "exploratory": "exploratory" if on else "complement (descriptive)"}[r.role],
                           "prespecified primary" if r.role == "primary" else r.role))
    for _, r in d4[d4.analysis == "primary"].iterrows():
        role = "primary" if r.state in ("stable", "worsening") else "secondary"
        rows.append(_c_row(r, "Direction 4: IMWG response trajectory at V(t)", "trajectory", r.state,
                           STATE_LABEL[r.state], "trajectory", role, role))
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "fig3C_treatment_and_trajectory_strata.csv", index=False)

    irows = []
    for _, x in d3i.iterrows():
        irows.append({"panel": "C", "source_analysis": "Direction 3", "contrast": f"on_{x.treatment}_minus_not_on",
                      "contrast_label": f"(D - C on {TREAT_LABEL[x.treatment]}) - (D - C not on)",
                      **metric_cols(x.task, x.metric), "role": x.role, "delta_group": x.delta_on,
                      "delta_reference": x.delta_not_on, "interaction": x.interaction_on_minus_not,
                      "ci_lower": x.ci_lo, "ci_upper": x.ci_hi, "p_value": x.p_boot_two_sided,
                      "adjusted_p": x.adjusted_p, "adjustment": f"{x.adjustment_method} ({x.adjustment_family})"})
    for _, x in d4i[d4i.analysis == "primary"].iterrows():
        irows.append({"panel": "C", "source_analysis": "Direction 4", "contrast": x.contrast,
                      "contrast_label": x.contrast.replace("_minus_", " - ").capitalize() + " (D - C differences)",
                      **metric_cols(x.task, x.metric), "role": "secondary", "delta_group": x.delta_state,
                      "delta_reference": x.delta_reference, "interaction": x.interaction_delta,
                      "ci_lower": x.ci_lo, "ci_upper": x.ci_hi, "p_value": x.p_boot_two_sided,
                      "adjusted_p": x.adjusted_p, "adjustment": f"{x.adjustment_method} ({x.adjustment_family})"})
    pd.DataFrame(irows).to_csv(OUT / "fig3C_interactions_secondary.csv", index=False)

    k = c.set_index(["source_analysis", "stratum", "task", "metric"])
    D3, D4 = "Direction 3: treatment class active at V(t)", "Direction 4: IMWG response trajectory at V(t)"
    for strat, exp_p in [("on_pi", 0.044978), ("on_imid", 0.041979), ("on_steroid", 0.041979)]:
        check(f"C {strat} six-class RPS Holm p", round(float(k.loc[(D3, strat, "multiclass", "rps"), "adjusted_p"]), 6),
              exp_p, SRC["d3_delta"], 1e-9)
    for strat, exp_p in [("on_pi", 0.467766), ("on_imid", 0.233883), ("on_steroid", 0.467766)]:
        check(f"C {strat} binary AUPRC Holm p", round(float(k.loc[(D3, strat, "binary", "auprc"), "adjusted_p"]), 6),
              exp_p, SRC["d3_delta"], 1e-9)
    check("C stable six-class dRPS", float(k.loc[(D4, "stable", "multiclass", "rps"), "delta"]), 0.001135, SRC["d4_delta"], 5e-7)
    check("C stable six-class RPS Holm p", round(float(k.loc[(D4, "stable", "multiclass", "rps"), "adjusted_p"]), 6),
          0.001999, SRC["d4_delta"], 1e-6)
    check("C worsening six-class RPS Holm p", round(float(k.loc[(D4, "worsening", "multiclass", "rps"), "adjusted_p"]), 6),
          0.49975, SRC["d4_delta"], 1e-6)
    check("C reference row equals panel A direct six-class dRPS",
          float(k.loc[(D3, "all", "multiclass", "rps"), "delta"]), 0.0007515795024, SRC["d3_delta"], 1e-10)
    check("C on_cd38 six-class n pairs", int(k.loc[(D3, "on_cd38", "multiclass", "rps"), "n_visit_pairs"]), 559, SRC["d3_delta"])
    check("C D4 primary rows only", int((c.source_analysis == D4).sum()), 20, SRC["d4_delta"])
    return c


def _c_row(r, source, grp, stratum, lab, family, role, role_detail):
    task, metric = r.task, r.metric
    adj_method = str(r.adjustment_method)
    return {"panel": "C", "source_analysis": source, "stratum_group": grp, "stratum": stratum,
            "stratum_label": lab, "family": family, **metric_cols(task, metric),
            "is_primary_metric": bool(r.is_primary_metric), "role": role, "role_detail": role_detail,
            "model_definition": MODEL_CV, "population": POP[task],
            "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients),
            "layer_c_value": r.C_mean, "layer_d_value": r.D_mean, "delta": r.delta_D_minus_C,
            "ci_lower": r.ci_lo, "ci_upper": r.ci_hi, "p_value": r.p_boot_two_sided,
            "adjusted_p": r.adjusted_p, "adjustment": adj_method + (f" ({r.adjustment_family})"
                                                                     if isinstance(r.adjustment_family, str) else ""),
            "significant_after_adjustment": bool(pd.notna(r.adjusted_p) and r.adjusted_p < 0.05),
            "pct_change_vs_layer_c": 100 * r.delta_D_minus_C / r.C_mean,
            "interpretation_flag": _flag(r.delta_D_minus_C, r.ci_lo, r.ci_hi, METRIC_INFO[metric][1]),
            "note": ("Treatment strata overlap (a patient can be on several classes at V(t))"
                     if grp == "treatment" else "")}


# ------------------------------------------------------------------------------------------------
# Panel D
# ------------------------------------------------------------------------------------------------
FAMILY5 = ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle", "Myc Targets V1", "Myc Targets V2"]


def panel_d():
    disc, test = jload("px_disc"), jload("px_test")
    s3 = pd.read_csv(SRC["d2_s3"])
    glob = pd.read_csv(SRC["d2_global"])
    outcome = ("next-visit IMWG category on the ordered scale PD < SD < PR < VGPR < CR < sCR; "
               "cumulative-logit ordinal GEE, patient-clustered robust SEs")
    rows = []
    for pw in FAMILY5:
        r = s3[(s3.family == "ordinal") & (s3.analysis == "S3_adjusted") & (s3.pathway == pw)].iloc[0]
        rows.append({"panel": "D", "evidence_stage": "1_selection_evidence", "estimate_label": f"{pw} (single pathway)",
                     "stage_description": ("Pooled covariate-adjusted association (Direction 2 S3 sensitivity "
                                           "regression). Included test patients; these results motivated the "
                                           "post-hoc choice of the five-pathway family."),
                     "patients_included": "all RNA-available patients, train+validation+test (complete cases)",
                     "includes_test_patients": True,
                     "adjustment_model": ("S3: 16 continuous Model-C covariates + RNA age + strata "
                                          "(V(t) response x treatment line x PI/IMiD/CD38)"),
                     "exposure": f"{pw} score, per SD", "outcome_model": outcome,
                     "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients),
                     "beta": r.estimate, "ci_lower": r.ci_lo, "ci_upper": r.ci_hi, "odds_ratio": r.odds_ratio_per_SD,
                     "or_ci_lower": np.exp(r.ci_lo), "or_ci_upper": np.exp(r.ci_hi),
                     "p_value": r.p_wald_robust, "p_method": "two-sided robust Wald",
                     "adjusted_p": r.q_bh, "adjustment": "Benjamini-Hochberg q across 50 pathways",
                     "role": "sensitivity analysis (Direction 2); pathway-selection evidence"})
    de = disc["discovery_estimate"]
    rows.append({"panel": "D", "evidence_stage": "2_discovery", "estimate_label": "Proliferation/MYC composite: discovery",
                 "stage_description": ("Composite fixed outcome-blind on train+validation patients (first principal "
                                       "component of the five pathway scores); association estimated in the same "
                                       "patients; gate required beta < 0 with one-sided p < 0.025."),
                 "patients_included": "train+validation patients only", "includes_test_patients": False,
                 "adjustment_model": ("full covariate model: composite + 16 continuous Model-C covariates + "
                                      "log1p(RNA age) + exact strata (V(t) response x line group x PI/IMiD/CD38)"),
                 "exposure": "composite score, per discovery SD", "outcome_model": outcome,
                 "n_visit_pairs": int(disc["accounting"]["discovery_rows"]),
                 "n_patients": int(disc["accounting"]["discovery_patients"]),
                 "beta": de["beta_disc"], "ci_lower": de["ci95_disc"][0], "ci_upper": de["ci95_disc"][1],
                 "odds_ratio": np.exp(de["beta_disc"]), "or_ci_lower": np.exp(de["ci95_disc"][0]),
                 "or_ci_upper": np.exp(de["ci95_disc"][1]), "p_value": de["one_sided_p_disc"],
                 "p_method": "one-sided robust Wald (H1: beta < 0)", "adjusted_p": np.nan,
                 "adjustment": "none (single gate test)", "role": "discovery (post hoc family)"})
    tr = test["result"]
    rows.append({"panel": "D", "evidence_stage": "3_split_sample_test",
                 "estimate_label": "Proliferation/MYC composite: post-hoc internal split-sample test",
                 "stage_description": ("Frozen composite and frozen clinical linear predictor applied to held-out "
                                       "test patients; one directional test run once. Not independent replication "
                                       "and not external validation (family chosen after pooled results that "
                                       "included these test patients)."),
                 "patients_included": "held-out test patients only", "includes_test_patients": True,
                 "adjustment_model": ("recalibration model: composite + frozen discovery clinical linear predictor "
                                      "(eta), thresholds re-estimated; NOT the same adjustment model as discovery"),
                 "exposure": "composite score, per discovery SD", "outcome_model": outcome,
                 "n_visit_pairs": int(test["accounting"]["test_rows"]), "n_patients": int(test["accounting"]["test_patients"]),
                 "beta": tr["beta_test"], "ci_lower": tr["ci95_wald"][0], "ci_upper": tr["ci95_wald"][1],
                 "odds_ratio": tr["odds_ratio_per_discovery_SD"], "or_ci_lower": np.exp(tr["ci95_wald"][0]),
                 "or_ci_upper": np.exp(tr["ci95_wald"][1]), "p_value": tr["one_sided_p_H1_beta_lt_0"],
                 "p_method": "one-sided robust Wald (H1: beta < 0)", "adjusted_p": np.nan,
                 "adjustment": "none (one primary test within this analysis)",
                 "bootstrap_ci_lower_descriptive": tr["bootstrap_ci95_descriptive"][0],
                 "bootstrap_ci_upper_descriptive": tr["bootstrap_ci95_descriptive"][1],
                 "recalibration_slope_eta": tr["recalibration_slope_b_eta"],
                 "role": "post-hoc internal split-sample test; prespecified rule (95% CI entirely < 0) met"})
    dd = pd.DataFrame(rows)
    dd["direction_note"] = "beta < 0 / odds ratio < 1: higher score associated with a worse next IMWG category"
    dd["comparability_note"] = ("Estimates come from different patients and different adjustment models; "
                                "they are shown side by side, not as directly comparable effect sizes.")
    dd.to_csv(OUT / "fig3D_proliferation_myc_association.csv", index=False)

    load = pd.read_csv(SRC["px_load"])
    load["variance_explained_by_pc1"] = disc["composite"]["variance_explained"]
    load["n_distinct_rna_samples_used"] = disc["composite"]["n_distinct_samples"]
    load.to_csv(OUT / "fig3D_composite_loadings.csv", index=False)

    acc = pd.DataFrame([
        {"stage": "discovery", "rows_before_exclusions": disc["accounting"]["discovery_rows_before_exclusions"],
         "patients_before": disc["accounting"]["discovery_patients_before"],
         "rows_lost_incomplete_covariates": disc["accounting"]["rows_lost_incomplete_covariates"],
         "rows_lost_sparse_or_unretained_strata": disc["accounting"]["rows_lost_sparse_strata"],
         "rows_analysed": disc["accounting"]["discovery_rows"], "patients_analysed": disc["accounting"]["discovery_patients"],
         **{f"class_{k}": v for k, v in disc["accounting"]["outcome_class_counts"].items()}},
        {"stage": "split_sample_test", "rows_before_exclusions": test["accounting"]["test_rows_before_exclusions"],
         "patients_before": test["accounting"]["test_patients_before"],
         "rows_lost_incomplete_covariates": test["accounting"]["rows_lost_incomplete_covariates"],
         "rows_lost_sparse_or_unretained_strata": test["accounting"]["rows_lost_strata_not_retained_in_discovery"],
         "rows_analysed": test["accounting"]["test_rows"], "patients_analysed": test["accounting"]["test_patients"],
         **{f"class_{k}": v for k, v in test["support"]["outcome_class_counts"].items()}},
    ])
    acc["universe_note"] = ("Direction 2 universe requires an RNA sample at or before V(t) AND a non-missing current "
                            "treatment line; one RNA-available pair lacks the line, hence 9,172 (not 9,173) six-class "
                            "and 8,261 (not 8,262) binary pairs, and 1,448 (not 1,449) test pairs.")
    acc.to_csv(OUT / "fig3D_attrition.csv", index=False)

    gt = glob[glob.analysis == "primary"].copy()
    gt["description"] = ("Direction 2 PRIMARY analysis: matched-pair global sign-flip test across 50 pathways "
                         "(within RNA-discordant matched patient pairs); null result")
    gt.to_csv(OUT / "fig3D_primary_matched_pair_global_test.csv", index=False)

    k = dd.set_index("evidence_stage")
    check("D discovery beta", float(k.loc["2_discovery", "beta"]), -0.2786100802788457, SRC["px_disc"], 1e-15)
    check("D test beta", float(k.loc["3_split_sample_test", "beta"]), -0.2035164250232188, SRC["px_test"], 1e-15)
    check("D test CI upper", float(k.loc["3_split_sample_test", "ci_upper"]), -0.029619044949844076, SRC["px_test"], 1e-15)
    check("D test OR", float(k.loc["3_split_sample_test", "odds_ratio"]), 0.8158568037456768, SRC["px_test"], 1e-12)
    check("D test OR equals exp(beta)", float(np.exp(tr["beta_test"])), float(tr["odds_ratio_per_discovery_SD"]), SRC["px_test"], 1e-12)
    check("D test n rows / patients", (int(k.loc["3_split_sample_test", "n_visit_pairs"]),
                                       int(k.loc["3_split_sample_test", "n_patients"])), (1221, 84), SRC["px_test"])
    check("D discovery n rows / patients", (int(k.loc["2_discovery", "n_visit_pairs"]),
                                            int(k.loc["2_discovery", "n_patients"])), (6212, 448), SRC["px_disc"])
    check("D all loadings positive", bool((load.loading > 0).all()), True, SRC["px_load"])
    check("D loadings pathways = family", sorted(load.pathway), sorted(FAMILY5), SRC["px_load"])
    check("D test interpretation", tr["interpretation"].split(" ")[0], "REPLICATED", SRC["px_test"])
    check("D primary global ordinal p", float(gt[gt.family == "ordinal"].global_p.iloc[0]), 0.14908509, SRC["d2_global"], 1e-8)
    check("D primary global binary p", float(gt[gt.family == "binary"].global_p.iloc[0]), 0.44145585, SRC["d2_global"], 1e-8)
    attr = pd.read_csv(SRC["d2_attr"]).set_index("family")
    check("D2 universe six-class rows", int(attr.loc["ordinal", "universe_rows"]), 9172, SRC["d2_attr"])
    check("D2 universe binary rows", int(attr.loc["binary", "universe_rows"]), 8261, SRC["d2_attr"])
    check("D test rows before exclusions = D2 test universe", int(test["accounting"]["test_rows_before_exclusions"]), 1448, SRC["px_test"])
    check("D discovery rows before = panel A six-class rows", int(disc["accounting"]["discovery_rows_before_exclusions"]), 7724, SRC["px_disc"])
    check("D S3 selection rows all negative beta", bool((dd[dd.evidence_stage == "1_selection_evidence"].beta < 0).all()), True, SRC["d2_s3"])
    return dd


# ------------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_manifests()
    panel_a()
    panel_b()
    panel_c()
    panel_d()

    glossary = {
        "information_layers": {"C": "Layer C = current clinical state + longitudinal history + treatment context (143 source input features)",
                               "D": "Layer D = Layer C + 50 Hallmark RNA pathway scores + RNA age (194 source input features + 1 derived RNA-availability indicator)"},
        "algorithm": "XGBoost (reporting reference algorithm; not a validation-selected best model)",
        "evaluation_population_A_to_C": ("Out-of-fold predictions, 5 repeats x 5-fold patient-grouped cross-validation on "
                                          "train+validation patients; RNA-available visit pairs only; binary 6,939 pairs / "
                                          "596 patients, six-class 7,724 pairs / 600 patients; held-out test patients not used"),
        "evaluation_population_D": ("Selection evidence: pooled, all splits; discovery: train+validation; split-sample test: "
                                    "held-out test patients"),
        "comparison_A_to_C": "Layer D - Layer C on identical visit pairs (paired)",
        "metrics": {k: {"label": v[0], "better_when": v[1], "sign": v[2]} for k, v in METRIC_INFO.items()},
        "uncertainty_quartiles": "quartiles of mean normalized Shannon entropy of Layer C out-of-fold probabilities (outcome-blind)",
        "trajectory": "IMWG category at V(t) compared with the previous visit: improving / stable / worsening; no_history = first visit pair",
        "treatment": "drug class active at V(t) (currently_on_*), not treatment received between V(t) and V(t+1)",
        "multiplicity": {"B": "Holm across 2 primary interaction tests",
                         "C_treatment": "Holm across 3 primary treatments (PI, IMiD, steroid) per task on the primary metric; BH for Tier-2 and interactions",
                         "C_trajectory": "Holm across stable + worsening per task on the primary metric; BH for secondary states and interactions",
                         "D_selection": "BH across 50 pathways (pooled S3)"},
        "pct_change_vs_layer_c": "100 x delta / Layer C value; derived arithmetic on stored values (optional annotation)",
        "evidence_level": "All Figure 3 analyses are post-hoc exploratory; none is external validation",
    }
    json.dump(glossary, open(OUT / "fig3_definitions.json", "w"), indent=2)

    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "fig3_verification_checks.csv", index=False)
    outputs = sorted(p.name for p in OUT.iterdir() if p.name != "fig3_provenance.json")
    json.dump({"script": "figures/fig3_extract_data.py",
               "inputs_sha256": {str(p): sha(p) for p in SRC.values()},
               "outputs": outputs + ["fig3_provenance.json"]},
              open(OUT / "fig3_provenance.json", "w"), indent=2)
    failed = ck[ck.passed == False]  # noqa: E712
    print(f"{len(ck)} checks: {int((ck.passed == True).sum())} passed, {len(failed)} failed, "  # noqa: E712
          f"{int(ck.passed.isna().sum())} informational")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
