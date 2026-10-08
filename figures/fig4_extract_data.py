"""
Figure 4 data extraction and verification (read-only).

Run from repo root:
    python figures/fig4_extract_data.py

Reads locked survival results only; never retrains, refits, or recomputes any metric.
Writes verified Figure 4 source data to figures/data/Fig4/.

Panels
    A  C-index of the clinical baseline model vs clinical baseline + RNA (PFS and OS), with the five
       per-repeat values (background) and Delta C with its 95% patient-bootstrap CI (main comparison).
    B  Forest plot of Delta C for the primary analysis and the two sensitivity analyses
       (expanded clinical baseline; RNA timing restricted to day -30 to 0), PFS and OS.
       The OS timing sensitivity was not run (specification limits it to PFS); it is recorded as
       "not evaluated", never imputed.
    IBS (integrated Brier score) is extracted for the caption / supplement only (point estimates;
    no CI exists).

Evaluation: 5 repeats x 5-fold patient-level cross-validation within the survival cohort
(INTERNAL validation only; not external validation). Day 0 = CoMMpass index date = start of first
myeloma treatment (MMRF data dictionary: every CoMMpass patient entered at first treatment;
subject_deid.csv index_date = "first_treatment"; checked below).

The script exits non-zero if any check fails.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RC = Path("artifacts/results_clean_rerun")
SI = RC / "survival_incremental"
SR = RC / "survival_richer_baseline_sensitivity"
OUT = Path("figures/data/Fig4")

SRC = {
    "si_final": SI / "final_summary.json",
    "si_spec": SI / "analysis_spec.json",
    "si_pfs_sum": SI / "primary_pfs_summary.json",
    "si_os_sum": SI / "secondary_os_summary.json",
    "si_sens_sum": SI / "sensitivity_m30_0_pfs_summary.json",
    "si_pfs_rep": SI / "primary_pfs_repeat_metrics.csv",
    "si_os_rep": SI / "secondary_os_repeat_metrics.csv",
    "si_sens_rep": SI / "sensitivity_m30_0_pfs_repeat_metrics.csv",
    "si_pfs_oof": SI / "primary_pfs_oof_predictions.csv",
    "si_os_oof": SI / "secondary_os_oof_predictions.csv",
    "si_sens_oof": SI / "sensitivity_m30_0_pfs_oof_predictions.csv",
    "si_done": SI / "DONE.json",
    "sr_final": SR / "final_summary.json",
    "sr_prespec": SR / "prespecification.json",
    "sr_gate": SR / "gate_report.json",
    "sr_cyto": SR / "cytogenetics_availability.json",
    "sr_pfs_rep": SR / "primary_pfs_richer_repeat_metrics.csv",
    "sr_os_rep": SR / "secondary_os_richer_repeat_metrics.csv",
    "sr_pfs_oof": SR / "primary_pfs_richer_oof_predictions.csv",
    "sr_os_oof": SR / "secondary_os_richer_oof_predictions.csv",
    "sr_done": SR / "DONE.json",
    "subject": Path("data/clinical/subject_deid.csv"),
    "pairs": Path("data/clinical/visit_pairs_with_rna.csv"),
}

BASELINE = "Clinical baseline: age at index, gender, ISS, exact first-line regimen (ridge Cox)"
EXPANDED = ("Expanded clinical baseline: clinical baseline + 7 prespecified baseline values (LDH, beta-2 "
            "microglobulin, albumin, creatinine, haemoglobin, calcium, bone-marrow plasma-cell %)")
RNA_ADD = "+ 50 Hallmark RNA pathway scores from one pre-treatment RNA sample"
CI_METHOD = ("95% patient-bootstrap percentile CI (2,000 resamples of patients; fixed out-of-fold "
             "predictions, all 5 repeats kept per patient; covers patient sampling, not model refitting)")
EVAL = ("Harrell's C on pooled out-of-fold predictions within each repeat, averaged over 5 repeats of "
        "5-fold patient-level cross-validation (internal validation; not external validation)")
DAY0 = "Day 0 = CoMMpass index date = start of first myeloma treatment"

checks = []


def check(name, observed, expected, source, tol=None):
    if tol is not None:
        ok = abs(float(observed) - float(expected)) <= tol
    else:
        ok = observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(ok),
                   "source": str(source)})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(k):
    return json.load(open(SRC[k], encoding="utf-8"))


def counts(oof_key):
    d = pd.read_csv(SRC[oof_key]).drop_duplicates("public_id")
    return d, len(d), int(d.event.sum())


def verify_manifests():
    for folder in (SI, SR):
        man = json.load(open(folder / "output_manifest.json", encoding="utf-8"))["files"]
        for k, p in SRC.items():
            if p.parent == folder and p.name in man:
                check(f"manifest hash {folder.name}/{p.name}", sha(p), man[p.name], folder / "output_manifest.json")
    check("survival_incremental DONE", jl("si_done").get("status"), "complete", SRC["si_done"])
    check("richer baseline DONE status present", "status" in jl("sr_done"), True, SRC["sr_done"])
    check("richer baseline gate passed", jl("sr_gate").get("passed"), True, SRC["sr_gate"])
    check("richer baseline uses the locked spec", jl("sr_final")["locked_spec_sha256"],
          jl("si_final")["spec_sha256"], SRC["sr_final"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_manifests()
    fin, srf = jl("si_final"), jl("sr_final")

    # ---- cohorts ----
    pfs_ids, n_pfs, e_pfs = counts("si_pfs_oof")
    os_ids, n_os, e_os = counts("si_os_oof")
    sen_ids, n_sen, e_sen = counts("si_sens_oof")
    rp_ids, n_rp, e_rp = counts("sr_pfs_oof")
    ro_ids, n_ro, e_ro = counts("sr_os_oof")
    check("primary cohort patients / PFS events", (n_pfs, e_pfs), (674, 459), SRC["si_pfs_oof"])
    check("primary cohort OS deaths", (n_os, e_os), (674, 247), SRC["si_os_oof"])
    check("timing sensitivity patients / PFS events", (n_sen, e_sen), (624, 423), SRC["si_sens_oof"])
    check("expanded-baseline PFS patients / events", (n_rp, e_rp), (674, 459), SRC["sr_pfs_oof"])
    check("expanded-baseline OS patients / deaths", (n_ro, e_ro), (674, 247), SRC["sr_os_oof"])
    P = set(pfs_ids.public_id)
    check("PFS and OS use the same 674 patients", set(os_ids.public_id) == P, True, SRC["si_os_oof"])
    check("expanded baseline uses the same patients", set(rp_ids.public_id) == P == set(ro_ids.public_id),
          True, SRC["sr_pfs_oof"])
    check("timing cohort is a subset of the primary cohort", set(sen_ids.public_id) <= P, True, SRC["si_sens_oof"])
    check("same event indicators in primary and expanded PFS",
          bool((pfs_ids.set_index("public_id").event.sort_index()
                == rp_ids.set_index("public_id").event.sort_index()).all()), True, SRC["sr_pfs_oof"])

    # cohort definition from locked visit-pair data: one pre-treatment RNA day per patient
    vp = pd.read_csv(SRC["pairs"], usecols=["public_id", "vt_days_to_visit", "days_since_rna_sample"],
                     low_memory=False).dropna()
    vp["rna_day"] = vp.vt_days_to_visit - vp.days_since_rna_sample
    pre = vp[vp.rna_day <= 0].groupby("public_id").rna_day.agg(["nunique", "first"])
    check("survival cohort = patients with RNA sample day <= 0", set(pre.index) == P, True, SRC["pairs"])
    check("one pre-treatment RNA day per patient", bool((pre["nunique"] == 1).all()), True, SRC["pairs"])
    win = set(pre[(pre["first"] >= -30) & (pre["first"] <= 0)].index)
    check("timing cohort = RNA day in [-30, 0]", win == set(sen_ids.public_id), True, SRC["pairs"])
    subj = pd.read_csv(SRC["subject"], usecols=["public_id", "index_date"])
    check("index_date = first_treatment for all survival patients",
          set(subj[subj.public_id.isin(P)].index_date.unique()), {"first_treatment"}, SRC["subject"])
    check("index_date = first_treatment for every subject", set(subj.index_date.unique()), {"first_treatment"},
          SRC["subject"])

    # ---- Panel A ----
    rep_rows, a_rows = [], []
    for ep, key, rep_key, n, e, ev in [("PFS", "primary_pfs", "si_pfs_rep", n_pfs, e_pfs, "events"),
                                       ("OS", "secondary_os", "si_os_rep", n_os, e_os, "deaths")]:
        s = fin[key]
        rep = pd.read_csv(SRC[rep_key])
        check(f"A {ep} 5 repeats", len(rep), 5, SRC[rep_key])
        check(f"A {ep} baseline C = mean of repeats", rep.c_C0.mean(), s["c_C0"], SRC[rep_key], 1e-12)
        check(f"A {ep} + RNA C = mean of repeats", rep.c_C0_RNA.mean(), s["c_C0_RNA"], SRC[rep_key], 1e-12)
        check(f"A {ep} delta C = mean of repeat deltas", rep.delta_c.mean(), s["delta_c"], SRC[rep_key], 1e-12)
        b = s["bootstrap_delta_c"]
        check(f"A {ep} CI brackets delta", b["ci_lower"] < s["delta_c"] < b["ci_upper"], True, SRC["si_final"])
        check(f"A {ep} n_boot", b["n_boot"], 2000, SRC["si_final"])
        for _, r in rep.iterrows():
            for model, col in [("Clinical baseline", "c_C0"), ("Clinical baseline + RNA", "c_C0_RNA")]:
                rep_rows.append({"panel": "A", "endpoint": ep, "repeat": int(r["repeat"]), "model": model,
                                 "c_index": r[col], "role": "background: per-repeat value (not a CI)"})
        perm = s.get("permutation_delta_c")
        a_rows.append({
            "panel": "A", "endpoint": ep, "endpoint_label": "Progression-free survival" if ep == "PFS" else "Overall survival",
            "n_patients": n, "n_events": e, "event_type": ev,
            "c_index_clinical_baseline": s["c_C0"], "c_index_baseline_plus_rna": s["c_C0_RNA"],
            "c_index_ci": "not available (only Delta C was bootstrapped)",
            "repeat_min_baseline": rep.c_C0.min(), "repeat_max_baseline": rep.c_C0.max(),
            "repeat_min_plus_rna": rep.c_C0_RNA.min(), "repeat_max_plus_rna": rep.c_C0_RNA.max(),
            "delta_c": s["delta_c"], "delta_c_ci_lower": b["ci_lower"], "delta_c_ci_upper": b["ci_upper"],
            "delta_c_repeat_min": rep.delta_c.min(), "delta_c_repeat_max": rep.delta_c.max(),
            "ci_method": CI_METHOD, "bootstrap_seed": b["seed"],
            "permutation_p_one_sided": perm["one_sided_p"] if perm else np.nan,
            "permutation_detail": (f"{perm['k_null_ge_observed']} of {perm['B']} permuted Delta C >= observed; "
                                   f"smallest attainable p = 1/{perm['B'] + 1}") if perm else
                                  "not performed (specification: permutation for primary PFS only)",
            "comparison": "Delta C = C(clinical baseline + RNA) - C(clinical baseline); positive = RNA improves",
            "evaluation": EVAL, "baseline_model": BASELINE, "rna_addition": RNA_ADD,
            "evidence_level": "post-hoc exploratory; internal cross-validation; not external validation",
        })
        if perm:
            check(f"A {ep} permutation p = (k+1)/(B+1)", perm["one_sided_p"],
                  (perm["k_null_ge_observed"] + 1) / (perm["B"] + 1), SRC["si_final"], 1e-12)
    a = pd.DataFrame(a_rows)
    a.to_csv(OUT / "fig4A_cindex_summary.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT / "fig4A_cindex_by_repeat.csv", index=False)
    check("A PFS delta C", float(a.loc[0, "delta_c"]), 0.04049462663357797, SRC["si_final"], 1e-15)
    check("A OS delta C CI upper", float(a.loc[1, "delta_c_ci_upper"]), 0.07449073673410617, SRC["si_final"], 1e-15)
    check("A PFS permutation p", float(a.loc[0, "permutation_p_one_sided"]), 1 / 201, SRC["si_final"], 1e-12)
    check("A OS has no permutation test", bool(np.isnan(a.loc[1, "permutation_p_one_sided"])), True, SRC["si_final"])

    # ---- Panel B ----
    def brow(ep, analysis, label, baseline, cohort, n, e, s, c0k, crk, src, rep_key, role):
        b = s["bootstrap_delta_c"]
        rep = pd.read_csv(SRC[rep_key])
        dcol = "delta_c"
        check(f"B {ep} {analysis} delta = mean of repeats", rep[dcol].mean(), s["delta_c"], SRC[rep_key], 1e-12)
        return {"panel": "B", "endpoint": ep, "analysis": analysis, "row_label": label, "role": role,
                "evaluated": True, "baseline_model": baseline, "cohort": cohort, "n_patients": n, "n_events": e,
                "c_index_baseline": s[c0k], "c_index_plus_rna": s[crk], "delta_c": s["delta_c"],
                "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"], "ci_method": CI_METHOD,
                "permutation_p_one_sided": s.get("permutation_delta_c", {}).get("one_sided_p", np.nan),
                "hypothesis_test": "permutation (primary PFS only)" if analysis == "primary" and ep == "PFS"
                else "none performed", "source": str(src)}

    full = "Survival cohort: one pre-treatment RNA sample (RNA day <= 0), n = 674"
    tim = "RNA day -30 to 0 (RNA within 30 days before first treatment), n = 624"
    rows = [
        brow("PFS", "primary", "Primary", BASELINE, full, n_pfs, e_pfs, fin["primary_pfs"], "c_C0", "c_C0_RNA",
             SRC["si_final"], "si_pfs_rep", "primary"),
        brow("PFS", "expanded_baseline", "Expanded clinical baseline", EXPANDED, full, n_rp, e_rp,
             srf["primary_pfs_richer"], "c_C0_richer", "c_C0_richer_RNA", SRC["sr_final"], "sr_pfs_rep", "sensitivity"),
        brow("PFS", "rna_timing_m30_0", "RNA timing restricted (day −30 to 0)", BASELINE, tim, n_sen, e_sen,
             fin["sensitivity_pfs"], "c_C0", "c_C0_RNA", SRC["si_final"], "si_sens_rep", "sensitivity"),
        brow("OS", "primary", "Primary", BASELINE, full, n_os, e_os, fin["secondary_os"], "c_C0", "c_C0_RNA",
             SRC["si_final"], "si_os_rep", "primary"),
        brow("OS", "expanded_baseline", "Expanded clinical baseline", EXPANDED, full, n_ro, e_ro,
             srf["secondary_os_richer"], "c_C0_richer", "c_C0_richer_RNA", SRC["sr_final"], "sr_os_rep", "sensitivity"),
        {"panel": "B", "endpoint": "OS", "analysis": "rna_timing_m30_0", "row_label": "RNA timing restricted (day −30 to 0)",
         "role": "sensitivity", "evaluated": False, "baseline_model": BASELINE, "cohort": tim,
         "hypothesis_test": "none", "source": str(SRC["si_spec"]),
         "not_evaluated_reason": "not run: the frozen specification defines the timing sensitivity for PFS only"},
    ]
    bdf = pd.DataFrame(rows)
    bdf["comparison"] = "Delta C = C(baseline + RNA) - C(baseline); positive = RNA improves discrimination"
    bdf["evaluation"] = EVAL
    bdf["rna_addition"] = RNA_ADD
    bdf["comparability_note"] = ("Rows differ in comparator (expanded baseline) or cohort (timing); no test "
                                 "between rows was performed")
    bdf.to_csv(OUT / "fig4B_delta_c_forest.csv", index=False)
    spec = jl("si_spec")["spec"]
    check("spec: timing sensitivity defined", spec["timing_sensitivity"], "inferred RNA day in [-30, 0]", SRC["si_spec"])
    check("final summary has no OS timing result", "sensitivity_os" not in fin, True, SRC["si_final"])
    check("expanded baseline: no permutation test", "permutation test" in jl("sr_prespec")["not_done"], True, SRC["sr_prespec"])
    exp_b = {("PFS", "expanded_baseline"): (0.035005165674360114, 0.014404939020709072, 0.05583149996539829),
             ("PFS", "rna_timing_m30_0"): (0.02789262630916951, 0.0027834175306825416, 0.0530004547899617),
             ("OS", "expanded_baseline"): (0.04867700338261301, 0.021710373146108842, 0.07584158614007878)}
    for (ep, an), (dv, lo, hi) in exp_b.items():
        r = bdf[(bdf.endpoint == ep) & (bdf.analysis == an)].iloc[0]
        check(f"B {ep} {an} delta / CI", (round(r.delta_c, 12), round(r.ci_lower, 12), round(r.ci_upper, 12)),
              (round(dv, 12), round(lo, 12), round(hi, 12)), r.source)
    check("B timing baseline C differs from primary", round(fin["sensitivity_pfs"]["c_C0"], 4), 0.6185, SRC["si_final"])

    # ---- IBS (caption / supplement only) ----
    ibs = []
    for ep, key, an, n, e in [("PFS", "primary_pfs", "primary", n_pfs, e_pfs), ("OS", "secondary_os", "primary", n_os, e_os),
                              ("PFS", "sensitivity_pfs", "rna_timing_m30_0", n_sen, e_sen)]:
        s = fin[key]
        ibs.append({"endpoint": ep, "analysis": an, "n_patients": n, "n_events": e, "ibs_clinical_baseline": s["ibs_C0"],
                    "ibs_baseline_plus_rna": s["ibs_C0_RNA"], "delta_ibs": s["delta_ibs"],
                    "metric": f"{spec['secondary_metric']}; lower is better; negative delta = RNA improves",
                    "uncertainty": "point estimate only (no CI computed)"})
    ibs.append({"endpoint": "PFS/OS", "analysis": "expanded_baseline", "uncertainty": "IBS not computed for this analysis"})
    pd.DataFrame(ibs).to_csv(OUT / "fig4_ibs_caption_only.csv", index=False)

    defs = {
        "evidence_level": "post-hoc exploratory; internal cross-validation within the survival cohort; NOT external validation",
        "day0": DAY0,
        "day0_source": ("MMRF data dictionary v25: 'each patient entered at the time of first treatment ... "
                        "\"first_treatment\", recorded as the index_date'; subject_deid.csv index_date checked"),
        "endpoints": {"PFS": "official MMRF-supplied progression-free survival (primary endpoint)",
                      "OS": "official MMRF-supplied overall survival (secondary endpoint)"},
        "models": {"clinical_baseline": BASELINE, "expanded_baseline": EXPANDED, "rna_addition": RNA_ADD,
                   "fitting": spec["model"] + "; " + spec["inner_cv"], "preprocessing": spec["preprocessing"]},
        "evaluation": EVAL, "ci_method": CI_METHOD,
        "permutation": spec["permutation"] + " (performed for primary PFS only)",
        "not_available": ["CIs for absolute C-index values", "OS permutation test", "OS RNA-timing sensitivity",
                          "permutation test / IBS for the expanded-baseline sensitivity",
                          "cytogenetic (FISH) adjustment: " + jl("sr_cyto")["conclusion"]],
        "expanded_baseline_exclusions": srf["excluded"],
        "overlap_with_next_visit_test_set": ("the 674 survival patients include patients from all three next-visit "
                                             "splits (train 473, validation 101, test 100; figures/data/Fig1)"),
        "terminology": {"C-index": "Harrell's concordance index; higher is better; 0.5 = chance",
                        "Delta C": "C(baseline + RNA) - C(baseline)",
                        "IBS": "integrated Brier score (IPCW, 365-1,095 days); lower is better"},
    }
    json.dump(defs, open(OUT / "fig4_definitions.json", "w"), indent=2)

    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "fig4_verification_checks.csv", index=False)
    outputs = sorted(p.name for p in OUT.iterdir() if p.name != "fig4_provenance.json")
    json.dump({"script": "figures/fig4_extract_data.py",
               "inputs_sha256": {str(p): sha(p) for p in SRC.values()},
               "outputs": outputs + ["fig4_provenance.json"]},
              open(OUT / "fig4_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
