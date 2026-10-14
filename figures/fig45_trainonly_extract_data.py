"""
Figures 4-5 and Table S5 (train-only survival redesign): data extraction and verification (read-only).

Run from repo root:
    python figures/fig45_trainonly_extract_data.py

Replaces the data source of figures/fig4_extract_data.py and figures/fig5_extract_data.py, which are left as they were (they read pooled
5x5-CV outputs and gitignored CSVs that are not available here). Everything is extracted from the frozen survival redesign
(explainability/SURVIVAL_TRAINONLY_PROTOCOL.md; Stage D / Stage H):

    artifacts/results_clean_rerun/survival_trainonly/
        heldout_report.json, heldout_manifest.json, heldout_results_table.csv, development_cv_and_figure5_refit_summary.json,
        development_gate_report.json, frozen_models.json, frozen_models.sha256,
        figure5_refit_pathway_stability_{pfs,os,both_endpoints}.csv, figure5_refit_fold_coefficients_{pfs,os}.csv
    explainability/SURVIVAL_TRAINONLY_PROTOCOL.{md,sha256}
    artifacts/results_clean_rerun/{survival_incremental,survival_richer_baseline_sensitivity}/final_summary.json   (earlier pooled results; reference rows only)
    artifacts/results_clean_rerun/direction2_replication_trainonly/selection_artifact.json  (Figure 3D family, for the Figure 5 note)

Nothing is refit. The script exits non-zero if any check fails. Writes figures/data/Fig45_trainonly/ (gitignored; aggregates only).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("artifacts/results_clean_rerun/survival_trainonly")
OUT = Path("figures/data/Fig45_trainonly")
PROTO = Path("explainability/SURVIVAL_TRAINONLY_PROTOCOL.md")
PROTO_SHA = Path("explainability/SURVIVAL_TRAINONLY_PROTOCOL.sha256")
EARLIER = Path("artifacts/results_clean_rerun")
DISPLAY_FIX = {"PI3K/AKT/mTOR  Signaling": "PI3K/AKT/mTOR Signaling", "Pperoxisome": "Peroxisome", "heme Metabolism": "Heme Metabolism"}
checks = []


def check(name, observed, expected, source, tol=None):
    ok = abs(float(observed) - float(expected)) <= tol if tol is not None else observed == expected
    checks.append({"check": name, "observed": str(observed)[:160], "expected": str(expected)[:160], "passed": bool(ok), "source": str(source)})


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(R / "heldout_report.json", encoding="utf-8"))
    fm = json.load(open(R / "frozen_models.json", encoding="utf-8"))
    man = json.load(open(R / "heldout_manifest.json", encoding="utf-8"))
    gate = json.load(open(R / "development_gate_report.json", encoding="utf-8"))
    cvs = json.load(open(R / "development_cv_and_figure5_refit_summary.json", encoding="utf-8"))
    res = rep["results"]

    # ---- provenance / hash chain ----
    for f_, h in man["files"].items():
        if (R / f_).exists():
            check(f"manifest hash {f_}", sha_lf(R / f_) == h, True, R / "heldout_manifest.json")
    rec = PROTO_SHA.read_text(encoding="utf-8").split()[0]
    check("protocol hash: file == recorded == frozen artifact == report", sha_lf(PROTO) == rec == fm["protocol_sha256"] == rep["protocol_sha256"], True, PROTO)
    fz = (R / "frozen_models.sha256").read_text(encoding="utf-8").split()[0]
    check("frozen artifact hash: file == recorded == report == gate report", sha_lf(R / "frozen_models.json") == fz == rep["frozen_models_sha256"] == gate["frozen_models_sha256"], True, R)
    check("Stage D gates all passed", all(g["passed"] for g in gate["gates"]), True, R)
    check("Stage H checks all passed", all(c["passed"] for c in rep["checks"]), True, R / "heldout_report.json")
    check("held-out set: 201 patients, 149 PFS events, 82 deaths", (rep["heldout"]["patients"], rep["heldout"]["pfs_events"], rep["heldout"]["deaths"]), (201, 149, 82), R)

    # ---- Figure 4A / 4B / Table S5 ----
    def row(rid, ep, an, label, evaluation, r, perm=None, evaluated=True):
        if not evaluated:
            return {"row_id": rid, "endpoint": ep, "analysis": an, "label": label, "evaluation": evaluation, "evaluated": False}
        b = r["bootstrap_delta_c"]
        return {"row_id": rid, "endpoint": ep, "analysis": an, "label": label, "evaluation": evaluation, "evaluated": True,
                "n_patients": r["n_patients"], "n_events": r["n_events"], "c_index_baseline": r["c_C0"], "c_index_plus_rna": r["c_C0_RNA"],
                "delta_c": r["delta_c"], "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"], "permutation_p_one_sided": perm,
                "ibs_baseline": r.get("ibs_C0"), "ibs_plus_rna": r.get("ibs_C0_RNA"), "lambda_baseline": r.get("lambda_C0"), "lambda_plus_rna": r.get("lambda_C0_RNA")}

    ho = "held-out (validation + test)"
    rows = [row("A1", "PFS", "primary", "Primary", ho, res["A1_pfs"], res["A1_pfs"]["permutation"]["one_sided_p"]),
            row("S2_pfs", "PFS", "expanded_baseline", "Expanded clinical baseline", ho, res["S2_pfs_rich"]),
            row("S1", "PFS", "rna_timing_m30_0", "RNA day -30 to 0 only", ho, res["S1_pfs_m30_0"]),
            row("S4_val", "PFS", "validation_only", "Validation patients only (descriptive)", ho + " partition", res["S4_partitions_A1"]["val"]),
            row("S4_test", "PFS", "test_only", "Test patients only (descriptive)", ho + " partition", res["S4_partitions_A1"]["test"]),
            row("A2", "OS", "primary", "Primary", ho, res["A2_os"]),
            row("S2_os", "OS", "expanded_baseline", "Expanded clinical baseline", ho, res["S2_os_rich"]),
            row("S1_os", "OS", "rna_timing_m30_0", "RNA day -30 to 0 only", "not run", None, evaluated=False)]
    for ep in ("pfs", "os"):
        c = cvs[ep]
        rows.append(row(f"S3_{ep}", ep.upper(), "development_only_cv", "Development-only 5x5 CV (reference)", "train patients, out-of-fold",
                        {"n_patients": c["n_patients"], "n_events": c["n_events"], "c_C0": c["c_C0"], "c_C0_RNA": c["c_C0_RNA"], "delta_c": c["delta_c"],
                         "bootstrap_delta_c": c["bootstrap_delta_c"]}))
    pooled = json.load(open(EARLIER / "survival_incremental/final_summary.json", encoding="utf-8"))
    rich = json.load(open(EARLIER / "survival_richer_baseline_sensitivity/final_summary.json", encoding="utf-8"))
    for ep, key in (("PFS", "primary_pfs"), ("OS", "secondary_os")):
        p = pooled[key]
        rows.append(row(f"P_{ep.lower()}", ep, "earlier_pooled_cv", "Earlier pooled 5x5 CV (all 674 patients)", "pooled; held-out patients took part in fitting",
                        {"n_patients": 674, "n_events": 459 if ep == "PFS" else 247, "c_C0": p["c_C0"], "c_C0_RNA": p["c_C0_RNA"], "delta_c": p["delta_c"],
                         "bootstrap_delta_c": p["bootstrap_delta_c"]}, perm=p.get("permutation_delta_c", {}).get("one_sided_p") if ep == "PFS" else None))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "fig4B_trainonly_delta_c.csv", index=False)
    a1 = res["A1_pfs"]
    check("A1 criterion not met: delta C > 0, CI lower < 0, permutation p <= 0.025", (a1["criterion"]["delta_c_gt_0"], a1["criterion"]["bootstrap_ci_lower_gt_0"],
          a1["criterion"]["permutation_one_sided_p_le_0.025"], a1["criterion"]["met"]), (True, False, True, False), R / "heldout_report.json")
    check("A1 permutation p = (1 + k)/(B + 1)", (1 + a1["permutation"]["k_null_ge_observed"]) / (a1["permutation"]["B"] + 1), a1["permutation"]["one_sided_p"], R, 1e-15)
    for r_ in T[T.evaluated].itertuples():
        check(f"{r_.row_id}: CI brackets delta C and delta C = C(+RNA) - C(base)", bool(r_.ci_lower < r_.delta_c < r_.ci_upper) and abs(r_.c_index_plus_rna - r_.c_index_baseline - r_.delta_c) < 1e-12,
              True, R)
    for key, lam in (("A1_pfs", (100.0, 1.0)), ("A2_os", (100.0, 1.0))):
        check(f"{key} frozen lambdas (baseline, +RNA)", (res[key]["lambda_C0"], res[key]["lambda_C0_RNA"]), lam, R / "frozen_models.json")
    check("pooled earlier PFS delta C reproduced from its locked summary", round(pooled["primary_pfs"]["delta_c"], 4), 0.0405, EARLIER / "survival_incremental", 1e-12)

    A = pd.DataFrame([{"endpoint": "PFS" if k == "A1" else "OS", "n_patients": res["A1_pfs" if k == "A1" else "A2_os"]["n_patients"],
                       "n_events": res["A1_pfs" if k == "A1" else "A2_os"]["n_events"],
                       "event_type": "PFS events" if k == "A1" else "deaths", "c_index_baseline": res["A1_pfs" if k == "A1" else "A2_os"]["c_C0"],
                       "c_index_plus_rna": res["A1_pfs" if k == "A1" else "A2_os"]["c_C0_RNA"], "delta_c": res["A1_pfs" if k == "A1" else "A2_os"]["delta_c"],
                       "ci_lower": res["A1_pfs" if k == "A1" else "A2_os"]["bootstrap_delta_c"]["ci_lower"],
                       "ci_upper": res["A1_pfs" if k == "A1" else "A2_os"]["bootstrap_delta_c"]["ci_upper"],
                       "permutation_p_one_sided": a1["permutation"]["one_sided_p"] if k == "A1" else np.nan} for k in ("A1", "A2")])
    A.to_csv(OUT / "fig4A_trainonly_cindex_summary.csv", index=False)
    part = pd.DataFrame([{"partition": p, "endpoint": "PFS", **{k: v[k] for k in ("n_patients", "n_events", "c_C0", "c_C0_RNA", "delta_c")}}
                         for p, v in res["S4_partitions_A1"].items()])
    part.to_csv(OUT / "fig4A_trainonly_partition_descriptive.csv", index=False)

    # ---- Figure 5 ----
    st = {ep: pd.read_csv(R / f"figure5_refit_pathway_stability_{ep}.csv") for ep in ("pfs", "os")}
    both = pd.read_csv(R / "figure5_refit_pathway_stability_both_endpoints.csv")
    folds = {ep: pd.read_csv(R / f"figure5_refit_fold_coefficients_{ep}.csv") for ep in ("pfs", "os")}
    paths = fm["pathways"]
    for ep in ("pfs", "os"):
        f_ = folds[ep]
        check(f"[{ep}] 25 development-CV fits (5 repeats x 5 folds)", (len(f_), f_.repeat.nunique(), f_.fold.nunique()), (25, 5, 5), R)
        cf = f_[paths]
        s = st[ep].set_index("pathway")
        check(f"[{ep}] stored mean = mean of the fold coefficients", float((cf.mean() - s.loc[paths, "beta_mean"]).abs().max()), 0.0, R, 1e-12)
        check(f"[{ep}] stored min / max = fold min / max", float(max((cf.min() - s.loc[paths, "beta_min"]).abs().max(), (cf.max() - s.loc[paths, "beta_max"]).abs().max())), 0.0, R, 1e-12)
        lab = pd.Series(np.where((cf > 0).all(), "consistently_positive", np.where((cf < 0).all(), "consistently_negative", "mixed")), index=cf.columns)
        check(f"[{ep}] direction labels follow the 25/25 sign rule", bool((lab == s.loc[paths, "direction_label"]).all()), True, R)
        check(f"[{ep}] no zero coefficients", int((cf == 0).sum().sum()), 0, R)
    conc = ((both.pfs_direction_label == both.os_direction_label) & (both.pfs_direction_label != "mixed"))
    check("concordance flag = same non-mixed direction in both endpoints", bool((conc == both.consistent_same_direction_both_endpoints).all()), True, R)
    check("12 concordant pathways (earlier pooled design: 19)", int(conc.sum()), 12, R)
    hi, lo = int((conc & (both.pfs_beta_mean > 0)).sum()), int((conc & (both.pfs_beta_mean < 0)).sum())
    check("concordant split higher / lower hazard (9 positive, 3 negative; counted from the concordant list)", (hi, lo), (9, 3), R)
    fam = json.load(open(EARLIER / "direction2_replication_trainonly/selection_artifact.json", encoding="utf-8"))["family"]
    in_c = {p: bool(both.set_index("pathway").loc[p, "consistent_same_direction_both_endpoints"]) for p in fam}
    check("Figure 3D family members concordant here: E2F, G2-M, Myc V1, Mitotic Spindle yes; Spermatogenesis no",
          in_c, {"E2F Targets": True, "G2-M Checkpoint": True, "Myc Targets V1": True, "Mitotic Spindle": True, "Spermatogenesis": False}, R)
    r_ = float(np.corrcoef(both.pfs_beta_mean, both.os_beta_mean)[0, 1])
    A5 = pd.DataFrame({"pathway_source_name": both.pathway, "pathway_display_name": both.pathway.map(lambda s: DISPLAY_FIX.get(s, s)),
                       "pfs_beta_mean": both.pfs_beta_mean, "os_beta_mean": both.os_beta_mean, "pfs_direction_label": both.pfs_direction_label,
                       "os_direction_label": both.os_direction_label, "concordant_both_endpoints": conc,
                       "concordance_group": np.where(~conc, "not concordant", np.where(both.pfs_beta_mean > 0, "higher hazard in both", "lower hazard in both")),
                       "figure3D_family_member": both.pathway.isin(fam)})
    A5.to_csv(OUT / "fig5A_trainonly_coefficients.csv", index=False)
    rows5 = []
    for _, x in both[conc].iterrows():
        for ep, labl in (("pfs", "PFS"), ("os", "OS")):
            rows5.append({"pathway_source_name": x.pathway, "pathway_display_name": DISPLAY_FIX.get(x.pathway, x.pathway),
                          "concordance_group": "higher hazard in both" if x.pfs_beta_mean > 0 else "lower hazard in both", "endpoint": labl,
                          "beta_mean": x[f"{ep}_beta_mean"], "beta_min_across_fits": x[f"{ep}_beta_min"], "beta_max_across_fits": x[f"{ep}_beta_max"],
                          "HR_of_mean_beta": x[f"{ep}_HR_of_mean_beta"], "n_fits": int(x[f"{ep}_n_fits"]),
                          "range_definition": "min-max across 25 overlapping development-CV fits; NOT a confidence interval"})
    B5 = pd.DataFrame(rows5)
    B5.to_csv(OUT / "fig5B_trainonly_concordant.csv", index=False)
    check("B: 24 rows (12 pathways x 2 endpoints), ranges bracket means and exclude 0", bool(len(B5) == 24 and ((B5.beta_min_across_fits <= B5.beta_mean) & (B5.beta_mean <= B5.beta_max_across_fits)).all()
          and ((B5.beta_min_across_fits > 0) | (B5.beta_max_across_fits < 0)).all()), True, R)
    stats = {"pearson_r_mean_beta_pfs_vs_os": r_, "n_concordant": int(conc.sum()), "n_concordant_higher_hazard": hi, "n_concordant_lower_hazard": lo,
             "direction_counts": cvs["figure5_refit"]["direction_counts"], "figure3D_family_concordant": in_c,
             "hr_per_sd_range": {ep: [float(np.exp(B5[B5.endpoint == ep.upper()].beta_mean.min())), float(np.exp(B5[B5.endpoint == ep.upper()].beta_mean.max()))] for ep in ("pfs", "os")},
             "note": "25 development-CV fits within the 473 train patients; fits share about 80% of their training patients; descriptive; no hypothesis tests"}
    json.dump(stats, open(OUT / "fig5_trainonly_summary_stats.json", "w"), indent=2)
    check("direction counts equal the Stage D summary", {ep: {k: int((st[ep].direction_label == k).sum()) for k in ("consistently_positive", "consistently_negative", "mixed")} for ep in ("pfs", "os")},
          cvs["figure5_refit"]["direction_counts"], R)
    defs = {"evaluation": "A1/A2/S1/S2/S4: frozen models fitted on the 473 train patients, scored once on the 201 held-out validation + test patients",
            "evidence_level": "post hoc, exploratory; the earlier pooled results had been seen; not independent validation; not external validation",
            "primary_criterion": "delta C > 0 AND 95% bootstrap CI lower bound > 0 AND one-sided permutation p <= 0.025 (pre-specified; not met)",
            "permutation": "RNA vectors permuted among held-out patients, frozen RNA model re-scored, B = 10,000; the null is centred below 0 because scrambled RNA adds noise to the frozen model",
            "earlier_pooled_rows": "5x5 CV over all 674 patients: validation and test patients trained the fold models; shown as reference only",
            "figure5": "coefficients of the 'clinical baseline + RNA' ridge Cox model in 25 development-CV fits (train patients only), lambda = nested-CV value per fit"}
    json.dump(defs, open(OUT / "fig45_trainonly_definitions.json", "w"), indent=2)
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "fig45_trainonly_verification_checks.csv", index=False)
    json.dump({"script": "figures/fig45_trainonly_extract_data.py", "outputs": sorted(p.name for p in OUT.iterdir() if p.name != "fig45_trainonly_provenance.json")},
              open(OUT / "fig45_trainonly_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed; Pearson r = {r_:.3f}")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
