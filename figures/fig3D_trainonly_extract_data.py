"""
Figure 3D (train-selected pathway composite) data extraction and verification (read-only).

Run from repo root:
    python figures/fig3D_trainonly_extract_data.py

Replaces the data source of panel D of figures/fig3_extract_data.py, which is left unchanged (its pooled stage-1 selection and
train+validation discovery are superseded; they cannot be re-run on this machine without gitignored panel A/B inputs). Panel D
is self-contained, so it is extracted here from the train-only redesign outputs only:

    artifacts/results_clean_rerun/direction2_replication_trainonly/
        S3_trainonly_regression_results.csv, S3_trainonly_row_accounting.json, selection_artifact.json, selection_manifest.json,
        discovery_report.json, frozen_composite_artifact.json, discovery_composite_loadings.csv, heldout_report.json,
        run_summary.json, output_manifest.json
    artifacts/results_clean_rerun/direction2/association/global_tests.csv   (primary matched-pair test; pooled over all patients)
    explainability/PATHWAY_TRAINONLY_PROTOCOL.md

Writes figures/data/Fig3D_trainonly/ (gitignored with the rest of figures/data; aggregates only). Nothing is refit. Every value is
checked against its source and the script exits non-zero if any check fails.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RC = Path("artifacts/results_clean_rerun")
D = RC / "direction2_replication_trainonly"
OUT = Path("figures/data/Fig3D_trainonly")
SRC = {
    "s3": D / "S3_trainonly_regression_results.csv", "s3_acc": D / "S3_trainonly_row_accounting.json",
    "sel": D / "selection_artifact.json", "sel_man": D / "selection_manifest.json", "disc": D / "discovery_report.json",
    "frozen": D / "frozen_composite_artifact.json", "load": D / "discovery_composite_loadings.csv",
    "held": D / "heldout_report.json", "run": D / "run_summary.json", "man": D / "output_manifest.json",
    "glob": RC / "direction2/association/global_tests.csv", "protocol": Path("explainability/PATHWAY_TRAINONLY_PROTOCOL.md"),
}
checks = []


def check(name, observed, expected, source, tol=None):
    ok = abs(float(observed) - float(expected)) <= tol if tol is not None else observed == expected
    checks.append({"check": name, "observed": str(observed)[:200], "expected": str(expected)[:200], "passed": bool(ok),
                   "source": str(source)})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def jl(k):
    return json.load(open(SRC[k], encoding="utf-8"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel, disc, held, frozen = jl("sel"), jl("disc"), jl("held"), jl("frozen")
    s3, load, glob = pd.read_csv(SRC["s3"]), pd.read_csv(SRC["load"]), pd.read_csv(SRC["glob"])
    fam = sel["family"]

    # ---- provenance / hash chain (line endings normalised: git on Windows may convert them) ----
    man = jl("man")["files"]
    for f, h in man.items():
        p = D / f
        if p.exists():
            check(f"manifest hash {f}", sha(p) == h or sha_lf(p) == h, True, SRC["man"])
    check("selection artifact sha = Stage-1 manifest", sha(SRC["sel"]) == jl("sel_man")["selection_artifact_sha256"]
          or sha_lf(SRC["sel"]) == jl("sel_man")["selection_artifact_sha256"], True, SRC["sel_man"])
    check("frozen artifact records the selection artifact", frozen["selection_artifact_sha256"],
          jl("sel_man")["selection_artifact_sha256"], SRC["frozen"])
    check("discovery report records the same artifacts", (disc["selection_artifact_sha256"], disc["frozen_artifact_sha256"]),
          (frozen["selection_artifact_sha256"], jl("man")["frozen_artifact_sha256"]), SRC["disc"])
    check("protocol hash recorded = current protocol file", sel["protocol_sha256"] in (sha(SRC["protocol"]), sha_lf(SRC["protocol"])),
          True, SRC["protocol"])

    # ---- selection ----
    check("five pathways selected, none stopped", (len(fam), sel["stop_reason"]), (5, None), SRC["sel"])
    s3r = s3.set_index("pathway")
    check("S3 train-only: 50 pathways, none failed", (len(s3), int(s3.status.str.startswith("failed").sum())), (50, 0), SRC["s3"])
    top5 = list(s3.sort_values("p_wald_robust").pathway[:5])
    check("family = five smallest p-values of the train-only S3 regression (rule unchanged)", top5, fam, SRC["s3"])
    check("selection used train patients only", sel["development_splits"], ["train"], SRC["sel"])
    acc = jl("s3_acc")
    check("S3 train-only analysis complete", acc["analysis_status"], "complete", SRC["s3_acc"])
    check("S3 rows / patients (frozen exclusions applied to train rows)", (int(s3.n_rows.iloc[0]), int(s3.n_patients.iloc[0])),
          (acc["final_rows"], acc["final_patients"]), SRC["s3"])

    outcome = ("next-visit IMWG category on the ordered scale PD < SD < PR < VGPR < CR < sCR; "
               "cumulative-logit ordinal GEE, patient-clustered robust SEs")
    rows = []
    for rk, pw in enumerate(fam, 1):
        r = s3r.loc[pw]
        rows.append({"panel": "D", "evidence_stage": "1_selection_evidence", "estimate_label": f"{pw} (single pathway)",
                     "selection_rank": rk,
                     "stage_description": ("Covariate-adjusted single-pathway association in train patients only (frozen Direction 2 S3 "
                                           "ordinal formulation, z-scores from train RNA samples only); the five smallest p-values "
                                           "defined the family."),
                     "patients_included": "train patients only (original training partition)", "includes_validation_or_test_patients": False,
                     "adjustment_model": ("S3: 16 continuous Model-C covariates + RNA age + strata "
                                          "(V(t) response x treatment line x PI/IMiD/CD38)"),
                     "exposure": f"{pw} score, per SD", "outcome_model": outcome,
                     "n_visit_pairs": int(r.n_rows), "n_patients": int(r.n_patients), "beta": r.estimate,
                     "ci_lower": r.ci_lo, "ci_upper": r.ci_hi, "odds_ratio": r.odds_ratio_per_SD,
                     "or_ci_lower": np.exp(r.ci_lo), "or_ci_upper": np.exp(r.ci_hi), "p_value": r.p_wald_robust,
                     "p_method": "two-sided robust Wald", "adjusted_p": r.q_bh,
                     "adjustment": "Benjamini-Hochberg q across 50 pathways (train-only S3)",
                     "role": "selection evidence (same patients as stage 2)"})
    de = disc["discovery_estimate"]
    rows.append({"panel": "D", "evidence_stage": "2_discovery", "estimate_label": "Train-selected pathway composite: discovery",
                 "selection_rank": np.nan,
                 "stage_description": ("Composite fixed outcome-blind on the same train patients (first principal component of the five "
                                       "selected pathway scores); association estimated in those patients, so it is optimistic "
                                       "after selection; gate required beta < 0 with one-sided p < 0.025."),
                 "patients_included": "train patients only (original training partition)", "includes_validation_or_test_patients": False,
                 "adjustment_model": ("full covariate model: composite + 16 continuous Model-C covariates + log1p(RNA age) + exact strata "
                                      "(V(t) response x line group x PI/IMiD/CD38)"),
                 "exposure": "composite score, per discovery SD", "outcome_model": outcome,
                 "n_visit_pairs": int(disc["accounting"]["discovery_rows"]), "n_patients": int(disc["accounting"]["discovery_patients"]),
                 "beta": de["beta_disc"], "ci_lower": de["ci95_disc"][0], "ci_upper": de["ci95_disc"][1],
                 "odds_ratio": np.exp(de["beta_disc"]), "or_ci_lower": np.exp(de["ci95_disc"][0]),
                 "or_ci_upper": np.exp(de["ci95_disc"][1]), "p_value": de["one_sided_p_disc"],
                 "p_method": "one-sided robust Wald (H1: beta < 0)", "adjusted_p": np.nan,
                 "adjustment": "none (single gate test)", "role": "discovery (post hoc)"})
    hr = held["result"]
    rows.append({"panel": "D", "evidence_stage": "3_heldout_evaluation",
                 "estimate_label": "Train-selected pathway composite: held-out evaluation", "selection_rank": np.nan,
                 "stage_description": ("Frozen composite and frozen clinical linear predictor applied to held-out validation and test "
                                       "patients; one pre-specified directional test run once. Post hoc and exploratory: earlier "
                                       "pooled results had been seen; not independent validation."),
                 "patients_included": "held-out validation + test patients (not used in selection or discovery)",
                 "includes_validation_or_test_patients": True,
                 "adjustment_model": ("recalibration model: composite + frozen discovery clinical linear predictor (eta), thresholds "
                                      "re-estimated; NOT the same adjustment model as discovery"),
                 "exposure": "composite score, per discovery SD", "outcome_model": outcome,
                 "n_visit_pairs": int(held["accounting"]["heldout_rows"]), "n_patients": int(held["accounting"]["heldout_patients"]),
                 "beta": hr["beta_heldout"], "ci_lower": hr["ci95_wald"][0], "ci_upper": hr["ci95_wald"][1],
                 "odds_ratio": hr["odds_ratio_per_discovery_SD"], "or_ci_lower": np.exp(hr["ci95_wald"][0]),
                 "or_ci_upper": np.exp(hr["ci95_wald"][1]), "p_value": hr["one_sided_p_H1_beta_lt_0"],
                 "p_method": "one-sided robust Wald (H1: beta < 0)", "adjusted_p": np.nan,
                 "adjustment": "none (one primary test within this analysis)",
                 "bootstrap_ci_lower_descriptive": hr["bootstrap_ci95_descriptive"][0],
                 "bootstrap_ci_upper_descriptive": hr["bootstrap_ci95_descriptive"][1],
                 "recalibration_slope_eta": hr["recalibration_slope_b_eta"],
                 "role": "held-out evaluation; pre-specified rule (95% CI entirely < 0) met"})
    dd = pd.DataFrame(rows)
    dd["direction_note"] = "beta < 0 / odds ratio < 1: higher score associated with a worse next IMWG category"
    dd["comparability_note"] = ("Estimates come from different patients and different adjustment models; shown side by side, "
                                "not as directly comparable effect sizes.")
    dd.to_csv(OUT / "fig3D_trainonly_association.csv", index=False)

    # ---- loadings ----
    check("loadings pathways = family (same order)", list(load.pathway), fam, SRC["load"])
    check("all five loadings positive", bool((load.loading > 0).all()), True, SRC["load"])
    check("loadings file = frozen artifact loadings", float(np.max(np.abs(load.loading.to_numpy() - np.array(frozen["composite"]["loadings"])))),
          0.0, SRC["frozen"], 1e-9)
    check("frozen artifact family = selection family", frozen["family"], fam, SRC["frozen"])
    check("frozen artifact discovery splits", frozen["discovery_splits"], ["train"], SRC["frozen"])
    load = load.copy()
    load["variance_explained_by_pc1"] = disc["composite"]["variance_explained"]
    load["n_distinct_rna_samples_used"] = disc["composite"]["n_distinct_samples"]
    load.to_csv(OUT / "fig3D_trainonly_composite_loadings.csv", index=False)

    # ---- primary matched-pair test (pooled over ALL patients; shown for context) ----
    gt = glob[glob.analysis == "primary"].copy()
    gt["description"] = ("Direction 2 PRIMARY analysis: matched-pair global sign-flip test across 50 pathways (within RNA-discordant "
                         "matched patient pairs), pooled over all patients; null result")
    gt.to_csv(OUT / "fig3D_primary_matched_pair_global_test.csv", index=False)
    check("primary global ordinal p", float(gt[gt.family == "ordinal"].global_p.iloc[0]), 0.14908509, SRC["glob"], 1e-8)
    check("primary global binary p", float(gt[gt.family == "binary"].global_p.iloc[0]), 0.44145585, SRC["glob"], 1e-8)

    # ---- stage consistency ----
    k = dd.set_index("evidence_stage")
    check("stage 1 rows/patients = stage 2 rows/patients (same train patients)",
          (int(dd[dd.evidence_stage == "1_selection_evidence"].n_visit_pairs.iloc[0]),
           int(dd[dd.evidence_stage == "1_selection_evidence"].n_patients.iloc[0])),
          (int(k.loc["2_discovery", "n_visit_pairs"]), int(k.loc["2_discovery", "n_patients"])), SRC["disc"])
    check("discovery gates all passed", bool(disc["discovery_gates_passed"]) and all(v for kk, v in disc["gates"].items() if kk != "failure_reason"),
          True, SRC["disc"])
    check("discovery OR = exp(beta)", float(k.loc["2_discovery", "odds_ratio"]), float(np.exp(de["beta_disc"])), SRC["disc"], 1e-12)
    check("held-out OR = exp(beta)", float(np.exp(hr["beta_heldout"])), float(hr["odds_ratio_per_discovery_SD"]), SRC["held"], 1e-12)
    check("held-out 95% CI entirely below 0 (pre-specified criterion)", bool(hr["ci95_wald"][1] < 0), True, SRC["held"])
    check("held-out one-sided p < 0.025", bool(hr["one_sided_p_H1_beta_lt_0"] < 0.025), True, SRC["held"])
    check("held-out support: >= 80 patients, six classes, eta non-degenerate",
          (held["support"]["patients_at_least_80"], held["support"]["all_six_classes_represented"], held["support"]["eta_non_degenerate"]),
          (True, True, True), SRC["held"])
    by = held["accounting"]["heldout_patients_by_split"]
    check("held-out patients = validation + test", (sorted(by), sum(by.values())), (["test", "val"], held["accounting"]["heldout_patients"]), SRC["held"])
    check("held-out validity flag (eta slope > 0)", bool(hr["validity_flag_b_eta_positive"]), True, SRC["held"])
    check("no discovery patient is a held-out patient (artifact assertion) and selection excluded val/test",
          sel["excluded_from_selection"]["validation_and_test_patients"], 309, SRC["sel"])
    check("held-out splits recorded", jl("run")["heldout_splits"], ["val", "test"], SRC["run"])

    defs = {
        "evidence_level": ("post hoc, exploratory association; selection and discovery in original-training-partition patients only; "
                           "held-out evaluation in validation + test patients; NOT untouched confirmation, NOT independent validation, "
                           "NOT external validation"),
        "composite_name": "train-selected pathway composite",
        "family": fam, "selection_rule": sel["rule"],
        "family_vs_earlier_analysis": ("the earlier pooled analysis selected Myc Targets V2 in place of Spermatogenesis "
                                       "(comparison only; see selection_comparison_with_earlier_family.json)"),
        "composite": {"construction": "PC1 of the correlation matrix of the five z-scored pathway scores (train means/SDs, ddof=1), loadings > 0",
                      "variance_explained": disc["composite"]["variance_explained"], "loadings": dict(zip(fam, disc["composite"]["loadings"]))},
        "stage_populations": {"1_selection_evidence": "train patients (S3 regression rows)", "2_discovery": "same train patients",
                              "3_heldout_evaluation": "validation + test patients with RNA (patients listed by split in heldout_report.json)"},
        "optimism_note": "stage 2 uses the same patients that selected the pathways, so its estimate is optimistic",
        "multiplicity": {"selection": "BH across 50 pathways (train-only S3); selection by raw p, top five",
                         "heldout": "one primary directional test; project-level post hoc selection remains"},
        "primary_analysis_note": "matched-pair global test is pooled over all patients (included validation and test patients)",
    }
    json.dump(defs, open(OUT / "fig3D_trainonly_definitions.json", "w"), indent=2)
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "fig3D_trainonly_verification_checks.csv", index=False)
    json.dump({"script": "figures/fig3D_trainonly_extract_data.py", "inputs_sha256": {str(p): sha(p) for p in SRC.values()},
               "outputs": sorted(p.name for p in OUT.iterdir() if p.name != "fig3D_trainonly_provenance.json") +
                          ["fig3D_trainonly_provenance.json"]}, open(OUT / "fig3D_trainonly_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
