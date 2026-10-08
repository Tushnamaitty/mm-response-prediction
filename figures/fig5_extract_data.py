"""
Figure 5 data extraction and verification (read-only).

Run from repo root:
    python figures/fig5_extract_data.py

Reads the locked survival pathway-interpretation outputs only; never refits or recomputes a model.
The only derived quantity is the Pearson correlation of the stored mean PFS and OS coefficients
across the 50 pathways (arithmetic on saved values), plus display-name clean-up.

Panels
    A  Mean PFS vs mean OS ridge-Cox coefficient for all 50 Hallmark pathways; the 19 pathways that
       are consistently directed (same sign in all 25 fits) in both endpoints are highlighted.
    B  The 19 concordant pathways: mean coefficient per endpoint with the min-max range across the
       25 overlapping outer cross-validation fits (NOT a confidence interval).

Coefficient: beta of the 50 pathway scores inside the "clinical baseline + RNA" ridge Cox model of
Figure 4, on the training-fold-standardized scale (log-hazard per training-fold SD of the pathway
score; HR = exp(beta)); beta > 0 = higher score associated with higher hazard (worse survival).

The script exits non-zero if any check fails.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("artifacts/results_clean_rerun/survival_pathway_interpretation")
OUT = Path("figures/data/Fig5")
SRC = {
    "both": D / "pathway_stability_both_endpoints.csv",
    "pfs": D / "pfs_pathway_stability.csv",
    "os": D / "os_pathway_stability.csv",
    "pfs_fold": D / "pfs_fold_coefficients.csv",
    "os_fold": D / "os_fold_coefficients.csv",
    "summary": D / "summary.json",
    "repro": D / "reproduction_check.json",
    "manifest": D / "output_manifest.json",
    "done": D / "DONE.json",
    "script": Path("explainability/survival_pathway_interpretation.py"),
}
LABELLED = ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle", "Myc Targets V1", "Myc Targets V2",
            "Notch Signaling", "mTORC1 Signaling"]
PROLIF_MYC = LABELLED[:5]
DISPLAY_FIX = {"PI3K/AKT/mTOR  Signaling": "PI3K/AKT/mTOR Signaling", "Pperoxisome": "Peroxisome",
               "heme Metabolism": "Heme Metabolism"}
BETA_DEF = ("beta = ridge-Cox log-hazard coefficient per training-fold SD of the pathway score in the "
            "'clinical baseline + RNA' model (HR = exp(beta)); beta > 0 = higher pathway score associated "
            "with higher hazard (worse survival); clinical covariates penalized jointly with the pathways")

checks = []


def check(name, observed, expected, source, tol=None):
    ok = abs(float(observed) - float(expected)) <= tol if tol is not None else observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(ok),
                   "source": str(source)})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.load(open(SRC["manifest"], encoding="utf-8"))["files"]
    for k, p in SRC.items():
        if p.parent == D and p.name in man:
            check(f"manifest hash {p.name}", sha(p), man[p.name], SRC["manifest"])
    check("analysis complete", json.load(open(SRC["done"]))["status"], "complete", SRC["done"])
    summ = json.load(open(SRC["summary"], encoding="utf-8"))
    check("reproduction gate passed", summ["reproduction_gate"]["passed"], True, SRC["summary"])
    check("reproduction_check passed", str(json.load(open(SRC["repro"], encoding="utf-8"))["passed"]), "True", SRC["repro"])

    b = pd.read_csv(SRC["both"])
    check("50 pathways", len(b), 50, SRC["both"])
    check("25 fits per endpoint (PFS)", int(b.pfs_n_fits.unique()[0]), 25, SRC["both"])
    check("25 fits per endpoint (OS)", int(b.os_n_fits.unique()[0]), 25, SRC["both"])

    # ---- per-fit coefficients: recheck the stored summaries and the direction rule ----
    folds = {}
    for ep in ("pfs", "os"):
        f = pd.read_csv(SRC[f"{ep}_fold"])
        folds[ep] = f
        check(f"{ep} 25 outer fits (5 repeats x 5 folds)", (len(f), f.repeat.nunique(), f.fold.nunique()), (25, 5, 5),
              SRC[f"{ep}_fold"])
        check(f"{ep} lambda = 1.0 in every fit", sorted(f["lambda"].unique().tolist()), [1.0], SRC[f"{ep}_fold"])
        bb = b.set_index("pathway")
        cf = f[bb.index]
        check(f"{ep} stored mean = mean of fold coefficients", float((cf.mean() - bb[f"{ep}_beta_mean"]).abs().max()), 0.0,
              SRC[f"{ep}_fold"], 1e-12)
        check(f"{ep} stored min/max = fold min/max",
              float(max((cf.min() - bb[f"{ep}_beta_min"]).abs().max(), (cf.max() - bb[f"{ep}_beta_max"]).abs().max())),
              0.0, SRC[f"{ep}_fold"], 1e-12)
        lab = np.where((cf > 0).all(), "consistently_positive", np.where((cf < 0).all(), "consistently_negative", "mixed"))
        check(f"{ep} direction labels follow the 25/25 sign rule", bool((pd.Series(lab, index=cf.columns)
                                                                         == bb[f"{ep}_direction_label"]).all()), True,
              SRC[f"{ep}_fold"])
        counts = bb[f"{ep}_direction_label"].value_counts().to_dict()
        exp = {"pfs": {"consistently_positive": 16, "consistently_negative": 9, "mixed": 25},
               "os": {"consistently_positive": 21, "consistently_negative": 10, "mixed": 19}}[ep]
        check(f"{ep} direction counts", counts, exp, SRC["both"])
        check(f"{ep} no zero coefficients", int((cf == 0).sum().sum()), 0, SRC[f"{ep}_fold"])

    conc = b.consistent_same_direction_both_endpoints.astype(bool)
    derived = (((b.pfs_direction_label == "consistently_positive") & (b.os_direction_label == "consistently_positive"))
               | ((b.pfs_direction_label == "consistently_negative") & (b.os_direction_label == "consistently_negative")))
    check("concordance flag = consistent same direction in both", bool((conc == derived).all()), True, SRC["both"])
    check("19 concordant pathways", int(conc.sum()), 19, SRC["both"])
    check("13 higher-hazard / 6 lower-hazard concordant", (int((conc & (b.pfs_beta_mean > 0)).sum()),
                                                         int((conc & (b.pfs_beta_mean < 0)).sum())), (13, 6), SRC["both"])
    check("concordant list = summary.json", sorted(b[conc].pathway), sorted(summ["consistent_same_direction_in_both_endpoints"]),
          SRC["summary"])
    check("no pathway consistent in opposite directions",
          int((((b.pfs_direction_label == "consistently_positive") & (b.os_direction_label == "consistently_negative"))
               | ((b.pfs_direction_label == "consistently_negative") & (b.os_direction_label == "consistently_positive"))).sum()),
          0, SRC["both"])
    for p in LABELLED:
        check(f"labelled pathway present and concordant: {p}", bool(b.set_index("pathway").loc[p, "consistent_same_direction_both_endpoints"]),
              True, SRC["both"])

    # ---- correlation (only derived statistic) ----
    r = float(np.corrcoef(b.pfs_beta_mean, b.os_beta_mean)[0, 1])
    rs = float(b.pfs_beta_mean.rank().corr(b.os_beta_mean.rank()))
    check("Pearson r of mean betas (50 pathways)", round(r, 4), 0.8275, SRC["both"])
    same_sign = int((np.sign(b.pfs_beta_mean) == np.sign(b.os_beta_mean)).sum())
    check("same sign of mean beta: 39 pathways", same_sign, 39, SRC["both"])
    check("same-sign column agrees", int(b.same_sign_of_mean_beta_pfs_os.sum()), same_sign, SRC["both"])

    # ---- outputs ----
    def group(row):
        if not row.consistent_same_direction_both_endpoints:
            return "not concordant"
        return "higher hazard in both" if row.pfs_beta_mean > 0 else "lower hazard in both"

    a = pd.DataFrame({
        "pathway_source_name": b.pathway,
        "pathway_display_name": b.pathway.map(lambda s: DISPLAY_FIX.get(s, s)),
        "pfs_beta_mean": b.pfs_beta_mean, "os_beta_mean": b.os_beta_mean,
        "pfs_HR_of_mean_beta": b.pfs_HR_of_mean_beta, "os_HR_of_mean_beta": b.os_HR_of_mean_beta,
        "pfs_direction_label": b.pfs_direction_label, "os_direction_label": b.os_direction_label,
        "concordant_both_endpoints": conc, "concordance_group": b.apply(group, axis=1),
        "label_in_panel": b.pathway.isin(LABELLED), "proliferation_myc_family": b.pathway.isin(PROLIF_MYC),
    })
    a.to_csv(OUT / "fig5A_pfs_vs_os_coefficients.csv", index=False)

    rows = []
    for _, x in b[conc].iterrows():
        for ep, lab in (("pfs", "PFS"), ("os", "OS")):
            rows.append({"pathway_source_name": x.pathway, "pathway_display_name": DISPLAY_FIX.get(x.pathway, x.pathway),
                         "concordance_group": group(x), "endpoint": lab, "beta_mean": x[f"{ep}_beta_mean"],
                         "beta_median": x[f"{ep}_beta_median"], "beta_min_across_fits": x[f"{ep}_beta_min"],
                         "beta_max_across_fits": x[f"{ep}_beta_max"], "beta_sd_across_fits": x[f"{ep}_beta_sd"],
                         "HR_of_mean_beta": x[f"{ep}_HR_of_mean_beta"], "HR_min_fit": x[f"{ep}_HR_min_fit"],
                         "HR_max_fit": x[f"{ep}_HR_max_fit"], "n_fits": int(x[f"{ep}_n_fits"]),
                         "range_definition": "min-max across 25 overlapping outer CV fits; NOT a confidence interval",
                         "label_in_panel": x.pathway in LABELLED})
    bdf = pd.DataFrame(rows)
    bdf["sort_key"] = bdf.groupby("pathway_source_name").beta_mean.transform("mean")
    bdf = bdf.sort_values(["concordance_group", "sort_key"], ascending=[True, False]).drop(columns="sort_key")
    bdf.to_csv(OUT / "fig5B_concordant_pathways.csv", index=False)
    pd.concat([folds["pfs"], folds["os"]]).to_csv(OUT / "fig5B_fold_coefficients_long_source.csv", index=False)
    check("B 38 rows (19 pathways x 2 endpoints)", len(bdf), 38, OUT / "fig5B_concordant_pathways.csv")
    check("B ranges bracket means", bool(((bdf.beta_min_across_fits <= bdf.beta_mean) & (bdf.beta_mean <= bdf.beta_max_across_fits)).all()),
          True, OUT / "fig5B_concordant_pathways.csv")
    check("B concordant ranges exclude 0 (25/25 sign)",
          bool(((bdf.beta_min_across_fits > 0) | (bdf.beta_max_across_fits < 0)).all()), True, OUT / "fig5B_concordant_pathways.csv")

    stats = {
        "pearson_r_mean_beta_pfs_vs_os": r, "spearman_r_mean_beta_pfs_vs_os": rs, "n_pathways": 50,
        "r_note": ("Derived arithmetic on stored per-pathway means (not stored in the locked outputs). PFS and OS share "
                   "the same 674 patients, RNA and clinical covariates, and OS deaths are PFS events, so a high r is "
                   "expected and is not independent corroboration. No test of r is reported."),
        "n_same_sign_mean_beta": same_sign,
        "direction_counts": {"PFS": {"consistently_positive": 16, "consistently_negative": 9, "mixed": 25},
                             "OS": {"consistently_positive": 21, "consistently_negative": 10, "mixed": 19}},
        "n_concordant": 19, "n_concordant_higher_hazard": 13, "n_concordant_lower_hazard": 6,
        "concordant_HR_per_SD_range": {"PFS": [float(bdf[bdf.endpoint == "PFS"].HR_of_mean_beta.min()),
                                               float(bdf[bdf.endpoint == "PFS"].HR_of_mean_beta.max())],
                                       "OS": [float(bdf[bdf.endpoint == "OS"].HR_of_mean_beta.min()),
                                              float(bdf[bdf.endpoint == "OS"].HR_of_mean_beta.max())]},
    }
    json.dump(stats, open(OUT / "fig5_summary_stats.json", "w"), indent=2)

    defs = {
        "evidence_level": "post-hoc exploratory predictive association; internal cross-validation only; not causal; not independent validation",
        "coefficient": BETA_DEF,
        "model": ("Figure 4 'clinical baseline + RNA' ridge Cox model (age at index, gender, ISS, exact first-line regimen "
                  "+ 50 Hallmark ssGSEA pathway scores), refit in each of the 25 outer folds with the locked penalty "
                  "(lambda = 1.0 in every fit); reproduction gate: refit out-of-fold risks match the locked run within 1e-9"),
        "cohort": "674 patients with one pre-treatment RNA sample; PFS 459 events, OS 247 deaths",
        "direction_rule": summ["direction_rule"],
        "concordance_rule": "consistently directed (25/25 fits) in BOTH endpoints with the same sign",
        "range_definition": "min-max of beta across the 25 outer CV fits; NOT a confidence interval",
        "labelled_pathways": LABELLED,
        "display_name_fixes": DISPLAY_FIX,
        "limitations": summ["caveats"] + [
            "The 25 fits come from 5 repeats of 5-fold CV and share ~80% of training patients: agreement across fits is not independent replication.",
            "Correlated pathways (e.g. the five proliferation/MYC scores share 71.5% of variance; Figure 3D) have non-independent ridge coefficients.",
            "No pathway-level confidence intervals, p-values, q-values or causal effects exist.",
            "Overlaps the patients used in Figures 3 and 4; agreement with Figure 3D is not independent evidence.",
        ],
    }
    json.dump(defs, open(OUT / "fig5_definitions.json", "w"), indent=2)

    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "fig5_verification_checks.csv", index=False)
    outputs = sorted(p.name for p in OUT.iterdir() if p.name != "fig5_provenance.json")
    json.dump({"script": "figures/fig5_extract_data.py", "inputs_sha256": {str(p): sha(p) for p in SRC.values()},
               "outputs": outputs + ["fig5_provenance.json"]}, open(OUT / "fig5_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed; Pearson r = {r:.4f}")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
