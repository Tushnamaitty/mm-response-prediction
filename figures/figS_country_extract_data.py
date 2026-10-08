"""
Supplementary Figure S2 (country robustness) data extraction and verification (read-only).

Run from repo root:
    python figures/figS_country_extract_data.py

Sources
    * Locked country-robustness outputs on branch origin/modeling-country-robustness, read with
      `git show` (no checkout; working tree untouched):
        results/robustness_country/binary_c_xgboost_us_to_spain_canada_metrics.json
        results/robustness_country/multiclass_c_xgboost_us_to_spain_canada_metrics.json
        results/robustness_country/full_run_audit_and_summary.json
        results/robustness_country/italy_descriptive_only_NOT_USED_IN_MODEL.json
        results/robustness_country/README.md
        modeling/us_to_spain_canada_model_c.py
    * Original (main-analysis) patient split and country, current branch:
        data/splits/patient_split_assignments.csv, data/splits/task_eligibility.csv,
        data/clinical/subject_deid.csv
    * Locked main-analysis hyperparameters and tuning grid:
        artifacts/results_clean_rerun/{binary,multiclass}_c_xgboost_metrics.json

Nothing is refit or rescored. Event rates are events / visit pairs from the stored counts;
original-split overlap counts are tallies of the split and country files.

The script exits non-zero if any check fails.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

BRANCH = "origin/modeling-country-robustness"
GIT = {
    "binary": "results/robustness_country/binary_c_xgboost_us_to_spain_canada_metrics.json",
    "multiclass": "results/robustness_country/multiclass_c_xgboost_us_to_spain_canada_metrics.json",
    "audit": "results/robustness_country/full_run_audit_and_summary.json",
    "italy": "results/robustness_country/italy_descriptive_only_NOT_USED_IN_MODEL.json",
    "readme": "results/robustness_country/README.md",
    "script": "modeling/us_to_spain_canada_model_c.py",
}
LOCAL = {
    "splits": Path("data/splits/patient_split_assignments.csv"),
    "elig": Path("data/splits/task_eligibility.csv"),
    "subject": Path("data/clinical/subject_deid.csv"),
    "lock_bin": Path("artifacts/results_clean_rerun/binary_c_xgboost_metrics.json"),
    "lock_mc": Path("artifacts/results_clean_rerun/multiclass_c_xgboost_metrics.json"),
}
OUT = Path("figures/data/FigS_country")
EVIDENCE = ("Internal geographic robustness within MMRF CoMMpass; NOT independent external validation. "
            "Hyperparameters were historically selected (main analysis) using data that included 80 of the 116 "
            "evaluated Spain/Canada patients; the country experiment's preprocessing, model fitting, calibration "
            "selection and threshold selection used US data only.")
MODEL = ("XGBoost, information layer C (current clinical + longitudinal history + treatment context; 143 source "
         "input features), refit on US training patients only")
CI_METHOD = "95% patient-level bootstrap percentile CI (1,000 resamples of evaluation patients)"
GROUPS = [("spain_canada_pooled", "Spain + Canada (pooled)", "pooled", "filled", "spain_canada_pooled", ""),
          ("spain_only", "Spain", "country subset", "hollow", "spain", "_spain"),
          ("canada_only", "Canada", "country subset", "hollow", "canada", "_canada")]

checks = []


def check(name, observed, expected, source, tol=None):
    ok = abs(float(observed) - float(expected)) <= tol if tol is not None else observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(ok),
                   "source": str(source)})


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git_show(path):
    blob = subprocess.run(["git", "rev-parse", f"{BRANCH}:{path}"], capture_output=True, text=True, check=True).stdout.strip()
    raw = subprocess.run(["git", "show", f"{BRANCH}:{path}"], capture_output=True, check=True).stdout
    return raw, blob


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {k: git_show(p) for k, p in GIT.items()}
    J = {k: json.loads(raw[k][0]) for k in ("binary", "multiclass", "audit", "italy")}
    b, m, au, it = J["binary"], J["multiclass"], J["audit"], J["italy"]
    script = raw["script"][0].decode("utf-8")
    readme = raw["readme"][0].decode("utf-8")
    src = lambda k: f"{BRANCH}:{GIT[k]}"  # noqa: E731

    # ---- consistency between per-task files and the full audit ----
    for task, d in (("binary", b), ("multiclass", m)):
        t = au["tasks"][task]
        for key in ("n_train", "n_val", "n_test_pooled", "n_test_spain", "n_test_canada", "params"):
            check(f"{task} {key}: metrics file = audit file", d[key], t[key], src("audit"))
        for g, *_ in GROUPS:
            check(f"{task} {g} metrics: metrics file = audit file", d[f"{g}_metrics"], t[f"{g}_metrics"], src("audit"))

    # ---- disjointness and cohort sizes ----
    a = au["audit"]
    check("US train / US val overlap", len(a["us_train_val_overlap"]), 0, src("audit"))
    check("US train / Spain+Canada overlap", len(a["us_train_sc_overlap"]), 0, src("audit"))
    check("US val / Spain+Canada overlap", len(a["us_val_sc_overlap"]), 0, src("audit"))
    check("cohort patients (US train, US val, Spain+Canada)",
          (a["n_us_train_patients"], a["n_us_val_patients"], a["n_spain_canada_patients"]), (632, 139, 116), src("audit"))
    check("disjointness asserted in script", script.count("assert not audit[") , 3, src("script"))
    check("Italy excluded from rob_split in script", "Italy: excluded entirely from rob_split" in script, True, src("script"))
    check("script states not external validation", "NOT external validation" in script, True, src("script"))
    check("README states no independent cohort", "No independent cohort was used" in readme, True, src("readme"))

    # ---- what the country experiment fitted on US data only ----
    for phrase in ("fit_preprocessor_on_us_train(df, feature_cols, tr_mask)", "cross-fitting on US-val ONLY",
                   "threshold = select_threshold(y_va, proba_va_cal)"):
        check(f"script uses US-only step: {phrase[:40]}", phrase in script, True, src("script"))

    # ---- historical hyperparameter reuse ----
    lb, lm = json.load(open(LOCAL["lock_bin"])), json.load(open(LOCAL["lock_mc"]))
    check("binary hyperparameters = locked main-analysis values", b["params"], lb["best_params"], LOCAL["lock_bin"])
    check("six-class hyperparameters = locked main-analysis values", m["params"], lm["best_params"], LOCAL["lock_mc"])
    check("tuning grid size (binary)", len(lb["cv_results"]), 3, LOCAL["lock_bin"])

    splits = pd.read_csv(LOCAL["splits"])
    subj = pd.read_csv(LOCAL["subject"], usecols=["public_id", "country_of_residence_at_enrollment"])
    sp = splits.merge(subj, on="public_id", how="left", validate="one_to_one")
    check("every patient has a country", int(sp.country_of_residence_at_enrollment.isna().sum()), 0, LOCAL["subject"])
    tab = pd.crosstab(sp.country_of_residence_at_enrollment, sp.split)
    tab.to_csv(OUT / "figS2_original_split_by_country.csv")
    ov = {c: {s: int(tab.loc[c, s]) for s in ("train", "val", "test")} for c in ("spain", "canada", "italy", "united_states")}
    check("Spain patients by original split (train/val/test)", (ov["spain"]["train"], ov["spain"]["val"], ov["spain"]["test"]),
          (53, 18, 9), LOCAL["splits"])
    check("Canada patients by original split (train/val/test)", (ov["canada"]["train"], ov["canada"]["val"], ov["canada"]["test"]),
          (27, 4, 5), LOCAL["splits"])
    check("Spain+Canada in original tuning population (train)", ov["spain"]["train"] + ov["canada"]["train"], 80, LOCAL["splits"])
    check("Spain+Canada total = 116 evaluated patients", sum(ov["spain"].values()) + sum(ov["canada"].values()),
          a["n_spain_canada_patients"], LOCAL["splits"])
    check("Spain total = 80", sum(ov["spain"].values()), b["n_patients_test_spain"], LOCAL["splits"])
    check("Canada total = 36", sum(ov["canada"].values()), b["n_patients_test_canada"], LOCAL["splits"])
    check("US total = US train + US val", sum(ov["united_states"].values()),
          a["n_us_train_patients"] + a["n_us_val_patients"], LOCAL["splits"])
    elig = pd.read_csv(LOCAL["elig"]).merge(subj, on="public_id")
    tune_pairs = {}
    for task in ("binary", "multiclass"):
        e = elig[(elig[f"eligible_for_{task}"].astype(str).str.lower() == "true") & (elig.split == "train")]
        tune_pairs[task] = {c: int((e.country_of_residence_at_enrollment == c).sum()) for c in ("spain", "canada")}
    check("binary Spain/Canada visit pairs in original train", (tune_pairs["binary"]["spain"], tune_pairs["binary"]["canada"]),
          (646, 465), LOCAL["elig"])
    overlap = {
        "statement": ("Historical hyperparameter reuse: the main analysis chose XGBoost layer C hyperparameters by "
                      "patient-grouped cross-validation within its training split, choosing among 3 configurations. "
                      "That training split contained 80 of the 116 Spain/Canada patients evaluated here. This is not "
                      "leakage into the country experiment's fitting, preprocessing, calibration or threshold selection, "
                      "which used US data only."),
        "spain_canada_patients_in_original_split": {"spain": ov["spain"], "canada": ov["canada"],
                                                    "total_in_original_train_tuning_population": 80, "of_evaluated": 116},
        "spain_canada_visit_pairs_in_original_train": tune_pairs,
        "tuning_grid": {"binary": lb["cv_results"], "multiclass": lm["cv_results"]},
        "selected": {"binary": lb["best_params"], "multiclass": lm["best_params"]},
        "original_val_and_test_note": ("22 Spain/Canada patients were in the original validation split and 14 in the "
                                       "original test split; neither split was used for hyperparameter selection"),
    }
    json.dump(overlap, open(OUT / "figS2_tuning_overlap.json", "w"), indent=2)

    # ---- S2A / S2B / AUROC ----
    rowsA, rowsB, rowsR = [], [], []
    ev = {"spain_canada_pooled": b["improvement_events_pooled"], "spain_only": b["improvement_events_spain"],
          "canada_only": b["improvement_events_canada"]}
    for g, label, role, marker, nkey, pkey in GROUPS:
        npat = b[f"n_patients_test{pkey}" if pkey else "n_patients_test_pooled"]
        bm, bc = b[f"{g}_metrics"], b[f"{g}_bootstrap_ci"]
        n_bin = bm["n"]
        rowsA.append({"cohort": label, "cohort_key": g, "role": role, "marker": marker, "n_patients": npat,
                      "n_visit_pairs": n_bin, "n_improvement_events": ev[g], "event_rate": ev[g] / n_bin,
                      "auprc": bm["auprc"], "ci_lower": bc["auprc"]["ci_lower_2.5"], "ci_upper": bc["auprc"]["ci_upper_97.5"],
                      "bootstrap_mean": bc["auprc"]["mean"], "n_resamples_used": bc["auprc"]["n_resamples_used"],
                      "ci_method": CI_METHOD, "reference_line": "event rate = expected AUPRC of a non-informative model"})
        rowsR.append({"cohort": label, "cohort_key": g, "auroc": bm["auroc"], "ci_lower": bc["auroc"]["ci_lower_2.5"],
                      "ci_upper": bc["auroc"]["ci_upper_97.5"], "n_resamples_used": bc["auroc"]["n_resamples_used"],
                      "use": "caption / supplementary table only"})
        mm, mc = m[f"{g}_metrics"], au["tasks"]["multiclass"][f"{g}_bootstrap_ci"]["macro_f1"]
        rowsB.append({"cohort": label, "cohort_key": g, "role": role, "marker": marker, "n_patients": npat,
                      "n_visit_pairs": mm["n"], "macro_f1": mm["macro_f1"], "ci_lower": mc["ci_lower_2.5"],
                      "ci_upper": mc["ci_upper_97.5"], "bootstrap_mean": mc["mean"], "n_resamples_used": mc["n_resamples_used"],
                      "ci_method": CI_METHOD,
                      "class_supports": json.dumps({k: v["support"] for k, v in mm.get("per_class", {}).items()})})
        check(f"{g} AUPRC CI brackets estimate", bc["auprc"]["ci_lower_2.5"] < bm["auprc"] < bc["auprc"]["ci_upper_97.5"],
              True, src("binary"))
        check(f"{g} macro-F1 CI brackets estimate", mc["ci_lower_2.5"] < mm["macro_f1"] < mc["ci_upper_97.5"], True, src("audit"))
        check(f"{g} 1,000 valid resamples (AUPRC, macro-F1)", (bc["auprc"]["n_resamples_used"], mc["n_resamples_used"]),
              (1000, 1000), src("audit"))
    uv = b["us_val_metrics"]
    ref = {"cohort": "US validation (reference)", "cohort_key": "us_val", "role": "descriptive reference", "marker": "grey",
           "n_patients": b["n_patients_val"], "note": ("informed calibration-method and threshold selection; no "
                                                        "confidence interval was computed")}
    rowsA.append({**ref, "n_visit_pairs": uv["n"], "n_improvement_events": b["improvement_events_val"],
                  "event_rate": b["improvement_events_val"] / uv["n"], "auprc": uv["auprc"], "ci_lower": None, "ci_upper": None,
                  "ci_method": "none (not computed)", "reference_line": "event rate"})
    rowsB.append({**ref, "n_patients": m["n_patients_val"], "n_visit_pairs": m["us_val_metrics"].get("n", m["n_val"]),
                  "macro_f1": m["us_val_metrics"]["macro_f1"], "ci_lower": None, "ci_upper": None,
                  "ci_method": "none (not computed)"})
    rowsR.append({"cohort": "US validation (reference)", "cohort_key": "us_val", "auroc": uv["auroc"], "ci_lower": None,
                  "ci_upper": None, "use": "caption / supplementary table only"})
    A, Bd, R = pd.DataFrame(rowsA), pd.DataFrame(rowsB), pd.DataFrame(rowsR)
    A.to_csv(OUT / "figS2A_binary_auprc.csv", index=False)
    Bd.to_csv(OUT / "figS2B_sixclass_macro_f1.csv", index=False)
    R.to_csv(OUT / "figS2_auroc_caption_only.csv", index=False)

    a_ = A.set_index("cohort_key")
    b_ = Bd.set_index("cohort_key")
    expA = {"spain_canada_pooled": (116, 1588, 283, 0.4514, 0.4012, 0.5071), "spain_only": (80, 949, 185, 0.4626, 0.4066, 0.5318),
            "canada_only": (36, 639, 98, 0.4435, 0.3691, 0.5413)}
    for g, (np_, nv, ne, est, lo, hi) in expA.items():
        r = a_.loc[g]
        check(f"S2A {g} denominators", (int(r.n_patients), int(r.n_visit_pairs), int(r.n_improvement_events)), (np_, nv, ne), src("binary"))
        check(f"S2A {g} AUPRC [CI] (4 dp)", (round(r.auprc, 4), round(r.ci_lower, 4), round(r.ci_upper, 4)), (est, lo, hi), src("binary"))
    check("S2A pooled = Spain + Canada pairs and events",
          (int(a_.loc["spain_only", "n_visit_pairs"] + a_.loc["canada_only", "n_visit_pairs"]),
           int(a_.loc["spain_only", "n_improvement_events"] + a_.loc["canada_only", "n_improvement_events"])),
          (1588, 283), src("binary"))
    check("S2A event rates (pooled, Spain, Canada, US val; 3 dp)",
          tuple(round(a_.loc[k, "event_rate"], 3) for k in ("spain_canada_pooled", "spain_only", "canada_only", "us_val")),
          (0.178, 0.195, 0.153, 0.184), src("binary"))
    check("S2A US val AUPRC", round(a_.loc["us_val", "auprc"], 4), 0.5247, src("binary"))
    expB = {"spain_canada_pooled": (1694, 0.6441, 0.5978, 0.6757), "spain_only": (1034, 0.6388, 0.5891, 0.6743),
            "canada_only": (660, 0.6343, 0.4916, 0.6860)}
    for g, (nv, est, lo, hi) in expB.items():
        r = b_.loc[g]
        check(f"S2B {g} macro-F1 [CI] and pairs", (int(r.n_visit_pairs), round(r.macro_f1, 4), round(r.ci_lower, 4),
                                                   round(r.ci_upper, 4)), (nv, est, lo, hi), src("audit"))
    check("S2B US val macro-F1", round(b_.loc["us_val", "macro_f1"], 4), 0.6475, src("multiclass"))
    check("README pooled AUPRC text", "0.451 (95% patient-bootstrap CI 0.401–0.507" in readme, True, src("readme"))
    check("README pooled macro-F1 text", "0.644 (95% CI 0.598–0.676" in readme, True, src("readme"))

    # ---- Italy ----
    check("Italy marked descriptive only", it["status"].startswith("DESCRIPTIVE ONLY"), True, src("italy"))
    check("Italy patients", it["descriptive_stats_final_table"]["patients"], 138, src("italy"))
    check("Italy count = original split total", sum(ov["italy"].values()), 138, LOCAL["splits"])
    italy = {"status": it["status"], "reason": it["reason"], "descriptive_stats": it["descriptive_stats_final_table"],
             "recommendation": it["recommendation"],
             "note": ("Exclusion was decided after inspecting the outcome distribution (post hoc); the 138 Italian patients "
                      "remain in the main analyses (Figures 1-5)."),
             "original_split": ov["italy"]}
    json.dump(italy, open(OUT / "figS2_italy_exclusion.json", "w"), indent=2)

    defs = {"evidence_level": EVIDENCE, "model": MODEL,
            "hyperparameters": {"binary": b["params"], "six_class": m["params"], "source": "reused from locked main analysis"},
            "calibration_selected_on_us_val": {"binary": b["calibration_method_selected_on_us_val"],
                                               "six_class": m["calibration_method_selected_on_us_val"]},
            "binary_threshold_selected_on_us_val": b["threshold_selected_on_us_val"],
            "cohorts": {"us_train": {"patients": a["n_us_train_patients"], "role": "model fitting and preprocessing"},
                        "us_val": {"patients": a["n_us_val_patients"], "role": "calibration and threshold selection; reference only"},
                        "spain_canada": {"patients": a["n_spain_canada_patients"], "role": "evaluated once"},
                        "us_split_note": "new random 82/18 patient-level split of US patients (seed 42), not the main-analysis split"},
            "disjointness": "0 shared patients between US train, US validation and Spain+Canada (asserted in script, recorded in audit)",
            "ci_method": CI_METHOD,
            "metrics": {"AUPRC": "binary improvement; higher is better; chance level = each cohort's own event rate",
                        "macro-F1": "six-class IMWG response; unweighted mean F1 over classes; higher is better; no simple chance line",
                        "AUROC": "caption / supplementary table only"},
            "unseen_category_audit_binary": b["unseen_category_audit"],
            "not_supported": ["statistical comparison between Spain and Canada", "statistical comparison with US validation",
                              "CI for US validation (not computed)", "external validation claims"],
            "canada_note": ("36 patients; six-class bootstrap mean 0.599 vs point 0.634 (skewed resampling distribution; "
                            "rare classes SD 27 and sCR 21 pairs)")}
    json.dump(defs, open(OUT / "figS2_definitions.json", "w"), indent=2)

    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "figS2_verification_checks.csv", index=False)
    json.dump({"script": "figures/figS_country_extract_data.py",
               "git_sources": {f"{BRANCH}:{p}": {"git_blob": raw[k][1], "sha256": sha(raw[k][0])} for k, p in GIT.items()},
               "inputs_sha256": {str(p): sha(p.read_bytes()) for p in LOCAL.values()},
               "outputs": sorted(p.name for p in OUT.iterdir() if p.name != "figS2_provenance.json") + ["figS2_provenance.json"]},
              open(OUT / "figS2_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
