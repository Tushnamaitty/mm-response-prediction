"""
Supplementary Figure S2, US-TUNED revision: data extraction and verification (read-only).

Run from repo root:
    python figures/figS_country_ustuned_extract_data.py

This replaces the data source of figures/figS_country_extract_data.py, which is left unchanged. The earlier
version read results in which XGBoost hyperparameters had been reused from the main analysis, whose tuning
split contained 80 of the 116 evaluated Spain/Canada patients. Here the source is the re-tuned run:

    artifacts/results_clean_rerun/geographic_us_tuned/{binary,multiclass}_c_xgboost_us_to_spain_canada_metrics.json
    artifacts/results_clean_rerun/geographic_us_tuned/full_run_audit_and_summary.json
    (produced by modeling/us_to_spain_canada_model_c_ustuned.py)

Also read, never modified:
    * earlier country results on origin/modeling-country-robustness via `git show` (comparison only, and the
      Italy descriptive file)
    * data/splits/patient_split_assignments.csv, data/splits/task_eligibility.csv, data/clinical/subject_deid.csv,
      data/clinical/visit_pairs_with_rna.csv (patient ids only; used for an independent recount of the tuning
      pool; no identifier is written to any output)
    * artifacts/results_clean_rerun/{binary,multiclass}_c_xgboost_metrics.json (main-analysis parameters)

Nothing is refit or rescored. Outputs: figures/data/FigS_country_ustuned/ (the directory is gitignored with the rest
of figures/data; it holds aggregate numbers only). The script exits non-zero if any check fails.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RES = Path("artifacts/results_clean_rerun/geographic_us_tuned")
BRANCH = "origin/modeling-country-robustness"
PREV = {"binary": "results/robustness_country/binary_c_xgboost_us_to_spain_canada_metrics.json",
        "multiclass": "results/robustness_country/multiclass_c_xgboost_us_to_spain_canada_metrics.json",
        "italy": "results/robustness_country/italy_descriptive_only_NOT_USED_IN_MODEL.json"}
LOCAL = {"splits": Path("data/splits/patient_split_assignments.csv"), "elig": Path("data/splits/task_eligibility.csv"),
         "subject": Path("data/clinical/subject_deid.csv"), "pairs": Path("data/clinical/visit_pairs_with_rna.csv"),
         "main_bin": Path("artifacts/results_clean_rerun/binary_c_xgboost_metrics.json"),
         "main_mc": Path("artifacts/results_clean_rerun/multiclass_c_xgboost_metrics.json"),
         "res_bin": RES / "binary_c_xgboost_us_to_spain_canada_metrics.json",
         "res_mc": RES / "multiclass_c_xgboost_us_to_spain_canada_metrics.json",
         "res_audit": RES / "full_run_audit_and_summary.json"}
OUT = Path("figures/data/FigS_country_ustuned")
EVIDENCE = ("Internal geographic robustness within MMRF CoMMpass; NOT independent external validation. XGBoost "
            "hyperparameters were re-tuned (patient-grouped 5-fold CV over the main analysis's 3-configuration grid) "
            "on US patients from the original training partition only; no Spain, Canada, Italy, US-validation, "
            "original-validation or original-test patient entered tuning. Preprocessing and model fitting used US "
            "training data, and calibration method and threshold were chosen on US validation. The earlier version of "
            "this experiment had been seen before the redesign, so the analysis is post hoc and exploratory.")
MODEL = ("XGBoost, information layer C (current clinical + longitudinal history + treatment context; 143 source "
         "input features), refit on US training patients only")
CI_METHOD = "95% patient-level bootstrap percentile CI (1,000 resamples of evaluation patients)"
GROUPS = [("spain_canada_pooled", "Spain + Canada (pooled)", "pooled", "filled", "spain_canada_pooled", ""),
          ("spain_only", "Spain", "country subset", "hollow", "spain", "_spain"),
          ("canada_only", "Canada", "country subset", "hollow", "canada", "_canada")]
# cohort facts (definition of the experiment; independent of tuning)
COHORT = {"spain_canada_pooled": (116, 1588, 283, 1694), "spain_only": (80, 949, 185, 1034), "canada_only": (36, 639, 98, 660)}
US_VAL = (139, 1911, 351, 2076)  # patients, binary pairs, binary events, six-class pairs

checks = []


def check(name, observed, expected, source, tol=None):
    ok = abs(float(observed) - float(expected)) <= tol if tol is not None else observed == expected
    checks.append({"check": name, "observed": str(observed)[:200], "expected": str(expected)[:200], "passed": bool(ok),
                   "source": str(source)})


def sha(b):
    return hashlib.sha256(b).hexdigest()


def digest(ids):
    return sha("\n".join(sorted(ids)).encode("utf-8"))


def git_show(path):
    return subprocess.run(["git", "show", f"{BRANCH}:{path}"], capture_output=True, check=True).stdout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = json.load(open(LOCAL["res_bin"], encoding="utf-8"))
    m = json.load(open(LOCAL["res_mc"], encoding="utf-8"))
    au = json.load(open(LOCAL["res_audit"], encoding="utf-8"))
    a, env = au["audit"], au["environment"]
    pb, pm = json.loads(git_show(PREV["binary"])), json.loads(git_show(PREV["multiclass"]))
    it = json.loads(git_show(PREV["italy"]))
    src = lambda k: str(LOCAL[k])  # noqa: E731

    # ---- consistency between per-task files and the full audit ----
    for task, d in (("binary", b), ("multiclass", m)):
        t = au["tasks"][task]
        for key in ("n_train", "n_val", "n_test_pooled", "n_test_spain", "n_test_canada", "params", "tuning"):
            check(f"{task} {key}: metrics file = audit file", d[key], t[key], src("res_audit"))
        for g, *_ in GROUPS:
            check(f"{task} {g} metrics: metrics file = audit file", d[f"{g}_metrics"], t[f"{g}_metrics"], src("res_audit"))
    check("run mode is the primary US-tuned run", a["mode"], "primary_us_tuned", src("res_audit"))

    # ---- disjointness and cohort sizes ----
    for k in ("n_us_train_val_overlap", "n_us_train_sc_overlap", "n_us_val_sc_overlap"):
        check(f"audit {k} = 0", a[k], 0, src("res_audit"))
    check("cohort patients (US train, US val, Spain+Canada)",
          (a["n_us_train_patients"], a["n_us_val_patients"], a["n_spain_canada_patients"]), (632, 139, 116), src("res_audit"))

    # ---- tuning audit: what the hyperparameters were tuned on ----
    tuning_out = {}
    for task, d, main_key in (("binary", b, "main_bin"), ("multiclass", m, "main_mc")):
        t = d["tuning"]
        main = json.load(open(LOCAL[main_key], encoding="utf-8"))
        check(f"{task} tuning was performed (not a control)", t["performed"], True, src("res_audit"))
        for k in ("n_overlap_with_spain_canada_evaluation", "n_overlap_with_us_val", "n_overlap_with_italy",
                  "n_overlap_with_original_val_or_test"):
            check(f"{task} tuning {k} = 0", t[k], 0, src("res_audit"))
        check(f"{task} tuning/evaluation overlap field = 0", d["tuning_evaluation_patient_overlap"], 0, src("res_audit"))
        check(f"{task} grid = main-analysis grid", [c["params"] for c in t["cv_results"]],
              [c["params"] for c in main["cv_results"]], src(main_key))
        best = max(t["cv_results"], key=lambda r: r["mean"])
        check(f"{task} selected params = argmax of tuning-pool CV score", t["selected_params"], best["params"], src("res_audit"))
        check(f"{task} model params = selected params", d["params"], t["selected_params"], src("res_audit"))
        check(f"{task} 5 folds per configuration", [len(c["fold_scores"]) for c in t["cv_results"]], [5, 5, 5], src("res_audit"))
        tuning_out[task] = {k: t[k] for k in ("n_patients", "n_pairs", "n_improvement_events", "definition", "selection_metric",
                                              "n_cv_folds", "grid", "cv_results", "selected_params",
                                              "previous_params_reused_from_main_analysis", "selected_params_equal_previous",
                                              "us_train_patients_not_in_tuning_pool",
                                              "us_train_composition_by_original_split")}
        tuning_out[task]["main_analysis_selected_params"] = main["best_params"]

    # ---- independent recount of the tuning pool from the split / subject / pair files ----
    splits = pd.read_csv(LOCAL["splits"]).set_index("public_id")["split"]
    subj = pd.read_csv(LOCAL["subject"], usecols=["public_id", "country_of_residence_at_enrollment"])
    ctry = subj.set_index("public_id")["country_of_residence_at_enrollment"]
    check("every patient has a country", int(subj.country_of_residence_at_enrollment.isna().sum()), 0, src("subject"))
    pairs = pd.read_csv(LOCAL["pairs"], usecols=["pair_id", "public_id"], low_memory=False)
    elig = pd.read_csv(LOCAL["elig"])
    us = sorted(p for p in set(pairs.public_id) if ctry[p] == "united_states")
    us_tr, us_va = train_test_split(us, train_size=0.82, random_state=42)
    us_tr, us_va = set(us_tr), set(us_va)
    sc = {p for p in set(pairs.public_id) if ctry[p] in ("spain", "canada")}
    check("independent recount: US-train / US-val / evaluation digests match the run",
          (digest(us_tr), digest(us_va), digest(sc)),
          (a["id_set_sha256"]["us_train"], a["id_set_sha256"]["us_val"], a["id_set_sha256"]["spain_canada_evaluation"]), src("res_audit"))
    for task, d in (("binary", b), ("multiclass", m)):
        e = set(elig.loc[elig[f"eligible_for_{task}"].astype(str).str.lower() == "true", "pair_id"])
        pp = pairs[pairs.pair_id.isin(e)]
        pool = {p for p in pp.public_id if p in us_tr and splits[p] == "train"}
        check(f"independent recount: {task} tuning pool digest", digest(pool), d["tuning"]["id_set_sha256"], src("res_audit"))
        check(f"independent recount: {task} tuning pool patients / pairs", (len(pool), int(pp.public_id.isin(pool).sum())),
              (d["tuning"]["n_patients"], d["tuning"]["n_pairs"]), src("res_audit"))
        check(f"independent recount: {task} tuning pool shares no patient with evaluation", len(pool & sc), 0, "recount")
    tab = pd.crosstab(pd.Series({p: ctry[p] for p in sc}, name="country"), pd.Series({p: splits[p] for p in sc}, name="split"))
    tab.to_csv(OUT / "figS2_evaluation_patients_by_original_split.csv")
    ov = {c: {s: int(tab.loc[c, s]) for s in ("train", "val", "test")} for c in ("spain", "canada")}
    check("evaluation patients by original split (Spain 53/18/9; Canada 27/4/5)",
          (tuple(ov["spain"].values()), tuple(ov["canada"].values())), ((53, 18, 9), (27, 4, 5)), "recount")

    # ---- comparison with the earlier (hyperparameter-reuse) experiment ----
    cmp_out = {"source": f"{BRANCH}:results/robustness_country/", "note": ("comparison only; the earlier results are not "
               "used for any figure or table value"), "tasks": {}}
    for task, new, old, key in (("binary", b, pb, "auprc"), ("multiclass", m, pm, "macro_f1")):
        diffs = {g: new[f"{g}_metrics"][key] - old[f"{g}_metrics"][key] for g in ("us_val", "spain_canada_pooled", "spain_only", "canada_only")}
        cmp_out["tasks"][task] = {"params_previous": old["params"], "params_new": new["params"], "params_equal": old["params"] == new["params"],
                                  "calibration_previous": old["calibration_method_selected_on_us_val"],
                                  "calibration_new": new["calibration_method_selected_on_us_val"],
                                  "threshold_previous": old.get("threshold_selected_on_us_val"),
                                  "threshold_new": new.get("threshold_selected_on_us_val"),
                                  "primary_metric": key, "difference_new_minus_previous": diffs,
                                  "previous": {g: old[f"{g}_metrics"][key] for g in diffs},
                                  "new": {g: new[f"{g}_metrics"][key] for g in diffs}}
    check("binary: re-tuning kept the same hyperparameters", cmp_out["tasks"]["binary"]["params_equal"], True, "comparison")
    for g in ("us_val", "spain_canada_pooled", "spain_only", "canada_only"):
        check(f"binary {g} AUPRC identical to earlier run (same params, same pipeline)",
              cmp_out["tasks"]["binary"]["difference_new_minus_previous"][g], 0.0, "comparison", tol=1e-12)
    check("six-class: re-tuning changed the hyperparameters", cmp_out["tasks"]["multiclass"]["params_equal"], False, "comparison")
    cmp_out["reproduction_control"] = ("run with --control-previous-params (earlier hyperparameters, same code) reproduced the earlier "
                                       "metrics, CIs, thresholds and calibration choices exactly (max abs difference 0.0); see "
                                       "artifacts/results_clean_rerun/geographic_us_tuned_control_previous_params/")
    cmp_out["environment"] = env
    json.dump(cmp_out, open(OUT / "figS2_previous_comparison.json", "w"), indent=2)
    json.dump({"statement": EVIDENCE, "tuning": tuning_out,
               "evaluation_patients_by_original_split": ov,
               "evaluation_patients_by_original_split_note": ("descriptive only: with US-only tuning, none of these patients "
                                                              "was in any tuning fold, whatever their original split"),
               "calibration_note": {"binary": b["calibration_method_selected_on_us_val"],
                                    "six_class_applied_rule_selection": m["calibration_method_selected_on_us_val"],
                                    "six_class_main_rule_check": m["calibration_cross_fit_scores"]["_main_rule_check"],
                                    "six_class_macro_f1_invariant_to_temperature": ("a single temperature rescales log-probabilities "
                                                                                    "monotonically within each row, so argmax and "
                                                                                    "macro-F1 are unchanged")}},
              open(OUT / "figS2_tuning_audit.json", "w"), indent=2)

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
        check(f"{g} AUPRC CI brackets estimate", bc["auprc"]["ci_lower_2.5"] < bm["auprc"] < bc["auprc"]["ci_upper_97.5"], True, src("res_bin"))
        check(f"{g} macro-F1 CI brackets estimate", mc["ci_lower_2.5"] < mm["macro_f1"] < mc["ci_upper_97.5"], True, src("res_audit"))
        check(f"{g} 1,000 valid resamples (AUPRC, macro-F1)", (bc["auprc"]["n_resamples_used"], mc["n_resamples_used"]), (1000, 1000), src("res_audit"))
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

    a_, b_ = A.set_index("cohort_key"), Bd.set_index("cohort_key")
    for g, (np_, nv, ne, nm_) in COHORT.items():
        r = a_.loc[g]
        check(f"S2A {g} denominators", (int(r.n_patients), int(r.n_visit_pairs), int(r.n_improvement_events)), (np_, nv, ne), src("res_bin"))
        check(f"S2B {g} denominators", (int(b_.loc[g].n_patients), int(b_.loc[g].n_visit_pairs)), (np_, nm_), src("res_mc"))
    check("US validation denominators (patients, binary pairs, events, six-class pairs)",
          (int(a_.loc["us_val"].n_patients), int(a_.loc["us_val"].n_visit_pairs), int(a_.loc["us_val"].n_improvement_events),
           int(b_.loc["us_val"].n_visit_pairs)), US_VAL, src("res_bin"))
    check("S2A pooled = Spain + Canada pairs and events",
          (int(a_.loc["spain_only", "n_visit_pairs"] + a_.loc["canada_only", "n_visit_pairs"]),
           int(a_.loc["spain_only", "n_improvement_events"] + a_.loc["canada_only", "n_improvement_events"])), (1588, 283), src("res_bin"))
    check("S2B pooled = Spain + Canada pairs", int(b_.loc["spain_only", "n_visit_pairs"] + b_.loc["canada_only", "n_visit_pairs"]), 1694, src("res_mc"))
    check("binary threshold (selected on US validation)", b["threshold_selected_on_us_val"], 0.25, src("res_bin"))
    check("binary calibration selected on US validation", b["calibration_method_selected_on_us_val"], "uncalibrated", src("res_bin"))

    # ---- Italy (unchanged decision; descriptive file from the earlier branch) ----
    check("Italy marked descriptive only", it["status"].startswith("DESCRIPTIVE ONLY"), True, f"{BRANCH}:{PREV['italy']}")
    check("Italy patients", it["descriptive_stats_final_table"]["patients"], 138, f"{BRANCH}:{PREV['italy']}")
    check("Italy excluded from tuning and evaluation (audit)", a["n_italy_patients_excluded"], 138, src("res_audit"))
    json.dump({"status": it["status"], "reason": it["reason"], "descriptive_stats": it["descriptive_stats_final_table"],
               "recommendation": it["recommendation"],
               "note": ("Exclusion was decided after inspecting the outcome distribution (post hoc); the 138 Italian patients "
                        "remain in the main analyses (Figures 1-5).")},
              open(OUT / "figS2_italy_exclusion.json", "w"), indent=2)

    defs = {"evidence_level": EVIDENCE, "model": MODEL,
            "hyperparameters": {"binary": b["params"], "six_class": m["params"],
                                "source": "re-tuned on US patients in the original training partition (see figS2_tuning_audit.json)"},
            "calibration_selected_on_us_val": {"binary": b["calibration_method_selected_on_us_val"],
                                               "six_class": m["calibration_method_selected_on_us_val"]},
            "binary_threshold_selected_on_us_val": b["threshold_selected_on_us_val"],
            "cohorts": {"us_train": {"patients": a["n_us_train_patients"], "role": "preprocessing and model fitting"},
                        "tuning_pool": {"patients": {"binary": b["tuning"]["n_patients"], "six_class": m["tuning"]["n_patients"]},
                                        "role": "hyperparameter tuning only (US patients in the original training partition, within US-train)"},
                        "us_val": {"patients": a["n_us_val_patients"], "role": "calibration and threshold selection; reference only"},
                        "spain_canada": {"patients": a["n_spain_canada_patients"], "role": "evaluated once"},
                        "us_split_note": "new random 82/18 patient-level split of US patients (seed 42), not the main-analysis split"},
            "disjointness": "0 shared patients among US train, US validation and Spain+Canada; 0 between the tuning pool and any other cohort",
            "ci_method": CI_METHOD,
            "metrics": {"AUPRC": "binary improvement; higher is better; chance level = each cohort's own event rate",
                        "macro-F1": "six-class IMWG response; unweighted mean F1 over classes; higher is better; no simple chance line",
                        "AUROC": "caption / supplementary table only"},
            "unseen_category_audit_binary": b["unseen_category_audit"],
            "not_supported": ["statistical comparison between Spain and Canada", "statistical comparison with US validation",
                              "CI for US validation (not computed)", "external validation claims"],
            "canada_note": ("36 patients; six-class bootstrap mean %.3f vs point %.3f (skewed resampling distribution)" %
                            (au["tasks"]["multiclass"]["canada_only_bootstrap_ci"]["macro_f1"]["mean"],
                             m["canada_only_metrics"]["macro_f1"]))}
    json.dump(defs, open(OUT / "figS2_definitions.json", "w"), indent=2)

    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "figS2_verification_checks.csv", index=False)
    json.dump({"script": "figures/figS_country_ustuned_extract_data.py",
               "analysis_script": "modeling/us_to_spain_canada_model_c_ustuned.py",
               "analysis_script_sha256_at_run": env["script_sha256"], "environment_at_run": env,
               "git_sources": {f"{BRANCH}:{p}": sha(git_show(p)) for p in PREV.values()},
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
