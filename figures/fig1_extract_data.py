"""
Figure 1 data extraction and verification (read-only).

Run from repo root:
    python figures/fig1_extract_data.py

Reads locked inputs only; never modifies them and never retrains or rescores models.
Writes verified Figure 1 source data to figures/data/Fig1/.

Inputs
    data/clinical/visit_pairs_with_rna.csv           visit pairs + as-of RNA join (Step 9)
    data/splits/task_eligibility.csv                 task eligibility per pair
    data/splits/patient_split_assignments.csv        patient-level train/val/test split
    data_pipeline/model_feature_sets.json            source input features per information layer
    artifacts/preprocessing/model_{a,b,c,d}_metadata.json   fitted model-input columns
    artifacts/results_clean_rerun/{task}_{layer}_xgboost_metrics.json   locked split sizes
    artifacts/results_clean_rerun/direction2/association/run_summary.json   distinct RNA samples
    artifacts/results/rna_freshness_sensitivity_PRELIMINARY.json   earlier RNA-age distribution
    artifacts/results_clean_rerun/survival_incremental/*_oof_predictions.csv   survival cohorts

Any failed check is recorded in verification_checks.csv; the script exits non-zero if a
locked count fails to reproduce.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("figures/data/Fig1")
RC = Path("artifacts/results_clean_rerun")
PAIRS = Path("data/clinical/visit_pairs_with_rna.csv")
ELIG = Path("data/splits/task_eligibility.csv")
SPLITS = Path("data/splits/patient_split_assignments.csv")
FEATS = Path("data_pipeline/model_feature_sets.json")
PRE = Path("artifacts/preprocessing")
D2_SUMMARY = RC / "direction2/association/run_summary.json"
FRESH_PRELIM = Path("artifacts/results/rna_freshness_sensitivity_PRELIMINARY.json")
SURV = {
    "PFS_primary": RC / "survival_incremental/primary_pfs_oof_predictions.csv",
    "OS_secondary": RC / "survival_incremental/secondary_os_oof_predictions.csv",
    "PFS_rna_day_-30_to_0": RC / "survival_incremental/sensitivity_m30_0_pfs_oof_predictions.csv",
}

RESPONSE_RANK = {  # same scale as explainability/direction2_common.py
    "progressive_disease": 0, "stable_disease": 1, "partial_response": 2,
    "very_good_partial_response": 3, "complete_response": 4, "stringent_complete_response": 5,
}
LAYER_NAMES = {
    "a": "A: Current clinical state",
    "b": "B: A + longitudinal history",
    "c": "C: B + treatment context",
    "d": "D: C + RNA pathway scores",
}

checks = []


def check(name, observed, expected, source, pdf_value=None):
    ok = bool(observed == expected) if expected is not None else None
    checks.append({"check": name, "observed": observed, "expected_locked": expected,
                   "previous_pdf_value": pdf_value, "passed": ok, "source": str(source)})
    return ok


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def as_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIRS, low_memory=False)
    elig = pd.read_csv(ELIG)
    splits = pd.read_csv(SPLITS)
    df = pairs.merge(elig[["pair_id", "split", "eligible_for_binary", "eligible_for_multiclass",
                           "exclusion_reason"]], on="pair_id", how="left", validate="one_to_one")
    df["eligible_for_binary"] = as_bool(df["eligible_for_binary"])
    df["eligible_for_multiclass"] = as_bool(df["eligible_for_multiclass"])
    df["rna_available"] = df["days_since_rna_sample"].notna()

    # ---------------- cohort and task counts ----------------
    check("pairs rows == eligibility rows", len(pairs), len(elig), ELIG)
    check("patients in split file", splits.public_id.nunique(), 1025, SPLITS, 1025)
    rows = []
    for task, col in [("multiclass", "eligible_for_multiclass"), ("binary", "eligible_for_binary")]:
        t = df[df[col]]
        for split in ["train", "val", "test", "all"]:
            s = t if split == "all" else t[t.split == split]
            r = s[s.rna_available]
            rows.append({"task": task, "split": split, "n_visit_pairs": len(s),
                         "n_patients": s.public_id.nunique(),
                         "n_visit_pairs_rna_available": len(r),
                         "n_patients_rna_available": r.public_id.nunique()})
    cohort = pd.DataFrame(rows)
    cohort.to_csv(OUT / "fig1_cohort_task_counts.csv", index=False)

    def get(task, split, col):
        return int(cohort.query("task == @task and split == @split")[col].iloc[0])

    check("multiclass visit pairs", get("multiclass", "all", "n_visit_pairs"), 13451, ELIG, 13451)
    check("multiclass patients", get("multiclass", "all", "n_patients"), 1025, ELIG, 1025)
    check("binary visit pairs", get("binary", "all", "n_visit_pairs"), 12298, ELIG, 12298)
    check("binary patients", get("binary", "all", "n_patients"), 1018, ELIG, 1018)
    check("RNA-available visit pairs (all pairs)", int(df.rna_available.sum()), 9173, PAIRS, 9173)
    check("RNA-available patients (all pairs)", df[df.rna_available].public_id.nunique(), 707, PAIRS, 707)
    excl = df.loc[~df.eligible_for_binary, "vt_disease_response"].value_counts().to_dict()
    check("binary exclusions are exactly V(t) = sCR",
          set(excl) == {"stringent_complete_response"}, True, ELIG)
    check("binary exclusions count", int((~df.eligible_for_binary).sum()), 1153, ELIG)

    # locked split sizes in clean-rerun metrics (identical for every layer)
    for task in ["binary", "multiclass"]:
        for layer in "abcd":
            m = json.load(open(RC / f"{task}_{layer}_xgboost_metrics.json"))
            for split, key in [("train", "n_train"), ("val", "n_val"), ("test", "n_test")]:
                check(f"{task} layer {layer.upper()} {split} rows match metrics JSON",
                      get(task, split, "n_visit_pairs"), int(m[key]),
                      RC / f"{task}_{layer}_xgboost_metrics.json")
            for split, key in [("train", "n_patients_train"), ("val", "n_patients_val"),
                               ("test", "n_patients_test")]:
                check(f"{task} layer {layer.upper()} {split} patients match metrics JSON",
                      get(task, split, "n_patients"), int(m[key]),
                      RC / f"{task}_{layer}_xgboost_metrics.json")

    # binary target definition: improved = next response ranks higher than current
    b = df[df.eligible_for_binary].copy()
    rank_t = b.vt_disease_response.map(RESPONSE_RANK)
    rank_t1 = b.vt1_disease_response.map(RESPONSE_RANK)
    derived = (rank_t1 > rank_t)
    improved = as_bool(b["improved"]) if b["improved"].dtype == object else b["improved"].astype(bool)
    check("binary target = next IMWG category better than current",
          bool((derived == improved).all()), True, PAIRS)
    prev = pd.DataFrame([{"split": s, "n_visit_pairs": int((b.split == s).sum()),
                          "n_improved": int(improved[b.split == s].sum()),
                          "improvement_rate": float(improved[b.split == s].mean())}
                         for s in ["train", "val", "test"]])
    prev.to_csv(OUT / "fig1_binary_improvement_prevalence.csv", index=False)
    check("binary test improvement rate (rounded 3 dp)",
          round(float(prev.query("split == 'test'").improvement_rate.iloc[0]), 3), 0.183, ELIG, "18.3%")

    mc = df[df.eligible_for_multiclass]
    cls = (mc.groupby(["split", "vt1_disease_response"]).size().unstack(0)
           .reindex(list(RESPONSE_RANK)).reset_index().rename(columns={"vt1_disease_response": "next_imwg_response"}))
    cls.to_csv(OUT / "fig1_multiclass_class_counts.csv", index=False)

    # ---------------- RNA timing ----------------
    r = df[df.rna_available].copy()
    # sample collection day (days from CoMMpass index) recovered from the as-of join
    r["rna_sample_day"] = r["vt_days_to_visit"] - r["days_since_rna_sample"]
    samples = r[["public_id", "rna_sample_day"]].drop_duplicates()
    per_pat = samples.groupby("public_id").size()
    ages = r["days_since_rna_sample"]
    d2 = json.load(open(D2_SUMMARY, encoding="utf-8"))
    fresh = json.load(open(FRESH_PRELIM, encoding="utf-8"))["binary"]["staleness_distribution_days"]
    within30 = samples.rna_sample_day.abs() <= 30
    first = samples.groupby("public_id").rna_sample_day.min()
    timing = {
        "definition_rna_age": "days_since_rna_sample at V(t): days from latest RNA sample at or before V(t) to V(t)",
        "definition_sample_day": "RNA sample day relative to CoMMpass index = vt_days_to_visit - days_since_rna_sample",
        "n_rna_visit_pairs": int(len(r)),
        "n_rna_patients": int(r.public_id.nunique()),
        "n_distinct_rna_samples": int(len(samples)),
        "n_patients_1_sample": int((per_pat == 1).sum()),
        "n_patients_ge2_samples": int((per_pat >= 2).sum()),
        "max_samples_per_patient": int(per_pat.max()),
        "n_samples_within_30d_of_index": int(within30.sum()),
        "pct_samples_within_30d_of_index": float(within30.mean() * 100),
        "n_patients_first_sample_within_30d_of_index": int((first.abs() <= 30).sum()),
        "pct_patients_first_sample_within_30d_of_index": float((first.abs() <= 30).mean() * 100),
        "n_patients_first_sample_on_or_before_index": int((first <= 0).sum()),
        "n_patients_first_sample_day_-30_to_0": int(((first >= -30) & (first <= 0)).sum()),
        "rna_age_median_days": float(ages.median()),
        "rna_age_q1_days": float(ages.quantile(0.25)),
        "rna_age_q3_days": float(ages.quantile(0.75)),
        "rna_age_mean_days": float(ages.mean()),
        "rna_age_min_days": float(ages.min()),
        "rna_age_max_days": float(ages.max()),
        "rna_age_median_binary_eligible_days": float(r.loc[r.eligible_for_binary, "days_since_rna_sample"].median()),
    }
    for task, col in [("binary", "eligible_for_binary"), ("multiclass", "eligible_for_multiclass")]:
        rt = r[r[col] & (r.split == "test")]
        timing[f"rna_age_median_{task}_test_days"] = float(rt.days_since_rna_sample.median())
    json.dump(timing, open(OUT / "fig1_rna_timing_summary.json", "w"), indent=2)
    samples.sort_values(["public_id", "rna_sample_day"]).to_csv(OUT / "fig1_rna_samples_distinct.csv", index=False)
    r[["pair_id", "public_id", "split", "eligible_for_binary", "eligible_for_multiclass",
       "vt_days_to_visit", "rna_sample_day", "days_since_rna_sample"]].to_csv(
        OUT / "fig1_rna_age_by_visit_pair.csv", index=False)

    check("distinct RNA samples vs direction2 run_summary", timing["n_distinct_rna_samples"],
          int(d2["rna_z_scoring"]["n_distinct_samples"]), D2_SUMMARY, 751)
    check("RNA-available rows vs direction2 run_summary", timing["n_rna_visit_pairs"],
          int(d2["rna_z_scoring"]["n_rna_rows"]), D2_SUMMARY, 9173)
    check("median RNA age vs PRELIMINARY freshness file", timing["rna_age_median_days"],
          float(fresh["median"]), FRESH_PRELIM, 866)
    check("RNA-age row count vs PRELIMINARY freshness file", timing["n_rna_visit_pairs"],
          int(fresh["count"]), FRESH_PRELIM)
    check("patients with 1 RNA sample", timing["n_patients_1_sample"], None, PAIRS, 669)
    check("patients with >=2 RNA samples", timing["n_patients_ge2_samples"], None, PAIRS, 38)
    check("RNA age IQR (days)", f'{timing["rna_age_q1_days"]:.0f}-{timing["rna_age_q3_days"]:.0f}', None,
          PAIRS, "380-1,465")
    check("% RNA samples within +/-30 d of index", round(timing["pct_samples_within_30d_of_index"], 1),
          None, PAIRS, "~85%")

    # ---------------- survival cohorts ----------------
    surv_rows = []
    for name, p in SURV.items():
        s = pd.read_csv(p).drop_duplicates("public_id")
        surv_rows.append({"analysis": name, "n_patients": len(s), "n_events": int(s.event.sum()),
                          "source": str(p)})
    surv = pd.DataFrame(surv_rows)
    surv.to_csv(OUT / "fig1_survival_cohort_counts.csv", index=False)
    check("survival PFS patients", int(surv.n_patients[0]), 674, SURV["PFS_primary"], 674)
    check("survival PFS events", int(surv.n_events[0]), 459, SURV["PFS_primary"], 459)
    check("survival OS deaths", int(surv.n_events[1]), 247, SURV["OS_secondary"], 247)
    check("survival -30 to 0 d patients", int(surv.n_patients[2]), 624, SURV["PFS_rna_day_-30_to_0"], 624)
    surv_ids = set(pd.read_csv(SURV["PFS_primary"]).public_id)
    rna_ids = set(r.public_id)
    check("survival patients that are in the visit-pair RNA subset", len(surv_ids & rna_ids), None, PAIRS)
    check("RNA visit-pair patients with first sample day <= 0 (survival-eligibility proxy)",
          timing["n_patients_first_sample_on_or_before_index"], None, PAIRS, 674)
    split_of = splits.set_index("public_id").split
    surv_split = pd.Series([split_of.get(i, "not_in_split") for i in surv_ids]).value_counts().to_dict()
    json.dump({"survival_patients_by_next_visit_split": surv_split},
              open(OUT / "fig1_survival_split_overlap.json", "w"), indent=2)

    # ---------------- information layers ----------------
    fs = json.load(open(FEATS))
    order = ["model_a", "model_b", "model_c", "model_d"]
    feat_rows, layer_rows, prev_set = [], [], set()
    for key in order:
        layer = key[-1]
        cur = fs[key]
        added = [f for f in cur if f not in prev_set]
        check(f"layer {layer.upper()} is a strict superset of previous", prev_set <= set(cur), True, FEATS)
        meta = json.load(open(PRE / f"model_{layer}_metadata.json"))
        buckets = {k: len(v) for k, v in meta["column_buckets"].items()}
        outn = meta["output_feature_names"]
        layer_rows.append({
            "layer": layer.upper(), "layer_name": LAYER_NAMES[layer],
            "source_input_features": len(cur), "added_source_features": len(added),
            "derived_indicator_added": meta.get("rna_indicator_added") or "",
            "preprocessor_input_columns": len(meta["input_columns"]),
            "fitted_model_input_columns": meta["output_feature_count"],
            "fitted_missing_indicator_columns": sum(n.endswith("_missing") for n in outn),
            "high_cardinality_columns_capped": len(meta.get("high_cardinality_columns_capped") or []),
            **{f"bucket_{k}": v for k, v in buckets.items()},
        })
        for f in added:
            feat_rows.append({"feature": f, "layer_added": layer.upper()})
        prev_set = set(cur)
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(OUT / "fig1_information_layers.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(OUT / "fig1_features_by_layer_added.csv", index=False)
    for layer, n_src, n_fit in zip("ABCD", [78, 115, 143, 194], [148, 185, 302, 354]):
        row = layers[layers.layer == layer].iloc[0]
        check(f"layer {layer} source input features", int(row.source_input_features), n_src, FEATS, n_src)
        check(f"layer {layer} fitted model-input columns", int(row.fitted_model_input_columns), n_fit,
              PRE / f"model_{layer.lower()}_metadata.json",
              "195 (stated for D)" if layer == "D" else None)
    check("layer D preprocessor input columns = 194 source + rna_available",
          int(layers[layers.layer == "D"].preprocessor_input_columns.iloc[0]), 195,
          PRE / "model_d_metadata.json", "195")
    d_added = [f["feature"] for f in feat_rows if f["layer_added"] == "D"]
    check("layer D adds days_since_rna_sample + 50 pathways",
          ("days_since_rna_sample" in d_added) and len(d_added) == 51, True, FEATS, "+51")

    # ---------------- provenance ----------------
    inputs = [PAIRS, ELIG, SPLITS, FEATS, D2_SUMMARY, FRESH_PRELIM, *SURV.values(),
              *[PRE / f"model_{l}_metadata.json" for l in "abcd"]]
    json.dump({"script": "figures/fig1_extract_data.py",
               "inputs_sha256": {str(p): sha(p) for p in inputs},
               "outputs": sorted(p.name for p in OUT.iterdir())},
              open(OUT / "fig1_provenance.json", "w"), indent=2)
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "fig1_verification_checks.csv", index=False)
    failed = ck[ck.passed == False]  # noqa: E712
    print(ck.to_string(index=False, max_colwidth=60))
    if len(failed):
        print(f"\n{len(failed)} locked check(s) FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
