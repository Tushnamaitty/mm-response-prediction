"""
Build Table 1 (cohort characteristics) and Supplementary Table S6 (full Figure 2 performance), read-only.

Run from repo root:
    python figures/supplementary_tables/build_table1_and_S6.py

Table 1 is a descriptive summary of locked clinical inputs (aggregate counts only; no patient-level rows, no
identifiers). Table S6 reports the locked held-out test results of all 24 task x information layer x algorithm
models with their locked 95% patient-bootstrap CIs; Figure 2C paired differences are deliberately NOT included
(their provenance is pending verification). Writes separate verification and provenance files and leaves
Tables S1-S5 and their files untouched.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("figures/supplementary_tables")
SRC = {
    "pairs": Path("data/clinical/visit_pairs_with_rna.csv"),
    "subject": Path("data/clinical/subject_deid.csv"),
    "splits": Path("data/splits/patient_split_assignments.csv"),
    "elig": Path("data/splits/task_eligibility.csv"),
    "fig1_counts": Path("figures/data/Fig1/fig1_cohort_task_counts.csv"),
    "fig1_surv": Path("figures/data/Fig1/fig1_survival_cohort_counts.csv"),
    "surv_oof": Path("artifacts/results_clean_rerun/survival_incremental/primary_pfs_oof_predictions.csv"),
    "cis": Path("artifacts/results_clean_rerun/uncertainty/main_model_cis.csv"),
    "cis_script": Path("modeling/regenerate_main_model_cis_clean_rerun.py"),
}
LOCKED_PAIRS_SHA = "7697743bbddb405edc61213c5bfa690eb73154df6b20b6f4f7a1cb5224ea5a58"
RESP = ["progressive_disease", "stable_disease", "partial_response", "very_good_partial_response", "complete_response",
        "stringent_complete_response"]
RESP_LAB = {"progressive_disease": "Progressive disease (PD)", "stable_disease": "Stable disease (SD)",
            "partial_response": "Partial response (PR)", "very_good_partial_response": "Very good partial response (VGPR)",
            "complete_response": "Complete response (CR)", "stringent_complete_response": "Stringent complete response (sCR)"}
ALGO = {"logistic": "Logistic Regression", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
LAYER = {"a": "A: current clinical state", "b": "B: A + longitudinal history", "c": "C: B + treatment context",
         "d": "D: C + RNA pathway scores"}
checks = []


def check(table, name, ok, observed="", expected=""):
    checks.append({"table": table, "check": name, "passed": bool(ok), "observed": str(observed)[:160],
                   "expected": str(expected)[:160]})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md_table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
def table1():
    T = "Table 1"
    check(T, "visit_pairs_with_rna.csv = locked file (SHA-256)", sha(SRC["pairs"]) == LOCKED_PAIRS_SHA)
    m = pd.read_csv(SRC["pairs"], low_memory=False)
    e = pd.read_csv(SRC["elig"])
    m = m.merge(e[["pair_id", "split", "eligible_for_binary"]], on="pair_id", how="left", validate="one_to_one")
    m["eligible_for_binary"] = m["eligible_for_binary"].astype(str).str.lower() == "true"
    m["rna"] = m["days_since_rna_sample"].notna()
    subj = pd.read_csv(SRC["subject"], usecols=["public_id", "gender", "age_at_index", "country_of_residence_at_enrollment"])
    sp = pd.read_csv(SRC["splits"])
    surv = set(pd.read_csv(SRC["surv_oof"]).public_id)

    first = m.sort_values(["public_id", "vt_days_to_visit", "pair_id"]).drop_duplicates("public_id")
    pat = (first[["public_id", "age_at_diagnosis", "iss_stage", "current_line_number"]]
           .rename(columns={"current_line_number": "line_first_pair"})
           .merge(m.groupby("public_id").agg(max_line=("current_line_number", "max"), n_pairs=("pair_id", "size"),
                                             any_rna=("rna", "any")).reset_index(), on="public_id")
           .merge(sp, on="public_id", how="left", validate="one_to_one")
           .merge(subj, on="public_id", how="left", validate="one_to_one"))
    pat["survival_cohort"] = pat.public_id.isin(surv)
    for c in ("age_at_diagnosis", "iss_stage"):
        check(T, f"{c} constant within patient", int(m.groupby("public_id")[c].nunique().max()) <= 1)
    both = pat.dropna(subset=["age_at_diagnosis", "age_at_index"])
    check(T, "age at diagnosis = age at index where both recorded", bool((both.age_at_diagnosis == both.age_at_index).all()),
          len(both))
    pat["age"] = pat.age_at_diagnosis.fillna(pat.age_at_index)
    n_age_from_index = int((pat.age_at_diagnosis.isna() & pat.age_at_index.notna()).sum())
    check(T, "age taken from index for 2 patients without age at diagnosis", n_age_from_index == 2, n_age_from_index)

    cols = [("All patients", pat.index == pat.index), ("Training", pat.split == "train"), ("Validation", pat.split == "val"),
            ("Test", pat.split == "test"), ("RNA available (≥1 pair)", pat.any_rna), ("No RNA", ~pat.any_rna),
            ("Survival cohort", pat.survival_cohort)]
    rows = []

    def add(section, label, fn, kind="count"):
        r = {"section": section, "characteristic": label, "kind": kind}
        for name, mask in cols:
            r[name] = fn(pat[mask])
        rows.append(r)

    n = lambda g: len(g)  # noqa: E731
    def npct(cond):
        return lambda g: f"{int(cond(g).sum())} ({100 * cond(g).sum() / len(g):.1f}%)"

    def med(col):
        def _f(g):
            x = g[col].dropna()
            return f"{x.median():.0f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})"
        return _f

    add("Patients", "Patients, n", n)
    add("Demographics", "Age at diagnosis, years, median (IQR)", med("age"), "median")
    add("Demographics", "Age missing, n (%)", npct(lambda g: g.age.isna()))
    add("Demographics", "Male, n (%)", npct(lambda g: g.gender == "male"))
    add("Demographics", "Female, n (%)", npct(lambda g: g.gender == "female"))
    add("Demographics", "Sex missing, n (%)", npct(lambda g: g.gender.isna()))
    for st, lab in ((1, "I"), (2, "II"), (3, "III")):
        add("International Staging System (ISS)", f"ISS {lab}, n (%)", npct(lambda g, st=st: g.iss_stage == st))
    add("International Staging System (ISS)", "ISS missing, n (%)", npct(lambda g: g.iss_stage.isna()))
    add("Treatment line", "Line at first visit pair: 1, n (%)", npct(lambda g: g.line_first_pair == 1))
    add("Treatment line", "Line at first visit pair: ≥2, n (%)", npct(lambda g: g.line_first_pair >= 2))
    add("Treatment line", "Line at first visit pair missing, n (%)", npct(lambda g: g.line_first_pair.isna()))
    for lo, hi, lab in ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 99, "≥4")):
        add("Treatment line", f"Highest line reached during follow-up: {lab}, n (%)",
            npct(lambda g, lo=lo, hi=hi: (g.max_line >= lo) & (g.max_line <= hi)))
    add("Follow-up", "Visit pairs per patient, median (IQR)", med("n_pairs"), "median")
    for c, lab in (("united_states", "United States"), ("italy", "Italy"), ("spain", "Spain"), ("canada", "Canada")):
        add("Country of enrollment", f"{lab}, n (%)", npct(lambda g, c=c: g.country_of_residence_at_enrollment == c))
    add("RNA", "Patients with ≥1 RNA-available visit pair, n (%)", npct(lambda g: g.any_rna))

    # visit-pair level (patient columns map to their pairs)
    split_of = pat.set_index("public_id").split
    m["split_p"] = m.public_id.map(split_of)
    m["any_rna_p"] = m.public_id.map(pat.set_index("public_id").any_rna)
    m["surv_p"] = m.public_id.isin(surv)
    pcols = [("All patients", m.index == m.index), ("Training", m.split_p == "train"), ("Validation", m.split_p == "val"),
             ("Test", m.split_p == "test"), ("RNA available (≥1 pair)", m.any_rna_p), ("No RNA", ~m.any_rna_p),
             ("Survival cohort", m.surv_p)]

    def addp(section, label, fn, kind="count"):
        r = {"section": section, "characteristic": label, "kind": kind}
        for name, mask in pcols:
            r[name] = fn(m[mask])
        rows.append(r)

    def pnp(cond, denom=None):
        return lambda g: (f"{int(cond(g).sum()):,} ({100 * cond(g).sum() / (len(g) if denom is None else denom(g).sum()):.1f}%)")

    addp("Visit pairs", "Six-class visit pairs, n", lambda g: f"{len(g):,}")
    addp("Visit pairs", "Binary-eligible visit pairs, n (% of pairs)", pnp(lambda g: g.eligible_for_binary))
    addp("Visit pairs", "RNA-available visit pairs, n (% of pairs)", pnp(lambda g: g.rna))
    addp("Visit pairs", "Improvement at V(t+1), n (% of binary-eligible)",
         pnp(lambda g: g.eligible_for_binary & (g.improved.astype(int) == 1), lambda g: g.eligible_for_binary))
    for c in RESP:
        addp("IMWG response at current visit V(t), n (% of pairs)", RESP_LAB[c], pnp(lambda g, c=c: g.vt_disease_response == c))
    for c in RESP:
        addp("IMWG response at next visit V(t+1), n (% of pairs)", RESP_LAB[c], pnp(lambda g, c=c: g.exact_next_response == c))
    df = pd.DataFrame(rows)

    # verification against Figure 1 verified counts
    f1 = pd.read_csv(SRC["fig1_counts"])
    get = lambda task, split, col: int(f1[(f1.task == task) & (f1.split == split)][col].iloc[0])  # noqa: E731
    pc = {"All patients": len(pat), "Training": int((pat.split == "train").sum()), "Validation": int((pat.split == "val").sum()),
          "Test": int((pat.split == "test").sum()), "RNA available (≥1 pair)": int(pat.any_rna.sum()),
          "No RNA": int((~pat.any_rna).sum()), "Survival cohort": int(pat.survival_cohort.sum())}
    check(T, "patients by split = Figure 1 (1,025 / 716 / 154 / 155)",
          (pc["All patients"], pc["Training"], pc["Validation"], pc["Test"]) ==
          (get("multiclass", "all", "n_patients"), get("multiclass", "train", "n_patients"), get("multiclass", "val", "n_patients"),
           get("multiclass", "test", "n_patients")), pc)
    check(T, "RNA patients = 707; RNA + no-RNA = 1,025", pc["RNA available (≥1 pair)"] == 707 and
          pc["RNA available (≥1 pair)"] + pc["No RNA"] == 1025)
    check(T, "survival cohort = 674 and within RNA patients", pc["Survival cohort"] == 674 and
          bool(pat[pat.survival_cohort].any_rna.all()))
    check(T, "pairs: 13,451 six-class; 12,298 binary; 9,173 RNA",
          (len(m), int(m.eligible_for_binary.sum()), int(m.rna.sum())) == (13451, 12298, 9173))
    check(T, "test pairs 2,079; binary test 1,930; test improvement 353",
          (int((m.split_p == "test").sum()), int(((m.split_p == "test") & m.eligible_for_binary).sum()),
           int(((m.split_p == "test") & m.eligible_for_binary & (m.improved.astype(int) == 1)).sum())) ==
          (2079, 1930, 353))
    check(T, "pair split = patient split for every pair", bool((m.split == m.split_p).all()))
    check(T, "binary exclusions = sCR at V(t)", bool((~m.eligible_for_binary == (m.vt_disease_response == "stringent_complete_response")).all()))
    for name, mask in cols:
        g = pat[mask]
        check(T, f"{name}: sex categories sum to N", int((g.gender == "male").sum() + (g.gender == "female").sum() + g.gender.isna().sum()) == len(g))
        check(T, f"{name}: ISS categories sum to N", int(g.iss_stage.isin([1, 2, 3]).sum() + g.iss_stage.isna().sum()) == len(g))
        check(T, f"{name}: highest-line categories sum to N", int(g.max_line.notna().sum()) == len(g))
        check(T, f"{name}: four countries sum to N", int(g.country_of_residence_at_enrollment.isin(
            ["united_states", "italy", "spain", "canada"]).sum()) == len(g))
    for name, mask in pcols:
        g = m[mask]
        check(T, f"{name}: V(t) and V(t+1) categories sum to pairs", int(g.vt_disease_response.isin(RESP).sum()) == len(g) and
              int(g.exact_next_response.isin(RESP).sum()) == len(g))
    s = pd.read_csv(SRC["fig1_surv"]).set_index("analysis")
    check(T, "survival cohort = Figure 1 survival count", pc["Survival cohort"] == int(s.loc["PFS_primary", "n_patients"]))
    # small-cell check: smallest count printed
    counts = []
    for _, r in df.iterrows():
        for name, _ in cols:
            v = str(r[name])
            if r.kind == "count" and "(" in v:
                counts.append(int(v.split(" ")[0].replace(",", "")))
    df.to_csv(OUT / "Table_1_cohort_characteristics.csv", index=False)
    colnames = [c for c, _ in cols]
    hdr = [f"{c} (n = {pc[c]:,})" for c in colnames]
    md = ["# Table 1. Characteristics of the MMRF CoMMpass cohort used for next-visit prediction", "",
          "Patient-level characteristics are per patient; visit-pair rows count consecutive visit pairs. Training, validation "
          "and test are the patient-grouped partitions used for all next-visit models. RNA available = at least one visit pair "
          "with an RNA sample collected on or before V(t). Survival cohort = patients with one RNA sample collected on or "
          "before the start of first treatment (CoMMpass index date), used in Figures 4–5. Age is age at diagnosis (identical "
          "to age at the CoMMpass index date where both are recorded; age at index was used for " + str(n_age_from_index) + " patients without a recorded age at diagnosis). Treatment line varies over follow-up and is summarised "
          "at the patient's first visit pair and as the highest line reached during follow-up. Percentages use the column "
          "denominator unless stated. IQR, interquartile range; IMWG, International Myeloma Working Group.", ""]
    for sec in df.section.drop_duplicates():
        sub = df[df.section == sec]
        md += [f"**{sec}**", "", md_table(sub, ["characteristic"] + colnames, ["Characteristic"] + hdr), ""]
    md += [f"Smallest cell count shown: {min(counts)}. Sources: `data/clinical/visit_pairs_with_rna.csv` (SHA-256 matches "
           "the locked file), `data/clinical/subject_deid.csv`, `data/splits/`; counts cross-checked against Figure 1 data."]
    (OUT / "Table_1_cohort_characteristics.md").write_text("\n".join(md), encoding="utf-8")
    return min(counts)


# ------------------------------------------------------------------------------------------------
def table_s6():
    T = "S6"
    ci = pd.read_csv(SRC["cis"])
    code = SRC["cis_script"].read_text(encoding="utf-8")
    check(T, "locked CI script uses 1,000 patient-bootstrap resamples, seed 42",
          "N_BOOTSTRAP = 1000" in code and "RANDOM_SEED = 42" in code)
    rows = []
    for task in ("binary", "multiclass"):
        mets = ["auprc", "auroc"] if task == "binary" else ["macro_f1"]
        for stage in "abcd":
            for algo in ("logistic", "lightgbm", "xgboost"):
                j = json.load(open(f"artifacts/results_clean_rerun/{task}_{stage}_{algo}_metrics.json"))
                r = {"task": "Binary improvement" if task == "binary" else "Six-class IMWG response", "task_key": task,
                     "information_layer": LAYER[stage], "layer": stage.upper(), "algorithm": ALGO[algo],
                     "n_test_visit_pairs": j["n_test"], "n_test_patients": j["n_patients_test"]}
                for met in mets:
                    c = ci[(ci.task == task) & (ci.model_stage == stage) & (ci.algorithm == algo) & (ci.metric == met)]
                    check(T, f"{task}/{stage}/{algo}/{met}: one locked CI row", len(c) == 1)
                    c = c.iloc[0]
                    tm = j["test_metrics_PRELIMINARY"][met]
                    check(T, f"{task}/{stage}/{algo}/{met}: CI-file point = metrics JSON", abs(c.point_estimate - tm) < 1e-9,
                          c.point_estimate, tm)
                    check(T, f"{task}/{stage}/{algo}/{met}: CI brackets estimate", c.ci_lower <= c.point_estimate <= c.ci_upper)
                    r.update({f"test_{met}": c.point_estimate, f"test_{met}_ci_lower": c.ci_lower, f"test_{met}_ci_upper": c.ci_upper})
                    r[f"validation_{met}"] = j["val_metrics"][met]
                if task == "binary":
                    r["test_brier"] = j["test_metrics_PRELIMINARY"]["brier_score"]
                    r["decision_threshold_from_validation"] = j["test_metrics_PRELIMINARY"]["threshold_used"]
                rows.append(r)
    df = pd.DataFrame(rows)
    check(T, "24 models (2 tasks x 4 layers x 3 algorithms)", len(df) == 24)
    check(T, "binary test 1,930 pairs / 155 patients; six-class 2,079 / 155",
          set(zip(df.task_key, df.n_test_visit_pairs, df.n_test_patients)) == {("binary", 1930, 155), ("multiclass", 2079, 155)})
    df["ci_method"] = "95% patient-level bootstrap percentile CI, 1,000 resamples of test patients (seed 42); unadjusted"
    df["note"] = "Figure 2C paired layer differences not included (bootstrap provenance pending verification)"
    df.to_csv(OUT / "Table_S6_full_figure2_performance.csv", index=False)
    fmt = lambda v, lo, hi: f"{v:.3f} ({lo:.3f}–{hi:.3f})"  # noqa: E731
    b = df[df.task_key == "binary"].copy()
    b["auprc"] = [fmt(*x) for x in zip(b.test_auprc, b.test_auprc_ci_lower, b.test_auprc_ci_upper)]
    b["auroc"] = [fmt(*x) for x in zip(b.test_auroc, b.test_auroc_ci_lower, b.test_auroc_ci_upper)]
    b["val"] = b.validation_auprc.map(lambda v: f"{v:.3f}")
    b["brier"] = b.test_brier.map(lambda v: f"{v:.4f}")
    mc = df[df.task_key == "multiclass"].copy()
    mc["f1"] = [fmt(*x) for x in zip(mc.test_macro_f1, mc.test_macro_f1_ci_lower, mc.test_macro_f1_ci_upper)]
    mc["val"] = mc.validation_macro_f1.map(lambda v: f"{v:.3f}")
    md = ["# Supplementary Table S6. Held-out test performance of all 24 next-visit models (Figure 2)", "",
          "Locked clean-rerun models; held-out test partition (155 patients), evaluated once. Information layers A–D as in "
          "Figure 1. Binary improvement (1,930 visit pairs; 353 improvement events, 18.3%) and six-class IMWG response (2,079 "
          "visit pairs) use different visit-pair populations and metrics and are not directly comparable. AUPRC, AUROC and "
          "macro-F1: higher is better; Brier score: lower is better (point estimate only). 95% patient-level bootstrap CIs "
          "(1,000 resamples of test patients; locked `artifacts/results_clean_rerun/uncertainty/main_model_cis.csv`), "
          "unadjusted. Validation-set values (point estimates) are shown for transparency; hyperparameters were tuned by "
          "cross-validation within training patients. No algorithm was selected as best; XGBoost is the reporting reference "
          "algorithm. Paired layer-to-layer differences (Figure 2C) are not reported here pending verification of their "
          "bootstrap provenance.", "",
          "## Binary improvement", "",
          md_table(b, ["information_layer", "algorithm", "auprc", "auroc", "brier", "val"],
                   ["Information layer", "Algorithm", "Test AUPRC (95% CI)", "Test AUROC (95% CI)", "Test Brier score",
                    "Validation AUPRC"]), "",
          "## Six-class IMWG response", "",
          md_table(mc, ["information_layer", "algorithm", "f1", "val"],
                   ["Information layer", "Algorithm", "Test macro-F1 (95% CI)", "Validation macro-F1"]), ""]
    (OUT / "Table_S6_full_figure2_performance.md").write_text("\n".join(md), encoding="utf-8")


def main():
    smallest = table1()
    table_s6()
    for p in (OUT / "Table_1_cohort_characteristics.csv", OUT / "Table_1_cohort_characteristics.md",
              OUT / "Table_S6_full_figure2_performance.csv", OUT / "Table_S6_full_figure2_performance.md"):
        txt = p.read_text(encoding="utf-8")
        check("all", f"no patient identifiers in {p.name}", "mmrf_" not in txt and "public_id" not in txt)
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "table1_S6_verification.csv", index=False)
    json.dump({"script": "figures/supplementary_tables/build_table1_and_S6.py", "inputs_sha256": {str(p): sha(p) for p in SRC.values()},
               "outputs": ["Table_1_cohort_characteristics.csv", "Table_1_cohort_characteristics.md",
                           "Table_S6_full_figure2_performance.csv", "Table_S6_full_figure2_performance.md"],
               "smallest_cell_count_table1": smallest},
              open(OUT / "table1_S6_provenance.json", "w"), indent=2)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed; smallest Table 1 cell = {smallest}")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
