"""
Direction 4 -- STEP 1 ONLY: trajectory construction and support audit.
(No predictive model, no molecular association, no RNA values are used here.)

Trajectory state of a visit pair (row) = change in the IMWG response from the PREVIOUS row's
vt_disease_response to the CURRENT row's vt_disease_response, using the order
PD < SD < PR < VGPR < CR < sCR (RESPONSE_RANK, data_pipeline/build_visit_pairs.py):
    improving   current rank > previous rank
    stable      current rank == previous rank
    worsening   current rank < previous rank
    no_history  first row of the patient (no previous row)
    ambiguous   the audited continuity rows (previous row's Vt+1 label != this row's Vt label on the same
                day): EXCLUDED from trajectory contrasts and accounted for explicitly (ambiguous_rows.csv).
                They are NOT removed from the frozen cohort (predictive rows are untouched).

Ordering: rows are sorted within patient by vt_days_to_visit; vt1_days_to_visit is used ONLY to VERIFY the
ordering/continuity (the previous row's vt1 day must equal this row's vt day; sorting by vt1 day must give the
same order). The state itself reads only Vt-side fields of rows up to and including the current one
(vt_disease_response, vt_days_to_visit; both are Model-C/B inputs); this is proven by a truncation test that
recomputes every state from rows <= Vt only. The previous row's vt1_* fields (which equal the CURRENT visit, i.e.
past information) are used only to flag the ambiguous rows.

RNA availability follows the existing project definition (days_since_rna_sample notna, as in
explainability/direction3_common.load_master). RNA VALUES are never used (pathway columns are dropped right after
loading); RNA age (days_since_rna_sample) is used only to COUNT rows in the descriptive sensitivity universes.
Outcomes (improved / exact_next_response) are used ONLY to count events / classes for the support audit.

Populations
    all                 every visit pair (multiclass-type; class counts)
    all_binary_eligible eligible_for_binary rows (binary-type; events) -- reported only
    rna_multiclass      RNA-available rows (class counts)
    rna_binary          RNA-available AND eligible_for_binary rows (events / non-events)

Previously audited counts are REPRODUCED and asserted (the run fails loudly on any difference):
    all             improving 2057, stable 8752, worsening 1611, no_history 1025, ambiguous 6
    rna_multiclass  improving 1421, stable 5957, worsening 1100, no_history 690
    rna_binary      improving 1237, stable 5242, worsening 1100, no_history 682

Outputs ONLY under artifacts/results_clean_rerun/direction4/support/  (smoke: direction4/_smoke/support/).
Existing results are never overwritten (--force is allowed only together with --smoke).

Run from the repo root:
    python -m explainability.direction4_support
    python -m explainability.direction4_support --smoke [--force]
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import pandas as pd

from explainability.direction2_common import (
    RESPONSE_RANK, STALENESS_COL, expect, load_feature_sets, require, sha256_file, write_csv, write_json,
)
from explainability.direction3_common import (
    MULTICLASS_CLASSES, MULTICLASS_MIN_PER_CLASS, TREATMENT_COLS, assert_predictors_clean, load_master,
    pathway_columns, tier_of, write_output_manifest,
)

OUT_ROOT = "artifacts/results_clean_rerun/direction4"
LABEL = "DIRECTION 4 STEP 1: TRAJECTORY CONSTRUCTION AND SUPPORT AUDIT (no models, no RNA values, no association)"

STATES = ["improving", "stable", "worsening", "no_history"]
AMBIG = "ambiguous"
ALL_STATES = STATES + [AMBIG]
HISTORY_STATES = ["improving", "stable", "worsening"]
TREATMENTS_AUDITED = ["pi", "imid", "steroid", "cd38", "chemo"]

# ---- previously audited counts (asserted, never written as outputs) ----
EXPECTED_STATE_COUNTS = {
    "all": {"improving": 2057, "stable": 8752, "worsening": 1611, "no_history": 1025, "ambiguous": 6},
    "rna_multiclass": {"improving": 1421, "stable": 5957, "worsening": 1100, "no_history": 690},
    "rna_binary": {"improving": 1237, "stable": 5242, "worsening": 1100, "no_history": 682},
}
EXPECTED_AMBIGUOUS_IN_RNA = {"rna_multiclass": 5, "rna_binary": 1}      # from the same audit run
EXPECTED_POP_ROWS = {"all": 13451, "all_binary_eligible": 12298, "rna_multiclass": 9173, "rna_binary": 8262}
EXPECTED_CONTINUITY = {"n_rows": 13451, "n_patients": 1025, "n_first_rows": 1025, "n_links": 12426,
                       "n_day_mismatch_links": 0, "n_label_conflict_links": 6, "n_visit_id_mismatch_links": 6,
                       "n_duplicate_patient_day": 0, "n_duplicate_patient_visit": 0}
# informational design expectation of the support rule (Direction 4 audit); NOT asserted
DESIGN_EXPECTED_TIER = {"partA_trainval": {"improving": 2, "stable": 1, "worsening": 1, "no_history": 1},
                        "pooled": {"improving": 1, "stable": 1, "worsening": 1, "no_history": 1}}


def get_dirs4(smoke=False):
    root = f"{OUT_ROOT}/_smoke" if smoke else OUT_ROOT
    return {"root": root, "support": f"{root}/support"}


def guard(marker_path, smoke, force):
    if force:
        require(smoke, "--force is only allowed together with --smoke (real outputs are never overwritten)")
    if os.path.exists(marker_path) and not force:
        raise SystemExit(f"[direction4] {marker_path} already exists: existing outputs are never overwritten. "
                         f"Delete the folder deliberately to rerun (or use --smoke --force for smoke runs).")


# ---------------------------------------------------------------------------
# Trajectory construction
# ---------------------------------------------------------------------------
def build_trajectory(m):
    """Order rows within patient, verify ordering/continuity, assign states. Returns (frame, continuity QA dict)."""
    cols = ["pair_id", "public_id", "vt_study_visit", "vt_days_to_visit", "vt_disease_response",
            "vt1_study_visit", "vt1_days_to_visit", "vt1_disease_response", "improved", "exact_next_response",
            "split", "rna_avail", "eligible_for_binary", STALENESS_COL] + [TREATMENT_COLS[t] for t in TREATMENTS_AUDITED]
    d = m[cols].copy()
    d = d.sort_values(["public_id", "vt_days_to_visit", "pair_id"]).reset_index(drop=True)
    qa = {"n_rows": int(len(d)), "n_patients": int(d["public_id"].nunique())}

    # ---- ordering checks (vt_days_to_visit defines the order; vt1_days_to_visit verifies it) ----
    qa["n_duplicate_patient_day"] = int(d.duplicated(["public_id", "vt_days_to_visit"]).sum())
    qa["n_duplicate_patient_visit"] = int(d.duplicated(["public_id", "vt_study_visit"]).sum())
    require(qa["n_duplicate_patient_day"] == 0, "duplicate (patient, vt_days_to_visit) rows")
    require(qa["n_duplicate_patient_visit"] == 0, "duplicate (patient, vt_study_visit) rows")
    require(bool((d["vt1_days_to_visit"] > d["vt_days_to_visit"]).all()), "a row has vt1_days_to_visit <= vt_days_to_visit")
    d_by_vt1 = d.sort_values(["public_id", "vt1_days_to_visit", "pair_id"])
    require(np.array_equal(d_by_vt1["pair_id"].to_numpy(), d["pair_id"].to_numpy()),
            "ordering by vt1_days_to_visit disagrees with ordering by vt_days_to_visit")
    qa["order_by_vt1_identical_to_order_by_vt"] = True

    g = d.groupby("public_id", sort=False)
    d["visit_index"] = g.cumcount()
    for c in ["vt_disease_response", "vt_days_to_visit", "vt1_days_to_visit", "vt1_disease_response", "vt1_study_visit"]:
        d["prev_" + c] = g[c].shift(1)
    links = (d["visit_index"] > 0).to_numpy()
    day_mismatch = links & (d["prev_vt1_days_to_visit"] != d["vt_days_to_visit"]).to_numpy()
    label_conflict = links & (d["prev_vt1_disease_response"] != d["vt_disease_response"]).to_numpy()
    visit_mismatch = links & (d["prev_vt1_study_visit"] != d["vt_study_visit"]).to_numpy()
    qa.update({"n_first_rows": int((~links).sum()), "n_links": int(links.sum()),
               "n_day_mismatch_links": int(day_mismatch.sum()), "n_label_conflict_links": int(label_conflict.sum()),
               "n_visit_id_mismatch_links": int(visit_mismatch.sum())})
    require(qa["n_day_mismatch_links"] == 0, "a previous row's Vt+1 day differs from the next row's Vt day (visit skipped)")
    require(np.array_equal(label_conflict, visit_mismatch),
            "rows with a label conflict and rows with a visit-id mismatch are not the same set")
    qa["label_conflict_set_equals_visit_id_mismatch_set"] = True

    d["rank_vt"] = d["vt_disease_response"].map(RESPONSE_RANK)
    require(d["rank_vt"].notna().all(), "unmapped vt_disease_response label")
    d["prior_rank"] = g["rank_vt"].shift(1)
    d["prior_response"] = d["prev_vt_disease_response"]
    d["current_response"] = d["vt_disease_response"]
    d["rank_change"] = d["rank_vt"] - d["prior_rank"]
    d["gap_prev_days"] = d["vt_days_to_visit"] - d["prev_vt_days_to_visit"]
    d["continuity_label_conflict"] = label_conflict
    nxt = pd.Series(label_conflict, index=d.index).groupby(d["public_id"]).shift(-1)
    d["prior_pair_target_conflict"] = nxt.fillna(False).astype(bool).to_numpy()   # retained (only its TARGET label is affected)

    state = np.where(~links, "no_history",
                     np.where(label_conflict, AMBIG,
                              np.where(d["rank_vt"].to_numpy() > d["prior_rank"].to_numpy(), "improving",
                                       np.where(d["rank_vt"].to_numpy() < d["prior_rank"].to_numpy(), "worsening", "stable"))))
    d["trajectory_state"] = state
    require(set(d["trajectory_state"].unique()) <= set(ALL_STATES), "unexpected trajectory state")
    return d, qa


def truncation_leakage_test(d, max_patients=None):
    """Recompute every state from ONLY the Vt rows up to and including the current row; must equal the assigned
    state for every non-ambiguous row (ambiguous rows are compared on their Vt-only state but excluded)."""
    rank = d["rank_vt"].to_numpy()
    state = d["trajectory_state"].to_numpy()
    n_checked, n_bad = 0, 0
    for pos, (_, idx) in enumerate(d.groupby("public_id", sort=False).indices.items()):
        if max_patients is not None and pos >= max_patients:
            break
        for i in range(len(idx)):
            hist = rank[idx[: i + 1]]                          # rows <= current only
            if len(hist) < 2:
                s = "no_history"
            else:
                s = "improving" if hist[-1] > hist[-2] else ("worsening" if hist[-1] < hist[-2] else "stable")
            if state[idx[i]] == AMBIG:
                continue
            n_checked += 1
            n_bad += int(s != state[idx[i]])
    require(n_bad == 0, f"truncation leakage test failed on {n_bad} rows")
    return {"rows_checked": n_checked, "mismatches": n_bad, "patients_limit": max_patients}


# ---------------------------------------------------------------------------
# Count tables
# ---------------------------------------------------------------------------
def populations(d):
    return {"all": (d, "multiclass"),
            "all_binary_eligible": (d[d["eligible_for_binary"]], "binary"),
            "rna_multiclass": (d[d["rna_avail"]], "multiclass"),
            "rna_binary": (d[d["rna_avail"] & d["eligible_for_binary"]], "binary")}


def scopes(df):
    yield "overall", df
    for s in ["train", "val", "test"]:
        yield s, df[df["split"] == s]
    yield "trainval", df[df["split"] != "test"]


def state_counts_table(pops):
    rows = []
    for pname, (df, kind) in pops.items():
        for st in ALL_STATES:
            sub = df[df["trajectory_state"] == st]
            for scope, ss in scopes(sub):
                r = {"population": pname, "population_type": kind, "state": st, "scope": scope,
                     "rows": int(len(ss)), "patients": int(ss["public_id"].nunique())}
                if kind == "binary":
                    r["events"], r["non_events"] = int(ss["improved"].sum()), int((1 - ss["improved"]).sum())
                else:
                    vc = ss["exact_next_response"].value_counts().reindex(MULTICLASS_CLASSES, fill_value=0)
                    for c in MULTICLASS_CLASSES:
                        r[f"n_{c}"] = int(vc[c])
                rows.append(r)
    return pd.DataFrame(rows)


def reproduction_table(pops):
    """Observed vs previously audited counts (rows by state); fails loudly on any difference."""
    rows = []
    for pname, exp in EXPECTED_STATE_COUNTS.items():
        df = pops[pname][0]
        obs = df["trajectory_state"].value_counts()
        for st, e in exp.items():
            o = int(obs.get(st, 0))
            rows.append({"population": pname, "state": st, "observed": o, "audited": e, "match": o == e})
    for pname, e in EXPECTED_AMBIGUOUS_IN_RNA.items():
        o = int((pops[pname][0]["trajectory_state"] == AMBIG).sum())
        rows.append({"population": pname, "state": AMBIG, "observed": o, "audited": e, "match": o == e})
    for pname, e in EXPECTED_POP_ROWS.items():
        o = int(len(pops[pname][0]))
        rows.append({"population": pname, "state": "(all rows)", "observed": o, "audited": e, "match": o == e})
    t = pd.DataFrame(rows)
    bad = t[~t["match"]]
    require(bad.empty, "previously audited counts NOT reproduced:\n" + bad.to_string(index=False))
    return t


def ambiguous_tables(d, pops):
    amb = d[d["trajectory_state"] == AMBIG]
    require(len(amb) == EXPECTED_STATE_COUNTS["all"][AMBIG], "unexpected number of ambiguous rows")
    cur = amb[["pair_id", "public_id", "vt_study_visit", "vt_days_to_visit", "prior_response", "current_response",
               "prev_vt1_disease_response", "prev_vt1_study_visit", "split", "rna_avail", "eligible_for_binary"]].copy()
    cur = cur.rename(columns={"prev_vt1_disease_response": "previous_pair_vt1_label_conflicting",
                              "prev_vt1_study_visit": "previous_pair_vt1_study_visit"})
    cur["role"] = "excluded_from_trajectory_contrasts (current row)"
    prec = d[d["prior_pair_target_conflict"]]
    require(len(prec) == len(amb), "number of preceding pairs with a conflicted target differs from the ambiguous rows")
    prec = prec[["pair_id", "public_id", "vt_study_visit", "vt_days_to_visit", "vt_disease_response", "vt1_disease_response",
                 "vt1_study_visit", "split", "rna_avail", "eligible_for_binary"]].copy()
    prec["role"] = "retained (preceding pair: only its Vt+1 target label conflicts with the next row's Vt label)"
    rows = []
    for pname, (df, _) in pops.items():
        n = int((df["trajectory_state"] == AMBIG).sum())
        rows.append({"population": pname, "rows_in_population": int(len(df)), "ambiguous_rows_excluded": n,
                     "ambiguous_patients": int(df.loc[df["trajectory_state"] == AMBIG, "public_id"].nunique()),
                     "rows_remaining_with_defined_state": int(len(df) - n),
                     "treatment": "excluded from trajectory contrasts; retained in the frozen cohort (predictive rows unchanged)"})
    return pd.concat([cur, prec], ignore_index=True), pd.DataFrame(rows)


def continuity_table(d, qa):
    rows = []
    for k, v in qa.items():
        e = EXPECTED_CONTINUITY.get(k)
        rows.append({"metric": k, "value": v if not isinstance(v, bool) else bool(v),
                     "audited": e if e is not None else "", "match": (v == e) if e is not None else ""})
    gap = d.loc[d["visit_index"] > 0, "gap_prev_days"]
    for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        rows.append({"metric": f"gap_prev_days_q{int(q * 100)}", "value": float(gap.quantile(q)), "audited": "", "match": ""})
    rows.append({"metric": "gap_prev_gt_180_share", "value": float((gap > 180).mean()), "audited": "", "match": ""})
    rows.append({"metric": "gap_prev_gt_365_share", "value": float((gap > 365).mean()), "audited": "", "match": ""})
    t = pd.DataFrame(rows)
    bad = t[t["match"] == False]      # noqa: E712
    require(bad.empty, "continuity QA does not reproduce the audited structure:\n" + bad.to_string(index=False))
    return t


def patient_overlap_table(pops):
    rows = []
    for pname in ["rna_multiclass", "rna_binary"]:
        df = pops[pname][0]
        h = df[df["trajectory_state"].isin(HISTORY_STATES)]
        per = h.groupby("public_id")["trajectory_state"].nunique()
        r = {"population": pname, "patients_with_history_rows": int(len(per))}
        for k in (1, 2, 3):
            r[f"patients_in_{k}_history_state(s)"] = int((per == k).sum())
        for st in HISTORY_STATES:
            r[f"patients_with_{st}"] = int(h.loc[h["trajectory_state"] == st, "public_id"].nunique())
        r["patients_with_no_history_row"] = int(df.loc[df["trajectory_state"] == "no_history", "public_id"].nunique())
        rows.append(r)
    return pd.DataFrame(rows)


def _counts(d):
    return {"rows": int(len(d)), "pts": int(d["public_id"].nunique()),
            "events": int(d["improved"].sum()), "non_events": int((1 - d["improved"]).sum())}


def _mc_min(d):
    vc = d["exact_next_response"].value_counts().reindex(MULTICLASS_CLASSES, fill_value=0)
    return int(vc.min())


def state_support_rule_table(pops):
    """Direction-3 support rule applied to each trajectory state vs the other defined states (ambiguous excluded)."""
    rb = pops["rna_binary"][0]
    rm = pops["rna_multiclass"][0]
    rb, rm = rb[rb["trajectory_state"] != AMBIG], rm[rm["trajectory_state"] != AMBIG]
    rows = []
    for universe in ["partA_trainval", "pooled"]:
        b = rb[rb["split"] != "test"] if universe == "partA_trainval" else rb
        mc = rm[rm["split"] != "test"] if universe == "partA_trainval" else rm
        for st in STATES:
            g, c = b[b["trajectory_state"] == st], b[b["trajectory_state"] != st]
            cg, cc = _counts(g), _counts(c)
            tg, tc = tier_of(cg["pts"], cg["events"], cg["non_events"]), tier_of(cc["pts"], cc["events"], cc["non_events"])
            r = {"state": st, "universe": universe, **{f"group_{k}": v for k, v in cg.items()},
                 **{f"complement_{k}": v for k, v in cc.items()}, "tier_group": tg, "tier_complement": tc,
                 "tier": max(tg, tc), "design_expected_tier_informational": DESIGN_EXPECTED_TIER[universe][st]}
            r["matches_design_expectation"] = bool(r["tier"] == r["design_expected_tier_informational"])
            if universe == "partA_trainval":
                mg, mcc = _mc_min(mc[mc["trajectory_state"] == st]), _mc_min(mc[mc["trajectory_state"] != st])
                r.update({"multiclass_min_class_group": mg, "multiclass_min_class_complement": mcc,
                          "multiclass_ok": bool(mg >= MULTICLASS_MIN_PER_CLASS and mcc >= MULTICLASS_MIN_PER_CLASS)})
            rows.append(r)
    return pd.DataFrame(rows)


def treatment_by_state_table(pops):
    rows = []
    for pname in ["rna_binary", "rna_multiclass"]:
        df, kind = pops[pname]
        for st in STATES:
            sub = df[df["trajectory_state"] == st]
            for t in TREATMENTS_AUDITED:
                on = sub[sub[TREATMENT_COLS[t]]]
                tv, te = on[on["split"] != "test"], on[on["split"] == "test"]
                r = {"population": pname, "state": st, "treatment": t, "state_rows": int(len(sub)),
                     "rows_on": int(len(on)), "share_state_rows_on": float(len(on) / len(sub)) if len(sub) else np.nan,
                     "patients_on": int(on["public_id"].nunique()),
                     "trainval_rows_on": int(len(tv)), "trainval_patients_on": int(tv["public_id"].nunique()),
                     "test_rows_on": int(len(te)), "test_patients_on": int(te["public_id"].nunique())}
                if kind == "binary":
                    r.update({"events_on": int(on["improved"].sum()), "non_events_on": int((1 - on["improved"]).sum()),
                              "trainval_events_on": int(tv["improved"].sum()),
                              "trainval_non_events_on": int((1 - tv["improved"]).sum()),
                              "test_events_on": int(te["improved"].sum()),
                              "cell_tier_trainval_same_thresholds_as_direction3": tier_of(
                                  int(tv["public_id"].nunique()), int(tv["improved"].sum()),
                                  int((1 - tv["improved"]).sum()))})
                else:
                    vc = tv["exact_next_response"].value_counts().reindex(MULTICLASS_CLASSES, fill_value=0)
                    r["trainval_min_class_count_on"] = int(vc.min())
                    r["trainval_class_counts_on"] = json.dumps({k: int(v) for k, v in vc.items()})
                rows.append(r)
    return pd.DataFrame(rows)


def sensitivity_universe_table(pops):
    """Descriptive support of the pre-declared sensitivity universes (history states only)."""
    masks = {
        "all_history_states": lambda x: pd.Series(True, index=x.index),
        "gap_prev_le_180d": lambda x: x["gap_prev_days"] <= 180,
        "gap_prev_le_365d": lambda x: x["gap_prev_days"] <= 365,
        "exclude_vt_PD": lambda x: x["vt_disease_response"] != "progressive_disease",
        "vt_VGPR_only": lambda x: x["vt_disease_response"] == "very_good_partial_response",
        "rna_age_le_365d": lambda x: x[STALENESS_COL] <= 365,
        "rna_age_le_730d": lambda x: x[STALENESS_COL] <= 730,
    }
    rows = []
    for pname in ["rna_binary", "rna_multiclass"]:
        df, kind = pops[pname]
        h = df[df["trajectory_state"].isin(HISTORY_STATES)]
        for sname, fn in masks.items():
            s = h[fn(h)]
            for st in HISTORY_STATES:
                ss = s[s["trajectory_state"] == st]
                r = {"population": pname, "sensitivity_universe": sname, "state": st, "rows": int(len(ss)),
                     "patients": int(ss["public_id"].nunique()),
                     "trainval_patients": int(ss.loc[ss["split"] != "test", "public_id"].nunique())}
                if kind == "binary":
                    r["events"] = int(ss["improved"].sum())
                    r["trainval_events"] = int(ss.loc[ss["split"] != "test", "improved"].sum())
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(LABEL)
    dirs = get_dirs4(args.smoke)
    out = dirs["support"]
    guard(f"{out}/output_manifest.json", args.smoke, args.force)
    os.makedirs(out, exist_ok=True)

    fs = load_feature_sets()
    assert_predictors_clean(fs)
    require({"vt_disease_response", "vt_days_to_visit"} <= set(fs["model_c"]),
            "trajectory source fields are not Vt-side Model-C inputs")
    m = load_master()                                   # asserts frozen cohort counts and patient-disjoint splits
    m = m.drop(columns=pathway_columns(fs))              # RNA values are never used in this step
    require(not (set(pathway_columns(fs)) & set(m.columns)), "pathway columns still present")

    d, qa = build_trajectory(m)
    leak = truncation_leakage_test(d, max_patients=300 if args.smoke else None)
    print(f"Truncation leakage test passed on {leak['rows_checked']} rows "
          f"({'first 300 patients (smoke)' if args.smoke else 'all patients'}).")

    pops = populations(d)
    repro = reproduction_table(pops)                     # asserts all previously audited counts
    print("Previously audited counts reproduced exactly.")
    cont = continuity_table(d, qa)
    amb_rows, amb_acc = ambiguous_tables(d, pops)

    assign_cols = ["pair_id", "public_id", "visit_index", "vt_days_to_visit", "prior_response", "current_response",
                   "prior_rank", "rank_change", "trajectory_state", "gap_prev_days", "continuity_label_conflict",
                   "prior_pair_target_conflict", "split", "rna_avail", "eligible_for_binary"]
    assign = d[assign_cols].copy()
    leaky = {"improved", "exact_next_response", "vt1_disease_response", "vt1_study_visit", "vt1_days_to_visit",
             "time_gap_days"}
    require(not (leaky & set(assign.columns)), "trajectory assignment table contains outcome / vt1_* columns")
    require(len(assign) == 13451 and assign["pair_id"].is_unique, "assignment table is not one row per pair")

    tables = {
        "trajectory_assignments.csv": assign,
        "state_counts_by_population_and_split.csv": state_counts_table(pops),
        "audit_reproduction.csv": repro,
        "continuity_qa.csv": cont,
        "ambiguous_rows.csv": amb_rows,
        "ambiguous_accounting.csv": amb_acc,
        "patient_state_overlap.csv": patient_overlap_table(pops),
        "state_support_rule.csv": state_support_rule_table(pops),
        "treatment_by_state_support.csv": treatment_by_state_table(pops),
        "sensitivity_universe_support.csv": sensitivity_universe_table(pops),
    }
    files = []
    for name, df in tables.items():
        p = f"{out}/{name}"
        write_csv(df, p, float_format="%.8g")
        files.append(p)

    rule = tables["state_support_rule.csv"]
    mism = rule[~rule["matches_design_expectation"]][["state", "universe", "tier", "design_expected_tier_informational"]]
    summary = {
        "label": LABEL, "smoke": bool(args.smoke),
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 1),
        "state_definition": "previous row's vt_disease_response -> current row's vt_disease_response "
                            "(PD < SD < PR < VGPR < CR < sCR); first row = no_history; audited continuity rows = ambiguous",
        "ordering": "within patient by vt_days_to_visit; vt1_days_to_visit used only to verify ordering/continuity",
        "rna_availability_definition": "days_since_rna_sample notna (existing project definition); RNA values never used",
        "leakage_tests": {"truncation_test": leak, "assignment_table_excludes_outcomes_and_vt1": True,
                          "trajectory_source_fields_in_model_c": ["vt_disease_response", "vt_days_to_visit"]},
        "audited_counts_reproduced": True,
        "state_counts": {p: {st: int((pops[p][0]["trajectory_state"] == st).sum()) for st in ALL_STATES} for p in pops},
        "ambiguous_rows_excluded": int((d["trajectory_state"] == AMBIG).sum()),
        "continuity": {k: v for k, v in qa.items()},
        "support_rule_vs_design_expectation_mismatches": mism.to_dict("records"),
        "assignments_sha256": sha256_file(f"{out}/trajectory_assignments.csv"),
        "note": "Step 1 only: no predictive model and no molecular association were run.",
    }
    p = f"{out}/run_summary.json"
    write_json(summary, p)
    files.append(p)
    man = write_output_manifest(out, files, {"label": LABEL, "smoke": bool(args.smoke),
                                             "input_master_sha256": sha256_file("data/clinical/visit_pairs_with_rna.csv"),
                                             "assignments_sha256": summary["assignments_sha256"]})
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs in {out}")
    print(f"Combined output SHA256: {man['combined_sha256']}")
    if len(mism):
        print("NOTE: support-rule tiers differ from the design expectation (informational):\n" + mism.to_string(index=False))


if __name__ == "__main__":
    main()
