"""
Direction 4 -- STEP 2: shared utilities for the later Part A (predictive) and Part B (molecular) scripts.
Utilities only: no model fitting, no pathway testing, no new scientific analysis.

What this module provides
  1. load_verified_assignments / load_trajectory_master: load the FROZEN Step-1 trajectory assignments from
     artifacts/results_clean_rerun/direction4/support/, verify the support-run manifest and every file hash,
     verify the master table is unchanged, verify row alignment with the frozen master, and attach the states.
  2. exclude_ambiguous / assert_no_ambiguous: the audited ambiguous rows never enter trajectory contrasts.
  3. State roles: primary states (stable, worsening) vs secondary (improving, no_history).
  4. sensitivity_masks: history-gap and RNA-age (and Vt-response) sensitivity masks, identical in definition to
     Step 1's sensitivity_universe_table and cross-checked against its CSV.
  5. Patient-grouped bootstrap utilities (same RNG scheme as the Direction 3 bootstrap: RandomState(42),
     one rng.randint(0, n, n) per draw) and a lazy accessor to the Direction 3 predictive utilities
     (Evaluator, percentile_ci, boot_p, OOF/CV helpers) so they are REUSED, never redefined.
  6. Multiple-testing helpers (Holm / BH) applied by the prespecified families.
  7. Direction 4 output folders, no-overwrite guard and hashed manifests (recording the upstream support hash).
  8. Lightweight self-checks callable by the later scripts (run_selfchecks).

Reused, not redefined:
  explainability/direction4_support.py : STATES, AMBIG, HISTORY_STATES, EXPECTED_* audited counts,
      populations(), reproduction_table(), get_dirs4(), guard()
  explainability/direction3_common.py  : load_master(), holm_adjust(), bh_qvalues(), selfcheck_metrics(),
      weighted metrics / RPS (used by direction3_predictive.Evaluator), write_output_manifest(), seeds
  explainability/direction2_common.py  : RESPONSE_RANK, hashing/IO helpers

Leakage guarantee: the trajectory state is a function of the PREVIOUS and CURRENT rows' vt_disease_response only.
The assignment table carries no outcome and no vt1_* column (re-verified on load); the only flag that looks at a
vt1 field (ambiguity) reads the PREVIOUS row's Vt+1 label, which is the current visit -- past information.
selfcheck_states_use_only_vt_fields() recomputes every non-ambiguous state from Vt fields alone.

RNA availability is the existing project definition (days_since_rna_sample notna) via direction3_common.load_master.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from explainability.direction2_common import (
    MASTER_PATH, RESPONSE_RANK, STALENESS_COL, combined_hash, expect, require, sha256_file, to_bool,
)
from explainability.direction3_common import (
    EXPLORATORY_TREATMENTS, LABEL_A, LABEL_B, N_BOOT, PRIMARY_TREATMENTS, RANDOM_SEED, RNA_INTERPRETATION,
    TREATMENT_COLS, bh_qvalues, holm_adjust, load_master, pathway_columns, selfcheck_metrics,
    write_output_manifest,
)
from explainability.direction4_support import (
    ALL_STATES, AMBIG, EXPECTED_STATE_COUNTS, HISTORY_STATES, STATES, get_dirs4, guard, populations,
    reproduction_table,
)

TASKS = ("binary", "multiclass")
PRIMARY_METRIC_D4 = {"binary": "auprc", "multiclass": "rps"}     # same as direction3_predictive.PRIMARY_METRIC

# ---- state roles (Direction 4 design) ----
PRIMARY_STATES = ["stable", "worsening"]          # Part A primary Holm family (2 tests per task)
SECONDARY_STATES = ["improving", "no_history"]    # secondary / exploratory
CONTRAST_STATES = list(HISTORY_STATES)            # Part B state contrasts: improving / stable / worsening
REFERENCE_STATE = "stable"                        # reference for state contrasts and Part B

ASSIGNMENT_FILE = "trajectory_assignments.csv"
ASSIGNMENT_COLUMNS_ATTACHED = ["trajectory_state", "visit_index", "prior_response", "prior_rank", "rank_change",
                               "gap_prev_days", "continuity_label_conflict", "prior_pair_target_conflict"]
FORBIDDEN_IN_ASSIGNMENTS = {"improved", "exact_next_response", "vt1_disease_response", "vt1_study_visit",
                            "vt1_days_to_visit", "time_gap_days"}

# sensitivity universes (definitions identical to direction4_support.sensitivity_universe_table)
SENSITIVITY_NAMES = ["all_history_states", "gap_prev_le_180d", "gap_prev_le_365d", "exclude_vt_PD",
                     "vt_VGPR_only", "rna_age_le_365d", "rna_age_le_730d"]

# prespecified multiple-testing families
FAMILY_SPEC = {
    "partA_primary": {"method": "Holm", "family": "per task: on-state D-C tests of the 2 primary states "
                      "(stable, worsening), primary metric (binary AUPRC / multiclass RPS); exactly 2 tests"},
    "partA_secondary_states": {"method": "BH", "family": "per task x metric: on-state D-C tests of the secondary "
                               "states (improving, no_history) -- BH as in the Direction 3 correction"},
    "partA_interactions": {"method": "BH", "family": "per task x metric: state interaction contrasts "
                           "(worsening - stable, improving - stable) -- BH as in the Direction 3 correction"},
    "partB_omnibus": {"method": "BH", "family": "the 50 Hallmark pathways, 2-df omnibus (any state difference)"},
    "partB_contrast_worsening_vs_stable": {"method": "BH", "family": "the 50 pathways"},
    "partB_contrast_improving_vs_stable": {"method": "BH", "family": "the 50 pathways"},
}


# ---------------------------------------------------------------------------
# Output folders / no-overwrite / manifests (reusing Step-1 and Direction-3 helpers)
# ---------------------------------------------------------------------------
def get_dirs4_all(smoke=False):
    d = get_dirs4(smoke)
    return {"root": d["root"], "support": d["support"], "partA": f"{d['root']}/partA_predictive",
            "partB": f"{d['root']}/partB_molecular"}


def guard_no_overwrite4(marker_path, smoke, force):
    """Never overwrite existing Direction 4 results (--force only together with --smoke)."""
    guard(marker_path, smoke, force)


def write_manifest4(folder, files, extra, smoke=False):
    """Hashed output manifest that also records the upstream (Step 1) support-run hash."""
    man, _ = verify_support_run(smoke)
    return write_output_manifest(folder, files, {**extra, "upstream_support_combined_sha256": man["combined_sha256"],
                                                 "upstream_assignments_sha256": man["assignments_sha256"]})


# ---------------------------------------------------------------------------
# 1. Verify + load the frozen Step-1 assignments
# ---------------------------------------------------------------------------
def verify_support_run(smoke=False):
    """Verify the Step-1 manifest, every file hash, the combined hash, the unchanged master table and the audited
    count reproduction. Returns (manifest, support_dir)."""
    sdir = get_dirs4(smoke)["support"]
    mpath = f"{sdir}/output_manifest.json"
    require(os.path.exists(mpath), f"{mpath} not found: run direction4_support first")
    with open(mpath, encoding="utf-8") as f:
        man = json.load(f)
    require(bool(man.get("smoke")) == bool(smoke), "support run smoke flag does not match the requested mode")
    for rel, h in man["files"].items():
        p = f"{sdir}/{rel}"
        require(os.path.exists(p), f"frozen support file missing: {p}")
        require(sha256_file(p) == h, f"FROZEN SUPPORT FILE CHANGED since Step 1: {rel}")
    require(combined_hash(man["files"]) == man["combined_sha256"], "combined support hash mismatch")
    require(ASSIGNMENT_FILE in man["files"], "assignment file is not in the support manifest")
    require(man["assignments_sha256"] == man["files"][ASSIGNMENT_FILE], "assignment hash differs from the manifest entry")
    require(sha256_file(MASTER_PATH) == man["input_master_sha256"], "MASTER TABLE CHANGED since the Step-1 support run")
    with open(f"{sdir}/run_summary.json", encoding="utf-8") as f:
        rs = json.load(f)
    require(rs.get("audited_counts_reproduced") is True, "Step-1 run did not record reproduction of the audited counts")
    require(rs.get("assignments_sha256") == man["assignments_sha256"], "run_summary / manifest assignment hash mismatch")
    rep = pd.read_csv(f"{sdir}/audit_reproduction.csv")
    require(bool(rep["match"].astype(bool).all()), "Step-1 audit_reproduction.csv contains mismatches")
    return man, sdir


def load_verified_assignments(master, smoke=False):
    """Load the frozen assignments, verify hashes (verify_support_run) and row alignment with `master`."""
    man, sdir = verify_support_run(smoke)
    a = pd.read_csv(f"{sdir}/{ASSIGNMENT_FILE}")
    bad = sorted(FORBIDDEN_IN_ASSIGNMENTS & set(a.columns))
    require(not bad, f"assignment table contains outcome / vt1_* columns: {bad}")
    require(a["pair_id"].is_unique, "duplicate pair_id in the assignment table")
    expect("assignment rows", len(a), len(master))
    require(set(a["pair_id"]) == set(master["pair_id"]), "assignment pair_ids differ from the master table")
    require(set(a["trajectory_state"].unique()) <= set(ALL_STATES), "unexpected trajectory state in assignments")
    chk = master[["pair_id", "public_id", "vt_days_to_visit", "split", "rna_avail", "eligible_for_binary"]].merge(
        a, on="pair_id", how="left", suffixes=("", "_a"), validate="one_to_one")
    for c in ["public_id", "vt_days_to_visit", "split"]:
        require(bool((chk[c] == chk[c + "_a"]).all()), f"row alignment failed on {c}")
    for c in ["rna_avail", "eligible_for_binary"]:
        require(bool((chk[c].astype(bool) == chk[c + "_a"].astype(bool)).all()), f"row alignment failed on {c}")
    return a, man


def attach_states(master, assignments):
    """Attach the frozen trajectory columns to the master rows (one-to-one on pair_id; master order kept)."""
    clash = sorted(set(ASSIGNMENT_COLUMNS_ATTACHED) & set(master.columns))
    require(not clash, f"master already has trajectory columns: {clash}")
    out = master.merge(assignments[["pair_id"] + ASSIGNMENT_COLUMNS_ATTACHED], on="pair_id", how="left",
                       validate="one_to_one")
    require(len(out) == len(master) and out["trajectory_state"].notna().all(), "state attachment failed")
    out["continuity_label_conflict"] = out["continuity_label_conflict"].astype(bool)
    out["prior_pair_target_conflict"] = out["prior_pair_target_conflict"].astype(bool)
    return out


def assert_expected_counts(m):
    """Assert the Step-1 (previously audited) trajectory / RNA population counts on the attached frame."""
    return reproduction_table(populations(m))


def load_trajectory_master(smoke=False, keep_rna_values=True):
    """Frozen master (Direction 3 loader: cohort counts + patient-disjoint splits asserted) + verified states.
    keep_rna_values=False drops the 50 pathway columns (Part A reads none of them directly)."""
    master = load_master()
    if not keep_rna_values:
        master = master.drop(columns=pathway_columns_safe(master))
    a, man = load_verified_assignments(master, smoke)
    m = attach_states(master, a)
    repro = assert_expected_counts(m)
    return m, {"manifest": man, "reproduction": repro, "assignments": a}


def pathway_columns_safe(master):
    """Pathway columns present in `master` (uses the frozen feature-set definition)."""
    from explainability.direction2_common import load_feature_sets
    cols = pathway_columns(load_feature_sets())
    return [c for c in cols if c in master.columns]


# ---------------------------------------------------------------------------
# 2-3. Ambiguous exclusion and state roles
# ---------------------------------------------------------------------------
def exclude_ambiguous(df):
    """Drop the audited ambiguous rows (they stay in the frozen cohort; they never enter trajectory contrasts).
    Returns (filtered frame, number excluded)."""
    mask = df["trajectory_state"] == AMBIG
    return df[~mask], int(mask.sum())


def assert_no_ambiguous(df):
    require(not (df["trajectory_state"] == AMBIG).any(), "ambiguous rows present in a trajectory contrast")


def state_role(state):
    if state in PRIMARY_STATES:
        return "primary"
    if state in SECONDARY_STATES:
        return "secondary"
    raise ValueError(f"no analysis role for state {state!r}")


def contrast_frame(df, states=None):
    """Rows usable for trajectory contrasts: ambiguous excluded, restricted to `states` (default: history states)."""
    d, _ = exclude_ambiguous(df)
    d = d[d["trajectory_state"].isin(CONTRAST_STATES if states is None else states)]
    assert_no_ambiguous(d)
    return d


# ---------------------------------------------------------------------------
# 4. Sensitivity masks (history rows; same definitions as direction4_support)
# ---------------------------------------------------------------------------
def sensitivity_masks(df):
    return {
        "all_history_states": pd.Series(True, index=df.index),
        "gap_prev_le_180d": df["gap_prev_days"] <= 180,
        "gap_prev_le_365d": df["gap_prev_days"] <= 365,
        "exclude_vt_PD": df["vt_disease_response"] != "progressive_disease",
        "vt_VGPR_only": df["vt_disease_response"] == "very_good_partial_response",
        "rna_age_le_365d": df[STALENESS_COL] <= 365,
        "rna_age_le_730d": df[STALENESS_COL] <= 730,
    }


def apply_sensitivity(df, name):
    """History-state rows (ambiguous and no_history excluded) satisfying the named sensitivity universe."""
    require(name in SENSITIVITY_NAMES, f"unknown sensitivity universe {name!r}")
    h = contrast_frame(df)
    return h[sensitivity_masks(h)[name]]


# ---------------------------------------------------------------------------
# 5. Patient-grouped bootstrap utilities (same scheme as the Direction 3 bootstrap)
# ---------------------------------------------------------------------------
def patient_codes(public_ids):
    """(unique patients, per-row patient index) exactly as direction3_predictive uses for weights."""
    return np.unique(np.asarray(public_ids, dtype=object).astype(str), return_inverse=True)


def patient_bootstrap_counts(n_patients, n_boot=N_BOOT, seed=RANDOM_SEED):
    """Yield per-patient resample COUNTS. Identical RNG scheme to direction3_predictive: RandomState(seed),
    one rng.randint(0, n, n) per draw; the same draw is meant to be shared across repeats, groups and metrics."""
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        idx = rng.randint(0, n_patients, n_patients)
        yield np.bincount(idx, minlength=n_patients).astype(float)


def row_weights(counts, pat_idx):
    return counts[pat_idx]


def d3_predictive_api():
    """Lazy accessor for the Direction 3 predictive utilities (kept lazy: importing direction3_predictive loads the
    frozen CV/XGBoost stack). Reuses Evaluator (weighted AUPRC/AUROC/log-loss/RPS), percentile_ci, boot_p,
    treatment_groups, oof_task, align_repeats, permuted_rna_frame, load_locked_params, make_task_df."""
    from explainability import direction3_predictive as p3
    require(p3.PRIMARY_METRIC == PRIMARY_METRIC_D4, "Direction 3 primary metrics differ from Direction 4's")
    return SimpleNamespace(
        Evaluator=p3.Evaluator, percentile_ci=p3.percentile_ci, boot_p=p3.boot_p,
        treatment_groups=p3.treatment_groups, oof_task=p3.oof_task, align_repeats=p3.align_repeats,
        permuted_rna_frame=p3.permuted_rna_frame, load_locked_params=p3.load_locked_params,
        make_task_df=p3.make_task_df, METRICS=p3.METRICS, PRIMARY_METRIC=p3.PRIMARY_METRIC,
        HIGHER_IS_BETTER=p3.HIGHER_IS_BETTER)


# ---------------------------------------------------------------------------
# 6. Multiple-testing helpers by prespecified family
# ---------------------------------------------------------------------------
def adjust_within_families(df, p_col, family_cols, method, mask=None, out_col="adjusted_p"):
    """Adjust p-values (Holm or BH) separately within each family (group of family_cols), in place.
    Holm requires finite p-values; BH leaves NaN p-values NaN."""
    require(method in ("Holm", "BH"), f"unknown method {method!r}")
    if out_col not in df.columns:
        df[out_col] = np.nan
    sel = df if mask is None else df[mask]
    for _, g in sel.groupby(list(family_cols), sort=False):
        p = g[p_col].to_numpy(dtype=float)
        df.loc[g.index, out_col] = holm_adjust(p) if method == "Holm" else bh_qvalues(p)
    return df


def adjust_partA_state_tests(df):
    """Part A on-state D-C tests. Needs columns: task, state, metric, p_boot_two_sided.
    Primary: Holm over stable + worsening on the primary metric, per task (exactly 2 tests).
    Secondary states (improving, no_history): BH per task x metric. All other rows are descriptive."""
    need = {"task", "state", "metric", "p_boot_two_sided"}
    require(need <= set(df.columns), f"missing columns {sorted(need - set(df.columns))}")
    d = df.copy()
    d["is_primary_metric"] = [m == PRIMARY_METRIC_D4[t] for t, m in zip(d["task"], d["metric"])]
    d["adjusted_p"] = np.nan
    d["adjustment_method"] = "none (descriptive: secondary metric / reference row)"
    d["adjustment_family"] = ""
    prim = d["state"].isin(PRIMARY_STATES) & d["is_primary_metric"]
    for task, g in d[prim].groupby("task", sort=False):
        require(len(g) == len(PRIMARY_STATES) and set(g["state"]) == set(PRIMARY_STATES),
                f"[{task}] the Holm family must contain exactly the primary states {PRIMARY_STATES}")
    adjust_within_families(d, "p_boot_two_sided", ["task"], "Holm", mask=prim)
    d.loc[prim, "adjustment_method"] = "Holm"
    d.loc[prim, "adjustment_family"] = [f"{t}|{PRIMARY_METRIC_D4[t]}|primary states (confirmatory)" for t in d.loc[prim, "task"]]
    sec = d["state"].isin(SECONDARY_STATES)
    adjust_within_families(d, "p_boot_two_sided", ["task", "metric"], "BH", mask=sec)
    d.loc[sec, "adjustment_method"] = "BH"
    d.loc[sec, "adjustment_family"] = [f"{t}|{m}|secondary states" for t, m in zip(d.loc[sec, "task"], d.loc[sec, "metric"])]
    return d


def adjust_partA_interactions(df):
    """Part A state-interaction contrasts. Needs columns: task, metric, p_boot_two_sided. BH per task x metric."""
    need = {"task", "metric", "p_boot_two_sided"}
    require(need <= set(df.columns), f"missing columns {sorted(need - set(df.columns))}")
    d = df.copy()
    adjust_within_families(d, "p_boot_two_sided", ["task", "metric"], "BH")
    d["adjustment_method"] = "BH"
    d["adjustment_family"] = [f"{t}|{m}|state interaction contrasts (secondary)" for t, m in zip(d["task"], d["metric"])]
    return d


def adjust_partB_family(df, p_col, family_name, n_expected=50):
    """BH across one Part B pathway family (omnibus or a state contrast). `family_name` must be a key of
    FAMILY_SPEC starting with 'partB_'. The family must contain n_expected pathways (use n_expected=None in smoke)."""
    require(family_name in FAMILY_SPEC and family_name.startswith("partB_"), f"unknown Part B family {family_name!r}")
    if n_expected is not None:
        expect(f"size of family {family_name}", len(df), n_expected)
    d = df.copy()
    d["adjusted_q_bh"] = bh_qvalues(d[p_col].to_numpy(dtype=float))
    d["adjustment_method"] = "BH"
    d["adjustment_family"] = family_name
    return d


# ---------------------------------------------------------------------------
# 8. Lightweight self-checks (callable by later scripts; no model fitting)
# ---------------------------------------------------------------------------
def recompute_states_vt_only(master):
    """State recomputed from public_id, vt_days_to_visit and vt_disease_response ONLY (no vt1, no outcome)."""
    d = master[["pair_id", "public_id", "vt_days_to_visit", "vt_disease_response"]].sort_values(
        ["public_id", "vt_days_to_visit", "pair_id"]).reset_index(drop=True)
    rank = d["vt_disease_response"].map(RESPONSE_RANK)
    require(rank.notna().all(), "unmapped vt_disease_response")
    prior = rank.groupby(d["public_id"]).shift(1)
    state = np.where(prior.isna(), "no_history",
                     np.where(rank > prior, "improving", np.where(rank < prior, "worsening", "stable")))
    return pd.Series(state, index=d["pair_id"].to_numpy())


def selfcheck_states_use_only_vt_fields(m):
    """Every non-ambiguous assigned state equals the Vt-fields-only recomputation; ambiguous rows are exactly the
    audited count; the assignment columns carry no outcome / vt1 information."""
    re_state = recompute_states_vt_only(m)
    assigned = m.set_index("pair_id")["trajectory_state"]
    same = assigned.loc[re_state.index] == re_state
    non_amb = assigned.loc[re_state.index] != AMBIG
    require(bool(same[non_amb].all()), "a non-ambiguous state differs from the Vt-only recomputation")
    expect("ambiguous rows", int((assigned == AMBIG).sum()), EXPECTED_STATE_COUNTS["all"][AMBIG])
    require(not (FORBIDDEN_IN_ASSIGNMENTS & set(ASSIGNMENT_COLUMNS_ATTACHED)), "attached columns include outcome info")
    return True


def selfcheck_ambiguity_flag(m):
    """The continuity-conflict flag equals the recomputation from the PREVIOUS row's Vt+1 label (past information)."""
    need = {"vt1_disease_response"}
    require(need <= set(m.columns), "vt1_disease_response not available for the ambiguity re-check")
    d = m[["pair_id", "public_id", "vt_days_to_visit", "vt_disease_response", "vt1_disease_response"]].sort_values(
        ["public_id", "vt_days_to_visit", "pair_id"]).reset_index(drop=True)
    g = d.groupby("public_id", sort=False)
    prev = g["vt1_disease_response"].shift(1)
    flag = (prev.notna() & (prev != d["vt_disease_response"])).to_numpy()
    got = m.set_index("pair_id")["continuity_label_conflict"].loc[d["pair_id"]].to_numpy()
    require(np.array_equal(flag, got), "continuity_label_conflict differs from its recomputation")
    return True


def selfcheck_bootstrap():
    """Bootstrap counts sum to the number of patients, are reproducible, and follow the Direction 3 RNG scheme."""
    n = 37
    a = list(patient_bootstrap_counts(n, 5, RANDOM_SEED))
    b = list(patient_bootstrap_counts(n, 5, RANDOM_SEED))
    require(all(np.array_equal(x, y) for x, y in zip(a, b)), "bootstrap draws are not reproducible")
    require(all(x.sum() == n and len(x) == n for x in a), "bootstrap counts do not sum to the number of patients")
    rng = np.random.RandomState(RANDOM_SEED)
    first = np.bincount(rng.randint(0, n, n), minlength=n).astype(float)
    require(np.array_equal(a[0], first), "bootstrap scheme differs from the Direction 3 scheme")
    pats, idx = patient_codes(["b", "a", "b", "c"])
    require(list(pats) == ["a", "b", "c"] and list(idx) == [1, 0, 1, 2], "patient_codes self-check failed")
    return True


def selfcheck_multiplicity():
    p = np.array([0.01, 0.04, 0.03])
    require(np.allclose(holm_adjust(p), [0.03, 0.06, 0.06]), "Holm self-check failed")
    require(np.allclose(bh_qvalues(p), [0.03, 0.04, 0.04]), "BH self-check failed")
    toy = pd.DataFrame({"task": ["binary"] * 4 + ["multiclass"] * 4,
                        "state": ["stable", "worsening", "improving", "no_history"] * 2,
                        "metric": ["auprc"] * 4 + ["rps"] * 4,
                        "p_boot_two_sided": [0.01, 0.02, 0.03, 0.2, 0.04, 0.01, 0.5, 0.05]})
    out = adjust_partA_state_tests(toy)
    require(out.loc[0, "adjustment_method"] == "Holm" and abs(out.loc[0, "adjusted_p"] - 0.02) < 1e-12,
            "Part A primary Holm family self-check failed")
    require((out.loc[[2, 3, 6, 7], "adjustment_method"] == "BH").all(), "Part A secondary BH family self-check failed")
    return True


def selfcheck_sensitivity_against_support(m, smoke=False):
    """sensitivity_masks reproduce Step 1's sensitivity_universe_support.csv (rows and patients)."""
    sdir = get_dirs4(smoke)["support"]
    ref = pd.read_csv(f"{sdir}/sensitivity_universe_support.csv")
    pops = populations(m)
    for pname in ["rna_binary", "rna_multiclass"]:
        df = pops[pname][0]
        for sname in SENSITIVITY_NAMES:
            s = apply_sensitivity(df, sname)
            for st in HISTORY_STATES:
                ss = s[s["trajectory_state"] == st]
                r = ref[(ref["population"] == pname) & (ref["sensitivity_universe"] == sname) & (ref["state"] == st)]
                require(len(r) == 1, f"no reference row for {pname}/{sname}/{st}")
                require(int(r["rows"].iloc[0]) == len(ss) and int(r["patients"].iloc[0]) == ss["public_id"].nunique(),
                        f"sensitivity universe {pname}/{sname}/{st} differs from the Step-1 support table")
    return True


def selfcheck_roles():
    require(set(PRIMARY_STATES) | set(SECONDARY_STATES) == set(STATES), "state roles do not cover the four states")
    require(not (set(PRIMARY_STATES) & set(SECONDARY_STATES)), "a state is both primary and secondary")
    require(AMBIG not in CONTRAST_STATES and "no_history" not in CONTRAST_STATES, "contrast states include an excluded state")
    require(set(PRIMARY_TREATMENTS) == {"pi", "imid", "steroid"} and set(EXPLORATORY_TREATMENTS) == {"cd38", "chemo"},
            "treatment roles differ from Direction 3")
    return True


def run_selfchecks(m, smoke=False, include_support_crosscheck=True):
    """All lightweight self-checks; raises on the first failure. `m` = output of load_trajectory_master()[0]
    (must still contain vt1_disease_response for the ambiguity re-check)."""
    selfcheck_metrics()                           # Direction 3 weighted metrics / RPS self-check (sklearn-compatible)
    selfcheck_roles()
    selfcheck_bootstrap()
    selfcheck_multiplicity()
    selfcheck_states_use_only_vt_fields(m)
    selfcheck_ambiguity_flag(m)
    assert_expected_counts(m)
    if include_support_crosscheck:
        selfcheck_sensitivity_against_support(m, smoke)
    return True
