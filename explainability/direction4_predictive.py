"""
Direction 4 -- PART A: predictive complementary value of RNA across longitudinal response trajectories.

Uses the frozen Direction-3 C-vs-D XGBoost OOF predictions for the observed 5x5 train+val analysis;
it does NOT refit observed C/D models. New model fitting is limited to the prespecified RNA-permutation negative control.
Trajectory state is frozen by Direction-4 Step 1 and verified by direction4_common.

Run:
  python -m explainability.direction4_predictive
  python -m explainability.direction4_predictive --smoke [--force]
"""
import argparse, datetime, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from explainability.direction3_common import (
    MULTICLASS_CLASSES, N_BOOT, N_FOLDS, N_PERM_NULL, N_REPEATS, RANDOM_SEED,
    STALENESS_COL, load_feature_sets, pathway_columns, require, sha256_file,
    write_csv, write_json,
)
from explainability.direction3_predictive import (
    Evaluator, HIGHER_IS_BETTER, METRICS, PRIMARY_METRIC, align_repeats,
    boot_p, load_locked_params, make_task_df, oof_task, percentile_ci,
    permuted_rna_frame,
)
from explainability.direction4_common import (
    AMBIG, CONTRAST_STATES, PRIMARY_STATES, REFERENCE_STATE, SECONDARY_STATES,
    adjust_partA_interactions, adjust_partA_state_tests, contrast_frame,
    get_dirs4_all, guard_no_overwrite4, load_trajectory_master,
    patient_bootstrap_counts, run_selfchecks, sensitivity_masks, write_manifest4,
)

LABEL = "DIRECTION 4 / PART A | TRAJECTORY-SPECIFIC PREDICTIVE COMPLEMENTARY VALUE"
SENSITIVITIES = ("gap_prev_le_180d", "exclude_vt_PD")
INTERACTIONS = (("worsening", "stable"), ("improving", "stable"))


def _d3_partA_dir(smoke):
    root = "artifacts/results_clean_rerun/direction3"
    if smoke:
        root += "/_smoke"
    return root + "/partA_predictive"


def verify_d3_oof(smoke):
    """Require a completed matching-mode Direction-3 Part A run and verify its manifest/hash."""
    d = _d3_partA_dir(smoke)
    mp = f"{d}/output_manifest.json"
    require(os.path.exists(mp), f"Direction 3 Part A manifest missing: {mp}. Run matching {'smoke' if smoke else 'real'} D3 Part A first.")
    with open(mp, encoding="utf-8") as f:
        man = json.load(f)
    require(bool(man.get("smoke")) == bool(smoke), "Direction 3 OOF smoke/real mode mismatch")
    paths = {}
    for task in ("binary", "multiclass"):
        p = f"{d}/oof_predictions/oof_{task}.csv"
        require(os.path.exists(p), f"Direction 3 OOF missing: {p}")
        rel = None
        for k in man.get("files", {}):
            if k.endswith(f"oof_predictions/oof_{task}.csv") or k.endswith(f"oof_{task}.csv"):
                rel = k; break
        if rel is not None:
            require(sha256_file(p) == man["files"][rel], f"Direction 3 OOF changed since manifest: {task}")
        expected = man.get("oof_sha256", {}).get(f"oof_{task}.csv")
        if expected:
            require(sha256_file(p) == expected, f"Direction 3 recorded OOF hash mismatch: {task}")
        paths[task] = p
    return paths, man


def load_oof(task, path, expected_repeats):
    d = pd.read_csv(path)
    need = {"repeat", "pair_id", "public_id", "y_true"}
    require(need <= set(d.columns), f"{path}: missing {sorted(need-set(d.columns))}")
    reps = sorted(d["repeat"].unique().tolist())
    require(reps == list(expected_repeats), f"{task}: OOF repeats {reps} != expected {list(expected_repeats)}")
    out = []
    for r in reps:
        x = d[d["repeat"] == r].sort_values("pair_id").reset_index(drop=True)
        require(x["pair_id"].is_unique, f"{task} repeat {r}: duplicate pair_id")
        if task == "binary":
            PC, PD = x["p_C"].to_numpy(float), x["p_D"].to_numpy(float)
        else:
            PC = x[[f"pC_{c}" for c in MULTICLASS_CLASSES]].to_numpy(float)
            PD = x[[f"pD_{c}" for c in MULTICLASS_CLASSES]].to_numpy(float)
        out.append((x, PC, PD))
    first = out[0][0]
    for x, _, _ in out[1:]:
        require(np.array_equal(x["pair_id"], first["pair_id"]), f"{task}: OOF row mismatch across repeats")
        require(np.array_equal(x["y_true"], first["y_true"]), f"{task}: OOF outcome mismatch across repeats")
    return (first["pair_id"].to_numpy(), first["public_id"].astype(str).to_numpy(),
            first["y_true"].to_numpy(int), np.stack([z[1] for z in out]), np.stack([z[2] for z in out]))


def state_groups(master, pair_id, include_no_history=True, mask=None):
    meta = master.set_index("pair_id").loc[pair_id]
    # The six frozen continuity conflicts are deliberately labelled ambiguous.
    # They may exist in the reused D3 OOF/test prediction files, but Direction 4
    # prespecifies that they are excluded from every trajectory contrast.
    state = meta["trajectory_state"].astype(str).to_numpy()
    allowed = set(CONTRAST_STATES) | {"no_history", AMBIG}
    require(set(np.unique(state)) <= allowed, f"unexpected trajectory state(s): {sorted(set(np.unique(state))-allowed)}")
    states = list(CONTRAST_STATES) + (["no_history"] if include_no_history else [])
    base = np.ones(len(meta), dtype=bool) if mask is None else np.array(mask, dtype=bool, copy=True)
    require(len(base) == len(meta), "trajectory mask length mismatch")
    base &= (state != AMBIG)
    groups = {}
    for s in states:
        groups[s] = np.flatnonzero(base & (state == s))
    require(not any(np.isin(v, np.flatnonzero(state == AMBIG)).any() for v in groups.values()),
            "ambiguous trajectory row entered a trajectory group")
    return groups, meta


def evaluate_state_analysis(task, pair_id, public_id, y, PC, PD, master, n_boot, sensitivity="primary"):
    groups, meta = state_groups(master, pair_id, include_no_history=(sensitivity == "primary"))
    if sensitivity != "primary":
        require(sensitivity in SENSITIVITIES, f"unknown sensitivity {sensitivity}")
        sm = sensitivity_masks(meta)[sensitivity].fillna(False).to_numpy(bool)
        groups, meta = state_groups(master, pair_id, include_no_history=False, mask=sm)
    for s in PRIMARY_STATES:
        require(len(groups[s]) > 0, f"{task}/{sensitivity}: primary state {s} has no rows")
    pats, pat_idx = np.unique(public_id, return_inverse=True)
    ev = Evaluator(task, y, pat_idx, PC, PD, groups)
    ones = np.ones(len(pats))
    obs = ev.evaluate(ones)
    delta = (obs[1] - obs[0]).mean(axis=0)
    boot = np.full((n_boot, len(groups), len(METRICS[task])), np.nan)
    for b, counts in enumerate(patient_bootstrap_counts(len(pats), n_boot, RANDOM_SEED)):
        boot[b] = ev.delta(counts)
    rows = []
    gnames = list(groups)
    for gi, s in enumerate(gnames):
        for ki, met in enumerate(METRICS[task]):
            lo, hi, nv = percentile_ci(boot[:, gi, ki])
            rows.append({"task": task, "analysis": sensitivity, "state": s, "metric": met,
                         "n_rows": len(groups[s]), "n_patients": int(np.unique(public_id[groups[s]]).size),
                         "C_mean": float(obs[0, :, gi, ki].mean()), "D_mean": float(obs[1, :, gi, ki].mean()),
                         "delta_D_minus_C": float(delta[gi, ki]), "ci_lo": lo, "ci_hi": hi,
                         "n_valid_bootstrap": nv, "p_boot_two_sided": boot_p(boot[:, gi, ki]),
                         "better_when": "higher" if HIGHER_IS_BETTER[met] else "lower"})
    inter = []
    if sensitivity == "primary":
        for a, ref in INTERACTIONS:
            ai, ri = gnames.index(a), gnames.index(ref)
            for ki, met in enumerate(METRICS[task]):
                v = boot[:, ai, ki] - boot[:, ri, ki]
                lo, hi, nv = percentile_ci(v)
                inter.append({"task": task, "analysis": sensitivity, "contrast": f"{a}_minus_{ref}", "metric": met,
                              "delta_state": float(delta[ai, ki]), "delta_reference": float(delta[ri, ki]),
                              "interaction_delta": float(delta[ai, ki]-delta[ri, ki]), "ci_lo": lo, "ci_hi": hi,
                              "n_valid_bootstrap": nv, "p_boot_two_sided": boot_p(v)})
    f1 = []
    if task == "multiclass":
        for s, rws in groups.items():
            if len(rws) == 0: continue
            for model, P in (("C", PC), ("D", PD)):
                vals = [f1_score(y[rws], P[ri][rws].argmax(1), labels=list(range(6)), average="macro", zero_division=0)
                        for ri in range(P.shape[0])]
                f1.append({"task": task, "analysis": sensitivity, "state": s, "model": model,
                           "macro_f1_mean_over_repeats": float(np.mean(vals)), "n_rows": len(rws),
                           "note": "DESCRIPTIVE ONLY; not bootstrapped"})
    return pd.DataFrame(rows), pd.DataFrame(inter), pd.DataFrame(f1)


def _find_unique(root, basename):
    hits = []
    for dp, _, fs in os.walk(root):
        if basename in fs: hits.append(os.path.join(dp, basename).replace("\\", "/"))
    require(len(hits) == 1, f"expected exactly one {basename} under {root}; found {hits}")
    return hits[0]


def _mc_prob_cols(df):
    cols = []
    for c in MULTICLASS_CLASSES:
        cand = [f"proba_{c}", f"p_{c}", c]
        hit = [x for x in cand if x in df.columns]
        require(len(hit) == 1, f"cannot uniquely identify probability column for {c}; columns={list(df.columns)}")
        cols.append(hit[0])
    return cols


def locked_test_descriptive(master):
    """Evaluate the already-generated locked C/D test probability files; no test refitting or inference."""
    root = "artifacts/results_clean_rerun"
    out = []
    for task in ("binary", "multiclass"):
        pc = _find_unique(root, f"{task}_c_xgboost_test.csv")
        pd_ = _find_unique(root, f"{task}_d_xgboost_test.csv")
        C, D = pd.read_csv(pc), pd.read_csv(pd_)
        require(set(C["pair_id"]) == set(D["pair_id"]), f"{task}: locked C/D test pair_ids differ")
        C = C.sort_values("pair_id").reset_index(drop=True); D = D.sort_values("pair_id").reset_index(drop=True)
        require(np.array_equal(C["pair_id"], D["pair_id"]), f"{task}: locked C/D test alignment failed")
        pair_id = C["pair_id"].to_numpy(); public_id = C["public_id"].astype(str).to_numpy()
        if task == "binary":
            y = C["y_true"].to_numpy(int)
            PC = C["proba"].to_numpy(float)[None, :]; PD = D["proba"].to_numpy(float)[None, :]
        else:
            # Locked multiclass test files store IMWG response labels as strings;
            # Evaluator expects the same 0..5 class indices used by Direction 3 OOF.
            require(C["y_true"].isin(MULTICLASS_CLASSES).all(),
                    "multiclass locked test contains an unknown y_true response label")
            y = np.array([MULTICLASS_CLASSES.index(v) for v in C["y_true"]], dtype=int)
            colsC, colsD = _mc_prob_cols(C), _mc_prob_cols(D)
            PC = C[colsC].to_numpy(float)[None, :, :]; PD = D[colsD].to_numpy(float)[None, :, :]
        groups, _ = state_groups(master, pair_id, include_no_history=True)
        pats, pat_idx = np.unique(public_id, return_inverse=True)
        ev = Evaluator(task, y, pat_idx, PC, PD, groups)
        obs = ev.evaluate(np.ones(len(pats)))
        delta = obs[1] - obs[0]
        for gi, s in enumerate(groups):
            for ki, met in enumerate(METRICS[task]):
                out.append({"task": task, "state": s, "metric": met, "n_rows": len(groups[s]),
                            "n_patients": int(np.unique(public_id[groups[s]]).size),
                            "C": float(obs[0,0,gi,ki]), "D": float(obs[1,0,gi,ki]),
                            "delta_D_minus_C": float(delta[0,gi,ki]),
                            "note": "LOCKED TEST DESCRIPTIVE ONLY; no inference / no model selection"})
    return pd.DataFrame(out)


def negative_control(task, task_df, master, fs, params, observed_pair_id, observed_public_id, y, PC0, n_perm):
    path_cols = pathway_columns(fs)
    groups, _ = state_groups(master, observed_pair_id, include_no_history=True)
    pats, pat_idx = np.unique(observed_public_id, return_inverse=True)
    ev0 = Evaluator(task, y, pat_idx, PC0, PC0, groups)  # only C side is used as baseline below
    ones = np.ones(len(pats))
    rows = []
    for pidx in range(n_perm):
        seed = RANDOM_SEED + 1000 + pidx
        perm = permuted_rna_frame(task_df, path_cols, seed)
        recs, _ = oof_task(task, perm, fs, params[(task,"c")], params[(task,"d")], [0], fit_c=False)
        pid, pub, yp, _, PDp = align_repeats(recs, [0], with_c=False)
        require(np.array_equal(pid, observed_pair_id) and np.array_equal(yp, y) and np.array_equal(pub.astype(str), observed_public_id.astype(str)),
                f"{task}: permuted run evaluated different rows")
        evp = Evaluator(task, y, pat_idx, PC0, PDp, groups)
        val = evp.evaluate(ones)
        d = val[1,0] - val[0,0]
        for gi, s in enumerate(groups):
            for ki, met in enumerate(METRICS[task]):
                rows.append({"task": task, "permutation": pidx, "seed": seed, "state": s, "metric": met,
                             "delta_Dperm_minus_C": float(d[gi,ki])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    repeats = [0] if args.smoke else list(range(N_REPEATS))
    n_boot = 50 if args.smoke else N_BOOT
    n_perm = 2 if args.smoke else N_PERM_NULL
    t0 = time.time(); print(LABEL)
    dirs = get_dirs4_all(args.smoke); out_dir = dirs["partA"]
    guard_no_overwrite4(f"{out_dir}/output_manifest.json", args.smoke, args.force)
    os.makedirs(out_dir, exist_ok=True)

    master, upstream = load_trajectory_master(args.smoke, keep_rna_values=True)
    run_selfchecks(master, args.smoke)
    oof_paths, d3man = verify_d3_oof(args.smoke)
    fs = load_feature_sets(); params, locked_hashes = load_locked_params()

    state_rows=[]; inter_rows=[]; f1_rows=[]; null_rows=[]
    for task in ("binary","multiclass"):
        print(f"\n[{task}] loading frozen Direction-3 OOF predictions...", flush=True)
        pair_id, public_id, y, PC, PD = load_oof(task, oof_paths[task], repeats)
        # OOF must be RNA-available train+val rows only and match task eligibility.
        mm = master.set_index("pair_id").loc[pair_id]
        require((mm["split"] != "test").all() and mm[STALENESS_COL].notna().all(), f"{task}: invalid OOF universe")
        if task == "binary": require(mm["eligible_for_binary"].astype(bool).all(), "binary OOF contains ineligible row")
        d,i,f = evaluate_state_analysis(task,pair_id,public_id,y,PC,PD,master,n_boot,"primary")
        state_rows.append(d); inter_rows.append(i); f1_rows.append(f)
        for sens in SENSITIVITIES:
            d,_,f = evaluate_state_analysis(task,pair_id,public_id,y,PC,PD,master,n_boot,sens)
            state_rows.append(d); f1_rows.append(f)

        print(f"[{task}] negative control: {n_perm} patient-level whole-RNA-vector permutations (repeat 0)...", flush=True)
        task_df = make_task_df(master, task)
        null_rows.append(negative_control(task, task_df, master, fs, params, pair_id, public_id, y, PC[:1], n_perm))

    states = pd.concat(state_rows, ignore_index=True)
    primary_mask = states["analysis"] == "primary"
    adj_primary = adjust_partA_state_tests(states[primary_mask].copy())
    states.loc[primary_mask, ["is_primary_metric","adjusted_p","adjustment_method","adjustment_family"]] = \
        adj_primary[["is_primary_metric","adjusted_p","adjustment_method","adjustment_family"]].to_numpy()
    states.loc[~primary_mask, "is_primary_metric"] = [m == PRIMARY_METRIC[t] for t,m in zip(states.loc[~primary_mask,"task"],states.loc[~primary_mask,"metric"])]
    states.loc[~primary_mask, "adjusted_p"] = np.nan
    states.loc[~primary_mask, "adjustment_method"] = "none (prespecified sensitivity analysis)"
    states.loc[~primary_mask, "adjustment_family"] = ""

    inter = adjust_partA_interactions(pd.concat(inter_rows, ignore_index=True))
    f1 = pd.concat([x for x in f1_rows if len(x)], ignore_index=True) if any(len(x) for x in f1_rows) else pd.DataFrame()
    null = pd.concat(null_rows, ignore_index=True)
    null_sum=[]
    # Compare null with observed repeat-0 D-C for each state/metric.
    for task in ("binary","multiclass"):
        pid,pub,y,PC,PD=load_oof(task,oof_paths[task],repeats)
        groups,_=state_groups(master,pid,True); pats,pi=np.unique(pub,return_inverse=True)
        ev=Evaluator(task,y,pi,PC[:1],PD[:1],groups); obs=ev.delta(np.ones(len(pats)))
        for gi,s in enumerate(groups):
            for ki,met in enumerate(METRICS[task]):
                v=null[(null.task==task)&(null.state==s)&(null.metric==met)].delta_Dperm_minus_C.to_numpy(float)
                o=float(obs[gi,ki]); sign=1.0 if HIGHER_IS_BETTER[met] else -1.0
                null_sum.append({"task":task,"state":s,"metric":met,"observed_repeat0_D_minus_C":o,
                                 "null_mean":float(v.mean()),"null_sd":float(v.std(ddof=1)) if len(v)>1 else np.nan,
                                 "null_p2.5":float(np.percentile(v,2.5)),"null_p97.5":float(np.percentile(v,97.5)),
                                 "n_permutations":len(v),"empirical_one_sided_p_improvement":float((1+(sign*v>=sign*o).sum())/(len(v)+1))})

    test_desc = locked_test_descriptive(master)
    outputs={"partA_state_delta.csv":states,"partA_state_interactions.csv":inter,
             "partA_macro_f1_descriptive.csv":f1,"partA_null_permutation_distribution.csv":null,
             "partA_null_permutation_summary.csv":pd.DataFrame(null_sum),"partA_locked_test_descriptive.csv":test_desc}
    files=[]
    for name,df in outputs.items():
        p=f"{out_dir}/{name}"; write_csv(df,p,float_format="%.8g"); files.append(p)
    summary={"label":LABEL,"smoke":bool(args.smoke),"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "runtime_seconds":round(time.time()-t0,1),"observed_models":"reused frozen Direction-3 OOF predictions; no observed refitting",
             "repeats":repeats,"folds":N_FOLDS,"n_bootstrap":n_boot,"bootstrap_seed":RANDOM_SEED,
             "primary_states":PRIMARY_STATES,"secondary_states":SECONDARY_STATES,"reference_state":REFERENCE_STATE,
             "primary_metric":PRIMARY_METRIC,"primary_multiplicity":"Holm over stable+worsening per task on primary metric",
             "secondary_interactions":"BH per task x metric: worsening-stable and improving-stable",
             "sensitivities":list(SENSITIVITIES),"locked_test":"descriptive only",
             "negative_control_permutations":n_perm,"negative_control_seed":"42+1000+permutation index",
             "note":"RNA is baseline/as-of tumour biology; trajectory uses Vt history only; association/prediction, not causal."}
    sp=f"{out_dir}/run_summary.json"; write_json(summary,sp); files.append(sp)
    man=write_manifest4(out_dir,files,{"label":LABEL,"smoke":bool(args.smoke),"d3_oof_sha256":{t:sha256_file(p) for t,p in oof_paths.items()},
                                              "locked_param_sha256":locked_hashes},args.smoke)
    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in {out_dir}\nCombined output SHA256: {man['combined_sha256']}")

if __name__ == "__main__":
    main()
