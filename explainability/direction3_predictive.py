"""
Direction 3 -- PART A: PREDICTIVE incremental value of RNA (Model D vs Model C) within treatment context.

*** PREDICTIVE analysis: out-of-fold, Vt-side inputs only. ***
RNA = baseline / as-of-sample tumour biology (latest sample at or before Vt); NOT current-state, NOT causal.

Design (frozen):
  * CV universe = TRAIN+VAL patients only (split != 'test'); the locked test split never enters.
    Binary task = eligible_for_binary rows; multiclass task = all rows.
  * 5 x 5 repeated patient-grouped CV (folds from make_repeat_folds, seeds 42+repeat), fold-local
    preprocessing (fit_fold_preprocessor), LOCKED XGBoost params from
    artifacts/results_clean_rerun/{task}_{c,d}_xgboost_metrics.json. Models train on ALL rows of the
    training patients; predictions are made for held-out RNA-AVAILABLE rows only (as in
    modeling/repeated_cv_c_vs_d_clean_rerun.py). Nothing is tuned or re-selected.
  * Groups: "currently on X" vs "not on X" for each treatment class (classes overlap; they are NOT
    mutually exclusive). Primary = PI, IMiD, steroid; exploratory = CD38, chemo (from the objective
    support rule); SLAMF7/BCMA unsupported (no inference). Multiclass additionally needs >= 20 rows per class.
  * Estimands per group: D - C, and interaction (D-C)_on - (D-C)_not_on.
  * Metrics: binary AUPRC (primary), AUROC, log-loss; multiclass ranked probability score RPS
    (primary; lower = better), log-loss; macro-F1 descriptive only (observed, not bootstrapped).
  * Inference: patient-cluster bootstrap, 2000 draws, seed 42, ONE patient resample shared across repeats,
    groups and metrics (bootstrap does not capture model-training variability). Holm across the 3 primary
    treatments per task on the primary metric (the ONLY confirmatory family).
  * SECONDARY analyses are BH-adjusted (never left unadjusted): (i) treatment-interaction contrasts, one BH
    family per task x metric across all treatments with inference; (ii) Tier-2 (exploratory: CD38, chemo)
    on-treatment D-C tests, one BH family per task x metric. Secondary metrics within primary treatments,
    the "all" row and the "not on X" complement rows are DESCRIPTIVE (adjustment columns say so).
  * Negative control: 20 patient-level RNA permutations (whole RNA vectors permuted across distinct
    (patient, sample) units; days_since_rna_sample and the availability indicator untouched), ONE 5-fold
    repeat (repeat 0) -- compared with the observed repeat-0 D-C.
  * Out-of-fold predictions are SAVED for reuse by Direction 4.

Run from the repo root:
    python -m explainability.direction3_predictive
    python -m explainability.direction3_predictive --smoke [--force]     (tiny, separate folder)
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, "modeling")
sys.path.insert(0, "data_pipeline")
from train_models import build_model, MULTICLASS_CLASSES as TM_CLASSES, RANDOM_SEED as TM_SEED  # noqa: E402
from repeated_cv_c_vs_d_clean_rerun import make_repeat_folds, fit_fold_preprocessor, transform_rows  # noqa: E402

from explainability.direction3_common import (  # noqa: E402
    LABEL_A, LOCKED_DIR, MULTICLASS_CLASSES, N_BOOT, N_FOLDS, N_PERM_NULL, N_REPEATS, PRIMARY_TREATMENTS, bh_qvalues,
    RANDOM_SEED, RNA_INTERPRETATION, STALENESS_COL, TREATMENT_COLS, TREATMENT_ROLE, assert_predictors_clean,
    assert_support_matches_declared_roles, assert_treatment_fields_clean, binary_logloss_rows, expect,
    get_dirs3, guard_no_overwrite, holm_adjust, inference_treatments, load_feature_sets, load_master,
    multiclass_logloss_rows, pathway_columns, rank_structure, require, rps_rows, selfcheck_metrics, sha256_file,
    support_table, weighted_ap_auroc, write_csv, write_json, write_output_manifest,
)

METRICS = {"binary": ["auprc", "auroc", "logloss"], "multiclass": ["rps", "logloss"]}
PRIMARY_METRIC = {"binary": "auprc", "multiclass": "rps"}
HIGHER_IS_BETTER = {"auprc": True, "auroc": True, "logloss": False, "rps": False}


# ---------------------------------------------------------------------------
def load_locked_params():
    params, hashes = {}, {}
    for task in ["binary", "multiclass"]:
        for stage in ["c", "d"]:
            p = f"{LOCKED_DIR}/{task}_{stage}_xgboost_metrics.json"
            require(os.path.exists(p), f"locked metrics file not found: {p}")
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            require("best_params" in d and isinstance(d["best_params"], dict), f"no best_params in {p}")
            params[(task, stage)] = d["best_params"]
            hashes[p] = sha256_file(p)
    return params, hashes


def make_task_df(master, task):
    d = master[master["split"] != "test"]
    if task == "binary":
        d = d[d["eligible_for_binary"]]
    d = d.reset_index(drop=True)
    require((d["split"] != "test").all(), f"[{task}] test rows present in the CV universe")
    return d


# ---------------------------------------------------------------------------
def oof_task(task, task_df, fs, params_c, params_d, repeats, fit_c=True):
    """Out-of-fold predictions for held-out RNA-available rows (same logic as the frozen CV script)."""
    model_c_cols, model_d_cols = list(fs["model_c"]), list(fs["model_d"])
    rna_cols = [c for c in model_d_cols if c not in set(model_c_cols)]
    require(STALENESS_COL in rna_cols, "days_since_rna_sample missing from the RNA column set")
    unique_patients = task_df["public_id"].unique()
    rna_all = task_df[STALENESS_COL].notna()
    recs, folds_out = [], []
    for repeat in repeats:
        folds = make_repeat_folds(unique_patients, N_FOLDS, seed=RANDOM_SEED + repeat)
        for fold_idx, test_patients in enumerate(folds):
            train_patients = np.setdiff1d(unique_patients, test_patients)
            require(set(train_patients).isdisjoint(set(test_patients)), "train/test patient overlap in a fold")
            train_mask = task_df["public_id"].isin(train_patients)
            eval_mask = task_df["public_id"].isin(test_patients) & rna_all
            require(not (task_df.loc[train_mask, "split"] == "test").any(), "test row in CV training")
            require(not (task_df.loc[eval_mask, "split"] == "test").any(), "test row in CV evaluation")
            require(int(eval_mask.sum()) >= 2 and int(train_mask.sum()) >= 2, "insufficient rows in a fold")
            for p in test_patients:
                folds_out.append((task, repeat, fold_idx, p))

            prep_d, wdf_d, cols_d = fit_fold_preprocessor(task_df, model_d_cols, train_mask, is_model_d=True,
                                                          rna_cols=rna_cols)
            require(len(cols_d) == len(model_d_cols) + 1, "Model D fold column count drifted")
            X_tr_d = transform_rows(prep_d, wdf_d, cols_d, train_mask)
            X_ev_d = transform_rows(prep_d, wdf_d, cols_d, eval_mask)
            if task == "binary":
                y_tr = task_df.loc[train_mask, "improved"].to_numpy()
                y_ev = task_df.loc[eval_mask, "improved"].to_numpy().astype(int)
            else:
                y_tr = np.array([MULTICLASS_CLASSES.index(v) for v in task_df.loc[train_mask, "exact_next_response"]])
                y_ev = np.array([MULTICLASS_CLASSES.index(v) for v in task_df.loc[eval_mask, "exact_next_response"]])
            md = build_model("xgboost", task, params_d)
            md.fit(X_tr_d, y_tr)
            pD = md.predict_proba(X_ev_d)
            pC = None
            if fit_c:
                prep_c, wdf_c, cols_c = fit_fold_preprocessor(task_df, model_c_cols, train_mask, is_model_d=False)
                require(len(cols_c) == len(model_c_cols), "Model C fold column count drifted")
                X_tr_c = transform_rows(prep_c, wdf_c, cols_c, train_mask)
                X_ev_c = transform_rows(prep_c, wdf_c, cols_c, eval_mask)
                mc = build_model("xgboost", task, params_c)
                mc.fit(X_tr_c, y_tr)
                pC = mc.predict_proba(X_ev_c)
            if task == "binary":
                pD = pD[:, 1]
                pC = pC[:, 1] if pC is not None else None
            else:
                require(pD.shape[1] == len(MULTICLASS_CLASSES), "multiclass model did not return 6 class probabilities")
            recs.append({"repeat": repeat, "fold": fold_idx, "pair_id": task_df.loc[eval_mask, "pair_id"].to_numpy(),
                         "public_id": task_df.loc[eval_mask, "public_id"].to_numpy(dtype=object),
                         "y": y_ev, "pC": pC, "pD": pD})
            print(f"  [{task}] repeat {repeat} fold {fold_idx}: train rows {int(train_mask.sum())}, "
                  f"eval rows {int(eval_mask.sum())} ({len(test_patients)} held-out patients)", flush=True)
    return recs, folds_out


def align_repeats(recs, repeats, with_c=True):
    """Stack fold records into arrays aligned over pair_id (identical row order in every repeat)."""
    out = {}
    for r in repeats:
        sub = [x for x in recs if x["repeat"] == r]
        pid = np.concatenate([x["pair_id"] for x in sub])
        order = np.argsort(pid, kind="stable")
        require(len(np.unique(pid)) == len(pid), f"repeat {r}: a row was predicted more than once")
        out[r] = {"pair_id": pid[order],
                  "public_id": np.concatenate([x["public_id"] for x in sub])[order],
                  "y": np.concatenate([x["y"] for x in sub])[order],
                  "pD": np.concatenate([x["pD"] for x in sub])[order],
                  "pC": np.concatenate([x["pC"] for x in sub])[order] if with_c else None}
    first = repeats[0]
    for r in repeats[1:]:
        require(np.array_equal(out[r]["pair_id"], out[first]["pair_id"]) and np.array_equal(out[r]["y"], out[first]["y"]),
                "repeats disagree on the evaluated rows")
    pair_id, y, public_id = out[first]["pair_id"], out[first]["y"], out[first]["public_id"]
    PD = np.stack([out[r]["pD"] for r in repeats])
    PC = np.stack([out[r]["pC"] for r in repeats]) if with_c else None
    return pair_id, public_id, y, PC, PD


# ---------------------------------------------------------------------------
class Evaluator:
    """Precomputes per (model, repeat, group) structures; evaluates metrics under patient weights."""

    def __init__(self, task, y, pat_idx, PC, PD, groups):
        self.task, self.y, self.pat_idx = task, y, pat_idx
        self.gnames = list(groups)
        self.metrics = METRICS[task]
        self.R = PD.shape[0]
        self.pre = {}
        for m, P in (("C", PC), ("D", PD)):
            for r in range(self.R):
                if task == "binary":
                    yf = y.astype(float)
                    loss_all = binary_logloss_rows(yf, P[r])
                else:
                    ll_all, rps_all = multiclass_logloss_rows(P[r], y), rps_rows(P[r], y)
                for g, rows in groups.items():
                    s = {"pat_rows": pat_idx[rows]}
                    if task == "binary":
                        order, distinct = rank_structure(P[r][rows])
                        rs = rows[order]
                        s.update({"y_sorted": y[rs].astype(float), "pat_sorted": pat_idx[rs], "distinct": distinct,
                                  "loss": loss_all[rows]})
                    else:
                        s.update({"rps": rps_all[rows], "ll": ll_all[rows]})
                    self.pre[(m, r, g)] = s

    def evaluate(self, counts):
        """Returns array (model C/D, repeat, group, metric)."""
        out = np.full((2, self.R, len(self.gnames), len(self.metrics)), np.nan)
        for mi, m in enumerate(("C", "D")):
            for r in range(self.R):
                for gi, g in enumerate(self.gnames):
                    s = self.pre[(m, r, g)]
                    wr = counts[s["pat_rows"]]
                    den = wr.sum()
                    if den <= 0:
                        continue
                    if self.task == "binary":
                        ap, auc = weighted_ap_auroc(s["y_sorted"], counts[s["pat_sorted"]], s["distinct"])
                        out[mi, r, gi, :] = (ap, auc, float((wr * s["loss"]).sum() / den))
                    else:
                        out[mi, r, gi, :] = (float((wr * s["rps"]).sum() / den), float((wr * s["ll"]).sum() / den))
        return out

    def delta(self, counts):
        o = self.evaluate(counts)
        return (o[1] - o[0]).mean(axis=0)                # (group, metric): D - C averaged over repeats


def percentile_ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))


def boot_p(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    lo, hi = (1 + (v <= 0).sum()) / (len(v) + 1), (1 + (v >= 0).sum()) / (len(v) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def treatment_groups(flags, treatments):
    groups = {"all": np.arange(len(flags))}
    for t in treatments:
        on = flags[TREATMENT_COLS[t]].to_numpy(dtype=bool)
        groups[f"on_{t}"] = np.flatnonzero(on)
        groups[f"not_{t}"] = np.flatnonzero(~on)
    return groups


# ---------------------------------------------------------------------------
def permuted_rna_frame(task_df, path_cols, seed):
    """Whole RNA vectors permuted across distinct (patient, sample) units; days_since_rna_sample and the
    RNA-availability pattern are untouched; rows sharing a sample keep sharing a (new) vector."""
    d = task_df.copy()
    r = d[STALENESS_COL].notna().to_numpy()
    vec = d.loc[r, path_cols]
    h = pd.util.hash_pandas_object(vec.round(8), index=False).to_numpy()
    pid = pd.factorize(d.loc[r, "public_id"])[0]
    key = pd.DataFrame({"pid": pid, "h": h})
    sample_idx = key.groupby(["pid", "h"], sort=False).ngroup().to_numpy()
    first = ~key.duplicated().to_numpy()
    V = vec.to_numpy()[first]
    require(V.shape[0] == sample_idx.max() + 1, "sample index / vector count mismatch")
    perm = np.random.RandomState(seed).permutation(V.shape[0])
    d.loc[r, path_cols] = V[perm][sample_idx]
    require(d[STALENESS_COL].equals(task_df[STALENESS_COL]), "days_since_rna_sample was modified")
    require(d[path_cols].isna().equals(task_df[path_cols].isna()), "RNA availability pattern changed")
    return d


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    repeats = [0] if args.smoke else list(range(N_REPEATS))
    n_boot = 50 if args.smoke else N_BOOT
    n_perm = 2 if args.smoke else N_PERM_NULL

    t0 = time.time()
    print(f"DIRECTION 3 / PART A  |  {LABEL_A}\n{RNA_INTERPRETATION}")
    dirs = get_dirs3(args.smoke)
    out_dir = dirs["partA"]
    guard_no_overwrite(f"{out_dir}/output_manifest.json", args.smoke, args.force)
    os.makedirs(f"{out_dir}/oof_predictions", exist_ok=True)

    require(TM_SEED == RANDOM_SEED, "train_models.RANDOM_SEED differs from the Direction 3 seed")
    require(list(TM_CLASSES) == MULTICLASS_CLASSES, "train_models.MULTICLASS_CLASSES order differs")
    selfcheck_metrics()

    fs = load_feature_sets()
    assert_predictors_clean(fs)
    assert_treatment_fields_clean(fs)
    path_cols = pathway_columns(fs)
    master = load_master()
    tbl = support_table(master)
    assert_support_matches_declared_roles(tbl)           # STOP if the objective rule disagrees with the roles
    params, locked_hashes = load_locked_params()
    files = []
    p = f"{out_dir}/partA_group_support_used.csv"
    write_csv(tbl[tbl["universe"] == "partA_trainval"], p)
    files.append(p)

    all_delta, all_inter, all_repeat, all_f1, all_null, all_null_sum, folds_all = [], [], [], [], [], [], []
    oof_hashes = {}
    for task in ["binary", "multiclass"]:
        print(f"\n=== {task}: out-of-fold C vs D ({len(repeats)} repeat(s) x {N_FOLDS} folds) ===")
        task_df = make_task_df(master, task)
        treatments = inference_treatments(tbl, task)
        require(all(t in treatments for t in PRIMARY_TREATMENTS), f"[{task}] a primary treatment lacks support")
        recs, folds_out = oof_task(task, task_df, fs, params[(task, "c")], params[(task, "d")], repeats)
        folds_all += folds_out
        pair_id, public_id, y, PC, PD = align_repeats(recs, repeats)

        # ---- save reusable OOF predictions (Direction 4) ----
        rows = []
        for ri, r in enumerate(repeats):
            base = pd.DataFrame({"task": task, "repeat": r, "pair_id": pair_id, "public_id": public_id, "y_true": y})
            fold_of = {(x["repeat"], int(pid_)): x["fold"] for x in recs for pid_ in x["pair_id"]}
            base["fold"] = [fold_of[(r, int(pid_))] for pid_ in pair_id]
            if task == "binary":
                base["p_C"], base["p_D"] = PC[ri], PD[ri]
            else:
                for k, cname in enumerate(MULTICLASS_CLASSES):
                    base[f"pC_{cname}"], base[f"pD_{cname}"] = PC[ri][:, k], PD[ri][:, k]
            rows.append(base)
        oof = pd.concat(rows, ignore_index=True)
        p = f"{out_dir}/oof_predictions/oof_{task}.csv"
        write_csv(oof, p, float_format="%.10g")
        files.append(p)
        oof_hashes[f"oof_{task}.csv"] = sha256_file(p)

        # ---- groups (Vt-side flags only) ----
        flags = master.set_index("pair_id").loc[pair_id, list(TREATMENT_COLS.values())]
        groups = treatment_groups(flags, treatments)
        pats, pat_idx = np.unique(public_id, return_inverse=True)
        ev = Evaluator(task, y, pat_idx, PC, PD, groups)
        gnames, metrics = ev.gnames, ev.metrics
        ones = np.ones(len(pats))
        obs = ev.evaluate(ones)                                   # (2, R, G, M)
        obs_delta = (obs[1] - obs[0]).mean(axis=0)
        for mi, m in enumerate(("C", "D")):
            for ri, r in enumerate(repeats):
                for gi, g in enumerate(gnames):
                    for ki, met in enumerate(metrics):
                        all_repeat.append({"task": task, "model": m, "repeat": r, "group": g, "metric": met,
                                           "value": obs[mi, ri, gi, ki], "n_rows": len(groups[g]),
                                           "n_patients": int(len(np.unique(pat_idx[groups[g]])))})
        # macro-F1 descriptive (multiclass, observed only)
        if task == "multiclass":
            for m, P in (("C", PC), ("D", PD)):
                for g, rws in groups.items():
                    vals = [f1_score(y[rws], P[ri][rws].argmax(axis=1), labels=list(range(6)), average="macro",
                                     zero_division=0) for ri in range(len(repeats))]
                    all_f1.append({"task": task, "model": m, "group": g, "macro_f1_mean_over_repeats": float(np.mean(vals)),
                                   "note": "DESCRIPTIVE ONLY (unstable in subgroups); not bootstrapped"})

        # ---- patient bootstrap (shared resample across repeats, groups, metrics) ----
        print(f"  bootstrapping ({n_boot} patient resamples, seed {RANDOM_SEED})...", flush=True)
        rng = np.random.RandomState(RANDOM_SEED)
        boot = np.full((n_boot, len(gnames), len(metrics)), np.nan)
        for b in range(n_boot):
            idx = rng.randint(0, len(pats), len(pats))
            boot[b] = ev.delta(np.bincount(idx, minlength=len(pats)).astype(float))

        pm = PRIMARY_METRIC[task]
        pm_i = metrics.index(pm)
        rows_d = []
        for gi, g in enumerate(gnames):
            trt = g.split("_", 1)[1] if g != "all" else "all"
            for ki, met in enumerate(metrics):
                lo, hi, nv = percentile_ci(boot[:, gi, ki])
                rows_d.append({"task": task, "group": g, "treatment": trt,
                               "role": TREATMENT_ROLE.get(trt, "reference"), "metric": met,
                               "is_primary_metric": met == pm, "n_rows": len(groups[g]),
                               "n_patients": int(len(np.unique(pat_idx[groups[g]]))),
                               "C_mean": float(obs[0, :, gi, ki].mean()), "D_mean": float(obs[1, :, gi, ki].mean()),
                               "delta_D_minus_C": float(obs_delta[gi, ki]), "ci_lo": lo, "ci_hi": hi,
                               "n_valid_bootstrap": nv, "p_boot_two_sided": boot_p(boot[:, gi, ki]),
                               "better_when": "higher" if HIGHER_IS_BETTER[met] else "lower",
                               "holm_p_primary_family": np.nan})
        dfd = pd.DataFrame(rows_d)
        fam = dfd[(dfd["is_primary_metric"]) & (dfd["role"] == "primary") & (dfd["group"].str.startswith("on_"))]
        require(len(fam) == 3, f"[{task}] Holm family must contain exactly the 3 primary treatments")
        dfd.loc[fam.index, "holm_p_primary_family"] = holm_adjust(fam["p_boot_two_sided"].to_numpy())
        # explicit adjustment bookkeeping for EVERY row
        dfd["adjusted_p"] = np.nan
        dfd["adjustment_method"] = "none (descriptive: secondary metric / complement / reference row)"
        dfd["adjustment_family"] = ""
        dfd.loc[fam.index, "adjusted_p"] = dfd.loc[fam.index, "holm_p_primary_family"]
        dfd.loc[fam.index, "adjustment_method"] = "Holm"
        dfd.loc[fam.index, "adjustment_family"] = f"{task}|{pm}|3 primary treatments (confirmatory)"
        tier2 = dfd[(dfd["role"] == "exploratory") & dfd["group"].str.startswith("on_")]
        for met_name, gsub in tier2.groupby("metric"):
            dfd.loc[gsub.index, "adjusted_p"] = bh_qvalues(gsub["p_boot_two_sided"].to_numpy())
            dfd.loc[gsub.index, "adjustment_method"] = "BH"
            dfd.loc[gsub.index, "adjustment_family"] = f"{task}|{met_name}|Tier-2 on-treatment D-C (secondary)"
        all_delta.append(dfd)

        inter_rows = []
        for t in treatments:
            on_i, not_i = gnames.index(f"on_{t}"), gnames.index(f"not_{t}")
            for ki, met in enumerate(metrics):
                inter = boot[:, on_i, ki] - boot[:, not_i, ki]
                lo, hi, nv = percentile_ci(inter)
                inter_rows.append({"task": task, "treatment": t, "role": TREATMENT_ROLE[t], "metric": met,
                                   "is_primary_metric": met == pm,
                                   "delta_on": float(obs_delta[on_i, ki]), "delta_not_on": float(obs_delta[not_i, ki]),
                                   "interaction_on_minus_not": float(obs_delta[on_i, ki] - obs_delta[not_i, ki]),
                                   "ci_lo": lo, "ci_hi": hi, "p_boot_two_sided": boot_p(inter)})
        di = pd.DataFrame(inter_rows)
        di["adjusted_p"] = np.nan
        for met_name, gsub in di.groupby("metric"):
            di.loc[gsub.index, "adjusted_p"] = bh_qvalues(gsub["p_boot_two_sided"].to_numpy())
        di["adjustment_method"] = "BH"
        di["adjustment_family"] = [f"{task}|{m_}|interaction contrasts across treatments (secondary)" for m_ in di["metric"]]
        di["note"] = "secondary; BH within task x metric across treatments; overlapping classes are not independent"
        all_inter.extend(di.to_dict("records"))

        # ---- negative control: 20 RNA permutations, ONE 5-fold repeat (repeat 0) ----
        print(f"  negative control: {n_perm} RNA permutations, repeat 0 only...", flush=True)
        ev0 = Evaluator(task, y, pat_idx, PC[:1], PD[:1], groups)
        obs0 = ev0.delta(ones)
        null_rows = []
        for pidx in range(n_perm):
            seed = RANDOM_SEED + 1000 + pidx
            perm_df = permuted_rna_frame(task_df, path_cols, seed)
            recs_p, _ = oof_task(task, perm_df, fs, params[(task, "c")], params[(task, "d")], [0], fit_c=False)
            pid_p, _, y_p, _, PDp = align_repeats(recs_p, [0], with_c=False)
            require(np.array_equal(pid_p, pair_id) and np.array_equal(y_p, y), "permuted run evaluated different rows")
            evp = Evaluator(task, y, pat_idx, PC[:1], PDp, groups)
            dp = evp.delta(ones)
            for gi, g in enumerate(gnames):
                for ki, met in enumerate(metrics):
                    null_rows.append({"task": task, "permutation": pidx, "seed": seed, "group": g, "metric": met,
                                      "delta_Dperm_minus_C": float(dp[gi, ki])})
        nd = pd.DataFrame(null_rows)
        all_null.append(nd)
        for gi, g in enumerate(gnames):
            for ki, met in enumerate(metrics):
                v = nd[(nd["group"] == g) & (nd["metric"] == met)]["delta_Dperm_minus_C"].to_numpy()
                o = float(obs0[gi, ki])
                sign = 1.0 if HIGHER_IS_BETTER[met] else -1.0
                all_null_sum.append({"task": task, "group": g, "metric": met, "observed_repeat0_D_minus_C": o,
                                     "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                                     "null_p2.5": float(np.percentile(v, 2.5)), "null_p97.5": float(np.percentile(v, 97.5)),
                                     "n_permutations": len(v),
                                     "empirical_one_sided_p_improvement": float((1 + (sign * v >= sign * o).sum()) / (len(v) + 1)),
                                     "note": "observed and null both use repeat 0 folds; D_perm keeps timing/availability features"})

    # ---- write outputs ----
    outputs = {"partA_delta_within_group.csv": pd.concat(all_delta), "partA_interaction.csv": pd.DataFrame(all_inter),
               "partA_repeat_level_metrics.csv": pd.DataFrame(all_repeat),
               "partA_macro_f1_descriptive.csv": pd.DataFrame(all_f1),
               "partA_null_permutation_distribution.csv": pd.concat(all_null),
               "partA_null_permutation_summary.csv": pd.DataFrame(all_null_sum),
               "oof_fold_assignments.csv": pd.DataFrame(folds_all, columns=["task", "repeat", "fold", "public_id"])}
    for name, df in outputs.items():
        path = f"{out_dir}/{name}" if not name.startswith("oof_") else f"{out_dir}/oof_predictions/{name}"
        write_csv(df, path, float_format="%.8g")
        files.append(path)
    summary = {"label": LABEL_A, "interpretation": RNA_INTERPRETATION, "smoke": bool(args.smoke),
               "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "runtime_seconds": round(time.time() - t0, 1), "repeats": repeats, "folds": N_FOLDS,
               "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED, "n_null_permutations": n_perm,
               "null_seeds": f"{RANDOM_SEED + 1000} + permutation index", "fold_seeds": f"{RANDOM_SEED} + repeat",
               "locked_params": {f"{k[0]}_{k[1]}": v for k, v in params.items()},
               "primary_metric": PRIMARY_METRIC, "holm_family": "3 primary treatments (PI, IMiD, steroid) per task, primary metric",
               "bh_families_secondary": ["interaction contrasts: per task x metric across treatments",
                                         "Tier-2 (CD38, chemo) on-treatment D-C: per task x metric"],
               "cv_universe": "train+val patients only; models trained on all rows of training patients; evaluated on held-out RNA-available rows",
               "note_bootstrap": "patient bootstrap does not capture model-training variability"}
    p = f"{out_dir}/run_summary.json"
    write_json(summary, p)
    files.append(p)
    man = write_output_manifest(out_dir, files, {"label": LABEL_A, "locked_input_sha256": locked_hashes,
                                                 "oof_sha256": oof_hashes,
                                                 "input_files_sha256": {"master": sha256_file("data/clinical/visit_pairs_with_rna.csv")},
                                                 "smoke": bool(args.smoke)})
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs in {out_dir}\nCombined output SHA256: {man['combined_sha256']}")
    print(LABEL_A)


if __name__ == "__main__":
    main()
