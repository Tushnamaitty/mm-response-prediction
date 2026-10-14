"""
RNA-contribution redesign -- STAGE D (development, train patients only) of explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md
(SHA-256 in explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.sha256).

Covers the analyses behind Figures 3A-C and Table S1 and the low-variance RNA add-on. Everything that is learned or decided here
(train-only out-of-fold predictions, uncertainty cut-points, treatment-support sets, the five frozen ridge RNA shifts) uses ONLY the
716 train patients. The master table is reduced to train patients right after loading; no held-out frame and no file that contains
held-out outcomes (the val/test probability CSVs) is read. Stage H (rna_contribution_trainonly_heldout.py) evaluates the frozen
objects once on validation + test patients.

All locked machinery is IMPORTED UNCHANGED: direction3_predictive (oof_task, Evaluator, bootstrap helpers), direction3_common (support
rule, Holm/BH), direction4_support (trajectory states), direction1_uncertainty (entropy), rna_low_variance_incremental (ridge shift).

POST HOC, EXPLORATORY. The earlier pooled results were seen first: not blind, not independent, not external validation.

Run from the repo root:
    python -m explainability.rna_contribution_trainonly_development --benchmark   # timings only
    python -m explainability.rna_contribution_trainonly_development
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, "modeling")
sys.path.insert(0, "data_pipeline")
from repeated_cv_c_vs_d_clean_rerun import fit_fold_preprocessor, make_repeat_folds, transform_rows  # noqa: E402
from train_models import build_model  # noqa: E402

from explainability import direction1_uncertainty as U1  # noqa: E402
from explainability import direction3_common as C3  # noqa: E402
from explainability import direction3_predictive as P3  # noqa: E402
from explainability import direction4_common as C4  # noqa: E402
from explainability import direction4_support as S4  # noqa: E402
from explainability import rna_low_variance_incremental as LV  # noqa: E402

OUT = Path("artifacts/results_clean_rerun/rna_contribution_trainonly")
PROTOCOL = Path("explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md")
PROTOCOL_HASH_FILE = Path("explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.sha256")
SPLITS = Path("data/splits/patient_split_assignments.csv")
N_BOOT = C3.N_BOOT
SEED = C3.RANDOM_SEED
N_REPEATS = C3.N_REPEATS
N_FOLDS = C3.N_FOLDS
CUT_Q = {"q25": 0.25, "q50": 0.50, "q75": 0.75, "q667": 2.0 / 3.0, "q80": 0.80}
EXPECTED_RNA = {"six_dev": (6484, 494), "six_ho": (2689, 213), "bin_dev": (5857, 490), "bin_ho": (2405, 213)}
LABEL = ("RNA-CONTRIBUTION TRAIN-ONLY REDESIGN, STAGE D (development): post hoc, exploratory; not independent validation; "
         "not external validation")
STATES = ["improving", "stable", "worsening", "no_history"]


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def sha_lf(path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=conv)
        f.write("\n")


def require(cond, msg):
    if not cond:
        raise AssertionError(f"[rna_contribution_trainonly] {msg}")


class Gates:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append({"gate": name, "passed": bool(ok), "detail": str(detail)})
        print(("  PASS " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        if not ok:
            raise SystemExit(f"[rna_contribution_trainonly] GATE FAILED: {name} {detail}. Stopped before any held-out outcome was used.")


def protocol_hash_ok():
    rec = PROTOCOL_HASH_FILE.read_text(encoding="utf-8").splitlines()[0].split()[0]
    return rec == sha_lf(PROTOCOL), rec


def entropy_for(task, P):
    return U1.entropy_binary(P) if task == "binary" else U1.entropy_multiclass(P)


def inf_str(x):
    return "inf" if (isinstance(x, float) and math.isinf(x)) else float(x)


def parse_inf(x):
    return math.inf if x == "inf" else float(x)


# ---------------------------------------------------------------------------------------------------------------------
# groups and evaluation (shared with Stage H)
# ---------------------------------------------------------------------------------------------------------------------
def make_groups(meta, ent, cut, treatments, partitions=None):
    """meta: DataFrame aligned to the prediction rows (treatment flags, trajectory_state[, split]); ent: Model-C entropy per row;
    cut: frozen entropy cut-points."""
    n = len(meta)
    groups = {"all": np.arange(n)}
    q = np.where(ent <= cut["q25"], 1, np.where(ent <= cut["q50"], 2, np.where(ent <= cut["q75"], 3, 4)))
    for k in (1, 2, 3, 4):
        groups[f"Q{k}"] = np.flatnonzero(q == k)
    groups.update({"Q1_Q3": np.flatnonzero(q <= 3), "top20": np.flatnonzero(ent > cut["q80"]),
                   "bottom80": np.flatnonzero(ent <= cut["q80"]), "top33": np.flatnonzero(ent > cut["q667"]),
                   "bottom67": np.flatnonzero(ent <= cut["q667"])})
    for t in treatments:
        on = meta[C3.TREATMENT_COLS[t]].to_numpy(dtype=bool)
        groups[f"on_{t}"], groups[f"not_{t}"] = np.flatnonzero(on), np.flatnonzero(~on)
    st = meta["trajectory_state"].astype(str).to_numpy()
    for s in STATES:                                         # ambiguous rows belong to no state group
        groups[f"state_{s}"] = np.flatnonzero(st == s)
    if partitions is not None:
        for p in ("val", "test"):
            groups[f"part_{p}"] = np.flatnonzero(partitions == p)
    return groups


def family_of(g):
    if g == "all" or g.startswith("part_"):
        return "overall"
    if g.startswith("Q") or g in ("top20", "bottom80", "top33", "bottom67"):
        return "uncertainty"
    if g.startswith("state_"):
        return "state"
    return "treatment"


def evaluate_groups(task, y, public_id, PC, PD, groups, n_boot, label):
    """Observed D - C and shared patient bootstrap for every group; same scheme as direction3_predictive."""
    pats, pat_idx = np.unique(public_id, return_inverse=True)
    ev = P3.Evaluator(task, y, pat_idx, PC, PD, groups)
    gnames, metrics = ev.gnames, ev.metrics
    ones = np.ones(len(pats))
    obs = ev.evaluate(ones)
    delta = (obs[1] - obs[0]).mean(axis=0)
    rng = np.random.RandomState(SEED)
    boot = np.full((n_boot, len(gnames), len(metrics)), np.nan)
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.randint(0, len(pats), len(pats))
        boot[b] = ev.delta(np.bincount(idx, minlength=len(pats)).astype(float))
    print(f"  [{label} {task}] bootstrap {n_boot} draws, {len(gnames)} groups: {time.time() - t0:.0f}s", flush=True)
    pm = P3.PRIMARY_METRIC[task]
    rows = []
    for gi, g in enumerate(gnames):
        for ki, met in enumerate(metrics):
            lo, hi, nv = P3.percentile_ci(boot[:, gi, ki])
            rows.append({"task": task, "group": g, "family": family_of(g), "metric": met, "is_primary_metric": met == pm,
                         "n_rows": int(len(groups[g])), "n_patients": int(len(np.unique(pat_idx[groups[g]]))) if len(groups[g]) else 0,
                         "C_mean": float(obs[0, :, gi, ki].mean()), "D_mean": float(obs[1, :, gi, ki].mean()),
                         "delta_D_minus_C": float(delta[gi, ki]), "ci_lo": lo, "ci_hi": hi, "n_valid_bootstrap": nv,
                         "p_boot_two_sided": P3.boot_p(boot[:, gi, ki]),
                         "better_when": "higher" if P3.HIGHER_IS_BETTER[met] else "lower"})
    within = pd.DataFrame(rows)
    specs = [("uncertainty", "PRIMARY_Q4_vs_Q1_Q3", "Q4", "Q1_Q3", "primary"),
             ("uncertainty", "SENS_top20_vs_bottom80", "top20", "bottom80", "sensitivity"),
             ("uncertainty", "SENS_top33_vs_bottom67", "top33", "bottom67", "sensitivity"),
             ("state", "worsening_minus_stable", "state_worsening", "state_stable", "secondary"),
             ("state", "improving_minus_stable", "state_improving", "state_stable", "secondary")]
    specs += [("treatment", f"on_minus_not_{t}", f"on_{t}", f"not_{t}", "secondary")
              for t in sorted({g[3:] for g in gnames if g.startswith("on_")}, key=lambda s: gnames.index(f"on_{s}"))]
    inter = []
    for fam, label_, a, b, role in specs:
        if a not in gnames or b not in gnames:
            continue
        ai, bi = gnames.index(a), gnames.index(b)
        for ki, met in enumerate(metrics):
            v = boot[:, ai, ki] - boot[:, bi, ki]
            lo, hi, nv = P3.percentile_ci(v)
            inter.append({"task": task, "family": fam, "contrast": label_, "role": role, "metric": met,
                          "is_primary_metric": met == pm, "delta_a": float(delta[ai, ki]), "delta_b": float(delta[bi, ki]),
                          "interaction_a_minus_b": float(delta[ai, ki] - delta[bi, ki]), "ci_lo": lo, "ci_hi": hi,
                          "n_valid_bootstrap": nv, "p_boot_two_sided": P3.boot_p(v)})
    f1 = []
    if task == "multiclass":
        for g in gnames:
            rws = groups[g]
            if len(rws) == 0:
                continue
            for m, P in (("C", PC), ("D", PD)):
                vals = [f1_score(y[rws], P[r][rws].argmax(1), labels=list(range(6)), average="macro", zero_division=0)
                        for r in range(P.shape[0])]
                f1.append({"task": task, "group": g, "model": m, "macro_f1_mean_over_repeats": float(np.mean(vals)),
                           "note": "DESCRIPTIVE ONLY (unstable in subgroups); not bootstrapped"})
    return within, pd.DataFrame(inter), pd.DataFrame(f1)


def adjust_families(within, inter):
    """Holm / BH families exactly as the earlier analyses (direction1, direction3, direction4)."""
    w, i = within.copy(), inter.copy()
    for d in (w, i):
        d["adjusted_p"] = np.nan
        d["adjustment_method"] = "none (descriptive)"
        d["adjustment_family"] = ""
    # D1: ONE Holm family = the two primary interactions (binary AUPRC, six-class RPS)
    m = (i["contrast"] == "PRIMARY_Q4_vs_Q1_Q3") & i["is_primary_metric"]
    if int(m.sum()) == 2:
        i.loc[m, "adjusted_p"] = C3.holm_adjust(i.loc[m, "p_boot_two_sided"].to_numpy())
        i.loc[m, "adjustment_method"], i.loc[m, "adjustment_family"] = "Holm", "uncertainty primary interaction: binary AUPRC + six-class RPS"
    # D3: Holm over PI, IMiD, steroid (per task, primary metric); BH for exploratory on-treatment tests per task x metric
    tw = w[w["family"] == "treatment"]
    for task, g in tw[(tw["is_primary_metric"]) & tw["group"].isin([f"on_{t}" for t in C3.PRIMARY_TREATMENTS])].groupby("task"):
        if len(g) == 3:
            w.loc[g.index, "adjusted_p"] = C3.holm_adjust(g["p_boot_two_sided"].to_numpy())
            w.loc[g.index, "adjustment_method"], w.loc[g.index, "adjustment_family"] = "Holm", f"{task}|primary metric|3 primary treatments"
    t2 = tw[tw["group"].isin([f"on_{t}" for t in C3.EXPLORATORY_TREATMENTS])]
    for (task, met), g in t2.groupby(["task", "metric"]):
        w.loc[g.index, "adjusted_p"] = C3.bh_qvalues(g["p_boot_two_sided"].to_numpy())
        w.loc[g.index, "adjustment_method"], w.loc[g.index, "adjustment_family"] = "BH", f"{task}|{met}|Tier-2 on-treatment"
    ti = i[i["family"] == "treatment"]
    for (task, met), g in ti.groupby(["task", "metric"]):
        i.loc[g.index, "adjusted_p"] = C3.bh_qvalues(g["p_boot_two_sided"].to_numpy())
        i.loc[g.index, "adjustment_method"], i.loc[g.index, "adjustment_family"] = "BH", f"{task}|{met}|treatment interaction contrasts"
    # D4: Holm over stable + worsening (per task, primary metric); BH for secondary states; BH for state interactions
    sw = w[w["family"] == "state"].copy()
    sw["state"] = sw["group"].str[len("state_"):]
    if len(sw):
        pm = {"binary": "auprc", "multiclass": "rps"}
        prim = sw["state"].isin(C4.PRIMARY_STATES) & (sw["metric"] == sw["task"].map(pm))
        for task, g in sw[prim].groupby("task"):
            if len(g) == len(C4.PRIMARY_STATES):
                w.loc[g.index, "adjusted_p"] = C3.holm_adjust(g["p_boot_two_sided"].to_numpy())
                w.loc[g.index, "adjustment_method"], w.loc[g.index, "adjustment_family"] = "Holm", f"{task}|{pm[task]}|primary states"
        sec = sw["state"].isin(C4.SECONDARY_STATES)
        for (task, met), g in sw[sec].groupby(["task", "metric"]):
            w.loc[g.index, "adjusted_p"] = C3.bh_qvalues(g["p_boot_two_sided"].to_numpy())
            w.loc[g.index, "adjustment_method"], w.loc[g.index, "adjustment_family"] = "BH", f"{task}|{met}|secondary states"
    si = i[i["family"] == "state"]
    for (task, met), g in si.groupby(["task", "metric"]):
        i.loc[g.index, "adjusted_p"] = C3.bh_qvalues(g["p_boot_two_sided"].to_numpy())
        i.loc[g.index, "adjustment_method"], i.loc[g.index, "adjustment_family"] = "BH", f"{task}|{met}|state interaction contrasts"
    return w, i


# ---------------------------------------------------------------------------------------------------------------------
# add-on helpers (train-only versions of the locked functions' data preparation)
# ---------------------------------------------------------------------------------------------------------------------
def prepare_units(master, paths):
    """Same construction as LV.prepare_master_and_units, on an already-restricted frame."""
    m = master.reset_index(drop=True).copy()
    m["_row"] = np.arange(len(m))
    r = m[C3.STALENESS_COL].notna().to_numpy()
    require(not m.loc[r, paths].isna().any().any(), "RNA-available row has missing pathway score")
    vec = m.loc[r, paths]
    h = pd.util.hash_pandas_object(vec.round(8), index=False).to_numpy()
    key = pd.DataFrame({"pid": pd.factorize(m.loc[r, "public_id"])[0], "h": h})
    unit = key.groupby(["pid", "h"], sort=False).ngroup().to_numpy()
    first = ~key.duplicated().to_numpy()
    V = vec.to_numpy(float)[first]
    require(V.shape[0] == int(unit.max()) + 1, "RNA unit indexing mismatch")
    full = np.full(len(m), -1, dtype=int)
    full[np.flatnonzero(r)] = unit
    m["_rna_unit"] = full
    return m, V


def crossfit_model_c(master, fs, cparams, repeat, train_pats):
    """Cross-fit the locked multiclass Model C inside the development patients (the earlier inner scheme, fold = 0)."""
    folds = make_repeat_folds(np.asarray(train_pats, dtype=object), N_FOLDS, seed=SEED + 10000 + 10 * repeat + 0)
    rec, model_c_cols = [], list(fs["model_c"])
    for j, val_pats in enumerate(folds):
        tr_pats = np.setdiff1d(train_pats, val_pats)
        tr_mask = master["public_id"].isin(tr_pats)
        va_mask = master["public_id"].isin(val_pats) & master[C3.STALENESS_COL].notna()
        prep, wdf, cols = fit_fold_preprocessor(master, model_c_cols, tr_mask, is_model_d=False)
        Xtr, Xva = transform_rows(prep, wdf, cols, tr_mask), transform_rows(prep, wdf, cols, va_mask)
        ytr = np.array([C3.MULTICLASS_CLASSES.index(v) for v in master.loc[tr_mask, "exact_next_response"]])
        mc = build_model("xgboost", "multiclass", cparams)
        mc.fit(Xtr, ytr)
        P = mc.predict_proba(Xva)
        rec.append(pd.DataFrame({"pair_id": master.loc[va_mask, "pair_id"].to_numpy(), "inner_fold": j,
                                 **{f"p{k}": P[:, k] for k in range(6)}}))
    d = pd.concat(rec, ignore_index=True).sort_values("pair_id")
    target = np.sort(master.loc[master["public_id"].isin(train_pats) & master[C3.STALENESS_COL].notna(), "pair_id"].to_numpy())
    require(np.array_equal(d["pair_id"].to_numpy(), target), "cross-fit did not cover the development RNA rows")
    return target, d[[f"p{k}" for k in range(6)]].to_numpy(float), d["inner_fold"].to_numpy(int)


def fit_addon_repeat(master, paths, V, fs, cparams, repeat, train_pats):
    pair_tr, Ptr, inner = crossfit_model_c(master, fs, cparams, repeat, train_pats)
    chosen, beta, mu, sd, scores, info = LV.choose_lambda_and_fit(master, paths, V, pair_tr, Ptr, inner)
    return {"repeat": int(repeat), "lambda": inf_str(chosen), "beta": [float(b) for b in beta], "mean": [float(v) for v in mu],
            "sd": [float(v) for v in sd], "beta_l2": float(np.linalg.norm(beta)),
            "inner_scores": [{"lambda": inf_str(x["lambda"]), "mean_inner_rps": x["mean_inner_rps"]} for x in scores]}, (pair_tr, Ptr)


# ---------------------------------------------------------------------------------------------------------------------
def build_meta(master_like, pair_id):
    """Treatment flags + trajectory state for the prediction rows (state from Vt-side fields only)."""
    return master_like.set_index("pair_id").loc[pair_id]


def dev_cv(dev, fs, params, trt_sets, g, files, n_boot):
    traj, _ = S4.build_trajectory(dev)                      # development patients only
    dev = dev.merge(traj[["pair_id", "trajectory_state"]], on="pair_id", how="left", validate="one_to_one")
    require(dev["trajectory_state"].notna().all(), "trajectory state attachment failed")
    res, cuts, withins, inters, f1s, oof_store = {}, {}, [], [], [], {}
    for task in ("binary", "multiclass"):
        task_df = dev[dev["eligible_for_binary"]] if task == "binary" else dev
        task_df = task_df.reset_index(drop=True)
        t0 = time.time()
        recs, _ = P3.oof_task(task, task_df, fs, params[(task, "c")], params[(task, "d")], list(range(N_REPEATS)))
        pair_id, public_id, y, PC, PD = P3.align_repeats(recs, list(range(N_REPEATS)))
        print(f"  [{task}] development OOF: {len(pair_id)} rows, {len(np.unique(public_id))} patients, {time.time() - t0:.0f}s", flush=True)
        ent = np.stack([entropy_for(task, PC[r]) for r in range(N_REPEATS)]).mean(axis=0)
        cut = {k: float(np.quantile(ent, q)) for k, q in CUT_Q.items()}
        cuts[task] = {**cut, "grid101": [float(v) for v in np.quantile(ent, np.linspace(0.0, 1.0, 101))]}   # ECDF reference for P7
        meta = build_meta(task_df, pair_id)
        groups = make_groups(meta, ent, cut, trt_sets[task])
        w, i, f1 = evaluate_groups(task, y, public_id, PC, PD, groups, n_boot, "devCV")
        withins.append(w), inters.append(i), f1s.append(f1)
        oof_store[task] = {"pair_id": pair_id, "public_id": public_id, "y": y}
    w, i = adjust_families(pd.concat(withins, ignore_index=True), pd.concat(inters, ignore_index=True))
    f1 = pd.concat(f1s, ignore_index=True)
    for name, df in (("devcv_within_groups.csv", w), ("devcv_interactions.csv", i), ("devcv_macro_f1_descriptive.csv", f1)):
        df.to_csv(OUT / name, index=False, float_format="%.10g")
        files.append(OUT / name)
    ov = w[(w["group"] == "all") & w["is_primary_metric"]]
    res = {"overall_primary": ov[["task", "metric", "n_rows", "n_patients", "delta_D_minus_C", "ci_lo", "ci_hi"]].to_dict("records"),
           "note": "development-only cross-validation reference; hyperparameters were tuned in the same patients"}
    return res, cuts, w, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    print(LABEL)
    g = Gates()
    ok, rec = protocol_hash_ok()
    g.check("protocol file hash equals the recorded SHA-256", ok, rec[:16] + "...")
    fs = C3.load_feature_sets()
    C3.assert_predictors_clean(fs)
    paths = C3.pathway_columns(fs)
    master = C3.load_master()                                   # frozen cohort counts / patient-disjoint splits asserted
    sp = pd.read_csv(SPLITS)
    cnt = sp["split"].value_counts().to_dict()
    g.check("partition patient counts 716 / 154 / 155", (cnt.get("train"), cnt.get("val"), cnt.get("test")) == (716, 154, 155), cnt)
    rna = master["rna_avail"]
    obs = {"six_dev": (int((rna & (master.split == "train")).sum()), int(master[rna & (master.split == "train")].public_id.nunique())),
           "six_ho": (int((rna & (master.split != "train")).sum()), int(master[rna & (master.split != "train")].public_id.nunique())),
           "bin_dev": (int((rna & master.eligible_for_binary & (master.split == "train")).sum()),
                       int(master[rna & master.eligible_for_binary & (master.split == "train")].public_id.nunique())),
           "bin_ho": (int((rna & master.eligible_for_binary & (master.split != "train")).sum()),
                      int(master[rna & master.eligible_for_binary & (master.split != "train")].public_id.nunique()))}
    g.check("G1 RNA-available rows / patients per population match the protocol", obs == EXPECTED_RNA, obs)
    params, locked_hashes = P3.load_locked_params()
    cparams = params[("multiclass", "c")]

    if args.benchmark:
        benchmark(master, fs, params, paths, cparams)
        return

    marker = OUT / "frozen_development.json"
    if marker.exists():
        raise SystemExit(f"[rna_contribution_trainonly] {marker} exists: existing outputs are never overwritten.")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- G2: machinery reproduction on the EARLIER train+validation universe (multiclass, repeat 0); aggregate only ----
    t0 = time.time()
    tv = P3.make_task_df(master, "multiclass")
    recs0, _ = P3.oof_task("multiclass", tv, fs, params[("multiclass", "c")], params[("multiclass", "d")], [0])
    _, _, y0, PC0, PD0 = P3.align_repeats(recs0, [0])
    rep = pd.read_csv("artifacts/results_clean_rerun/direction3/partA_predictive/partA_repeat_level_metrics.csv")
    worst = 0.0
    for mname, P in (("C", PC0), ("D", PD0)):
        ref = rep[(rep.task == "multiclass") & (rep.model == mname) & (rep.group == "all") & (rep.metric == "rps") & (rep.repeat == 0)]["value"]
        require(len(ref) == 1, "committed repeat-0 RPS row not found")
        worst = max(worst, abs(float(C3.rps_rows(P[0], y0).mean()) - float(ref.iloc[0])))
    g.check("G2 locked CV machinery reproduces the committed repeat-0 six-class RPS of Models C and D (< 1e-9)", worst < 1e-9,
            f"max |diff| = {worst:.2e}; {time.time() - t0:.0f}s")
    del tv, recs0, y0, PC0, PD0

    # ---- reduce to development patients; the held-out frame is deleted from memory ----
    heldout_ids = sorted(master.loc[master.split != "train", "public_id"].unique())
    dev = master[master.split == "train"].reset_index(drop=True)
    dev_ids = sorted(dev["public_id"].unique())
    del master
    g.check("G5 development and held-out patient sets are disjoint (716 / 309)",
            not (set(dev_ids) & set(heldout_ids)) and len(dev_ids) == 716 and len(heldout_ids) == 309)
    require(set(dev["split"]) == {"train"}, "a non-train row is in the development frame")

    # ---- G4 / D-B: support decided from development counts ----
    tbl = C3.support_table(dev)
    A = tbl[tbl["universe"] == "partA_trainval"]
    g.check("G4 support rule applied to development counts reproduces the declared treatment roles",
            bool(A["tier_matches_declared_role"].all()), A[["treatment", "tier"]].to_dict("records"))
    trt_sets = {t: C3.inference_treatments(tbl, t) for t in ("binary", "multiclass")}
    print("  treatments with inference:", trt_sets)

    # ---- add-on numerical gates (unchanged self-checks) ----
    rel = LV.finite_difference_gradient_check()
    sim = LV.planted_effect_selfcheck()
    g.check("G3 add-on gradient check, planted-effect recovery and beta=0 identity pass", True,
            f"grad rel err {rel:.1e}; planted delta RPS {sim['planted_effect_delta_rps']:.4f}")

    # ---- D-A: development-only 5 x 5 CV reference + D-B/D-C ----
    files = []
    cv_res, cuts, w_cv, i_cv = dev_cv(dev, fs, params, trt_sets, g, files, N_BOOT)

    # ---- D-D: five frozen add-on fits ----
    mtr, V = prepare_units(dev, paths)
    train_pats = np.array(dev_ids, dtype=object)
    addon, floor_diff = [], 0.0
    for r in range(N_REPEATS):
        t0 = time.time()
        fit, (pair_tr, Ptr) = fit_addon_repeat(mtr, paths, V, fs, cparams, r, train_pats)
        ytr = np.array([C3.MULTICLASS_CLASSES.index(v) for v in mtr.set_index("pair_id").loc[pair_tr, "exact_next_response"]])
        floor_diff = max(floor_diff, abs(float(C3.rps_rows(LV.floor_probs(Ptr), ytr).mean()) - float(C3.rps_rows(Ptr, ytr).mean())))
        addon.append(fit)
        print(f"  add-on repeat {r}: lambda={fit['lambda']}, |beta|={fit['beta_l2']:.4f}, {time.time() - t0:.0f}s", flush=True)
    g.check("G3 probability floor changes development out-of-fold mean RPS by < 1e-5", floor_diff < LV.FLOOR_RPS_TOL, f"{floor_diff:.2e}")

    # ---- freeze and hash ----
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           **{p: __import__(p).__version__ for p in ("numpy", "pandas", "scipy", "sklearn", "xgboost")}}
    art = {"label": LABEL, "protocol_sha256": rec, "code_sha256_lf": sha_lf(__file__), "environment": env,
           "development_patient_ids_sha256": hashlib.sha256("\n".join(dev_ids).encode()).hexdigest(),
           "heldout_patient_ids_sha256": hashlib.sha256("\n".join(heldout_ids).encode()).hexdigest(),
           "n_development": len(dev_ids), "n_heldout": len(heldout_ids), "pathways": paths,
           "locked_param_file_sha256": locked_hashes, "locked_params": {f"{k[0]}_{k[1]}": v for k, v in params.items()},
           "treatment_sets": trt_sets, "support_table_development": A.to_dict("records"),
           "entropy_cutpoints": cuts, "entropy_cut_quantiles": CUT_Q, "trajectory_states": STATES,
           "primary_states": C4.PRIMARY_STATES, "secondary_states": C4.SECONDARY_STATES,
           "addon": {"fits": addon, "lambda_grid": [inf_str(x) for x in LV.LAMBDA_GRID], "prob_floor_eps": LV.PROB_EPS,
                     "success_rule": "delta<0 AND patient-bootstrap 95% CI upper<0 AND one-sided permutation p<=0.05"},
           "n_boot": N_BOOT, "bootstrap_seed": SEED}
    dump_json(art, marker)
    art_sha = sha_lf(marker)
    (OUT / "frozen_development.sha256").write_text(f"{art_sha}  frozen_development.json\n", encoding="utf-8", newline="\n")
    dump_json(cv_res, OUT / "devcv_summary.json")
    dump_json({"gates": g.rows, "frozen_development_sha256": art_sha, "protocol_sha256": rec,
               "seconds": time.time() - t_start, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
              OUT / "development_gate_report.json")
    print(f"\nFrozen artifact SHA-256 (LF-normalised): {art_sha}")
    print(f"Stage D finished in {(time.time() - t_start) / 60:.1f} min. Held-out outcomes were not read.")
    for r_ in cv_res["overall_primary"]:
        print(f"  development-only CV reference, {r_['task']} {r_['metric']}: D-C = {r_['delta_D_minus_C']:+.5f} "
              f"({r_['ci_lo']:+.5f} to {r_['ci_hi']:+.5f})")


# ---------------------------------------------------------------------------------------------------------------------
def benchmark(master, fs, params, paths, cparams):
    """Timings only, on development patients; no scientific quantity is printed or saved."""
    OUT.mkdir(parents=True, exist_ok=True)
    dev = master[master.split == "train"].reset_index(drop=True)
    out = {"label": "benchmark: timings only", "cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    traj, _ = S4.build_trajectory(dev)
    dev = dev.merge(traj[["pair_id", "trajectory_state"]], on="pair_id", how="left")
    # one outer fold of the locked CV (C + D) per task
    for task in ("binary", "multiclass"):
        td = (dev[dev["eligible_for_binary"]] if task == "binary" else dev).reset_index(drop=True)
        pats = td["public_id"].unique()
        folds = make_repeat_folds(pats, N_FOLDS, seed=SEED)
        t0 = time.time()
        recs, _ = P3.oof_task(task, td, fs, params[(task, "c")], params[(task, "d")], [0])      # all 5 folds of repeat 0
        out[f"oof_one_repeat_seconds_{task}"] = time.time() - t0
        pair_id, public_id, y, PC, PD = P3.align_repeats(recs, [0])
        ent = entropy_for(task, PC[0])
        cut = {k: float(np.quantile(ent, q)) for k, q in CUT_Q.items()}
        meta = build_meta(td, pair_id)
        groups = make_groups(meta, ent, cut, ["pi", "imid", "steroid", "cd38", "chemo"] if task == "binary" else ["pi", "imid", "steroid", "cd38"])
        pats_u, pat_idx = np.unique(public_id, return_inverse=True)
        ev = P3.Evaluator(task, y, pat_idx, PC, PD, groups)
        rng = np.random.RandomState(1)
        t0 = time.time()
        for _ in range(20):
            idx = rng.randint(0, len(pats_u), len(pats_u))
            ev.delta(np.bincount(idx, minlength=len(pats_u)).astype(float))
        out[f"per_bootstrap_draw_seconds_{task}"] = (time.time() - t0) / 20
        out[f"n_groups_{task}"] = len(groups)
    # add-on: one cross-fit repeat + ridge selection
    mtr, V = prepare_units(dev, paths)
    t0 = time.time()
    fit_addon_repeat(mtr, paths, V, fs, cparams, 0, np.array(sorted(dev["public_id"].unique()), dtype=object))
    out["addon_one_repeat_seconds"] = time.time() - t0
    # add-on permutation scoring micro-benchmark (5 frozen fits, ~2,700 rows)
    rows = mtr[mtr[C3.STALENESS_COL].notna()].reset_index(drop=True)
    P = np.random.RandomState(0).dirichlet(np.ones(6), size=2700)
    off = LV.probs_to_offsets(LV.floor_probs(P))
    Z = np.random.RandomState(1).normal(size=(2700, 50))
    b = np.random.RandomState(2).normal(scale=0.01, size=50)
    t0 = time.time()
    for _ in range(100):
        for _r in range(5):
            q = LV.shifted_probs(off, Z, b)
            C3.rps_rows(q, np.zeros(2700, int))
    out["addon_per_permutation_seconds_est"] = (time.time() - t0) / 100
    # Model-D re-scoring with permuted RNA (per permutation) on ~2,400 held-out-sized development rows
    import joblib
    from train_models import build_feature_matrix, load_preprocessor_and_meta
    prep, meta = load_preprocessor_and_meta("d")
    mdl = joblib.load("artifacts/models_clean_rerun/binary_d_xgboost.joblib")
    td = dev[dev["rna_avail"] & dev["eligible_for_binary"]].sample(2400, random_state=0).reset_index(drop=True)
    t0 = time.time()
    for k in range(5):
        pf = P3.permuted_rna_frame(td, paths, k)
        X, _ = build_feature_matrix(pf, "d", prep, meta)
        mdl.predict_proba(X)
    out["modelD_rescoring_per_permutation_seconds_binary"] = (time.time() - t0) / 5
    est = {"G2_repeat0_reproduction": out["oof_one_repeat_seconds_multiclass"],
           "development_CV_5_repeats_both_tasks": 5 * (out["oof_one_repeat_seconds_binary"] + out["oof_one_repeat_seconds_multiclass"]),
           "development_CV_bootstraps": N_BOOT * 5 * (out["per_bootstrap_draw_seconds_binary"] + out["per_bootstrap_draw_seconds_multiclass"]),
           "addon_five_repeats": 5 * out["addon_one_repeat_seconds"],
           "stage_H_addon_10000_permutations": 10000 * out["addon_per_permutation_seconds_est"],
           "stage_H_modelD_permutations_B1000_both_tasks": 1000 * 2 * out["modelD_rescoring_per_permutation_seconds_binary"] * 1.3,
           "stage_H_bootstraps_R1": N_BOOT * (out["per_bootstrap_draw_seconds_binary"] + out["per_bootstrap_draw_seconds_multiclass"]) * 1.5}
    out["estimated_full_runtime_seconds"] = est
    out["estimated_total_minutes"] = sum(est.values()) / 60
    out["note"] = "development patients only; no held-out outcome read; CPU only"
    dump_json(out, OUT / "benchmark.json")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
