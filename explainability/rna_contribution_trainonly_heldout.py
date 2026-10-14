"""
RNA-contribution redesign -- STAGE H (held-out evaluation) of explainability/RNA_CONTRIBUTION_TRAINONLY_PROTOCOL.md.

Run ONCE after Stage D (rna_contribution_trainonly_development.py) has written and hashed
artifacts/results_clean_rerun/rna_contribution_trainonly/frozen_development.json. The protocol hash, the frozen-artifact hash and the
Stage-D gate report are verified BEFORE the master table (which carries the outcomes) or the held-out probability files are read.

Held-out set = validation + test patients (309; RNA-available rows only: six-class 2,689 rows / 213 patients, binary 2,405 / 213).
The locked train-only Layer C / Layer D XGBoost models are only SCORED (probabilities from calibration/probabilities/*.csv, checked
against a fresh inference of the locked joblib models); the five frozen add-on ridge fits are only APPLIED. Nothing is refit or
re-tuned; strata, treatments and uncertainty cut-points come from the frozen development artifact.

P1 overall D - C | P2 uncertainty groups | P3 add-on (success rule: delta RPS < 0 AND bootstrap CI upper < 0 AND one-sided permutation
p <= 0.05) | P4 treatment strata | P5 trajectory states | P6 negative control (RNA vectors permuted among held-out units, Model D
re-scored) | P7 continuous uncertainty slope (binary, descriptive) | R reference rows.

POST HOC, EXPLORATORY: not independent validation, not external validation.

Run from the repo root:   python -m explainability.rna_contribution_trainonly_heldout
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import sys
import time

import joblib
import numpy as np
import pandas as pd

from explainability import direction3_common as C3
from explainability import direction3_predictive as P3
from explainability import direction4_support as S4
from explainability import rna_contribution_trainonly_development as D
from explainability import rna_low_variance_incremental as LV
from train_models import build_feature_matrix, load_preprocessor_and_meta

OUT = D.OUT
REPORT = OUT / "heldout_report.json"
B_ADDON = 10_000
B_NEG = 1_000
PROB_DIR = "artifacts/results_clean_rerun/calibration/probabilities/"
LABEL = ("RNA-CONTRIBUTION TRAIN-ONLY REDESIGN, STAGE H (held-out validation + test evaluation): post hoc, exploratory; NOT "
         "independent validation; NOT external validation")


def load_heldout_probs(task):
    frames = {}
    for stage in ("c", "d"):
        d = pd.concat([pd.read_csv(f"{PROB_DIR}{task}_{stage}_xgboost_{s}.csv") for s in ("val", "test")], ignore_index=True)
        frames[stage] = d[d["rna_available"].astype(bool)].sort_values("pair_id").reset_index(drop=True)
    c, d = frames["c"], frames["d"]
    D.require(np.array_equal(c["pair_id"], d["pair_id"]) and np.array_equal(c["public_id"], d["public_id"]), "C/D probability rows differ")
    ycol = c["y_true"]
    if task == "binary":
        y = ycol.to_numpy(int)
        PC, PD = c["proba"].to_numpy(float), d["proba"].to_numpy(float)
    else:
        y = ycol.to_numpy(int) if pd.api.types.is_numeric_dtype(ycol) else np.array([C3.MULTICLASS_CLASSES.index(str(v)) for v in ycol])
        cols = [f"p_{k}" for k in C3.MULTICLASS_CLASSES]
        PC, PD = c[cols].to_numpy(float), d[cols].to_numpy(float)
        D.require(np.allclose(PC.sum(1), 1, atol=1e-5) and np.allclose(PD.sum(1), 1, atol=1e-5), "probabilities do not sum to one")
    return c["pair_id"].to_numpy(), c["public_id"].astype(str).to_numpy(), y, PC, PD, c["split"].to_numpy()


def addon_eval(P_raw, y, V, unit, fits, Vp=None):
    """Mean over the 5 frozen repeats of per-row delta RPS (RNA-shifted minus floored Model C)."""
    Pbase = LV.floor_probs(P_raw)
    off = LV.probs_to_offsets(Pbase)
    base_rps = C3.rps_rows(Pbase, y)
    rows = []
    Vuse = V if Vp is None else Vp
    for f in fits:
        mu, sd, beta = np.array(f["mean"]), np.array(f["sd"]), np.array(f["beta"])
        Z = (Vuse[unit] - mu) / sd
        rows.append(C3.rps_rows(LV.shifted_probs(off, Z, beta), y) - base_rps)
    return np.mean(rows, axis=0), np.array(rows)


def addon_bootstrap(delta_rows, public_id, B=C3.N_BOOT, seed=C3.RANDOM_SEED):
    pats, idx = np.unique(public_id, return_inverse=True)
    S = np.bincount(idx, weights=delta_rows, minlength=len(pats))
    n = np.bincount(idx, minlength=len(pats)).astype(float)
    rng = np.random.RandomState(seed)
    vals = np.empty(B)
    for b in range(B):
        cnt = np.bincount(rng.randint(0, len(pats), len(pats)), minlength=len(pats)).astype(float)
        vals[b] = float((cnt * S).sum() / (cnt * n).sum())
    return {"n_bootstrap": B, "seed": seed, "mean": float(vals.mean()), "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5))}


def slope_stat(b, u, w):
    ub, bb = (w * u).sum() / w.sum(), (w * b).sum() / w.sum()
    return float((w * (u - ub) * (b - bb)).sum() / (w * (u - ub) ** 2).sum())


def main():
    t_start = time.time()
    print(LABEL)
    if REPORT.exists():
        raise SystemExit(f"[rna_contribution_heldout] {REPORT} exists: Stage H is run once; outputs are never overwritten.")
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(("  PASS " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        if not ok:
            raise SystemExit(f"[rna_contribution_heldout] CHECK FAILED: {name} {detail}")

    # ---------------- verification BEFORE any held-out outcome is read ----------------
    ok, rec = D.protocol_hash_ok()
    chk("H1 protocol file hash equals the recorded SHA-256", ok)
    fp = OUT / "frozen_development.json"
    recorded = (OUT / "frozen_development.sha256").read_text(encoding="utf-8").split()[0]
    chk("H1 frozen_development.json hash equals the recorded hash", D.sha_lf(fp) == recorded, recorded[:16] + "...")
    gate = json.load(open(OUT / "development_gate_report.json", encoding="utf-8"))
    chk("H1 Stage-D gate report: all gates passed and records the same artifact hash",
        all(r["passed"] for r in gate["gates"]) and gate["frozen_development_sha256"] == recorded, f"{len(gate['gates'])} gates")
    art = json.load(open(fp, encoding="utf-8"))
    chk("H1 artifact records the protocol hash", art["protocol_sha256"] == rec)
    chk("H1 frozen artifact was written before this run started", fp.stat().st_mtime < t_start)

    # ---------------- only now: outcomes enter ----------------
    master = C3.load_master()
    fs = C3.load_feature_sets()
    paths = C3.pathway_columns(fs)
    chk("H2 pathway list equals the frozen one", paths == art["pathways"])
    ho_ids = sorted(master.loc[master.split != "train", "public_id"].unique())
    dev_ids = sorted(master.loc[master.split == "train", "public_id"].unique())
    chk("H3 held-out and development patient sets equal the frozen ones (hash) and are disjoint",
        hashlib.sha256("\n".join(ho_ids).encode()).hexdigest() == art["heldout_patient_ids_sha256"]
        and hashlib.sha256("\n".join(dev_ids).encode()).hexdigest() == art["development_patient_ids_sha256"]
        and not (set(ho_ids) & set(dev_ids)) and len(ho_ids) == 309)
    traj, _ = S4.build_trajectory(master)
    master = master.merge(traj[["pair_id", "trajectory_state"]], on="pair_id", how="left", validate="one_to_one")
    chk("H3 trajectory states attached to every row", bool(master["trajectory_state"].notna().all()))
    ho = master[master.split != "train"].reset_index(drop=True)
    prep = {s: load_preprocessor_and_meta(s) for s in ("c", "d")}
    models = {(t, s): joblib.load(f"artifacts/models_clean_rerun/{t}_{s}_xgboost.joblib") for t in ("binary", "multiclass") for s in "cd"}

    data, within_all, inter_all, f1_all = {}, [], [], []
    for task in ("binary", "multiclass"):
        pair_id, public_id, y, PC, PD, split = load_heldout_probs(task)
        rows_df = ho[ho["pair_id"].isin(pair_id)].sort_values("pair_id").reset_index(drop=True)
        exp_rows = ho[ho["rna_avail"] & (ho["eligible_for_binary"] if task == "binary" else True)]
        chk(f"H3 [{task}] held-out RNA-available rows / patients = protocol",
            (len(pair_id), len(np.unique(public_id))) == ((2405, 213) if task == "binary" else (2689, 213))
            and set(pair_id) == set(exp_rows["pair_id"]) and np.array_equal(rows_df["pair_id"], pair_id), f"{len(pair_id)} rows")
        worst = 0.0
        for stage, P in (("c", PC), ("d", PD)):
            X, _ = build_feature_matrix(rows_df, stage, *prep[stage])
            pr = models[(task, stage)].predict_proba(X)
            pr = pr[:, 1] if task == "binary" else pr
            worst = max(worst, float(np.max(np.abs(pr - P))))
        chk(f"H3 [{task}] saved held-out probabilities equal a fresh inference of the locked models (< 1e-6)", worst < 1e-6,
            f"max |diff| = {worst:.1e}")
        ent = D.entropy_for(task, PC)
        cut = art["entropy_cutpoints"][task]
        meta = rows_df.set_index("pair_id")
        groups = D.make_groups(meta, ent, cut, art["treatment_sets"][task], partitions=split)
        w, i, f1 = D.evaluate_groups(task, y, public_id, PC[None], PD[None], groups, D.N_BOOT, "held-out")
        within_all.append(w), inter_all.append(i), f1_all.append(f1)
        data[task] = {"pair_id": pair_id, "public_id": public_id, "y": y, "PC": PC, "PD": PD, "rows_df": rows_df, "groups": groups,
                      "ent": ent}
    w, i = D.adjust_families(pd.concat(within_all, ignore_index=True), pd.concat(inter_all, ignore_index=True))
    f1 = pd.concat(f1_all, ignore_index=True)
    # sanity: paired-structure checks of the evaluator on the observed held-out data
    for task in ("binary", "multiclass"):
        dd = data[task]
        ov = w[(w.task == task) & (w.group == "all") & w.is_primary_metric].iloc[0]
        pm = P3.PRIMARY_METRIC[task]
        if task == "binary":
            from sklearn.metrics import average_precision_score
            direct = average_precision_score(dd["y"], dd["PD"]) - average_precision_score(dd["y"], dd["PC"])
        else:
            direct = float((C3.rps_rows(dd["PD"], dd["y"]) - C3.rps_rows(dd["PC"], dd["y"])).mean())
        chk(f"H4 [{task}] overall {pm} D - C from the evaluator equals a direct recomputation", abs(direct - ov["delta_D_minus_C"]) < 1e-9,
            f"{direct:+.6f}")

    # ---------------- P3: low-variance add-on (frozen per-repeat fits) ----------------
    d6 = data["multiclass"]
    ho_rna = ho[ho["pair_id"].isin(d6["pair_id"])].sort_values("pair_id").reset_index(drop=True)
    mu_, V = D.prepare_units(ho_rna, paths)
    unit = mu_["_rna_unit"].to_numpy(int)
    chk("H5 add-on rows align with the held-out six-class rows", np.array_equal(mu_["pair_id"].to_numpy(), d6["pair_id"]))
    fits = art["addon"]["fits"]
    for f_ in fits:
        f_["lambda_num"] = D.parse_inf(f_["lambda"])
    delta_rows, per_rep = addon_eval(d6["PC"], d6["y"], V, unit, fits)
    obs_delta = float(delta_rows.mean())
    chk("H5 mean over repeats of per-repeat mean delta RPS equals the mean of the repeat-averaged rows",
        abs(float(per_rep.mean(axis=1).mean()) - obs_delta) < 1e-15, f"{obs_delta:+.3e}")
    pats_u = np.unique(d6["public_id"])
    rng_a, rng_b = np.random.RandomState(42), np.random.RandomState(42)
    chk("H5 bootstrap index stream equals the earlier rng.choice scheme",
        np.array_equal(rng_a.choice(pats_u, size=len(pats_u), replace=True), pats_u[rng_b.randint(0, len(pats_u), len(pats_u))]))
    boot = addon_bootstrap(delta_rows, d6["public_id"])
    t0 = time.time()
    null = np.empty(B_ADDON)
    for p in range(B_ADDON):
        Vp = LV.permute_unit_vectors(V, LV.PERM_SEED_BASE + p)
        null[p] = float(addon_eval(d6["PC"], d6["y"], V, unit, fits, Vp=Vp)[0].mean())
    k = int((null <= obs_delta).sum())
    p_perm = float((1 + k) / (B_ADDON + 1))
    comp = {"observed_delta_negative": bool(obs_delta < 0), "bootstrap_ci_upper_below_zero": bool(boot["ci_upper"] < 0),
            "permutation_p_le_0.05": bool(p_perm <= 0.05)}
    n_ok = sum(comp.values())
    decision = ("SUPPORTED (EXPLORATORY)" if n_ok == 3 else "NO INCREMENTAL VALUE DETECTED" if n_ok == 0 else "SUGGESTIVE OR INCONCLUSIVE")
    addon = {"observed_delta_rps": obs_delta, "per_repeat_delta_rps": [float(v) for v in per_rep.mean(axis=1)],
             "n_rows": int(len(delta_rows)), "n_patients": int(len(pats_u)), "patient_bootstrap": boot,
             "permutation": {"B": B_ADDON, "k_null_le_observed": k, "one_sided_p": p_perm, "null_mean": float(null.mean()),
                             "null_sd": float(null.std(ddof=1)), "null_p2_5": float(np.percentile(null, 2.5)),
                             "null_p97_5": float(np.percentile(null, 97.5)), "seconds": time.time() - t0,
                             "definition": "RNA vectors permuted among held-out distinct (patient, sample) units; frozen fits re-applied"},
             "success_components": comp, "decision": decision, "selected_lambdas": [f_["lambda"] for f_ in fits],
             "rule": "delta<0 AND patient-bootstrap 95% CI upper<0 AND one-sided permutation p<=0.05 (unchanged)"}
    chk("H5 add-on permutation null completed", len(null) == B_ADDON and np.isfinite(null).all(), f"{addon['permutation']['seconds']:.0f}s")

    # ---------------- P6: negative control, Model D re-scored with permuted RNA among held-out units ----------------
    neg_rows = []
    for task in ("binary", "multiclass"):
        dd = data[task]
        pats, pat_idx = np.unique(dd["public_id"], return_inverse=True)
        ones = np.ones(len(pats))
        ev0 = P3.Evaluator(task, dd["y"], pat_idx, dd["PC"][None], dd["PD"][None], dd["groups"])
        obs = ev0.delta(ones)
        gnames, metrics = ev0.gnames, ev0.metrics
        nulls = np.empty((B_NEG, len(gnames), len(metrics)))
        t0 = time.time()
        for b in range(B_NEG):
            pf = P3.permuted_rna_frame(dd["rows_df"], paths, C3.RANDOM_SEED + 1000 + b)
            X, _ = build_feature_matrix(pf, "d", *prep["d"])
            pr = models[(task, "d")].predict_proba(X)
            pr = pr[:, 1] if task == "binary" else pr
            nulls[b] = P3.Evaluator(task, dd["y"], pat_idx, dd["PC"][None], pr[None], dd["groups"]).delta(ones)
        print(f"  [{task}] negative control {B_NEG} permutations: {time.time() - t0:.0f}s", flush=True)
        for gi, g in enumerate(gnames):
            for ki, met in enumerate(metrics):
                v = nulls[:, gi, ki]
                sign = 1.0 if P3.HIGHER_IS_BETTER[met] else -1.0
                o = float(obs[gi, ki])
                neg_rows.append({"task": task, "group": g, "metric": met, "observed_D_minus_C": o, "null_mean": float(np.nanmean(v)),
                                 "null_sd": float(np.nanstd(v, ddof=1)), "null_p2.5": float(np.nanpercentile(v, 2.5)),
                                 "null_p97.5": float(np.nanpercentile(v, 97.5)), "n_permutations": int(np.isfinite(v).sum()),
                                 "empirical_one_sided_p_improvement": float((1 + (sign * v >= sign * o).sum()) / (len(v) + 1)),
                                 "note": "descriptive; Model D re-scored with RNA vectors permuted among held-out units"})
    neg = pd.DataFrame(neg_rows)
    chk("H6 negative-control permutations completed for both tasks", bool((neg["n_permutations"] > 0).all()))

    # ---------------- P7: continuous uncertainty slope (binary; descriptive) ----------------
    db = data["binary"]
    grid = np.array(art["entropy_cutpoints"]["binary"]["grid101"])
    u = np.interp(db["ent"], grid, np.linspace(0, 1, 101))
    bi = C3.binary_logloss_rows(db["y"].astype(float), db["PC"]) - C3.binary_logloss_rows(db["y"].astype(float), db["PD"])
    pats, pat_idx = np.unique(db["public_id"], return_inverse=True)
    slope_obs = slope_stat(bi, u, np.ones(len(bi)))
    rng = np.random.RandomState(C3.RANDOM_SEED)
    sv = np.array([slope_stat(bi, u, np.bincount(rng.randint(0, len(pats), len(pats)), minlength=len(pats)).astype(float)[pat_idx])
                   for _ in range(C3.N_BOOT)])
    slope = {"slope_logloss_benefit_per_unit_percentile": slope_obs, "ci_lo": float(np.percentile(sv, 2.5)), "ci_hi": float(np.percentile(sv, 97.5)),
             "benefit_at_90th_minus_10th_percentile": slope_obs * 0.8,
             "note": "positive = larger RNA (Model D) log-loss benefit at higher Model-C entropy; percentile = empirical CDF of the development entropy; descriptive"}

    # ---------------- R: reference rows from existing files ----------------
    R = "artifacts/results_clean_rerun/"
    ref = {"development_only_cv": json.load(open(OUT / "devcv_summary.json", encoding="utf-8"))}
    earlier = pd.read_csv(R + "direction3/partA_predictive/partA_delta_within_group.csv")
    ref["earlier_pooled_overall_primary"] = earlier[(earlier.group == "all") & earlier.is_primary_metric][
        ["task", "metric", "n_rows", "n_patients", "delta_D_minus_C", "ci_lo", "ci_hi"]].to_dict("records")
    ref["earlier_pooled_addon"] = json.load(open(R + "rna_low_variance_incremental/observed_summary.json", encoding="utf-8"))["patient_bootstrap"]
    ref["earlier_pooled_addon_delta_rps"] = json.load(open(R + "rna_low_variance_incremental/observed_summary.json", encoding="utf-8"))["delta_rps"]
    chk("H7 frozen artifact unchanged at the end of the run", D.sha_lf(fp) == recorded)

    # ---------------- write ----------------
    w.to_csv(OUT / "heldout_within_groups.csv", index=False, float_format="%.10g")
    i.to_csv(OUT / "heldout_interactions.csv", index=False, float_format="%.10g")
    f1.to_csv(OUT / "heldout_macro_f1_descriptive.csv", index=False, float_format="%.10g")
    neg.to_csv(OUT / "heldout_negative_control_summary.csv", index=False, float_format="%.10g")
    pd.DataFrame({"permutation": np.arange(B_ADDON), "seed": LV.PERM_SEED_BASE + np.arange(B_ADDON), "delta_rps": null}).to_csv(
        OUT / "heldout_addon_permutation_null.csv", index=False, float_format="%.10g")
    pm = {"binary": "auprc", "multiclass": "rps"}
    report = {"label": LABEL, "protocol_sha256": rec, "frozen_development_sha256": recorded, "code_sha256_lf": D.sha_lf(__file__),
              "heldout": {"patients": 309, "binary_rna_rows": 2405, "multiclass_rna_rows": 2689, "rna_patients": 213},
              "P1_overall": w[(w.group == "all") & w.is_primary_metric][["task", "metric", "n_rows", "n_patients", "delta_D_minus_C",
                                                                       "ci_lo", "ci_hi"]].to_dict("records"),
              "P2_primary_interactions": i[(i.contrast == "PRIMARY_Q4_vs_Q1_Q3") & i.is_primary_metric][
                  ["task", "metric", "interaction_a_minus_b", "ci_lo", "ci_hi", "p_boot_two_sided", "adjusted_p"]].to_dict("records"),
              "P3_addon": addon, "P7_slope": slope, "reference": ref, "checks": checks,
              "n_not_evaluable_rows": int(w["delta_D_minus_C"].isna().sum() + i["interaction_a_minus_b"].isna().sum()),
              "interpretation": ("post hoc, exploratory; evaluated in held-out validation and test patients with locked train-only models; "
                                 "the earlier pooled results had been seen; not independent; not external validation; no biological-value claim"),
              "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seconds": time.time() - t_start}
    D.dump_json(report, REPORT)
    pd.DataFrame(checks).to_csv(OUT / "heldout_checks.csv", index=False)
    man = {f.name: D.sha_lf(f) for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.csv")) if f.name != "heldout_manifest.json"}
    D.dump_json({"files": man, "frozen_development_sha256": recorded, "protocol_sha256": rec}, OUT / "heldout_manifest.json")

    print("\n=== P1 overall D - C (held-out) ===")
    for r in report["P1_overall"]:
        print(f"  {r['task']:10s} {r['metric']:6s} delta = {r['delta_D_minus_C']:+.5f} (95% CI {r['ci_lo']:+.5f} to {r['ci_hi']:+.5f}), n={r['n_rows']} rows / {r['n_patients']} patients")
    print("=== P2 primary interactions (Q4 - Q1..Q3) ===")
    for r in report["P2_primary_interactions"]:
        print(f"  {r['task']:10s} {r['metric']:6s} interaction = {r['interaction_a_minus_b']:+.5f} ({r['ci_lo']:+.5f} to {r['ci_hi']:+.5f}); Holm p = {r['adjusted_p']:.3f}")
    print("=== P3 add-on ===")
    print(f"  delta RPS = {obs_delta:+.3e} (95% CI {boot['ci_lower']:+.3e} to {boot['ci_upper']:+.3e}); permutation p = {p_perm:.4g} (k={k}/{B_ADDON}); {comp}")
    print(f"  decision: {decision}")
    print(f"\nStage H finished in {(time.time() - t_start) / 60:.1f} min. Report: {REPORT}")


if __name__ == "__main__":
    main()
