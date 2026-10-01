"""
Direction 1 -- Does RNA add predictive value specifically in clinically uncertain cases?

FROZEN DESIGN
* Reuse the already-locked Direction 3 out-of-fold Model C/D predictions. No model retraining and no test split.
* Uncertainty is defined ONLY from Model C (the non-RNA clinical+temporal+treatment model), never from outcomes or Model D.
  Binary: normalized Bernoulli entropy. Multiclass: Shannon entropy / log(6).
* For each pair, average Model-C entropy over the 5 OOF repeats, then freeze one patient-row subgroup assignment.
* PRIMARY uncertain subgroup = top quartile (Q4) of mean Model-C entropy; comparator = Q1-Q3.
* Primary scientific estimand = interaction: (D-C in Q4) - (D-C in Q1-Q3).
  Binary primary metric AUPRC (higher better); multiclass primary metric RPS (lower better).
* Also report D-C within Q4, all four entropy quartiles, AUROC/logloss, and multiclass macro-F1 descriptively.
* Patient-cluster bootstrap: 2000 resamples, seed 42, shared across repeats/groups/metrics.
* The two primary interaction tests (binary AUPRC, multiclass RPS) form one Holm family.
* Threshold sensitivities: top 20% and top 33% uncertain vs remainder; secondary/descriptive, no redefinition of primary.

Interpretation: predictive incremental value in model-defined uncertain cases; not causal and not a molecular association analysis.

Run from repo root:
    python -m explainability.direction1_uncertainty --smoke
    python -m explainability.direction1_uncertainty
"""

import argparse
import datetime
import json
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from explainability.direction3_common import (
    MULTICLASS_CLASSES, N_BOOT, RANDOM_SEED, holm_adjust, require, sha256_file,
    write_csv, write_json, write_output_manifest,
)
from explainability.direction3_predictive import Evaluator, percentile_ci, boot_p

D3_DIR = "artifacts/results_clean_rerun/direction3/partA_predictive"
EXPECTED_D3_COMBINED_SHA = "ce6c5170e5ceee1fe48d8f0a35b7a042f4ae3e5620866fb108be5c1bf4289a8a"
EXPECTED_OOF_SHA = {
    "binary": "2841bca0553b5afd6e8312721eb949ba64f8c958764b641ae9fe70a5cff1f82b",
    "multiclass": "9701df1fd3d2728f085f508daf1a472cfae263af48bf640cecfd62d600ee7716",
}
PRIMARY_METRIC = {"binary": "auprc", "multiclass": "rps"}
HIGHER_IS_BETTER = {"auprc": True, "auroc": True, "logloss": False, "rps": False}


def load_and_verify(task):
    manp = f"{D3_DIR}/output_manifest.json"
    require(os.path.exists(manp), f"missing locked D3 manifest: {manp}")
    with open(manp, encoding="utf-8") as f:
        man = json.load(f)
    require(man.get("combined_sha256") == EXPECTED_D3_COMBINED_SHA,
            "Direction 3 Part A combined SHA differs from the locked run")
    p = f"{D3_DIR}/oof_predictions/oof_{task}.csv"
    require(os.path.exists(p), f"missing D3 OOF file: {p}")
    require(sha256_file(p) == EXPECTED_OOF_SHA[task], f"locked OOF SHA mismatch for {task}")
    d = pd.read_csv(p)
    require(set(d["repeat"].unique()) == set(range(5)), f"{task}: expected repeats 0..4")
    require(not d.duplicated(["repeat", "pair_id"]).any(), f"{task}: duplicate repeat/pair rows")
    # Each pair must appear once in every repeat and preserve patient/outcome.
    g = d.groupby("pair_id", sort=False)
    require((g.size() == 5).all(), f"{task}: not every pair has 5 OOF predictions")
    require((g["public_id"].nunique() == 1).all() and (g["y_true"].nunique() == 1).all(),
            f"{task}: patient/outcome differs across repeats")
    return d, p


def entropy_binary(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / math.log(2.0)


def entropy_multiclass(P):
    P = np.clip(np.asarray(P, float), 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return -(P * np.log(P)).sum(axis=1) / math.log(P.shape[1])


def aligned_arrays(d, task):
    reps = sorted(d["repeat"].unique())
    first = d[d["repeat"] == reps[0]].sort_values("pair_id").reset_index(drop=True)
    pair_id = first["pair_id"].to_numpy()
    public_id = first["public_id"].astype(str).to_numpy()
    y = first["y_true"].to_numpy(dtype=int)
    PC, PD, ent = [], [], []
    for r in reps:
        x = d[d["repeat"] == r].sort_values("pair_id").reset_index(drop=True)
        require(np.array_equal(x["pair_id"].to_numpy(), pair_id), f"{task}: pair alignment drift repeat {r}")
        require(np.array_equal(x["y_true"].to_numpy(dtype=int), y), f"{task}: outcome alignment drift repeat {r}")
        if task == "binary":
            pc, pd_ = x["p_C"].to_numpy(float), x["p_D"].to_numpy(float)
            PC.append(pc); PD.append(pd_); ent.append(entropy_binary(pc))
        else:
            ccols = [f"pC_{c}" for c in MULTICLASS_CLASSES]
            dcols = [f"pD_{c}" for c in MULTICLASS_CLASSES]
            pc, pd_ = x[ccols].to_numpy(float), x[dcols].to_numpy(float)
            require(np.allclose(pc.sum(1), 1, atol=1e-5) and np.allclose(pd_.sum(1), 1, atol=1e-5),
                    f"{task}: probabilities do not sum to one")
            PC.append(pc); PD.append(pd_); ent.append(entropy_multiclass(pc))
    return pair_id, public_id, y, np.stack(PC), np.stack(PD), np.stack(ent).mean(axis=0)


def fixed_groups(u):
    # rank(method='first') makes ties deterministic by pair-id order, and yields exact-sized bins.
    pct = pd.Series(u).rank(method="first", pct=True).to_numpy()
    q = np.minimum(4, np.ceil(4 * pct).astype(int))
    groups = {
        "all": np.arange(len(u)),
        "Q1_low_uncertainty": np.flatnonzero(q == 1),
        "Q2": np.flatnonzero(q == 2),
        "Q3": np.flatnonzero(q == 3),
        "Q4_high_uncertainty": np.flatnonzero(q == 4),
        "Q1_Q3_lower_uncertainty": np.flatnonzero(q <= 3),
    }
    # threshold sensitivities, fixed before looking at outcomes
    groups["top20_uncertainty"] = np.flatnonzero(pct > 0.80)
    groups["bottom80_uncertainty"] = np.flatnonzero(pct <= 0.80)
    groups["top33_uncertainty"] = np.flatnonzero(pct > (2/3))
    groups["bottom67_uncertainty"] = np.flatnonzero(pct <= (2/3))
    require(len(groups["Q4_high_uncertainty"]) > 0, "empty Q4")
    require(set(groups["Q4_high_uncertainty"]).isdisjoint(set(groups["Q1_Q3_lower_uncertainty"])), "group overlap")
    require(len(groups["Q4_high_uncertainty"]) + len(groups["Q1_Q3_lower_uncertainty"]) == len(u), "group coverage")
    return groups, q, pct


def run_task(task, d, n_boot):
    pair_id, public_id, y, PC, PD, uncertainty = aligned_arrays(d, task)
    groups, quartile, pct = fixed_groups(uncertainty)
    pats, pat_idx = np.unique(public_id, return_inverse=True)
    ev = Evaluator(task, y, pat_idx, PC, PD, groups)
    ones = np.ones(len(pats))
    obs = ev.evaluate(ones)
    delta = (obs[1] - obs[0]).mean(axis=0)
    gnames, metrics = ev.gnames, ev.metrics

    rng = np.random.RandomState(RANDOM_SEED)
    boot = np.full((n_boot, len(gnames), len(metrics)), np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, len(pats), len(pats))
        boot[b] = ev.delta(np.bincount(idx, minlength=len(pats)).astype(float))

    rows = []
    for gi, g in enumerate(gnames):
        for ki, met in enumerate(metrics):
            lo, hi, nv = percentile_ci(boot[:, gi, ki])
            rows.append({
                "task": task, "group": g, "metric": met,
                "n_rows": int(len(groups[g])), "n_patients": int(len(np.unique(public_id[groups[g]]))),
                "C_mean": float(obs[0, :, gi, ki].mean()), "D_mean": float(obs[1, :, gi, ki].mean()),
                "delta_D_minus_C": float(delta[gi, ki]), "ci_lo": lo, "ci_hi": hi,
                "p_boot_two_sided": boot_p(boot[:, gi, ki]), "n_valid_bootstrap": nv,
                "better_when": "higher" if HIGHER_IS_BETTER[met] else "lower",
                "role": "primary_subgroup" if g == "Q4_high_uncertainty" else "descriptive_or_sensitivity",
            })
    within = pd.DataFrame(rows)

    # Interaction contrasts: uncertain subgroup D-C minus comparator D-C.
    specs = [
        ("PRIMARY_Q4_vs_Q1_Q3", "Q4_high_uncertainty", "Q1_Q3_lower_uncertainty", "primary"),
        ("SENS_top20_vs_bottom80", "top20_uncertainty", "bottom80_uncertainty", "sensitivity"),
        ("SENS_top33_vs_bottom67", "top33_uncertainty", "bottom67_uncertainty", "sensitivity"),
    ]
    inter = []
    for label, hi_g, lo_g, role in specs:
        hi_i, lo_i = gnames.index(hi_g), gnames.index(lo_g)
        for ki, met in enumerate(metrics):
            v = boot[:, hi_i, ki] - boot[:, lo_i, ki]
            lo, hi, nv = percentile_ci(v)
            inter.append({
                "task": task, "contrast": label, "role": role, "metric": met,
                "delta_uncertain": float(delta[hi_i, ki]), "delta_comparator": float(delta[lo_i, ki]),
                "interaction_uncertain_minus_comparator": float(delta[hi_i, ki] - delta[lo_i, ki]),
                "ci_lo": lo, "ci_hi": hi, "p_boot_two_sided": boot_p(v), "n_valid_bootstrap": nv,
                "better_when": "higher" if HIGHER_IS_BETTER[met] else "lower",
                "interpret_sign": ("positive interaction favors greater RNA benefit in uncertain cases"
                                   if HIGHER_IS_BETTER[met]
                                   else "negative interaction favors greater RNA benefit in uncertain cases"),
            })
    inter = pd.DataFrame(inter)

    # Descriptive multiclass macro-F1 by fixed uncertainty groups.
    f1rows = []
    if task == "multiclass":
        for g in ["all", "Q1_low_uncertainty", "Q2", "Q3", "Q4_high_uncertainty", "Q1_Q3_lower_uncertainty"]:
            rr = groups[g]
            for model, P in (("C", PC), ("D", PD)):
                vals = [f1_score(y[rr], P[r][rr].argmax(1), labels=list(range(6)), average="macro", zero_division=0)
                        for r in range(P.shape[0])]
                f1rows.append({"task": task, "group": g, "model": model,
                               "macro_f1_mean_over_repeats": float(np.mean(vals)),
                               "note": "DESCRIPTIVE ONLY; not part of Direction 1 inference"})

    assignment = pd.DataFrame({"task": task, "pair_id": pair_id, "public_id": public_id,
                               "modelC_mean_normalized_entropy": uncertainty,
                               "uncertainty_percentile_rank": pct, "uncertainty_quartile": quartile,
                               "primary_high_uncertainty_Q4": quartile == 4})
    return within, inter, pd.DataFrame(f1rows), assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_boot = 50 if args.smoke else N_BOOT
    out_dir = ("artifacts/results_clean_rerun/direction1/_smoke/uncertainty" if args.smoke
               else "artifacts/results_clean_rerun/direction1/uncertainty")
    manp = f"{out_dir}/output_manifest.json"
    if os.path.exists(manp) and not args.force:
        raise RuntimeError(f"refusing to overwrite existing output: {manp}; use --force only for an intentional rerun")
    os.makedirs(out_dir, exist_ok=True)
    print("DIRECTION 1 | RNA incremental value in clinically uncertain cases")
    print("Uncertainty = Model-C OOF entropy only; outcomes and Model D do not define the subgroup.")

    all_w, all_i, all_f, all_a, input_hashes = [], [], [], [], {}
    for task in ["binary", "multiclass"]:
        d, p = load_and_verify(task)
        input_hashes[p] = sha256_file(p)
        print(f"\n[{task}] {d['pair_id'].nunique()} unique RNA-available pairs; {d['public_id'].nunique()} patients")
        w, i, f, a = run_task(task, d, n_boot)
        all_w.append(w); all_i.append(i); all_f.append(f); all_a.append(a)

    within = pd.concat(all_w, ignore_index=True)
    inter = pd.concat(all_i, ignore_index=True)
    assign = pd.concat(all_a, ignore_index=True)
    f1 = pd.concat(all_f, ignore_index=True) if any(len(x) for x in all_f) else pd.DataFrame()

    # ONE confirmatory family: primary interaction on each task's primary metric.
    mask = ((inter["contrast"] == "PRIMARY_Q4_vs_Q1_Q3") &
            (((inter["task"] == "binary") & (inter["metric"] == "auprc")) |
             ((inter["task"] == "multiclass") & (inter["metric"] == "rps"))))
    require(int(mask.sum()) == 2, "expected exactly two primary interaction tests")
    inter["adjusted_p"] = np.nan
    inter["adjustment_method"] = "none (secondary/descriptive)"
    inter["adjustment_family"] = ""
    inter.loc[mask, "adjusted_p"] = holm_adjust(inter.loc[mask, "p_boot_two_sided"].to_numpy())
    inter.loc[mask, "adjustment_method"] = "Holm"
    inter.loc[mask, "adjustment_family"] = "Direction1 primary interaction: binary AUPRC + multiclass RPS"

    files = []
    for name, df in [
        ("direction1_within_uncertainty_groups.csv", within),
        ("direction1_uncertainty_interactions.csv", inter),
        ("direction1_uncertainty_assignments.csv", assign),
        ("direction1_multiclass_macro_f1_descriptive.csv", f1),
    ]:
        p = f"{out_dir}/{name}"
        write_csv(df, p, float_format="%.10g")
        files.append(p)

    prim = inter[mask].copy()
    summary = {
        "label": "DIRECTION 1: RNA incremental value in clinically uncertain cases",
        "smoke": bool(args.smoke), "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 1), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
        "locked_D3_partA_combined_sha256": EXPECTED_D3_COMBINED_SHA,
        "input_oof_sha256": input_hashes,
        "uncertainty_definition": "mean normalized Shannon entropy of Model C OOF probabilities across 5 repeats; no outcome/Model-D information",
        "primary_subgroup": "top quartile (Q4) uncertainty; fixed once per pair",
        "primary_estimand": "(D-C in Q4) - (D-C in Q1-Q3)",
        "primary_tests": prim[["task", "metric", "interaction_uncertain_minus_comparator", "ci_lo", "ci_hi",
                               "p_boot_two_sided", "adjusted_p"]].to_dict("records"),
        "multiplicity": "Holm across the two primary interaction tests; threshold sensitivities and other metrics are secondary/descriptive",
        "interpretation_limit": "predictive subgroup analysis only; RNA is baseline/as-of-sample biology; not causal",
    }
    p = f"{out_dir}/run_summary.json"; write_json(summary, p); files.append(p)
    man = write_output_manifest(out_dir, files, {
        "label": summary["label"], "smoke": bool(args.smoke),
        "locked_D3_partA_combined_sha256": EXPECTED_D3_COMBINED_SHA,
        "input_oof_sha256": input_hashes,
    })
    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in {out_dir}")
    print(f"Combined output SHA256: {man['combined_sha256']}")
    print("DIRECTION 1 UNCERTAINTY ANALYSIS")


if __name__ == "__main__":
    main()
