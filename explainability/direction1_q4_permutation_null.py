"""
Direction 1 -- STEP 1 of the fixed sequence: scrambled-RNA permutation null for the locked binary Q4 interaction.

*** EXPLORATORY POST-HOC ROBUSTNESS ANALYSIS. ***
It is another look at the same train+val OOF data that generated the original Q4 result. The empirical one-sided
permutation p is evidence about whether the observed directional interaction exceeds what a same-architecture
Model D with SCRAMBLED RNA produces; it does NOT make the original Direction 1 finding confirmatory. It is counted
among the Direction 1 looks (Q4, top-20%, top-33%, continuous rank, raw entropy, spline, this null, and Step 2).
Directions 1-4 and the continuous-sensitivity outputs are not modified.

ESTIMAND (identical to the locked D1 analysis, binary task, primary metric AUPRC)
    interaction = (D - C in Q4) - (D - C in Q1-Q3)
    with D - C averaged over the 5 locked OOF repeats and the frozen D1 groups (Q4 = top quartile of mean Model-C
    entropy; comparator Q1-Q3); unit weights (the observed analysis point estimate).

OBSERVED: recomputed from the locked OOF predictions by the same Evaluator as Direction 1.

NULL (each permutation = one scrambled RNA dataset evaluated over ALL 5 repeats, like the observed analysis):
    * permute WHOLE RNA vectors across distinct (patient, sample) units; days_since_rna_sample and RNA availability are
      untouched (direction3_predictive.permuted_rna_frame);
    * retrain Model D with the LOCKED XGBoost parameters on the locked fold splits of all 5 repeats (seeds 42 + repeat),
      fold-local preprocessing, train+val only, binary task (25 fold-fits per permutation);
    * Model C = the locked OOF predictions; groups = the frozen D1 groups (they depend on Model C only);
    * aggregate exactly as the observed statistic: mean over repeats of (AUPRC_Dperm - AUPRC_C) in Q4 minus the same in Q1-Q3.
    * 200 permutations, seeds 20042 + p, recorded; checkpoint per permutation; NO interim looks (no p-value or
      null summary is computed or printed before all 200 permutations are complete).

GATES (hard stops; no permutation runs unless BOTH pass)
    Gate 1: the observed interaction recomputed from the locked OOF (pinned SHA256) equals the locked D1 value
            +0.013948030608735174 (|diff| < 1e-12) and matches D1's run_summary.json and CSV value.
    Gate 2: refitting TRUE (unpermuted) Model D for ALL 5 repeats reproduces each repeat's locked OOF Model-D
            predictions: max |difference| < 1e-6 for every repeat.

DECISION (literal, fixed in advance): one-sided empirical p = (1 + k) / (B + 1), k = #null interactions >= observed.
    p <= 0.05 -> Step 2 (freshness-standardized log-loss check) is ALLOWED; p > 0.05 -> stop.
    Either way Direction 1 stops after this sequence: no further entropy or freshness searches.
    The exact 95% Monte Carlo interval of the permutation p is reported; a borderline p is reported with it, the rule is
    not bent. gate_decision.json records the decision.

Outputs only under artifacts/results_clean_rerun/direction1_sensitivity/q4_permutation_null/
(smoke: .../_smoke/q4_permutation_null/; --gates-only: .../gates_only/). Existing outputs are never overwritten.

Run from the repo root:
    python -m explainability.direction1_q4_permutation_null --gates-only
    python -m explainability.direction1_q4_permutation_null --smoke [--force]     (gates + 2 permutations)
    python -m explainability.direction1_q4_permutation_null                        (gates + 200 permutations)
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

from explainability.direction1_uncertainty import aligned_arrays, fixed_groups, load_and_verify
from explainability.direction3_common import (
    N_FOLDS, N_REPEATS, RANDOM_SEED, load_feature_sets, load_master, pathway_columns, require, sha256_file,
    write_csv, write_json, write_output_manifest,
)
from explainability.direction3_predictive import (
    Evaluator, align_repeats, load_locked_params, make_task_df, oof_task, permuted_rna_frame,
)

D1_DIR = "artifacts/results_clean_rerun/direction1/uncertainty"
OUT_ROOT = "artifacts/results_clean_rerun/direction1_sensitivity"
LABEL = ("DIRECTION 1 STEP 1 -- EXPLORATORY POST-HOC ROBUSTNESS ANALYSIS: scrambled-RNA permutation null for the "
         "locked binary Q4 interaction (does not make D1 confirmatory)")
LOCKED_OBSERVED_INTERACTION = 0.013948030608735174      # locked D1 binary AUPRC Q4 interaction
GATE1_TOL = 1e-12
GATE2_TOL = 1e-6
N_PERM_FULL = 200
N_PERM_SMOKE = 2
SEED_BASE = RANDOM_SEED + 20000                          # permutation p uses seed SEED_BASE + p (20042 + p)
ALPHA = 0.05
REPEATS = list(range(N_REPEATS))
HIGHER_IS_BETTER = {"auprc": True, "auroc": True, "logloss": False}


# ---------------------------------------------------------------------------
# Core computations (no file access; callable from tests)
# ---------------------------------------------------------------------------
def round10(a):
    """Round to 10 significant digits exactly like the locked OOF files (float_format='%.10g')."""
    a = np.asarray(a, dtype=float)
    return np.array([float("%.10g" % v) for v in a.ravel()]).reshape(a.shape)


def prepare_locked(d):
    """Locked arrays + frozen D1 groups (same functions as Direction 1)."""
    pair_id, public_id, y, PC, PD, unc = aligned_arrays(d, "binary")
    groups, quartile, pct = fixed_groups(unc)
    pats, pat_idx = np.unique(public_id, return_inverse=True)
    return {"pair_id": pair_id, "public_id": public_id, "y": y, "PC": PC, "PD": PD, "groups": groups,
            "pats": pats, "pat_idx": pat_idx, "ones": np.ones(len(pats))}


def interactions_from_delta(delta, gnames, metrics, hi="Q4_high_uncertainty", lo="Q1_Q3_lower_uncertainty"):
    hi_i, lo_i = gnames.index(hi), gnames.index(lo)
    return {m: float(delta[hi_i, k] - delta[lo_i, k]) for k, m in enumerate(metrics)}, \
           {m: (float(delta[hi_i, k]), float(delta[lo_i, k])) for k, m in enumerate(metrics)}


def observed_interaction(L, PD=None, full_groups=True):
    """Observed statistic via Direction 1's Evaluator (full D1 group set, or only the two groups needed)."""
    groups = L["groups"] if full_groups else {k: L["groups"][k] for k in
                                                ("Q4_high_uncertainty", "Q1_Q3_lower_uncertainty")}
    ev = Evaluator("binary", L["y"], L["pat_idx"], L["PC"], L["PD"] if PD is None else PD, groups)
    delta = ev.delta(L["ones"])
    return interactions_from_delta(delta, ev.gnames, ev.metrics)


def gate1(L, expected=LOCKED_OBSERVED_INTERACTION, d1_reference=None, tol=GATE1_TOL):
    """Observed interaction must reproduce the locked D1 value."""
    full, parts = observed_interaction(L, full_groups=True)
    two, _ = observed_interaction(L, full_groups=False)
    obs = full["auprc"]
    res = {"observed_auprc_interaction": obs, "expected_locked_value": expected, "abs_diff_vs_expected": abs(obs - expected),
           "two_group_evaluator_value": two["auprc"], "abs_diff_full_vs_two_group": abs(obs - two["auprc"]),
           "delta_Q4_auprc": parts["auprc"][0], "delta_Q1_Q3_auprc": parts["auprc"][1],
           "logloss_interaction_descriptive": full["logloss"], "tolerance": tol}
    require(abs(obs - expected) < tol,
            f"GATE 1 FAILED: observed interaction {obs!r} differs from the locked {expected!r} by {abs(obs - expected):.3e}")
    require(abs(obs - two["auprc"]) < 1e-14, "GATE 1 FAILED: two-group evaluator differs from the full D1 evaluator")
    if d1_reference is not None:
        res["d1_run_summary_value"] = d1_reference["json"]
        res["d1_csv_text_10sig"] = d1_reference["csv_text"]
        require(abs(obs - d1_reference["json"]) < tol, "GATE 1 FAILED: differs from D1 run_summary.json")
        # the D1 CSV stores this value with float_format="%.10g": compare as TEXT (no float-parser question)
        require("%.10g" % obs == d1_reference["csv_text"],
                f"GATE 1 FAILED: differs from the D1 CSV value ({'%.10g' % obs} vs {d1_reference['csv_text']})")
    res["passed"] = True
    return res


def refit_true_D(task_df, fs, params, repeats=REPEATS, quiet=True):
    """Refit TRUE (unpermuted) Model D on the locked folds of every repeat."""
    sink = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        recs, _ = oof_task("binary", task_df, fs, params[("binary", "c")], params[("binary", "d")], repeats, fit_c=False)
        pid, _, y, _, PD = align_repeats(recs, repeats, with_c=False)
    return pid, y, PD


def gate2_compare(refit_pid, refit_y, refit_PD, L, tol=GATE2_TOL):
    """Each repeat's refit Model-D predictions must equal the locked OOF predictions (max |diff| < tol, every repeat)."""
    require(np.array_equal(refit_pid, L["pair_id"]) and np.array_equal(refit_y, L["y"]),
            "GATE 2 FAILED: refit rows differ from the locked OOF rows")
    rows = []
    for r in range(refit_PD.shape[0]):
        diff = np.abs(refit_PD[r] - L["PD"][r])
        rows.append({"repeat": r, "n_rows": int(diff.size), "max_abs_diff": float(diff.max()),
                     "mean_abs_diff": float(diff.mean()), "tolerance": tol, "passed": bool(diff.max() < tol)})
    t = pd.DataFrame(rows)
    bad = t[~t["passed"]]
    require(bad.empty, "GATE 2 FAILED: refit Model D does not reproduce the locked OOF predictions:\n" + bad.to_string(index=False))
    return t


def permutation_statistic(L, task_df, fs, params, path_cols, p):
    """One permutation: scrambled RNA, retrain D on all 5 locked repeats, aggregate exactly like the observed analysis."""
    seed = SEED_BASE + p
    perm_df = permuted_rna_frame(task_df, path_cols, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        recs, _ = oof_task("binary", perm_df, fs, params[("binary", "c")], params[("binary", "d")], REPEATS, fit_c=False)
        pid, _, y, _, PDp = align_repeats(recs, REPEATS, with_c=False)
    require(np.array_equal(pid, L["pair_id"]) and np.array_equal(y, L["y"]), "permuted run evaluated different rows")
    PDp = round10(PDp)                                   # same 10-significant-digit storage as the locked OOF
    inter, parts = observed_interaction(L, PD=PDp, full_groups=False)
    return {"permutation": p, "seed": seed,
            **{f"interaction_{m}": inter[m] for m in inter},
            **{f"delta_Q4_{m}": parts[m][0] for m in parts}, **{f"delta_Q1_Q3_{m}": parts[m][1] for m in parts}}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def config_hash(B, locked_hashes, params, code_sha):
    blob = json.dumps({"B": B, "repeats": REPEATS, "folds": N_FOLDS, "seed_base": SEED_BASE, "locked_hashes": locked_hashes,
                       "params": {f"{k[0]}_{k[1]}": v for k, v in params.items() if k[0] == "binary"},
                       "code_sha": code_sha}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_checkpoint(path, p, chash):
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    require(c.get("config_hash") == chash and c.get("permutation") == p and c.get("seed") == SEED_BASE + p,
            f"checkpoint {path} does not match the current configuration (seed/config/code changed): refusing to mix")
    return c


def save_checkpoint(path, rec, chash, seconds):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({**rec, "config_hash": chash, "seconds": seconds}, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Summary (computed only after ALL permutations exist)
# ---------------------------------------------------------------------------
def clopper_pearson(k, B, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, B - k + 1))
    hi = 1.0 if k == B else float(beta.ppf(1 - a / 2, k + 1, B - k))
    return lo, hi


def summarize(obs_inter, null_df, B):
    require(len(null_df) == B, "the null distribution is incomplete")
    out = {}
    for m, better_high in HIGHER_IS_BETTER.items():
        v = null_df[f"interaction_{m}"].to_numpy(float)
        o = obs_inter[m]
        k = int((v >= o).sum()) if better_high else int((v <= o).sum())
        lo, hi = clopper_pearson(k, B)
        out[m] = {"observed_interaction": o, "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)),
                  "null_p2.5": float(np.percentile(v, 2.5)), "null_p97.5": float(np.percentile(v, 97.5)),
                  "standardized_excess": float((o - v.mean()) / v.std(ddof=1)), "n_exceeding_or_equal": k,
                  "n_permutations": B, "empirical_one_sided_p": float((1 + k) / (B + 1)),
                  "p_monte_carlo_95_interval": [lo, hi],
                  "direction": "higher = RNA benefit larger in uncertain cases" if better_high
                               else "lower (more negative) = RNA benefit larger in uncertain cases",
                  "role": "PRIMARY (AUPRC)" if m == "auprc" else "descriptive"}
    return out


# ---------------------------------------------------------------------------
def d1_reference_values():
    with open(f"{D1_DIR}/run_summary.json", encoding="utf-8") as f:
        rs = json.load(f)
    js = [t for t in rs["primary_tests"] if t["task"] == "binary" and t["metric"] == "auprc"]
    require(len(js) == 1, "D1 run_summary primary binary AUPRC test not found")
    it = pd.read_csv(f"{D1_DIR}/direction1_uncertainty_interactions.csv",
                     dtype={"interaction_uncertain_minus_comparator": str})
    it = it[(it["task"] == "binary") & (it["metric"] == "auprc") & (it["contrast"] == "PRIMARY_Q4_vs_Q1_Q3")]
    require(len(it) == 1, "D1 primary binary AUPRC interaction row not found")
    return {"json": float(js[0]["interaction_uncertain_minus_comparator"]),
            "csv_text": str(it["interaction_uncertain_minus_comparator"].iloc[0])}


def verify_d1_outputs_unchanged():
    with open(f"{D1_DIR}/output_manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    checked = []
    for rel, h in man["files"].items():
        p = f"{D1_DIR}/{rel}"
        if os.path.exists(p):
            require(sha256_file(p) == h, f"LOCKED D1 FILE CHANGED: {rel}")
            checked.append(rel)
    return checked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    B = N_PERM_SMOKE if args.smoke else N_PERM_FULL
    base = f"{OUT_ROOT}/_smoke/q4_permutation_null" if args.smoke else f"{OUT_ROOT}/q4_permutation_null"
    out = f"{base.rsplit('/', 1)[0]}/gates_only" if args.gates_only else base
    marker = f"{out}/output_manifest.json"
    if args.force:
        require(args.smoke, "--force is only allowed together with --smoke (real outputs are never overwritten)")
    if os.path.exists(marker) and not args.force:
        raise SystemExit(f"[q4_null] {marker} already exists: never overwritten (delete deliberately, or --smoke --force).")
    print(LABEL)

    checked = verify_d1_outputs_unchanged()
    d, oof_path = load_and_verify("binary")                    # pinned SHA256 of the locked D3 OOF predictions
    L = prepare_locked(d)
    fs = load_feature_sets()
    path_cols = pathway_columns(fs)
    master = load_master()
    task_df = make_task_df(master, "binary")
    params, locked_hashes = load_locked_params()
    locked_hashes = {**locked_hashes, oof_path: sha256_file(oof_path)}

    # ---- Gate 1 ----
    g1 = gate1(L, LOCKED_OBSERVED_INTERACTION, d1_reference_values())
    print(f"GATE 1 passed: observed binary Q4 AUPRC interaction = {g1['observed_auprc_interaction']!r} "
          f"(locked {LOCKED_OBSERVED_INTERACTION!r}, |diff| = {g1['abs_diff_vs_expected']:.2e})")
    # ---- Gate 2 (all 5 repeats) ----
    print("GATE 2: refitting TRUE Model D for all 5 repeats (25 fold fits)...", flush=True)
    rp, ry, rPD = refit_true_D(task_df, fs, params)
    g2 = gate2_compare(rp, ry, rPD, L)
    refit_obs, _ = observed_interaction(L, PD=rPD, full_groups=False)
    print("GATE 2 passed: max |refit - locked| per repeat = " + ", ".join(f"{x:.2e}" for x in g2["max_abs_diff"]))

    os.makedirs(out, exist_ok=True)
    files = []
    gates = {"label": LABEL, "gate1": g1, "gate2": {"per_repeat": g2.to_dict("records"), "tolerance": GATE2_TOL,
                                                     "refit_unpermuted_interaction_auprc": refit_obs["auprc"],
                                                     "passed": True}, "d1_files_verified_unchanged": checked}
    p = f"{out}/gates.json"
    write_json(gates, p)
    files.append(p)
    p = f"{out}/gate2_repeat_diffs.csv"
    write_csv(g2, p, float_format="%.10g")
    files.append(p)
    code_sha = sha256_file(os.path.abspath(__file__))
    if args.gates_only:
        man = write_output_manifest(out, files, {"label": LABEL, "mode": "gates only", "code_sha256": code_sha,
                                                 "locked_input_sha256": locked_hashes})
        print(f"\nBoth gates passed. No permutations were run. Outputs in {out}\nCombined SHA256: {man['combined_sha256']}")
        return

    # ---- permutations (checkpointed, fixed seeds, no interim looks) ----
    ck_dir = f"{out}/checkpoints"
    os.makedirs(ck_dir, exist_ok=True)
    chash = config_hash(B, locked_hashes, params, code_sha)
    recs, t_start, done_now = [], time.time(), 0
    print(f"\n{B} permutations x 5 repeats (seeds {SEED_BASE}..{SEED_BASE + B - 1}); checkpointing to {ck_dir}; "
          f"no interim p-values are computed.", flush=True)
    for pidx in range(B):
        ck = f"{ck_dir}/perm_{pidx:04d}.json"
        if os.path.exists(ck):
            recs.append(load_checkpoint(ck, pidx, chash))
            continue
        t1 = time.time()
        rec = permutation_statistic(L, task_df, fs, params, path_cols, pidx)
        secs = time.time() - t1
        save_checkpoint(ck, rec, chash, secs)
        recs.append({**rec, "config_hash": chash, "seconds": secs})
        done_now += 1
        el = time.time() - t_start
        print(f"  permutation {pidx + 1}/{B} done ({time.time() - t1:.0f}s; elapsed {el / 60:.1f} min; "
              f"est. remaining {el / done_now * (B - pidx - 1) / 60:.1f} min)", flush=True)
    null_df = pd.DataFrame(recs).drop(columns=["config_hash"])
    null_df = null_df.sort_values("permutation").reset_index(drop=True)

    # ---- summary (only now that all B permutations exist) ----
    obs_inter, _ = observed_interaction(L, full_groups=True)
    summ = summarize(obs_inter, null_df, B)
    prim = summ["auprc"]
    step2_allowed = bool(prim["empirical_one_sided_p"] <= ALPHA)
    decision = {"label": LABEL, "rule": f"literal: empirical one-sided p = (1+k)/(B+1) <= {ALPHA} -> Step 2 allowed; else stop",
                "observed_interaction_auprc": prim["observed_interaction"], "n_exceeding_or_equal": prim["n_exceeding_or_equal"],
                "n_permutations": B, "empirical_one_sided_p": prim["empirical_one_sided_p"],
                "p_monte_carlo_95_interval": prim["p_monte_carlo_95_interval"], "step2_allowed": step2_allowed,
                "stop_direction1_after_sequence": True,
                "note": ("exploratory post-hoc robustness analysis; does not make D1 confirmatory; a borderline p is "
                         "reported with its Monte Carlo interval and the rule is not bent; smoke runs are not decisions"
                         if not args.smoke else "SMOKE RUN: not a decision (B=2)")}
    p = f"{out}/permutation_null_distribution.csv"
    write_csv(null_df, p, float_format="%.12g")
    files.append(p)
    for name, obj in (("permutation_summary.json", {"label": LABEL, "smoke": bool(args.smoke), "summary": summ}),
                      ("gate_decision.json", decision)):
        p = f"{out}/{name}"
        write_json(obj, p)
        files.append(p)
    run = {"label": LABEL, "smoke": bool(args.smoke), "runtime_seconds": round(time.time() - t0, 1),
           "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "n_permutations": B,
           "repeats_per_permutation": len(REPEATS), "seed_base": SEED_BASE,
           "seeds": f"{SEED_BASE} + permutation index ({SEED_BASE}..{SEED_BASE + B - 1})",
           "locked_input_sha256": locked_hashes, "code_sha256": code_sha, "config_hash": chash,
           "status": "EXPLORATORY POST-HOC ROBUSTNESS ANALYSIS; does not make D1 confirmatory",
           "stopping": "D1 stops after this sequence; Step 2 only if p<=0.05; no further entropy/freshness searches"}
    p = f"{out}/run_summary.json"
    write_json(run, p)
    files.append(p)
    man = write_output_manifest(out, files, {"label": LABEL, "smoke": bool(args.smoke), "code_sha256": code_sha,
                                             "locked_input_sha256": locked_hashes})
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {out}\nCombined SHA256: {man['combined_sha256']}")
    if args.smoke:
        print("SMOKE RUN ONLY (B=2): not a decision.")
    else:
        print(f"Empirical one-sided p = {prim['empirical_one_sided_p']:.4f} (k = {prim['n_exceeding_or_equal']}/{B}); "
              f"Step 2 allowed: {step2_allowed}. Direction 1 stops after this sequence.")


if __name__ == "__main__":
    main()
