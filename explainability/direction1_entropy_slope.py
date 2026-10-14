"""
Direction 1 -- EXPLORATORY SENSITIVITY: does the RNA benefit change CONTINUOUSLY with Model-C entropy? (binary task)

*** EXPLORATORY / SENSITIVITY ANALYSIS. It follows the locked Direction 1 result, is not part of Direction 1's
Holm family, and supports no confirmatory claim. Direction 1 (and Directions 2-4) are not touched. ***

Outcome per visit pair i:   b_i = mean over the 5 locked OOF repeats of ( logloss_C - logloss_D )
    positive b = Model D (RNA) better.  Log-loss is linear, so the repeat average is exactly D1's repeat-averaged
    log-loss delta with the sign reversed.  One row per visit pair; the 5 repeats are NOT treated as independent rows.
Exposure: Model-C entropy exactly as Direction 1 froze it (mean normalized Shannon entropy of the Model-C OOF
    probabilities over the 5 repeats; rank percentile u_i in (0, 1] with D1's tie rule). No cutoff is chosen.
Inputs: the LOCKED Direction 3 Part A OOF predictions via direction1_uncertainty.load_and_verify (pinned SHA256s).
    No model training, no test split.

PRIMARY: weighted least-squares slope of b_i on the entropy RANK percentile u_i.
    Inference: patient-cluster bootstrap, 2000 resamples, seed 42, identical scheme to Direction 1
    (RandomState(42); one rng.randint(0, n_patients, n_patients) per draw; each visit pair carries the resample
    count of its patient as its weight, so all visit pairs of a patient move together).
    Reported with the estimated benefit at the 90th vs 10th entropy percentile and a cluster-robust (CR1) Wald
    cross-check (t, G-1 df).
SECONDARY robustness: the same slope on RAW (normalized) entropy.
SECONDARY / DESCRIPTIVE: restricted cubic spline on u (knots at the 10th, 50th, 90th percentiles of u) with a
    pointwise bootstrap band, a bootstrap-covariance Wald 2-df test of any dependence and a 1-df test of
    non-linearity; Spearman correlation (bootstrap CI); the five per-repeat slopes (no inference).

SANITY CHECKS (must pass or the run fails): the group means of b_i reproduce D1's committed log-loss group
deltas (direction1_within_uncertainty_groups.csv) and log-loss interactions
(direction1_uncertainty_interactions.csv) with the expected sign reversal (b = -(D - C)), and the existing D1
output files still match their manifest hashes.

Limitation: a slope shows how the C-to-D log-loss difference varies with Model-C entropy; with no row-level permuted-RNA
null it does not by itself identify RNA-specific signal.

Outputs only under artifacts/results_clean_rerun/direction1_sensitivity/entropy_slope/ (smoke: .../_smoke/entropy_slope/).
No existing file is modified; existing outputs are never overwritten.

Run from the repo root:
    python -m explainability.direction1_entropy_slope --smoke [--force]
    python -m explainability.direction1_entropy_slope
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata, t as t_dist

from explainability.direction1_uncertainty import aligned_arrays, fixed_groups, load_and_verify
from explainability.direction3_common import (
    N_BOOT, RANDOM_SEED, binary_logloss_rows, require, sha256_file, write_csv, write_json, write_output_manifest,
)
from explainability.direction3_predictive import boot_p, percentile_ci

D1_DIR = "artifacts/results_clean_rerun/direction1/uncertainty"
OUT_ROOT = "artifacts/results_clean_rerun/direction1_sensitivity"
LABEL = "DIRECTION 1 EXPLORATORY SENSITIVITY: continuous Model-C entropy vs RNA log-loss benefit (binary)"
KNOT_PCTS = (0.10, 0.50, 0.90)
GRID = np.round(np.arange(0.02, 0.9801, 0.02), 4)
SANITY_TOL = 1e-9


# ---------------------------------------------------------------------------
# Pure computation (no file access)
# ---------------------------------------------------------------------------
def benefit_per_pair(PC, PD, y):
    """b[r, i] = logloss_C - logloss_D (positive = D better); also the per-model losses. Same loss as D3's Evaluator."""
    yf = y.astype(float)
    ll_c = np.stack([binary_logloss_rows(yf, PC[r]) for r in range(PC.shape[0])])
    ll_d = np.stack([binary_logloss_rows(yf, PD[r]) for r in range(PD.shape[0])])
    return ll_c - ll_d, ll_c, ll_d


def wls_slope(x, b, w):
    """Closed-form weighted least-squares (intercept, slope)."""
    sw = w.sum()
    xb, bb = (w * x).sum() / sw, (w * b).sum() / sw
    sxx = (w * (x - xb) ** 2).sum()
    beta = (w * (x - xb) * (b - bb)).sum() / sxx
    return bb - beta * xb, beta


def wls_fit(X, b, w):
    return np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * b))


def rcs_basis(u, knots):
    """Restricted cubic spline, 3 knots (Harrell): columns [u, nonlinear term]."""
    t1, t2, t3 = knots
    p3 = lambda z: np.maximum(z, 0.0) ** 3
    nl = (p3(u - t1) - p3(u - t2) * (t3 - t1) / (t3 - t2) + p3(u - t3) * (t2 - t1) / (t3 - t2)) / (t3 - t1) ** 2
    return np.column_stack([u, nl])


def cluster_robust_slope(x, b, pat_idx):
    """OLS slope with patient-clustered CR1 covariance; t test with G-1 df."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ (X.T @ b)
    e = b - X @ beta
    G = int(pat_idx.max()) + 1
    Sg = np.zeros((G, 2))
    np.add.at(Sg, pat_idx, X * e[:, None])
    V = bread @ (Sg.T @ Sg) @ bread * (G / (G - 1)) * ((n - 1) / (n - 2))
    se = float(np.sqrt(V[1, 1]))
    tstat = float(beta[1] / se)
    return {"slope": float(beta[1]), "se_cr1": se, "t": tstat, "df": G - 1,
            "p_cr1_t": float(2 * t_dist.sf(abs(tstat), G - 1))}


def analyze(pair_id, public_id, y, PC, PD, entropy, n_boot, seed=RANDOM_SEED):
    """All statistics. Returns dict of tables/arrays. Uses D1's frozen entropy and rank percentile."""
    n = len(pair_id)
    b_r, ll_c, ll_d = benefit_per_pair(PC, PD, y)
    b = b_r.mean(axis=0)                                       # one value per visit pair
    groups, quartile, u = fixed_groups(entropy)                # D1's frozen rank percentile (no cutoff chosen here)
    pats, pat_idx = np.unique(public_id, return_inverse=True)
    npat = len(pats)
    knots = np.quantile(u, KNOT_PCTS)
    Xs = np.column_stack([np.ones(n), rcs_basis(u, knots)])
    Xg = np.column_stack([np.ones(len(GRID)), rcs_basis(GRID, knots)])
    ones = np.ones(n)

    # ---- observed ----
    a_rank, s_rank = wls_slope(u, b, ones)
    a_raw, s_raw = wls_slope(entropy, b, ones)
    beta_sp = wls_fit(Xs, b, ones)
    curve_obs = Xg @ beta_sp
    rho_obs = float(np.corrcoef(rankdata(b), rankdata(u))[0, 1])
    e10, e90 = np.quantile(entropy, [0.10, 0.90])

    # ---- patient-cluster bootstrap: same draws for every statistic (D1 scheme) ----
    rng = np.random.RandomState(seed)
    B = np.full(n_boot, np.nan)
    Braw = np.full(n_boot, np.nan)
    Bsp = np.full((n_boot, 3), np.nan)
    Bcurve = np.full((n_boot, len(GRID)), np.nan)
    Brho = np.full(n_boot, np.nan)
    for k in range(n_boot):
        idx = rng.randint(0, npat, npat)
        counts = np.bincount(idx, minlength=npat).astype(float)
        w = counts[pat_idx]                                   # each visit pair weighted by its patient's resample count
        B[k] = wls_slope(u, b, w)[1]
        Braw[k] = wls_slope(entropy, b, w)[1]
        Bsp[k] = wls_fit(Xs, b, w)
        Bcurve[k] = Xg @ Bsp[k]
        ex = np.repeat(np.arange(n), w.astype(int))
        Brho[k] = np.corrcoef(rankdata(b[ex]), rankdata(u[ex]))[0, 1]

    cr_rank, cr_raw = cluster_robust_slope(u, b, pat_idx), cluster_robust_slope(entropy, b, pat_idx)
    lo, hi, nv = percentile_ci(B)
    lo_r, hi_r, nv_r = percentile_ci(Braw)
    span_rank = 0.90 - 0.10                                     # benefit at the 90th vs 10th rank percentile
    main = pd.DataFrame([
        {"analysis": "PRIMARY_rank_slope", "role": "primary (exploratory sensitivity)",
         "exposure": "Model-C entropy rank percentile u in (0,1]", "n_pairs": n, "n_patients": npat,
         "slope_per_full_rank_range": float(s_rank), "ci_lo": lo, "ci_hi": hi, "p_boot_two_sided": boot_p(B),
         "n_valid_bootstrap": nv, "benefit_p90_minus_p10": float(s_rank * span_rank),
         "benefit_p90_minus_p10_ci_lo": lo * span_rank, "benefit_p90_minus_p10_ci_hi": hi * span_rank,
         "cr1_se": cr_rank["se_cr1"], "cr1_p_t": cr_rank["p_cr1_t"], "intercept": float(a_rank)},
        {"analysis": "SECONDARY_raw_entropy_slope", "role": "secondary robustness",
         "exposure": "raw mean normalized Model-C entropy (0-1)", "n_pairs": n, "n_patients": npat,
         "slope_per_full_rank_range": float(s_raw), "ci_lo": lo_r, "ci_hi": hi_r, "p_boot_two_sided": boot_p(Braw),
         "n_valid_bootstrap": nv_r, "benefit_p90_minus_p10": float(s_raw * (e90 - e10)),
         "benefit_p90_minus_p10_ci_lo": lo_r * (e90 - e10), "benefit_p90_minus_p10_ci_hi": hi_r * (e90 - e10),
         "cr1_se": cr_raw["se_cr1"], "cr1_p_t": cr_raw["p_cr1_t"], "intercept": float(a_raw)},
    ])
    main["unit_note"] = ["nats of log-loss per full rank range; positive = RNA better with higher uncertainty",
                         "nats of log-loss per unit of normalized entropy (p90-p10 benefit uses the observed entropy range)"]

    # ---- spline (secondary/descriptive) ----
    curve = pd.DataFrame({"rank_percentile": GRID, "benefit_estimate": curve_obs,
                          "ci_lo": np.nanpercentile(Bcurve, 2.5, axis=0), "ci_hi": np.nanpercentile(Bcurve, 97.5, axis=0),
                          "role": "secondary / descriptive"})
    theta = beta_sp[1:3]
    cov_b = np.cov(Bsp[:, 1:3].T)
    W2 = float(theta @ np.linalg.solve(cov_b, theta))
    W1 = float(beta_sp[2] ** 2 / cov_b[1, 1])
    spline_tests = pd.DataFrame([
        {"test": "any dependence on entropy (2 df)", "wald_chi2": W2, "df": 2, "p": float(chi2.sf(W2, 2)),
         "method": "Wald with bootstrap covariance of (linear, nonlinear) coefficients", "role": "secondary / descriptive"},
        {"test": "non-linearity (1 df)", "wald_chi2": W1, "df": 1, "p": float(chi2.sf(W1, 1)),
         "method": "Wald with bootstrap variance of the nonlinear coefficient", "role": "secondary / descriptive"}])
    spline_tests["knots_rank_percentile"] = json.dumps([float(k) for k in knots])

    # ---- descriptive: Spearman and per-repeat slopes ----
    lo_s, hi_s, _ = percentile_ci(Brho)
    desc = [{"statistic": "spearman_rho(b, rank_percentile)", "value": rho_obs, "ci_lo": lo_s, "ci_hi": hi_s,
             "note": "DESCRIPTIVE ONLY (patient-bootstrap percentile interval)"}]
    per_rep = []
    for r in range(b_r.shape[0]):
        s = float(wls_slope(u, b_r[r], ones)[1])
        per_rep.append(s)
        desc.append({"statistic": f"per_repeat_slope_repeat_{r}", "value": s, "ci_lo": np.nan, "ci_hi": np.nan,
                     "note": "DESCRIPTIVE ONLY: stability check, no inference; repeats share outcomes and RNA"})
    desc.append({"statistic": "per_repeat_slope_range", "value": float(max(per_rep) - min(per_rep)), "ci_lo": np.nan,
                 "ci_hi": np.nan, "note": "DESCRIPTIVE ONLY"})

    pair_tbl = pd.DataFrame({"pair_id": pair_id, "public_id": public_id, "modelC_mean_normalized_entropy": entropy,
                             "entropy_rank_percentile": u, "uncertainty_quartile": quartile,
                             "logloss_C_mean": ll_c.mean(axis=0), "logloss_D_mean": ll_d.mean(axis=0),
                             "benefit_b_mean_over_repeats": b})
    return {"main": main, "curve": curve, "spline_tests": spline_tests, "descriptive": pd.DataFrame(desc),
            "pairs": pair_tbl, "draws": pd.DataFrame({"draw": np.arange(n_boot), "slope_rank": B, "slope_raw": Braw}),
            "b": b, "ll_c": ll_c, "ll_d": ll_d, "groups": groups, "u": u, "entropy": entropy, "pat_idx": pat_idx}


# ---------------------------------------------------------------------------
# Sanity checks against the locked Direction 1 outputs
# ---------------------------------------------------------------------------
def verify_d1_outputs_unchanged():
    with open(f"{D1_DIR}/output_manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    checked, missing = [], []
    for rel, h in man["files"].items():
        p = f"{D1_DIR}/{rel}"
        if os.path.exists(p):
            require(sha256_file(p) == h, f"LOCKED D1 FILE CHANGED: {rel}")
            checked.append(rel)
        else:
            missing.append(rel)
    return checked, missing


def quartile_sanity(res):
    """Group means of b reproduce D1's log-loss group deltas and interactions with the sign reversed (b = -(D - C))."""
    b, ll_c, ll_d, groups = res["b"], res["ll_c"], res["ll_d"], res["groups"]
    g = pd.read_csv(f"{D1_DIR}/direction1_within_uncertainty_groups.csv")
    g = g[(g["task"] == "binary") & (g["metric"] == "logloss")].set_index("group")
    rows = []
    for name, idx in groups.items():
        require(name in g.index, f"group {name} missing from the D1 within-group table")
        mean_b = float(b[idx].mean())
        ref = g.loc[name]
        d_ok = abs(mean_b - (-float(ref["delta_D_minus_C"]))) < SANITY_TOL
        c_ok = abs(float(ll_c[:, idx].mean()) - float(ref["C_mean"])) < SANITY_TOL
        d2_ok = abs(float(ll_d[:, idx].mean()) - float(ref["D_mean"])) < SANITY_TOL
        rows.append({"check": "group_logloss", "group": name, "n_rows": int(len(idx)), "mean_b_recomputed": mean_b,
                     "D1_delta_D_minus_C_logloss": float(ref["delta_D_minus_C"]),
                     "expected_b_equals_minus_D1_delta": bool(d_ok), "C_mean_matches": bool(c_ok),
                     "D_mean_matches": bool(d2_ok)})
    it = pd.read_csv(f"{D1_DIR}/direction1_uncertainty_interactions.csv")
    it = it[(it["task"] == "binary") & (it["metric"] == "logloss")].set_index("contrast")
    pairs = {"PRIMARY_Q4_vs_Q1_Q3": ("Q4_high_uncertainty", "Q1_Q3_lower_uncertainty"),
             "SENS_top20_vs_bottom80": ("top20_uncertainty", "bottom80_uncertainty"),
             "SENS_top33_vs_bottom67": ("top33_uncertainty", "bottom67_uncertainty")}
    for contrast, (hi_g, lo_g) in pairs.items():
        require(contrast in it.index, f"contrast {contrast} missing from the D1 interaction table")
        mine = float(b[groups[hi_g]].mean() - b[groups[lo_g]].mean())
        ref = float(it.loc[contrast, "interaction_uncertain_minus_comparator"])
        rows.append({"check": "interaction_logloss", "group": contrast, "n_rows": np.nan, "mean_b_recomputed": mine,
                     "D1_delta_D_minus_C_logloss": ref,
                     "expected_b_equals_minus_D1_delta": bool(abs(mine - (-ref)) < SANITY_TOL),
                     "C_mean_matches": True, "D_mean_matches": True})
    t = pd.DataFrame(rows)
    bad = t[~(t["expected_b_equals_minus_D1_delta"] & t["C_mean_matches"] & t["D_mean_matches"])]
    require(bad.empty, "SANITY CHECK FAILED (recomputed benefit does not reproduce D1 log-loss results):\n"
            + bad.to_string(index=False))
    return t


def assignments_crosscheck(res):
    """If D1's (uncommitted) assignment file is present locally, the recomputed entropy/rank must equal it."""
    p = f"{D1_DIR}/direction1_uncertainty_assignments.csv"
    if not os.path.exists(p):
        return {"checked": False, "reason": "D1 assignments file not present locally; entropy/rank recomputed from the locked OOF"}
    # the rank column is read as TEXT: D1 stored it with float_format="%.10g", and comparing text avoids any
    # float-parser / tolerance question
    a = pd.read_csv(p, dtype={"uncertainty_percentile_rank": str})
    a = a[a["task"] == "binary"].sort_values("pair_id").reset_index(drop=True)
    order = np.argsort(res["pairs"]["pair_id"].to_numpy(), kind="stable")
    pr = res["pairs"].iloc[order].reset_index(drop=True)
    require(np.array_equal(a["pair_id"].to_numpy(), pr["pair_id"].to_numpy()), "assignment pair_ids differ from recomputed")
    require(np.allclose(a["modelC_mean_normalized_entropy"], pr["modelC_mean_normalized_entropy"], atol=1e-9, rtol=0),
            "recomputed entropy differs from the D1 assignments file")
    # D1 wrote this file with write_csv(..., float_format="%.10g"): the stored ranks carry 10 significant digits
    # (error up to 5e-11), so they can never equal the full-precision k/n within 1e-12. The faithful comparison is
    # EXACT equality with the recomputed rank formatted exactly as D1 formatted it (no tolerance involved).
    mine_txt = np.array(["%.10g" % v for v in pr["entropy_rank_percentile"].to_numpy(float)])
    locked_txt = a["uncertainty_percentile_rank"].to_numpy(dtype=str)
    n_bad = int((mine_txt != locked_txt).sum())
    require(n_bad == 0, f"recomputed rank percentile differs from the D1 assignments file on {n_bad} rows "
                        f"(max |diff| = {np.abs(locked_txt.astype(float) - pr['entropy_rank_percentile'].to_numpy(float)).max():.3e})")
    require(np.array_equal(a["uncertainty_quartile"].to_numpy(int), pr["uncertainty_quartile"].to_numpy(int)),
            "recomputed uncertainty quartile differs from the D1 assignments file")
    return {"checked": True, "reason": "matches the D1 assignments file"}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_boot = 50 if args.smoke else N_BOOT
    out = f"{OUT_ROOT}/_smoke/entropy_slope" if args.smoke else f"{OUT_ROOT}/entropy_slope"
    marker = f"{out}/output_manifest.json"
    if args.force:
        require(args.smoke, "--force is only allowed together with --smoke (real outputs are never overwritten)")
    if os.path.exists(marker) and not args.force:
        raise SystemExit(f"[entropy_slope] {marker} already exists: never overwritten. Delete deliberately to rerun "
                         f"(or --smoke --force for smoke runs).")
    print(f"{LABEL}\n(exploratory sensitivity; follows the locked D1 result; binary task only)")

    checked, missing = verify_d1_outputs_unchanged()
    d, oof_path = load_and_verify("binary")                    # pinned SHA256 of the locked D3 OOF predictions
    os.makedirs(out, exist_ok=True)                            # only after every locked input verified
    pair_id, public_id, y, PC, PD, entropy = aligned_arrays(d, "binary")
    require(PC.shape[0] == 5, "expected 5 OOF repeats")
    res = analyze(pair_id, public_id, y, PC, PD, entropy, n_boot)
    sanity = quartile_sanity(res)
    asg = assignments_crosscheck(res)
    print(f"Sanity checks passed ({len(sanity)} rows): group means of b reproduce D1 log-loss results with sign reversed.")
    print(f"Locked D1 files verified: {len(checked)}; not present locally: {missing}")

    files = []
    outputs = {"entropy_slope_primary_and_secondary.csv": res["main"], "spline_curve.csv": res["curve"],
               "spline_tests.csv": res["spline_tests"], "descriptive_spearman_and_per_repeat.csv": res["descriptive"],
               "quartile_sanity_check_vs_D1.csv": sanity, "pair_level_benefit.csv": res["pairs"],
               "bootstrap_slope_draws.csv": res["draws"]}
    for name, df in outputs.items():
        p = f"{out}/{name}"
        write_csv(df, p, float_format="%.10g")
        files.append(p)
    m = res["main"].set_index("analysis")
    summary = {
        "label": LABEL, "smoke": bool(args.smoke), "status": "EXPLORATORY SENSITIVITY (post hoc; follows the locked D1 result)",
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 1), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
        "design": "b_i = mean over 5 locked OOF repeats of logloss_C - logloss_D (positive = Model D better); "
                  "WLS slope on Model-C entropy rank percentile; patient-cluster bootstrap with visit-pair weights "
                  "equal to the patient's resample count; no cutoff, no retraining, no test split",
        "primary": {"slope": float(m.loc["PRIMARY_rank_slope", "slope_per_full_rank_range"]),
                    "ci": [float(m.loc["PRIMARY_rank_slope", "ci_lo"]), float(m.loc["PRIMARY_rank_slope", "ci_hi"])],
                    "p_boot_two_sided": float(m.loc["PRIMARY_rank_slope", "p_boot_two_sided"]),
                    "cr1_p_t": float(m.loc["PRIMARY_rank_slope", "cr1_p_t"])},
        "secondary_raw_entropy": {"slope": float(m.loc["SECONDARY_raw_entropy_slope", "slope_per_full_rank_range"]),
                                  "p_boot_two_sided": float(m.loc["SECONDARY_raw_entropy_slope", "p_boot_two_sided"])},
        "sanity_checks_passed": True, "d1_files_verified_unchanged": checked, "d1_files_not_present_locally": missing,
        "d1_assignments_crosscheck": asg,
        "locked_inputs": {"oof_path": oof_path, "oof_sha256": sha256_file(oof_path)},
        "limitation": "a slope does not by itself identify RNA-specific signal (no row-level permuted-RNA null); "
                      "exploratory, outside D1's Holm family",
    }
    p = f"{out}/run_summary.json"
    write_json(summary, p)
    files.append(p)
    man = write_output_manifest(out, files, {"label": LABEL, "smoke": bool(args.smoke),
                                             "input_oof_sha256": summary["locked_inputs"]["oof_sha256"]})
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs in {out}\nCombined output SHA256: {man['combined_sha256']}")
    print("EXPLORATORY SENSITIVITY -- no confirmatory claim.")


if __name__ == "__main__":
    main()
