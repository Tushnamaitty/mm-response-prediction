"""
Survival redesign -- STAGE D (development) of the train-only protocol (explainability/SURVIVAL_TRAINONLY_PROTOCOL.md,
SHA-256 in explainability/SURVIVAL_TRAINONLY_PROTOCOL.sha256).

Responds to the concern that validation patients influenced model development. Every learned quantity here (preprocessing,
ridge penalty, Cox coefficients, Breslow baseline, KM censoring) uses ONLY the 473 development (train) patients. The 201
held-out (validation + test) patients are removed from memory before any development fit; this script never reads their
survival outcomes except inside gate G1/G2, which only compute aggregate counts and the earlier pooled analysis (a quantity that
was already known), and whose results are boolean/aggregate and are discarded.

The locked analysis (survival_incremental_analysis.py, spec SHA fe0792db...db6e4) is IMPORTED UNCHANGED. Necessary changes N1-N5
of the protocol: a single frozen fit on the development patients; lambda by 5-fold inner CV repeated 5 times; (permutation null
and success rule are Stage H).

POST HOC, EXPLORATORY: the earlier pooled results were seen first; this is not blind, not independent and not external validation.

Outputs only under artifacts/results_clean_rerun/survival_trainonly/ ; existing outputs are never overwritten.

Run from the repo root:
    python -m explainability.survival_trainonly_development --benchmark   # timings only, no scientific output
    python -m explainability.survival_trainonly_development               # gates, development fits, freeze + hash
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from explainability import survival_incremental_analysis as L
from explainability.survival_pathway_interpretation import EXPECTED_SPEC_SHA, combined_table, stability_table

OUT = Path("artifacts/results_clean_rerun/survival_trainonly")
PROTOCOL = Path("explainability/SURVIVAL_TRAINONLY_PROTOCOL.md")
PROTOCOL_HASH_FILE = Path("explainability/SURVIVAL_TRAINONLY_PROTOCOL.sha256")
SPLITS = Path("data/splits/patient_split_assignments.csv")
N_INNER_REPEATS = 5                       # protocol N2
EXPECTED_COUNTS = {"development": (473, 310, 165), "val": (101, 71, 45), "test": (100, 78, 37),
                   "dev_sens": (438, 286, 154), "heldout_sens": (186, 137, 75)}      # patients, PFS events, deaths
ANALYSES = {            # id -> (endpoint, cohort, richer baseline)
    "A1_pfs": ("pfs", "primary", False), "A2_os": ("os", "primary", False),
    "S1_pfs_m30_0": ("pfs", "sens", False),
    "S2_pfs_rich": ("pfs", "primary", True), "S2_os_rich": ("os", "primary", True),
}
LABEL = ("SURVIVAL TRAIN-ONLY REDESIGN, STAGE D (development): post hoc, exploratory; not independent validation; "
         "not external validation")


# ---------------------------------------------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------------------------------------------
def sha_lf(path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=conv)
        f.write("\n")


def require(cond, msg):
    if not cond:
        raise AssertionError(f"[survival_trainonly] {msg}")


class Gates:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append({"gate": name, "passed": bool(ok), "detail": str(detail)})
        print(("  PASS " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        if not ok:
            raise SystemExit(f"[survival_trainonly] GATE FAILED: {name} {detail}. Stopped before any held-out outcome was used.")


def harrell_c_fast(t, e, r):
    """Vectorised Harrell C, algebraically identical to L.harrell_c (verified exactly in gate G5)."""
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    r = np.asarray(r, float)
    ev = np.flatnonzero(e == 1)
    later = t[None, :] > t[ev][:, None]
    comparable = int(later.sum())
    if comparable == 0:
        return float("nan")
    gt = int(((r[ev][:, None] > r[None, :]) & later).sum())
    eq = int(((r[ev][:, None] == r[None, :]) & later).sum())
    return float((gt + 0.5 * eq) / comparable)


def partition_table():
    sp = pd.read_csv(SPLITS)
    require(sp["public_id"].is_unique and set(sp["split"]) == {"train", "val", "test"}, "unexpected split file")
    return sp


def protocol_hash_ok():
    rec = PROTOCOL_HASH_FILE.read_text(encoding="utf-8").splitlines()[0].split()[0]
    return rec == sha_lf(PROTOCOL), rec


# ---------------------------------------------------------------------------------------------------------------
# preprocessing (de)serialisation
# ---------------------------------------------------------------------------------------------------------------
def pp_to_dict(pp, rich):
    d = {"age_median": float(pp.age_median), "categories": pp.categories, "mean": [float(v) for v in pp.mean],
         "sd": [float(v) for v in pp.sd], "feature_names": list(pp.feature_names if not rich else pp.names),
         "include_rna": bool(pp.include_rna), "pathways": list(pp.pathways)}
    if rich:
        d.update({"num_medians": {k: float(v) for k, v in pp.num_medians.items()},
                  "extra_numeric": list(pp.extra_numeric), "extra_cat": list(pp.extra_cat)})
    return d


def pp_from_dict(d, rich, R=None):
    if not rich:
        return L.Preprocessor(d["age_median"], d["categories"], np.array(d["mean"], float), np.array(d["sd"], float),
                              d["feature_names"], d["include_rna"], d["pathways"])
    return R.RichPP(d["age_median"], d["categories"], d["num_medians"], np.array(d["mean"], float),
                    np.array(d["sd"], float), d["feature_names"], d["include_rna"], d["pathways"], d["extra_numeric"],
                    d["extra_cat"])


def design(df, fm, R=None):
    rich = fm["kind"] == "rich"
    pp = pp_from_dict(fm["pp"], rich, R)
    return R.transform_pp(df, pp) if rich else L.transform(df, pp)


# ---------------------------------------------------------------------------------------------------------------
# frozen development fit (N1, N2)
# ---------------------------------------------------------------------------------------------------------------
def choose_lambda_repeated(train_df, paths, endpoint, include_rna, R=None, n_repeats=N_INNER_REPEATS):
    """Locked inner-CV rule (5-fold, mean validation Harrell C, ties -> largest lambda) averaged over `n_repeats` inner splits."""
    scores = {float(l): [] for l in L.LAMBDA_GRID}
    for r in range(n_repeats):
        for tr_idx, va_idx in L.make_inner_folds(len(train_df), r, 0):
            tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
            if R is None:
                pp = L.fit_preprocessor(tr, paths, include_rna)
                Xtr, Xva = L.transform(tr, pp), L.transform(va, pp)
            else:
                pp = R.fit_pp(tr, paths, include_rna, R.EXTRA_NUMERIC, R.EXTRA_CAT)
                Xtr, Xva = R.transform_pp(tr, pp), R.transform_pp(va, pp)
            ttr, etr = L.endpoint_arrays(tr, endpoint)
            tva, eva = L.endpoint_arrays(va, endpoint)
            for lam in L.LAMBDA_GRID:
                b = L.fit_cox(Xtr, ttr, etr, float(lam))
                c = harrell_c_fast(tva, eva, Xva @ b)
                require(np.isfinite(c), "non-finite inner C-index")
                scores[float(lam)].append(c)
    means = {l: float(np.mean(v)) for l, v in scores.items()}
    best_val = max(means.values())
    tied = [l for l, v in means.items() if abs(v - best_val) <= 1e-12]
    return max(tied), means


def fit_frozen(dev_df, paths, endpoint, include_rna, R=None):
    lam, means = choose_lambda_repeated(dev_df, paths, endpoint, include_rna, R)
    if R is None:
        pp = L.fit_preprocessor(dev_df, paths, include_rna)
        X = L.transform(dev_df, pp)
    else:
        pp = R.fit_pp(dev_df, paths, include_rna, R.EXTRA_NUMERIC, R.EXTRA_CAT)
        X = R.transform_pp(dev_df, pp)
    t, e = L.endpoint_arrays(dev_df, endpoint)
    beta = L.fit_cox(X, t, e, lam)
    lp = X @ beta
    bt, bh = L.breslow_baseline(t, e, lp)
    km_t, km_s = L.km_censoring(t, e)
    return {"kind": "rich" if R is not None else "base", "include_rna": bool(include_rna), "endpoint": endpoint,
            "lambda": float(lam), "inner_mean_c_by_lambda": {repr(k): v for k, v in means.items()},
            "n_inner_repeats": N_INNER_REPEATS, "pp": pp_to_dict(pp, R is not None), "beta": [float(b) for b in beta],
            "baseline_t": [float(v) for v in bt], "baseline_H": [float(v) for v in bh],
            "km_t": [float(v) for v in km_t], "km_s": [float(v) for v in km_s],
            "n_train": int(len(dev_df)), "n_train_events": int(e.sum())}, lp


# ---------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------
def gate_partition_counts(g, x):
    sp = partition_table()
    d = x.merge(sp, on="public_id", how="left", validate="one_to_one")
    require(d["split"].notna().all(), "cohort patient without partition")
    got = {}
    for name, m in (("development", d.split == "train"), ("val", d.split == "val"), ("test", d.split == "test")):
        s = d[m]
        got[name] = (int(len(s)), int(s["censor_pfs"].sum()), int(s["censor_os"].sum()))
    win = (d.rna_day >= L.SENS_WINDOW[0]) & (d.rna_day <= L.SENS_WINDOW[1])
    for name, m in (("dev_sens", win & (d.split == "train")), ("heldout_sens", win & (d.split != "train"))):
        s = d[m]
        got[name] = (int(len(s)), int(s["censor_pfs"].sum()), int(s["censor_os"].sum()))
    ho = d[d.split != "train"]
    got["heldout"] = (int(len(ho)), int(ho["censor_pfs"].sum()), int(ho["censor_os"].sum()))
    g.check("G1 partition counts (patients, PFS events, deaths) match the protocol",
            all(got[k] == v for k, v in EXPECTED_COUNTS.items()) and got["heldout"] == (201, 149, 82),
            f"development {got['development']}, held-out {got['heldout']}")
    return d


def gate_pooled_reproduction(g, x, paths):
    """G2: the imported locked machinery on all 674 patients (earlier pooled design, PFS) reproduces the locked summary."""
    t0 = time.time()
    pred, _ = L.run_oof(x, paths, "pfs", "gate", save_detail=False)
    summ, _ = L.summarize_oof(pred)
    with open(L.ROOT / "primary_pfs_summary.json", encoding="utf-8") as f:
        locked = json.load(f)
    worst = max(abs(summ[k] - float(locked[k])) for k in ("c_C0", "c_C0_RNA", "delta_c"))
    g.check("G2 pooled PFS reproduction of the locked summary (c_C0, c_C0_RNA, delta_c) < 1e-9", worst < 1e-9,
            f"max |diff| = {worst:.2e}; {time.time() - t0:.0f}s")


def gate_c_index(g, dev, paths):
    rng = np.random.RandomState(11)
    t, e = L.endpoint_arrays(dev, "pfs")
    worst = 0.0
    for _ in range(200):
        idx = rng.choice(len(dev), size=int(rng.randint(40, len(dev))), replace=False)
        r = np.round(rng.normal(size=len(idx)), 1)                        # rounded -> frequent ties
        worst = max(worst, abs(harrell_c_fast(t[idx], e[idx], r) - L.harrell_c(t[idx], e[idx], r)))
    g.check("G5 vectorised C-index equals the locked harrell_c exactly (200 random subsets with ties)", worst == 0.0,
            f"max |diff| = {worst:.1e}")


# ---------------------------------------------------------------------------------------------------------------
# benchmark (timings only; development outcomes only; no scientific output)
# ---------------------------------------------------------------------------------------------------------------
def benchmark(dev, x_all, paths):
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"label": "benchmark: timings only", "cpu_count": os.cpu_count(), "python": sys.version.split()[0]}

    def timed(fn, *a, **k):
        t0 = time.time()
        r = fn(*a, **k)
        return time.time() - t0, r

    t_c0_rep, _ = timed(choose_lambda_repeated, dev, paths, "pfs", False, None, 1)
    t_rna_rep, _ = timed(choose_lambda_repeated, dev, paths, "pfs", True, None, 1)
    out["lambda_selection_one_inner_repeat_seconds"] = {"C0": t_c0_rep, "C0_RNA": t_rna_rep}
    # one nested outer fold of the locked design on the development patients and on all 674 patients
    for name, frame in (("development_473", dev), ("pooled_674", x_all)):
        fold = L.make_outer_folds(len(frame), 0)
        tr, te = frame.iloc[np.flatnonzero(fold != 0)], frame.iloc[np.flatnonzero(fold == 0)]
        a, _ = timed(L.fit_outer, tr, te, paths, False, "pfs", 0, 0)
        b, _ = timed(L.fit_outer, tr, te, paths, True, "pfs", 0, 0)
        out[f"one_nested_outer_fold_seconds_{name}"] = {"C0": a, "C0_RNA": b}
    # frozen fit, then permutation / bootstrap micro-benchmarks on a 201-patient subsample of the DEVELOPMENT data
    rng = np.random.RandomState(5)
    perm_idx = rng.permutation(len(dev))
    sub, rest = dev.iloc[perm_idx[:201]].reset_index(drop=True), dev.iloc[perm_idx[201:]].reset_index(drop=True)
    fm0, _ = fit_frozen(rest, paths, "pfs", False)
    fm1, _ = fit_frozen(rest, paths, "pfs", True)
    X0, X1 = design(sub, fm0), design(sub, fm1)
    b0, b1 = np.array(fm0["beta"]), np.array(fm1["beta"])
    t, e = L.endpoint_arrays(sub, "pfs")
    c0 = harrell_c_fast(t, e, X0 @ b0)
    Xc, Xr = X1[:, :-len(paths)], X1[:, -len(paths):]
    lc, bc, br = Xc @ b1[:-len(paths)], None, b1[-len(paths):]
    n_perm = 200
    t0 = time.time()
    for p in range(n_perm):
        pi = np.random.RandomState(L.SEED + 20000 + p).permutation(len(sub))
        _ = harrell_c_fast(t, e, lc + Xr[pi] @ br) - c0
    per_perm = (time.time() - t0) / n_perm
    n_boot = 100
    t0 = time.time()
    r0, r1 = X0 @ b0, X1 @ b1
    brng = np.random.RandomState(1)
    for _ in range(n_boot):
        i = brng.choice(len(sub), len(sub), replace=True)
        _ = harrell_c_fast(t[i], e[i], r1[i]) - harrell_c_fast(t[i], e[i], r0[i])
    per_boot = (time.time() - t0) / n_boot
    # locked pandas-based bootstrap used for the development-CV reference (S3)
    rows = []
    for rep in range(L.N_REPEATS):
        for j in range(len(sub)):
            rows.append((rep, sub.iloc[j]["public_id"], t[j], e[j], r0[j], r1[j]))
    pred = pd.DataFrame(rows, columns=["repeat", "public_id", "time", "event", "risk_C0", "risk_C0_RNA"])
    t0 = time.time()
    _ = L.bootstrap_delta_c(pred, B=10, seed=1)
    per_boot_locked = (time.time() - t0) / 10
    out["per_permutation_seconds_n201"] = per_perm
    out["per_bootstrap_draw_seconds_fast_n201"] = per_boot
    out["per_bootstrap_draw_seconds_locked_pandas_n201_x5repeats"] = per_boot_locked
    # estimate of the full workload (conservative: scale locked-bootstrap time by 473/201)
    n_dev_boot = 2 * L.N_BOOT * per_boot_locked * (473 / 201)
    est = {
        "G2_pooled_PFS_reproduction": 25 * (out["one_nested_outer_fold_seconds_pooled_674"]["C0"]
                                           + out["one_nested_outer_fold_seconds_pooled_674"]["C0_RNA"]),
        "development_CV_PFS_and_OS": 2 * 25 * (out["one_nested_outer_fold_seconds_development_473"]["C0"]
                                              + out["one_nested_outer_fold_seconds_development_473"]["C0_RNA"]),
        "development_CV_bootstraps": n_dev_boot,
        "frozen_fits_A1_A2_S1_(3 analyses x 2 models x 5 inner repeats)": 3 * N_INNER_REPEATS * (t_c0_rep + t_rna_rep),
        "frozen_fits_S2_(2 analyses, richer design, x1.5 allowance)": 1.5 * 2 * N_INNER_REPEATS * (t_c0_rep + t_rna_rep),
        "stage_H_10000_permutations": 10000 * per_perm * 1.0,
        "stage_H_bootstraps_(about 9 sets x 2000 draws, fast)": 9 * L.N_BOOT * per_boot * 2,
    }
    out["estimated_full_runtime_seconds"] = est
    out["estimated_total_minutes"] = sum(est.values()) / 60
    out["note"] = ("timings use development patients only (201-patient subsample for the held-out-sized steps); no held-out "
                   "outcome was read; CPU is used by default; GPU/Kaggle are not needed if the total is practical")
    dump_json(out, OUT / "benchmark.json")
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=2, default=float))
    return out


# ---------------------------------------------------------------------------------------------------------------
# development-only cross-validation (S3) and the Figure 5 refit (A6)
# ---------------------------------------------------------------------------------------------------------------
def refit_coefficients(dev, paths, endpoint, pred, tune, g):
    lam_map = {(int(r.repeat), int(r.fold)): float(r.lambda_) for r in
               tune[tune["model"] == "C0_RNA"].rename(columns={"lambda": "lambda_"}).itertuples(index=False)}
    idx = pred.set_index(["repeat", "public_id"])["risk_C0_RNA"]
    rows, worst = [], 0.0
    for rep in range(L.N_REPEATS):
        fold = L.make_outer_folds(len(dev), rep)
        for f in range(L.N_FOLDS):
            te_idx, tr_idx = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            tr, te = dev.iloc[tr_idx], dev.iloc[te_idx]
            ttr, etr = L.endpoint_arrays(tr, endpoint)
            pp = L.fit_preprocessor(tr, paths, True)
            beta = L.fit_cox(L.transform(tr, pp), ttr, etr, lam_map[(rep, f)])
            lp = L.transform(te, pp) @ beta
            ref = idx.loc[[(rep, p) for p in te["public_id"]]].to_numpy(float)
            worst = max(worst, float(np.max(np.abs(lp - ref))))
            require(list(pp.feature_names[-len(paths):]) == list(paths), "pathway coefficient positions")
            rows.append({"endpoint": endpoint, "repeat": rep, "fold": f, "lambda": lam_map[(rep, f)],
                         **{p: float(b) for p, b in zip(paths, beta[-len(paths):])}})
    g.check(f"A6 [{endpoint}] refit risks reproduce the development-CV out-of-fold risks (< 1e-9)", worst < 1e-9, f"max={worst:.1e}")
    return pd.DataFrame(rows)


def development_cv(dev, paths, g, files):
    res, coefs = {}, {}
    for endpoint in ("pfs", "os"):
        t0 = time.time()
        pred, tune = L.run_oof(dev, paths, endpoint, f"devcv_{endpoint}", save_detail=False)
        summ, by_rep = L.summarize_oof(pred)
        boot = L.bootstrap_delta_c(pred, L.N_BOOT, L.SEED + (0 if endpoint == "pfs" else 1000))
        res[endpoint] = {"analysis": "S3_development_only_5x5_cv", "endpoint": endpoint, **summ, "bootstrap_delta_c": boot,
                         "n_patients": int(len(dev)), "n_events": int(L.endpoint_arrays(dev, endpoint)[1].sum()),
                         "seconds": time.time() - t0,
                         "note": "descriptive reference for optimism; fitted and evaluated inside the development patients only"}
        tune.to_csv(OUT / f"devcv_{endpoint}_tuning.csv", index=False)
        by_rep.to_csv(OUT / f"devcv_{endpoint}_repeat_metrics.csv", index=False)
        pred.to_csv(OUT / f"devcv_{endpoint}_oof_predictions.csv", index=False)       # contains patient ids; *.csv is gitignored
        files += [OUT / f"devcv_{endpoint}_tuning.csv", OUT / f"devcv_{endpoint}_repeat_metrics.csv"]
        coef = refit_coefficients(dev, paths, endpoint, pred, tune, g)
        coef.to_csv(OUT / f"figure5_refit_fold_coefficients_{endpoint}.csv", index=False)
        stab = stability_table(coef, paths, endpoint)
        stab.to_csv(OUT / f"figure5_refit_pathway_stability_{endpoint}.csv", index=False)
        files += [OUT / f"figure5_refit_fold_coefficients_{endpoint}.csv", OUT / f"figure5_refit_pathway_stability_{endpoint}.csv"]
        coefs[endpoint] = stab
        dump_json(res[endpoint], OUT / f"devcv_{endpoint}_summary.json")
        files.append(OUT / f"devcv_{endpoint}_summary.json")
    both = combined_table(coefs["pfs"], coefs["os"])
    both.to_csv(OUT / "figure5_refit_pathway_stability_both_endpoints.csv", index=False)
    files.append(OUT / "figure5_refit_pathway_stability_both_endpoints.csv")
    conc = both.loc[both["consistent_same_direction_both_endpoints"], "pathway"].tolist()
    res["figure5_refit"] = {"n_concordant": len(conc), "concordant_pathways": conc,
                            "direction_counts": {ep: {k: int((coefs[ep]["direction_label"] == k).sum())
                                                      for k in ("consistently_positive", "consistently_negative", "mixed")}
                                                 for ep in ("pfs", "os")},
                            "note": "descriptive; 25 fits share about 80% of their training patients; no hypothesis test"}
    return res


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    print(LABEL)
    g = Gates()
    ok, rec = protocol_hash_ok()
    g.check("protocol file hash equals the recorded SHA-256", ok, rec[:16] + "...")
    g.check("locked module spec SHA-256 (fe0792db...db6e4)", L.spec_sha() == EXPECTED_SPEC_SHA, L.spec_sha()[:12])
    x, paths = L.build_cohort()            # asserts 674 patients, 459 PFS events, 247 deaths, 624 sensitivity patients
    g.check("G1 cohort rebuilt with the locked asserts (674 / 459 / 247; 50 pathways)", len(x) == 674 and len(paths) == 50)
    d = gate_partition_counts(g, x)

    if args.benchmark:
        dev = d[d.split == "train"].drop(columns="split").reset_index(drop=True)
        benchmark(dev, x, paths)
        return

    marker = OUT / "frozen_models.json"
    if marker.exists():
        raise SystemExit(f"[survival_trainonly] {marker} exists: existing outputs are never overwritten.")
    OUT.mkdir(parents=True, exist_ok=True)

    # G2 uses all 674 patients (earlier pooled design); only a pass/fail and an aggregate difference are kept.
    gate_pooled_reproduction(g, x, paths)

    # ---- from here on the held-out patients are gone from memory ----
    heldout_ids = sorted(d.loc[d.split != "train", "public_id"])
    dev = d[d.split == "train"].drop(columns="split").sort_values("public_id").reset_index(drop=True)
    dev_ids = sorted(dev["public_id"])
    del x, d
    g.check("G3 development and held-out patient sets are disjoint (473 / 201)",
            not (set(dev_ids) & set(heldout_ids)) and len(dev_ids) == 473 and len(heldout_ids) == 201)
    g.check("development frame has no held-out patient", set(dev["public_id"]) == set(dev_ids))
    gate_c_index(g, dev, paths)

    # ---- optional richer baseline module (pinned pipeline modules from git; failure -> S2 not run) ----
    R, s2_status = None, "run"
    try:
        from explainability import survival_richer_baseline_sensitivity as R   # noqa: N813
    except SystemExit as e:                                                       # pinned module unavailable
        R, s2_status = None, f"NOT RUN: {e}"
    print("S2 richer baseline:", s2_status)

    frozen, dev_lp_check = {}, {}
    dev_win = dev[(dev.rna_day >= L.SENS_WINDOW[0]) & (dev.rna_day <= L.SENS_WINDOW[1])].reset_index(drop=True)
    require(len(dev_win) == EXPECTED_COUNTS["dev_sens"][0], "development sensitivity cohort size")
    dev_rich, audit_rich = (R.build_baseline_variables(dev) if R is not None else (None, None))
    for aid, (endpoint, cohort, rich) in ANALYSES.items():
        if rich and R is None:
            frozen[aid] = {"status": s2_status}
            continue
        frame = dev_rich if rich else (dev_win if cohort == "sens" else dev)
        fmods = {}
        for name, inc in (("C0", False), ("C0_RNA", True)):
            t0 = time.time()
            fm, lp = fit_frozen(frame, paths, endpoint, inc, R if rich else None)
            fm["seconds"] = time.time() - t0
            fmods[name] = fm
            # round trip: the serialised model must reproduce its in-memory development risks exactly
            fm2 = json.loads(json.dumps(fm))
            lp2 = design(frame, fm2, R) @ np.array(fm2["beta"])
            dev_lp_check[f"{aid}/{name}"] = float(np.max(np.abs(lp2 - lp)))
            print(f"  {aid} {name}: lambda={fm['lambda']:g}, n={fm['n_train']}, events={fm['n_train_events']}, "
                  f"{fm['seconds']:.0f}s", flush=True)
        frozen[aid] = {"endpoint": endpoint, "cohort": cohort, "richer_baseline": rich, "models": fmods,
                       "n_development": int(len(frame)), "n_development_events": int(L.endpoint_arrays(frame, endpoint)[1].sum())}
    g.check("G3b serialised frozen models reproduce the in-memory development risks exactly",
            max(dev_lp_check.values()) == 0.0, f"max |diff| = {max(dev_lp_check.values()):.1e}")

    # G3c: independent recomputation of preprocessing from the development rows only
    pp = frozen["A1_pfs"]["models"]["C0_RNA"]["pp"]
    ind_ok = (abs(pp["age_median"] - float(pd.to_numeric(dev["age_at_index"], errors="coerce").median())) == 0.0
              and pp["categories"]["first_line"] == sorted(dev["first_line"].astype(str).unique().tolist()))
    j = pp["feature_names"].index("E2F Targets")
    ind_ok &= abs(pp["mean"][j] - float(dev["E2F Targets"].mean())) < 1e-9 and abs(pp["sd"][j] - float(dev["E2F Targets"].std(ddof=0))) < 1e-9
    g.check("G3c preprocessing parameters equal an independent recomputation from the development rows", bool(ind_ok))

    # ---- development-only CV reference (S3) and the Figure 5 refit (A6) ----
    files = []
    cv = development_cv(dev, paths, g, files)

    # A6: coefficients of the frozen A1 / A2 models
    for aid, ep in (("A1_pfs", "pfs"), ("A2_os", "os")):
        fm = frozen[aid]["models"]["C0_RNA"]
        b = np.array(fm["beta"])[-len(paths):]
        pd.DataFrame({"pathway": paths, "beta_frozen_model": b, "HR_per_SD": np.exp(b)}).to_csv(
            OUT / f"figure5_frozen_model_coefficients_{ep}.csv", index=False)
        files.append(OUT / f"figure5_frozen_model_coefficients_{ep}.csv")

    # ---- freeze and hash ----
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           **{p: __import__(p).__version__ for p in ("numpy", "pandas", "scipy", "sklearn")}}
    artifact = {"label": LABEL, "protocol_sha256": rec, "locked_spec_sha256": EXPECTED_SPEC_SHA,
                "code_sha256_lf": sha_lf(__file__), "environment": env,
                "development_patient_ids_sha256": hashlib.sha256("\n".join(dev_ids).encode()).hexdigest(),
                "heldout_patient_ids_sha256": hashlib.sha256("\n".join(heldout_ids).encode()).hexdigest(),
                "n_development": len(dev_ids), "n_heldout": len(heldout_ids), "pathways": paths,
                "s2_status": s2_status, "analyses": frozen, "inner_cv_repeats": N_INNER_REPEATS,
                "lambda_grid": [float(v) for v in L.LAMBDA_GRID]}
    dump_json(artifact, marker)
    art_sha = sha_lf(marker)
    (OUT / "frozen_models.sha256").write_text(f"{art_sha}  frozen_models.json\n", encoding="utf-8", newline="\n")
    dump_json(cv, OUT / "development_cv_and_figure5_refit_summary.json")
    dump_json({"gates": g.rows, "frozen_models_sha256": art_sha, "protocol_sha256": rec, "richer_baseline_audit": audit_rich,
               "development_risk_roundtrip_max_abs_diff": dev_lp_check, "seconds": time.time() - t_start,
               "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, OUT / "development_gate_report.json")
    files += [marker, OUT / "frozen_models.sha256", OUT / "development_cv_and_figure5_refit_summary.json",
              OUT / "development_gate_report.json"]
    print(f"\nFrozen artifact SHA-256 (LF-normalised): {art_sha}")
    print(f"Stage D finished in {(time.time() - t_start) / 60:.1f} min. Held-out outcomes were not used for any fit.")
    print("Development-only 5x5 CV (descriptive): PFS delta C = %.4f; OS delta C = %.4f" %
          (cv["pfs"]["delta_c"], cv["os"]["delta_c"]))


if __name__ == "__main__":
    main()
