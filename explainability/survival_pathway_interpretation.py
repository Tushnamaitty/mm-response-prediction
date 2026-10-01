"""
Survival pathway interpretation -- EXPLORATORY and strictly separate from the locked survival incremental analysis.

QUESTION (descriptive only): in the locked ridge-Cox C0+RNA model, what are the 50 Hallmark pathway coefficients and how
stable is their DIRECTION across the 25 outer-CV fits, for PFS and OS separately?

This script performs NO hypothesis testing, NO pathway selection, NO alternative model, penalty, cohort, preprocessing,
fold, seed or endpoint. Coefficients describe PREDICTIVE ASSOCIATION inside a penalized prognostic model; they are NOT
causal effects and no pathway is a "driver".

DESIGN IS REUSED, NOT RE-IMPLEMENTED: everything (cohort construction, fold-local preprocessing and standardization, ridge
Cox solver, outer folds, endpoints) is imported from the locked module explainability/survival_incremental_analysis.py.
The lambda used for each outer fold is the value recorded in the locked tuning table (no re-selection); the final-fit step is
exactly the one in the locked fit_outer(): fit_preprocessor -> transform -> fit_cox(lambda).

REPRODUCTION GATE (runs first; any failure STOPS the script before any coefficient is exported):
  G1  locked spec SHA256 == fe0792db...db6e4 (recorded manifest, analysis_spec.json, and the imported module's spec);
      every needed locked file exists and matches the SHA256 in the locked output_manifest.json; the manifest's combined
      SHA256 recomputes; cohort rebuilds with the locked asserts (n = 674, 459 PFS events, 247 OS deaths).
  G2  locked tuning tables are well formed (50 rows, 25 outer folds x {C0, C0_RNA}) and each recorded lambda equals the
      deterministic rule applied to its recorded inner scores (best mean inner C-index; ties to the largest lambda).
  G3  locked OOF prediction tables align with the cohort (3,370 rows per endpoint = 5 repeats x 674 patients; same
      patients, times and events).
  G4  refitting all 25 outer folds for BOTH models (C0 and C0_RNA), per endpoint, reproduces the locked OOF risks:
      max |refit - locked| < 1e-9 for every patient, repeat and model.
  G5  the pooled per-repeat Harrell C-indices (C0, C0_RNA, delta) match the locked repeat-metrics tables, and their means
      match the locked summary JSON (c_C0, c_C0_RNA, delta_c): |difference| < 1e-9.
  The locked analysis is only READ (never written, rerun or modified).

COEFFICIENT EXPORT (only if every gate passes), per endpoint:
  * the 50 pathway coefficients of the C0_RNA model for each of the 25 outer fits (beta on the training-fold-standardized
    scale, i.e. per training-fold SD of the pathway score; HR = exp(beta) per SD);
  * per pathway: mean, median, SD, min, quartiles, max of beta; HR of the mean and median; number / proportion of positive,
    negative, zero coefficients; direction label (consistently_positive / consistently_negative only if all 25 fits share
    the sign, else mixed); a descriptive rank by |mean beta| (ordering for readability only, not a selection).
  Caveats written into the summary: the 25 fits share ~80% of their training patients, so direction agreement is optimistic
  relative to independent replication; ridge coefficients of correlated pathways trade off against each other and are not
  individually identifiable; clinical covariates are penalized jointly with the pathways (locked design).

Outputs only under artifacts/results_clean_rerun/survival_pathway_interpretation/ (gate-only runs under .../gate_only/).
Existing outputs are never overwritten.

Run from the repository root:
    python -m explainability.survival_pathway_interpretation --gate-only    # reproduction gate only, no coefficients
    python -m explainability.survival_pathway_interpretation                # gate, then export if (and only if) it passes
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from explainability import survival_incremental_analysis as L

LABEL = ("EXPLORATORY SURVIVAL PATHWAY INTERPRETATION (predictive association in the locked ridge-Cox model; "
         "not causal; no hypothesis testing)")
OUT = Path("artifacts/results_clean_rerun/survival_pathway_interpretation")
LOCKED = L.ROOT
EXPECTED_SPEC_SHA = "fe0792dbcb61ec62d3901aa7b268b7fcb51a5b60291b9dc4b3da75d0317db6e4"
AUDITED_SCRIPT_SHA = "bd188693a586538bb28b49c8fedcc73f5d27eaea15d94757bd7d4cc54394ad9a"   # copy audited before this script
RISK_TOL = 1e-9
C_TOL = 1e-9
ENDPOINTS = [("pfs", "primary"), ("os", "secondary")]          # (endpoint, locked analysis label)
MODELS = [("C0", False), ("C0_RNA", True)]
N_FITS = L.N_REPEATS * L.N_FOLDS
NEEDED_LOCKED_FILES = ["analysis_spec.json", "final_summary.json",
                       "primary_pfs_oof_predictions.csv", "primary_pfs_repeat_metrics.csv", "primary_pfs_summary.json",
                       "primary_pfs_tuning.csv", "secondary_os_oof_predictions.csv", "secondary_os_repeat_metrics.csv",
                       "secondary_os_summary.json", "secondary_os_tuning.csv"]
CAVEATS = [
    "Coefficients describe predictive association inside a penalized prognostic model; they are not causal effects and no "
    "pathway is a driver.",
    "The 25 outer-CV fits share about 80% of their training patients (5 overlapping repeats), so direction agreement across "
    "fits is optimistic relative to independent replication.",
    "Ridge coefficients of correlated Hallmark pathways trade off against each other; individual coefficients are not "
    "identifiable as separate effects.",
    "Coefficients are on the training-fold-standardized scale (HR per training-fold SD of the pathway score); clinical "
    "covariates are penalized jointly with the pathways, as in the locked design.",
    "Exploratory and descriptive: no hypothesis tests, no pathway selection, no alternative models; internal validation only.",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Gate:
    """Collects every gate check (name, pass/fail, detail) so a failure report is complete."""

    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append({"check": name, "passed": bool(ok), "detail": detail})
        print(("  PASS  " if ok else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        return bool(ok)

    @property
    def passed(self):
        return all(r["passed"] for r in self.rows)

    def failures(self):
        return [r for r in self.rows if not r["passed"]]


# ---------------------------------------------------------------------------
# G1: locked spec / manifest / files / cohort
# ---------------------------------------------------------------------------
def verify_locked_files(g: Gate):
    mpath = LOCKED / "output_manifest.json"
    if not g.check("locked output_manifest.json exists", mpath.exists(), str(mpath)):
        return None
    with open(mpath, encoding="utf-8") as f:
        man = json.load(f)
    g.check("manifest spec SHA256 == expected fe0792db...", man.get("spec_sha256") == EXPECTED_SPEC_SHA, str(man.get("spec_sha256")))
    g.check("imported locked module spec_sha() == expected", L.spec_sha() == EXPECTED_SPEC_SHA, L.spec_sha())
    sp = LOCKED / "analysis_spec.json"
    if sp.exists():
        with open(sp, encoding="utf-8") as f:
            g.check("analysis_spec.json spec SHA256 == expected", json.load(f).get("spec_sha256") == EXPECTED_SPEC_SHA)
    files = man.get("files", {})
    recomputed = L.sha256_text("\n".join(f"{k}:{files[k]}" for k in sorted(files)))
    g.check("manifest combined SHA256 recomputes from its file hashes", recomputed == man.get("combined_sha256"),
            str(man.get("combined_sha256")))
    for rel in NEEDED_LOCKED_FILES:
        p = LOCKED / rel
        if not g.check(f"needed locked file present: {rel}", p.exists() and rel in files):
            continue
        g.check(f"locked file hash matches manifest: {rel}", L.sha256_file(p) == files[rel])
    return man


def locked_script_info():
    p = Path("explainability/survival_incremental_analysis.py")
    h = L.sha256_file(p) if p.exists() else None
    return {"path": str(p), "sha256": h, "matches_audited_copy": bool(h == AUDITED_SCRIPT_SHA),
            "note": "informational only; reproduction of the locked OOF risks is the binding test"}


# ---------------------------------------------------------------------------
# G2/G3: tuning table and OOF table integrity
# ---------------------------------------------------------------------------
def lambda_rule(inner_json: str) -> float:
    means = {float(k): float(v) for k, v in json.loads(inner_json).items()}
    best = max(means.values())
    return float(max(l for l, v in means.items() if abs(v - best) <= 1e-12))      # locked tie rule: largest lambda


def check_tuning(g: Gate, tune: pd.DataFrame, endpoint: str):
    ok_shape = len(tune) == 2 * N_FITS and set(tune["model"]) == {"C0", "C0_RNA"}
    g.check(f"[{endpoint}] tuning table has {2 * N_FITS} rows (25 folds x 2 models)", ok_shape, f"rows={len(tune)}")
    key = tune[["repeat", "fold", "model"]]
    g.check(f"[{endpoint}] tuning keys unique and cover 5 repeats x 5 folds",
            (not key.duplicated().any()) and set(zip(tune.repeat, tune.fold)) ==
            {(r, f) for r in range(L.N_REPEATS) for f in range(L.N_FOLDS)})
    in_grid = tune["lambda"].isin([float(x) for x in L.LAMBDA_GRID]).all()
    g.check(f"[{endpoint}] every recorded lambda is in the locked grid", bool(in_grid))
    rule_ok = all(abs(lambda_rule(r.inner_scores_json) - float(r.lambda_)) == 0.0
                  for r in tune.rename(columns={"lambda": "lambda_"}).itertuples(index=False))
    g.check(f"[{endpoint}] recorded lambdas equal the locked selection rule applied to the recorded inner scores", rule_ok)
    return {(int(r.repeat), int(r.fold), r.model): float(r.lambda_)
            for r in tune.rename(columns={"lambda": "lambda_"}).itertuples(index=False)}


def check_oof_alignment(g: Gate, oof: pd.DataFrame, df: pd.DataFrame, endpoint: str):
    n = len(df)
    g.check(f"[{endpoint}] OOF table has {L.N_REPEATS * n} rows (5 repeats x {n} patients)", len(oof) == L.N_REPEATS * n, f"rows={len(oof)}")
    g.check(f"[{endpoint}] OOF (repeat, public_id) unique", not oof.duplicated(["repeat", "public_id"]).any())
    g.check(f"[{endpoint}] OOF patients equal the rebuilt cohort", set(oof["public_id"]) == set(df["public_id"]))
    t, e = L.endpoint_arrays(df, endpoint)
    ref = pd.DataFrame({"public_id": df["public_id"].to_numpy(), "t_ref": t, "e_ref": e})
    m = oof.merge(ref, on="public_id", how="left")
    g.check(f"[{endpoint}] OOF times and events equal the cohort's {endpoint.upper()} endpoint",
            bool((m["time"].to_numpy(float) == m["t_ref"].to_numpy(float)).all()
                 and (m["event"].to_numpy(int) == m["e_ref"].to_numpy(int)).all()))


# ---------------------------------------------------------------------------
# G4/G5: refit and compare
# ---------------------------------------------------------------------------
def refit_endpoint(df, paths, endpoint, lam_map):
    """Replays the locked final-fit step for all 25 outer folds and both models."""
    pred = {m: {} for m, _ in MODELS}                  # model -> {(repeat, public_id): risk}
    coef_rows = []
    t0 = time.time()
    for rep in range(L.N_REPEATS):
        fold = L.make_outer_folds(len(df), rep)
        for f in range(L.N_FOLDS):
            te_idx, tr_idx = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            ttr, etr = L.endpoint_arrays(tr, endpoint)
            for model_name, inc_rna in MODELS:
                lam = lam_map[(rep, f, model_name)]
                pp = L.fit_preprocessor(tr, paths, inc_rna)                    # locked fold-local preprocessing
                Xtr, Xte = L.transform(tr, pp), L.transform(te, pp)
                beta = L.fit_cox(Xtr, ttr, etr, lam)                           # locked ridge-Cox solver
                lp = Xte @ beta
                for j, idx in enumerate(te_idx):
                    pred[model_name][(rep, df.iloc[idx]["public_id"])] = float(lp[j])
                if inc_rna:
                    if list(pp.feature_names[-len(paths):]) != list(paths):
                        raise AssertionError("pathway coefficient positions do not match the pathway order")
                    coef_rows.append({"endpoint": endpoint, "repeat": rep, "fold": f, "lambda": lam,
                                      **{p: float(b) for p, b in zip(paths, beta[-len(paths):])}})
            print(f"    [{endpoint}] refit repeat {rep + 1}/{L.N_REPEATS} fold {f + 1}/{L.N_FOLDS} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return pred, pd.DataFrame(coef_rows)


def compare_endpoint(g: Gate, df, endpoint, label, pred, oof, rep_metrics, summary):
    idx = oof.set_index(["repeat", "public_id"])
    report = {"endpoint": endpoint}
    for model_name, _ in MODELS:
        col = "risk_C0_RNA" if model_name == "C0_RNA" else "risk_C0"
        keys = list(pred[model_name].keys())
        refit = np.array([pred[model_name][k] for k in keys])
        locked = idx.loc[keys, col].to_numpy(float)
        d = np.abs(refit - locked)
        report[f"max_abs_diff_{model_name}"] = float(d.max())
        report[f"mean_abs_diff_{model_name}"] = float(d.mean())
        g.check(f"[{endpoint}] refit {model_name} OOF risks reproduce the locked risks (max |diff| < {RISK_TOL:g})",
                bool(d.max() < RISK_TOL), f"max={d.max():.3e}, mean={d.mean():.3e}, n={len(d)}")
    # per-repeat pooled Harrell C-index from the REFIT risks
    t_all, e_all = L.endpoint_arrays(df, endpoint)
    pos = {pid: i for i, pid in enumerate(df["public_id"].to_numpy())}
    per_rep = []
    for rep in range(L.N_REPEATS):
        ids = [pid for (r, pid) in pred["C0"].keys() if r == rep]
        ii = np.array([pos[p] for p in ids])
        r0 = np.array([pred["C0"][(rep, p)] for p in ids])
        r1 = np.array([pred["C0_RNA"][(rep, p)] for p in ids])
        c0, c1 = L.harrell_c(t_all[ii], e_all[ii], r0), L.harrell_c(t_all[ii], e_all[ii], r1)
        per_rep.append({"repeat": rep, "c_C0": c0, "c_C0_RNA": c1, "delta_c": c1 - c0})
    pr = pd.DataFrame(per_rep).set_index("repeat")
    lm = rep_metrics.set_index("repeat")
    worst = 0.0
    for col in ["c_C0", "c_C0_RNA", "delta_c"]:
        worst = max(worst, float(np.max(np.abs(pr[col].to_numpy() - lm.loc[pr.index, col].to_numpy(float)))))
    g.check(f"[{endpoint}] per-repeat C-indices (C0, C0_RNA, delta) reproduce the locked repeat-metrics table (< {C_TOL:g})",
            worst < C_TOL, f"max|diff|={worst:.3e}")
    means = {"c_C0": float(pr["c_C0"].mean()), "c_C0_RNA": float(pr["c_C0_RNA"].mean()), "delta_c": float(pr["delta_c"].mean())}
    worst_s = max(abs(means[k] - float(summary[k])) for k in means)
    g.check(f"[{endpoint}] mean C-indices and delta C reproduce the locked summary JSON (< {C_TOL:g})", worst_s < C_TOL,
            f"delta_c refit={means['delta_c']:.12f} locked={float(summary['delta_c']):.12f}")
    report.update({"refit_means": means, "locked_summary": {k: float(summary[k]) for k in means},
                   "per_repeat_refit": per_rep, "max_abs_diff_repeat_metrics": worst, "max_abs_diff_summary": worst_s})
    return report


# ---------------------------------------------------------------------------
# Coefficient summaries (descriptive)
# ---------------------------------------------------------------------------
def stability_table(coef: pd.DataFrame, paths, endpoint):
    B = coef[paths].to_numpy(float)                     # 25 fits x 50 pathways
    nf = B.shape[0]
    rows = []
    for j, p in enumerate(paths):
        b = B[:, j]
        npos, nneg, nzero = int((b > 0).sum()), int((b < 0).sum()), int((b == 0).sum())
        label = ("consistently_positive" if npos == nf else "consistently_negative" if nneg == nf else "mixed")
        rows.append({"endpoint": endpoint, "pathway": p, "n_fits": nf, "beta_mean": float(b.mean()),
                     "beta_median": float(np.median(b)), "beta_sd": float(b.std(ddof=1)), "beta_min": float(b.min()),
                     "beta_q25": float(np.quantile(b, 0.25)), "beta_q75": float(np.quantile(b, 0.75)),
                     "beta_max": float(b.max()), "HR_of_mean_beta": float(np.exp(b.mean())),
                     "HR_of_median_beta": float(np.exp(np.median(b))), "HR_min_fit": float(np.exp(b.min())),
                     "HR_max_fit": float(np.exp(b.max())), "n_positive": npos, "n_negative": nneg, "n_zero": nzero,
                     "prop_positive": npos / nf, "prop_negative": nneg / nf,
                     "direction_agreement": max(npos, nneg) / nf, "direction_label": label})
    t = pd.DataFrame(rows)
    t["rank_abs_mean_beta_descriptive_only"] = t["beta_mean"].abs().rank(ascending=False, method="first").astype(int)
    return t


def combined_table(pfs: pd.DataFrame, osr: pd.DataFrame):
    a = pfs.add_prefix("pfs_").rename(columns={"pfs_pathway": "pathway"})
    b = osr.add_prefix("os_").rename(columns={"os_pathway": "pathway"})
    m = a.merge(b, on="pathway", validate="one_to_one")
    m["same_sign_of_mean_beta_pfs_os"] = np.sign(m["pfs_beta_mean"]) == np.sign(m["os_beta_mean"])
    m["consistent_same_direction_both_endpoints"] = (
        m["pfs_direction_label"].eq(m["os_direction_label"]) & m["pfs_direction_label"].ne("mixed"))
    return m


def build_summary(gate_report, tables, locked_man):
    out = {"label": LABEL, "status": "gate passed; coefficients exported", "caveats": CAVEATS,
           "locked_spec_sha256": EXPECTED_SPEC_SHA, "locked_manifest_combined_sha256": locked_man["combined_sha256"],
           "n_outer_fits_per_endpoint": N_FITS, "n_pathways": int(len(tables["pfs"])),
           "coefficient_scale": "log-hazard per training-fold SD of the standardized pathway score (HR = exp(beta))",
           "direction_rule": "consistently_positive / consistently_negative only if all 25 fits share the sign; else mixed",
           "endpoints": {}}
    for ep in ("pfs", "os"):
        t = tables[ep]
        out["endpoints"][ep] = {
            "n_consistently_positive": int((t["direction_label"] == "consistently_positive").sum()),
            "n_consistently_negative": int((t["direction_label"] == "consistently_negative").sum()),
            "n_mixed": int((t["direction_label"] == "mixed").sum()),
            "consistently_positive_pathways": t.loc[t["direction_label"] == "consistently_positive", "pathway"].tolist(),
            "consistently_negative_pathways": t.loc[t["direction_label"] == "consistently_negative", "pathway"].tolist()}
    both = tables["both"]
    out["consistent_same_direction_in_both_endpoints"] = both.loc[both["consistent_same_direction_both_endpoints"], "pathway"].tolist()
    out["reproduction_gate"] = {"passed": True, "tolerances": {"risk": RISK_TOL, "c_index": C_TOL},
                                "per_endpoint": gate_report["endpoints"]}
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-only", action="store_true", help="run the reproduction gate only; export no coefficients")
    args = ap.parse_args()
    t0 = time.time()
    print(LABEL)
    require_new = OUT / "output_manifest.json"
    if not args.gate_only and require_new.exists():
        raise SystemExit(f"[survival_pathway_interpretation] {require_new} exists: existing outputs are never overwritten.")
    require_dir = OUT.resolve() != LOCKED.resolve()
    if not require_dir:
        raise SystemExit("output directory must differ from the locked survival directory")

    g = Gate()
    print("\n=== G1: locked spec, manifest, files, cohort ===")
    man = verify_locked_files(g)
    df = paths = None
    try:
        df, paths = L.build_cohort()                                   # locked cohort construction with its asserts
        g.check("cohort rebuilt with the locked asserts (n=674, 459 PFS events, 247 OS deaths, 50 pathways)",
                len(df) == L.EXPECTED_N and len(paths) == L.EXPECTED_PATHWAYS, f"n={len(df)}, pathways={len(paths)}")
    except Exception as e:                                             # report, do not continue
        g.check("cohort rebuilt with the locked asserts", False, f"{type(e).__name__}: {e}")

    reports, coefs = {"endpoints": {}}, {}
    if man is not None and df is not None:
        for endpoint, label in ENDPOINTS:
            print(f"\n=== G2-G5: {endpoint.upper()} ===")
            try:
                tune = pd.read_csv(LOCKED / f"{label}_{endpoint}_tuning.csv")
                oof = pd.read_csv(LOCKED / f"{label}_{endpoint}_oof_predictions.csv")
                rep_metrics = pd.read_csv(LOCKED / f"{label}_{endpoint}_repeat_metrics.csv")
                with open(LOCKED / f"{label}_{endpoint}_summary.json", encoding="utf-8") as f:
                    summary = json.load(f)
                lam_map = check_tuning(g, tune, endpoint)
                check_oof_alignment(g, oof, df, endpoint)
                if g.passed:                                           # refit only if the locked tables are sound
                    pred, coef = refit_endpoint(df, paths, endpoint, lam_map)
                    reports["endpoints"][endpoint] = compare_endpoint(g, df, endpoint, label, pred, oof, rep_metrics, summary)
                    coefs[endpoint] = coef
                else:
                    g.check(f"[{endpoint}] refit skipped because an earlier gate failed", False)
            except Exception as e:
                g.check(f"[{endpoint}] gate execution", False, f"{type(e).__name__}: {e}")

    import scipy
    import sklearn
    versions_now = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    gate_record = {"label": LABEL, "created_utc": now_utc(), "passed": g.passed, "checks": g.rows,
                   "tolerances": {"risk": RISK_TOL, "c_index": C_TOL}, "endpoints": reports["endpoints"],
                   "locked_script": locked_script_info(), "versions_now": versions_now,
                   "versions_locked_run": (man or {}).get("versions"),
                   "note": "the locked analysis was only read; nothing in the locked directory was written"}

    if not g.passed:
        target = OUT / ("gate_only" if args.gate_only else "") / f"reproduction_check_FAILED_{now_utc()}.json"
        write_json(gate_record, target)
        print(f"\nREPRODUCTION GATE FAILED ({len(g.failures())} failed checks). No coefficients exported.\nReport: {target}")
        for r in g.failures():
            print("  -", r["check"], r["detail"])
        sys.exit(1)

    if args.gate_only:
        target = OUT / "gate_only" / f"reproduction_check_{now_utc()}.json"
        write_json(gate_record, target)
        print(f"\nREPRODUCTION GATE PASSED (gate-only run). No coefficients exported. Report: {target}")
        return

    # ------------------------------ export (gate passed) ------------------------------
    files = []
    write_json(gate_record, OUT / "reproduction_check.json")
    files.append(OUT / "reproduction_check.json")
    tables = {}
    for endpoint, _ in ENDPOINTS:
        p = OUT / f"{endpoint}_fold_coefficients.csv"
        coefs[endpoint].to_csv(p, index=False)
        files.append(p)
        tables[endpoint] = stability_table(coefs[endpoint], paths, endpoint)
        p = OUT / f"{endpoint}_pathway_stability.csv"
        tables[endpoint].to_csv(p, index=False)
        files.append(p)
    tables["both"] = combined_table(tables["pfs"], tables["os"])
    p = OUT / "pathway_stability_both_endpoints.csv"
    tables["both"].to_csv(p, index=False)
    files.append(p)
    summary = build_summary(gate_record, tables, man)
    summary["completed_utc"] = now_utc()
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    p = OUT / "summary.json"
    write_json(summary, p)
    files.append(p)

    hashes = {str(f.relative_to(OUT)).replace("\\", "/"): L.sha256_file(f) for f in files}
    combined = L.sha256_text("\n".join(f"{k}:{hashes[k]}" for k in sorted(hashes)))
    write_json({"label": LABEL, "files": hashes, "combined_sha256": combined, "locked_spec_sha256": EXPECTED_SPEC_SHA,
                "locked_manifest_combined_sha256": man["combined_sha256"], "locked_script": locked_script_info(),
                "this_script_sha256": L.sha256_file(Path(__file__)), "versions": versions_now},
               OUT / "output_manifest.json")
    write_json({"status": "complete", "combined_sha256": combined}, OUT / "DONE.json")

    print(f"\nREPRODUCTION GATE PASSED; coefficients exported in {(time.time() - t0) / 60:.1f} min.")
    for ep in ("pfs", "os"):
        e = summary["endpoints"][ep]
        print(f"  {ep.upper()}: consistently positive {e['n_consistently_positive']}, "
              f"consistently negative {e['n_consistently_negative']}, mixed {e['n_mixed']}")
    print(f"Output: {OUT}\nCombined SHA256: {combined}")
    print("EXPLORATORY: predictive association only; not causal; no pathway is a driver.")


if __name__ == "__main__":
    main()
