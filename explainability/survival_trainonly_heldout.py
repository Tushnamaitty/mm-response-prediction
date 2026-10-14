"""
Survival redesign -- STAGE H (held-out evaluation) of the train-only protocol (explainability/SURVIVAL_TRAINONLY_PROTOCOL.md).

Run ONCE, after Stage D (survival_trainonly_development.py) has written and hashed artifacts/results_clean_rerun/
survival_trainonly/frozen_models.json. Held-out survival outcomes are not read until the protocol hash, the frozen-artifact hash
and the Stage-D gate report have all been verified; the verification happens in this script before the cohort (which carries the
outcomes) is rebuilt.

Held-out set = validation + test patients (201; 149 PFS events; 82 deaths). Frozen models are only SCORED here; nothing is refit,
re-tuned or re-selected.

PRIMARY (A1): PFS delta C = C(C0 + RNA) - C(C0) on the held-out patients. Criterion met only if (1) delta C > 0, (2) the 2,000-draw
patient-bootstrap 95% percentile CI lower bound > 0 (seed 42) and (3) the one-sided permutation p <= 0.025, where the null permutes
the 50-pathway RNA vectors among the held-out patients and re-scores the frozen RNA model (B = 10,000; p = (1 + k) / (B + 1)).
Otherwise "not met" (inconclusive unless the CI upper bound is < 0). No other analysis can replace the primary verdict.

Other analyses (descriptive / sensitivity, no pass-fail role): A2 OS; A3 IBS; S1 RNA day [-30, 0]; S2 expanded baseline; S3
development-only CV (read from Stage D); S4 validation vs test.

POST HOC, EXPLORATORY: the earlier pooled results were seen first; not blind, not independent, not external validation.

Run from the repo root:   python -m explainability.survival_trainonly_heldout
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from explainability import survival_incremental_analysis as L
from explainability import survival_trainonly_development as D

OUT = D.OUT
REPORT = OUT / "heldout_report.json"
B_PERM = 10_000
ALPHA_ONE_SIDED = 0.025
LABEL = ("SURVIVAL TRAIN-ONLY REDESIGN, STAGE H (held-out validation + test evaluation): post hoc, exploratory; NOT independent "
         "validation; NOT external validation")


def boot_delta(t, e, r0, r1, seed, B=L.N_BOOT):
    rng = np.random.RandomState(seed)
    n = len(t)
    vals = np.empty(B)
    for b in range(B):
        i = rng.choice(np.arange(n), size=n, replace=True)
        vals[b] = D.harrell_c_fast(t[i], e[i], r1[i]) - D.harrell_c_fast(t[i], e[i], r0[i])
    return {"n_boot": int(B), "seed": int(seed), "ci_lower": float(np.quantile(vals, 0.025)),
            "ci_upper": float(np.quantile(vals, 0.975)), "bootstrap_mean": float(vals.mean())}


def ibs(lp, fm, t, e):
    grid = np.linspace(L.IBS_T0, L.IBS_T1, L.IBS_GRID_N)
    S = L.predict_survival(lp, np.array(fm["baseline_t"]), np.array(fm["baseline_H"]), grid)
    br = L.ipcw_brier_rows(t, e, S, grid, np.array(fm["km_t"]), np.array(fm["km_s"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.nanmean([np.nanmean(row) for row in br]))


def score(frame, ana, R, endpoint):
    m0, m1 = ana["models"]["C0"], ana["models"]["C0_RNA"]
    X0, X1 = D.design(frame, m0, R), D.design(frame, m1, R)
    r0, r1 = X0 @ np.array(m0["beta"]), X1 @ np.array(m1["beta"])
    t, e = L.endpoint_arrays(frame, endpoint)
    return {"t": t, "e": e, "r0": r0, "r1": r1, "X1": X1, "m0": m0, "m1": m1}


def evaluate(frame, ana, R, endpoint, seed, with_ibs=False):
    s = score(frame, ana, R, endpoint)
    c0, c1 = D.harrell_c_fast(s["t"], s["e"], s["r0"]), D.harrell_c_fast(s["t"], s["e"], s["r1"])
    out = {"n_patients": int(len(frame)), "n_events": int(s["e"].sum()), "c_C0": c0, "c_C0_RNA": c1, "delta_c": c1 - c0,
           "lambda_C0": s["m0"]["lambda"], "lambda_C0_RNA": s["m1"]["lambda"],
           "bootstrap_delta_c": boot_delta(s["t"], s["e"], s["r0"], s["r1"], seed)}
    if with_ibs:
        i0, i1 = ibs(s["r0"], s["m0"], s["t"], s["e"]), ibs(s["r1"], s["m1"], s["t"], s["e"])
        out["ibs_C0"], out["ibs_C0_RNA"], out["delta_ibs"] = i0, i1, i1 - i0
    return out, s


def permutation_null(s, n_paths, B=B_PERM):
    """Permute the 50-pathway vectors among the held-out patients; re-score the frozen RNA model (comparator fixed)."""
    X1, b1 = s["X1"], np.array(s["m1"]["beta"])
    Xc, Xr = X1[:, :-n_paths], X1[:, -n_paths:]
    lc, br = Xc @ b1[:-n_paths], b1[-n_paths:]
    c0 = D.harrell_c_fast(s["t"], s["e"], s["r0"])
    null = np.empty(B)
    for p in range(B):
        pi = np.random.RandomState(L.SEED + 20000 + p).permutation(len(lc))
        null[p] = D.harrell_c_fast(s["t"], s["e"], lc + Xr[pi] @ br) - c0
    return null


def main():
    t_start = time.time()
    print(LABEL)
    if REPORT.exists():
        raise SystemExit(f"[survival_trainonly_heldout] {REPORT} exists: Stage H is run once; outputs are never overwritten.")
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(("  PASS " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        if not ok:
            raise SystemExit(f"[survival_trainonly_heldout] CHECK FAILED: {name} {detail}")

    # ---- verification BEFORE any held-out outcome is read ----
    ok, rec = D.protocol_hash_ok()
    chk("H1 protocol file hash equals the recorded SHA-256", ok)
    frozen_p = OUT / "frozen_models.json"
    recorded = (OUT / "frozen_models.sha256").read_text(encoding="utf-8").split()[0]
    chk("H1 frozen_models.json hash equals the recorded hash", D.sha_lf(frozen_p) == recorded, recorded[:16] + "...")
    gate = json.load(open(OUT / "development_gate_report.json", encoding="utf-8"))
    chk("H1 Stage-D gate report: all gates passed and records the same artifact hash",
        all(r["passed"] for r in gate["gates"]) and gate["frozen_models_sha256"] == recorded, f"{len(gate['gates'])} gates")
    art = json.load(open(frozen_p, encoding="utf-8"))
    chk("H1 artifact records the protocol hash and the locked spec", art["protocol_sha256"] == rec
        and art["locked_spec_sha256"] == D.EXPECTED_SPEC_SHA)
    chk("H1 frozen artifact was written before this run started", frozen_p.stat().st_mtime < t_start)
    # ---- only now: rebuild the cohort (this reads the survival outcomes) ----
    x, paths = L.build_cohort()
    chk("H2 cohort rebuilt with the locked asserts and the same pathway list", len(x) == 674 and paths == art["pathways"])
    sp = D.partition_table()
    x = x.merge(sp, on="public_id", how="left", validate="one_to_one")
    ho = x[x.split != "train"].sort_values("public_id").reset_index(drop=True)
    dev_ids = sorted(x.loc[x.split == "train", "public_id"])
    ho_ids = sorted(ho["public_id"])
    chk("H3 held-out patient set equals the one frozen in Stage D (hash)",
        hashlib.sha256("\n".join(ho_ids).encode()).hexdigest() == art["heldout_patient_ids_sha256"] and len(ho) == 201)
    chk("H3 development patient set equals the frozen one and is disjoint from held-out",
        hashlib.sha256("\n".join(dev_ids).encode()).hexdigest() == art["development_patient_ids_sha256"]
        and not (set(dev_ids) & set(ho_ids)))
    chk("H3 held-out events: 149 PFS / 82 deaths (validation 71/45, test 71...78/37 as in the protocol)",
        (int(ho.censor_pfs.sum()), int(ho.censor_os.sum())) == (149, 82)
        and (int(ho[ho.split == "val"].censor_pfs.sum()), int(ho[ho.split == "test"].censor_pfs.sum())) == (71, 78))

    R = None
    if art["s2_status"] == "run":
        from explainability import survival_richer_baseline_sensitivity as R   # noqa: N813
    ho_win = ho[(ho.rna_day >= L.SENS_WINDOW[0]) & (ho.rna_day <= L.SENS_WINDOW[1])].reset_index(drop=True)
    chk("H3 held-out sensitivity cohort: 186 patients / 137 PFS events", (len(ho_win), int(ho_win.censor_pfs.sum())) == (186, 137))
    ho_rich = R.build_baseline_variables(ho)[0] if R is not None else None
    A = art["analyses"]
    res = {}

    # ---- A1: primary ----
    a1, s1 = evaluate(ho, A["A1_pfs"], None, "pfs", L.SEED, with_ibs=True)
    ref = [L.harrell_c(s1["t"], s1["e"], s1["r0"]), L.harrell_c(s1["t"], s1["e"], s1["r1"])]
    chk("H4 vectorised C-index equals the locked harrell_c on the observed held-out risks (exact)",
        ref == [a1["c_C0"], a1["c_C0_RNA"]])
    t0 = time.time()
    null = permutation_null(s1, len(paths))
    k = int((null >= a1["delta_c"]).sum())
    p_perm = (1 + k) / (B_PERM + 1)
    a1["permutation"] = {"B": B_PERM, "k_null_ge_observed": k, "one_sided_p": float(p_perm), "null_mean": float(null.mean()),
                         "null_sd": float(null.std(ddof=1)), "null_q95": float(np.quantile(null, 0.95)),
                         "seconds": time.time() - t0,
                         "definition": "RNA vectors permuted among held-out patients; frozen RNA model re-scored; comparator fixed"}
    chk("H5 permutation null completed (B = 10,000)", len(null) == B_PERM and np.isfinite(null).all(),
        f"{a1['permutation']['seconds']:.1f}s")
    lo, hi = a1["bootstrap_delta_c"]["ci_lower"], a1["bootstrap_delta_c"]["ci_upper"]
    crit = {"delta_c_gt_0": bool(a1["delta_c"] > 0), "bootstrap_ci_lower_gt_0": bool(lo > 0),
            "permutation_one_sided_p_le_0.025": bool(p_perm <= ALPHA_ONE_SIDED)}
    met = all(crit.values())
    verdict = ("CRITERION MET: held-out delta C > 0, bootstrap 95% CI excludes 0 and one-sided permutation p <= 0.025"
               if met else ("RNA did not improve discrimination in the held-out patients (criterion not met; CI upper bound < 0)"
                            if hi < 0 else "CRITERION NOT MET: inconclusive (not evidence of no RNA effect)"))
    a1["criterion"] = {**crit, "met": met, "verdict": verdict, "rule": "all three conditions (protocol section 6)"}
    res["A1_pfs"] = a1

    # ---- A2, A3 (IBS inside A1/A2), S1, S2 ----
    res["A2_os"], _ = evaluate(ho, A["A2_os"], None, "os", L.SEED + 1000, with_ibs=True)
    res["S1_pfs_m30_0"], _ = evaluate(ho_win, A["S1_pfs_m30_0"], None, "pfs", L.SEED)
    for aid, ep, seed in (("S2_pfs_rich", "pfs", L.SEED), ("S2_os_rich", "os", L.SEED + 1000)):
        if art["s2_status"] == "run":
            res[aid], _ = evaluate(ho_rich, A[aid], R, ep, seed)
        else:
            res[aid] = {"status": art["s2_status"]}

    # ---- S4: partition-level description of A1 (descriptive) ----
    res["S4_partitions_A1"] = {}
    for part in ("val", "test"):
        sub = ho[ho.split == part].reset_index(drop=True)
        res["S4_partitions_A1"][part], _ = evaluate(sub, A["A1_pfs"], None, "pfs", L.SEED)

    # ---- S3 development-only CV (computed in Stage D; read here) ----
    cvs = json.load(open(OUT / "development_cv_and_figure5_refit_summary.json", encoding="utf-8"))
    res["S3_development_only_cv"] = {ep: {k_: cvs[ep][k_] for k_ in ("c_C0", "c_C0_RNA", "delta_c", "bootstrap_delta_c", "n_patients",
                                                                    "n_events")} for ep in ("pfs", "os")}
    chk("H6 frozen artifact unchanged at the end of the run", D.sha_lf(frozen_p) == recorded)

    report = {"label": LABEL, "protocol_sha256": rec, "frozen_models_sha256": recorded,
              "code_sha256_lf": D.sha_lf(__file__), "heldout": {"patients": 201, "pfs_events": 149, "deaths": 82,
                                                                "validation": 101, "test": 100},
              "results": res, "checks": checks,
              "interpretation": ("post hoc, exploratory; evaluated in held-out validation and test patients; the earlier pooled "
                                 "results had been seen; not independent; not external validation; no causal or biological-value claim"),
              "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seconds": time.time() - t_start}
    D.dump_json(report, REPORT)
    pd.DataFrame({"perm": np.arange(B_PERM), "null_delta_c": null}).to_csv(OUT / "a1_permutation_null.csv", index=False)
    rows = []
    for aid, r in res.items():
        if aid == "S4_partitions_A1":
            for part, rr in r.items():
                rows.append({"analysis": f"S4_{part}", **{k_: rr[k_] for k_ in ("n_patients", "n_events", "c_C0", "c_C0_RNA", "delta_c")},
                             "ci_lower": rr["bootstrap_delta_c"]["ci_lower"], "ci_upper": rr["bootstrap_delta_c"]["ci_upper"]})
        elif aid != "S3_development_only_cv" and "delta_c" in r:
            rows.append({"analysis": aid, **{k_: r[k_] for k_ in ("n_patients", "n_events", "c_C0", "c_C0_RNA", "delta_c")},
                         "ci_lower": r["bootstrap_delta_c"]["ci_lower"], "ci_upper": r["bootstrap_delta_c"]["ci_upper"],
                         "ibs_C0": r.get("ibs_C0"), "ibs_C0_RNA": r.get("ibs_C0_RNA")})
    pd.DataFrame(rows).to_csv(OUT / "heldout_results_table.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "heldout_checks.csv", index=False)
    man = {f.name: D.sha_lf(f) for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.csv")) if f.name != "heldout_manifest.json"}
    D.dump_json({"files": man, "frozen_models_sha256": recorded, "protocol_sha256": rec}, OUT / "heldout_manifest.json")

    print("\n=== A1 PRIMARY (PFS, held-out validation + test) ===")
    print(f"C(C0) = {a1['c_C0']:.4f}; C(C0+RNA) = {a1['c_C0_RNA']:.4f}; delta C = {a1['delta_c']:+.4f} "
          f"(95% CI {lo:+.4f} to {hi:+.4f}); one-sided permutation p = {p_perm:.4g} (k = {k} of {B_PERM})")
    print(verdict)
    print(f"Stage H finished in {(time.time() - t_start):.0f}s. Report: {REPORT}")


if __name__ == "__main__":
    main()
