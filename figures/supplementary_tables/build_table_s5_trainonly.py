"""
Build Supplementary Table S5 (survival discrimination and prediction error with and without pre-treatment RNA), TRAIN-ONLY redesign (read-only).

Run from repo root, after figures/fig45_trainonly_extract_data.py:
    python figures/supplementary_tables/build_table_s5_trainonly.py

Writes Table_S5_survival_trainonly.{md,csv} and table_s5_trainonly_verification.csv next to this script. The earlier Table S5
(pooled 5x5 CV over all 674 patients; Table_S5_survival.md) and its builder are left unchanged. Values come from
figures/data/Fig45_trainonly/ and are cross-checked against artifacts/results_clean_rerun/survival_trainonly/ and the frozen protocol
explainability/SURVIVAL_TRAINONLY_PROTOCOL.md. No model is fitted here and no patient identifier is written.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("figures/supplementary_tables")
FD = Path("figures/data/Fig45_trainonly")
R = Path("artifacts/results_clean_rerun/survival_trainonly")
checks = []


def check(name, ok, observed="", expected=""):
    checks.append({"check": name, "passed": bool(ok), "observed": str(observed)[:160], "expected": str(expected)[:160]})


def f(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = 0.0 if round(x, nd) == 0 else x
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    d = pd.read_csv(FD / "fig4B_trainonly_delta_c.csv").set_index("row_id")
    res = pd.read_csv(R / "heldout_results_table.csv").set_index("analysis")
    null = pd.read_csv(R / "a1_permutation_null.csv")
    rep = {ep: pd.read_csv(R / f"devcv_{ep}_repeat_metrics.csv") for ep in ("pfs", "os")}
    devsum = {ep: json.load(open(R / f"devcv_{ep}_summary.json")) for ep in ("pfs", "os")}
    rpt = json.load(open(R / "heldout_report.json"))

    # ---- cross-checks against the locked held-out and development files ----
    for rid, an in (("A1", "A1_pfs"), ("A2", "A2_os")):
        s, r = res.loc[an], d.loc[rid]
        check(f"{rid}: C-index, delta C, CI and IBS equal the locked held-out table",
              all(abs(a - b) < 1e-12 for a, b in [(s.c_C0, r.c_index_baseline), (s.c_C0_RNA, r.c_index_plus_rna), (s.delta_c, r.delta_c),
                                                  (s.ci_lower, r.ci_lower), (s.ci_upper, r.ci_upper), (s.ibs_C0, r.ibs_baseline), (s.ibs_C0_RNA, r.ibs_plus_rna)])
              and int(s.n_patients) == int(r.n_patients) and int(s.n_events) == int(r.n_events))
    k = int((null.null_delta_c >= d.loc["A1", "delta_c"]).sum())
    p_chk = (k + 1) / (len(null) + 1)
    check("A1 permutation p recomputed from the saved null (B = 10,000)", len(null) == 10000 and abs(p_chk - d.loc["A1", "permutation_p_one_sided"]) < 1e-12, f"k={k}; p={p_chk:.6f}")
    for ep, rid in (("pfs", "S3_pfs"), ("os", "S3_os")):
        s, r = devsum[ep], d.loc[rid]
        check(f"{rid}: development CV summary equals the table row",
              abs(s["c_C0"] - r.c_index_baseline) < 1e-12 and abs(s["c_C0_RNA"] - r.c_index_plus_rna) < 1e-12 and abs(s["delta_c"] - r.delta_c) < 1e-12
              and int(s["n_patients"]) == 473 == int(r.n_patients) and int(s["n_events"]) == int(r.n_events))
        check(f"{rid}: ΔC is the mean of the five repeat ΔC values (pooled-OOF C averaged over repeats)", abs(rep[ep].delta_c.mean() - s["delta_c"]) < 1e-9, f"{rep[ep].delta_c.mean():.6f}", f"{s['delta_c']:.6f}")
    check("Protocol hash recorded in the held-out report is present", isinstance(rpt.get("protocol_sha256"), str) and len(rpt["protocol_sha256"]) == 64)
    check("held-out denominators 201 patients (PFS 149 events, OS 82 deaths)", (int(d.loc["A1", "n_patients"]), int(d.loc["A1", "n_events"]), int(d.loc["A2", "n_events"])) == (201, 149, 82))

    def row(rid, ep, lab, perm_note="not performed", ibs=True, rng=None):
        r = d.loc[rid]
        if not bool(r.evaluated):
            return [ep, lab, "—", "not evaluated", "—", "—", perm_note, "—", "—"]
        pm = f"{r.permutation_p_one_sided:.4f}" if pd.notna(r.permutation_p_one_sided) else perm_note
        ib = (f"{r.ibs_baseline:.3f} → {r.ibs_plus_rna:.3f} (Δ {f(r.ibs_plus_rna - r.ibs_baseline)}; no CI)" if ibs and pd.notna(r.ibs_baseline) else "not computed")
        lam = f"{r.lambda_baseline:g} / {r.lambda_plus_rna:g}" if pd.notna(r.lambda_plus_rna) else "—"
        return [ep, lab, f"{int(r.n_patients)} / {int(r.n_events)}", f"{r.c_index_baseline:.3f} → {r.c_index_plus_rna:.3f}",
                f"{f(r.delta_c)} ({f(r.ci_lower, 3, False)} to {f(r.ci_upper, 3, False)})", rng or "—", pm, ib, lam]

    def rng_of(ep):
        x = rep[ep].delta_c
        return f"{f(x.min(), 3, False)} to {f(x.max(), 3, False)}"

    hdr = ["Endpoint", "Analysis", "Patients / events", "C-index: baseline → + RNA", "ΔC (95% CI)", "ΔC range over 5 CV repeats", "Permutation p (one-sided)", "IBS: baseline → + RNA", "Frozen λ: baseline / + RNA"]
    held = [row("A1", "PFS", "Primary (held-out)"), row("S2_pfs", "PFS", "Expanded clinical baseline"), row("S1", "PFS", "RNA day −30 to 0 only"),
            row("S4_val", "PFS", "Validation patients only (descriptive)"), row("S4_test", "PFS", "Test patients only (descriptive)"),
            row("A2", "OS", "Primary (held-out)"), row("S2_os", "OS", "Expanded clinical baseline"), row("S1_os", "OS", "RNA day −30 to 0 only")]
    ref = [row("S3_pfs", "PFS", "Development-only 5×5 CV (train patients)", rng=rng_of("pfs")), row("S3_os", "OS", "Development-only 5×5 CV (train patients)", rng=rng_of("os"))]
    ibs_dev = {ep: (devsum[ep]["ibs_C0"], devsum[ep]["ibs_C0_RNA"]) for ep in devsum}
    for rr, ep in zip(ref, ("pfs", "os")):
        rr[7] = f"{ibs_dev[ep][0]:.3f} → {ibs_dev[ep][1]:.3f} (Δ {f(ibs_dev[ep][1] - ibs_dev[ep][0])}; no CI)"
    pooled = [row("P_pfs", "PFS", "Earlier pooled 5×5 CV (all 674 patients)", ibs=False), row("P_os", "OS", "Earlier pooled 5×5 CV (all 674 patients)", ibs=False)]
    # The pooled IBS and repeat ranges exist only in the superseded Table S5; its inputs are unavailable on this machine, so they are
    # transcribed verbatim (marked †) after checking that the legacy row agrees with the verified pooled C-index and delta C.
    legacy = {}
    lp = OUT / "Table_S5_survival.md"
    if lp.exists():
        for ln in lp.read_text(encoding="utf-8").splitlines():
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) == 8 and c[1] == "Primary" and c[0] in ("PFS", "OS"):
                legacy[c[0]] = c
    for rr, ep, rid in zip(pooled, ("PFS", "OS"), ("P_pfs", "P_os")):
        r = d.loc[rid]
        if ep in legacy:
            c = legacy[ep]
            ok = (c[2] == f"{int(r.n_patients)} / {int(r.n_events)}" and c[3] == f"{r.c_index_baseline:.3f} → {r.c_index_plus_rna:.3f}"
                  and c[4].startswith(f"{f(r.delta_c)} (") and c[4].endswith(f"{f(r.ci_upper, 3, False)})"))
            check(f"legacy Table S5 pooled {ep} row agrees with the verified pooled C-index, delta C and CI", ok, c[2:5], "")
            rr[5], rr[7] = f"{c[5]} †", f"{c[7]} †"
        else:
            check(f"legacy Table S5 pooled {ep} row available for transcription", False, "Table_S5_survival.md missing or row not found", "present")
            rr[5], rr[7] = "not available", "not available"
    check("held-out IBS is lower with RNA for both endpoints (descriptive)", d.loc["A1", "ibs_plus_rna"] < d.loc["A1", "ibs_baseline"] and d.loc["A2", "ibs_plus_rna"] < d.loc["A2", "ibs_baseline"])

    held_tbl, ref_tbl, pool_tbl = md_table(held, hdr), md_table(ref, hdr), md_table(pooled, hdr)
    p1 = d.loc["A1"]
    md = ["# Supplementary Table S5. Survival discrimination and overall prediction error with and without pre-treatment RNA: held-out evaluation (Figure 5), train-only redesign", "",
          "Survival cohort: patients with one RNA sample collected on or before the start of first treatment (CoMMpass index date); official MMRF PFS and OS. Ridge-penalized Cox models "
          "(Breslow); clinical baseline = age at index, gender, ISS, exact first-line regimen; expanded baseline adds LDH, β2-microglobulin, albumin, creatinine, haemoglobin, calcium and "
          "bone-marrow plasma-cell %; + RNA = 50 Hallmark pathway scores. **Models were fitted on the 473 train patients only** (penalties chosen by 5-fold inner CV repeated 5 times within "
          "those patients, then frozen) and **scored once on the 201 held-out validation + test patients** (101 + 100; PFS 149 events, OS 82 deaths). Harrell's C-index (higher is better); "
          "ΔC 95% CIs: patient bootstrap (2,000 resamples). Integrated Brier score (IBS; IPCW, 365–1,095 days; lower is better): point estimates only. Pre-specified PFS criterion: ΔC > 0 AND CI lower "
          "bound > 0 AND one-sided permutation p ≤ 0.025 (RNA vectors permuted among held-out patients, B = 10,000). Post hoc and exploratory: the earlier pooled results had been seen "
          "before the redesign, so this is not independent or external validation.", "",
          "## Held-out evaluation (validation + test patients)", "", held_tbl, "",
          f"**The pre-specified PFS criterion was not met**: ΔC = {f(p1.delta_c)} with a 95% CI that includes 0 ({f(p1.ci_lower, 3, False)} to {f(p1.ci_upper, 3, False)}), although the permutation p was "
          f"{p1.permutation_p_one_sided:.4f} (0 of 10,000 permuted statistics reached the observed ΔC). The permutation null is centred below 0 because permuted RNA adds noise to the frozen model, so the "
          "permutation result tests association between RNA and outcome, not ΔC > 0. The OS point estimate is also small with a CI that includes 0. Neither result is evidence of no effect. "
          "The validation-only and test-only rows are descriptive partitions of the 201 patients and are imprecise; they were not tested against each other. The OS RNA-timing analysis was "
          "specified for PFS only and was not run.", "",
          "## Development-only cross-validation reference (train patients; not an evaluation)", "",
          "5 × 5 patient-level CV within the 473 train patients (pooled out-of-fold C per repeat, averaged over repeats), shown for context and to gauge optimism. It cannot replace the held-out evaluation.", "", ref_tbl, "",
          "## Earlier pooled estimate (superseded; context only)", "",
          "5 × 5 CV over all 674 patients, in which the validation and test patients also trained the fold models. These values are not leakage-free and are not comparable with the held-out rows.", "", pool_tbl, "",
          "† Integrated Brier score and repeat range transcribed verbatim from the superseded Table S5 (`Table_S5_survival.md`); the pooled "
          "out-of-fold prediction files it was built from are not available on this machine (see `figures/LEGACY_MISSING_INPUTS.md`), so these two values "
          "cannot be re-derived or independently checked here. The pooled C-index, ΔC, CI and permutation p come from the verified extraction.", "",
          "No CIs exist for absolute C-index or IBS values. Rows differ in cohort or comparator and were not compared statistically. Sources: `figures/data/Fig45_trainonly/` (built by "
          "`figures/fig45_trainonly_extract_data.py`), cross-checked against `artifacts/results_clean_rerun/survival_trainonly/` and the frozen protocol `explainability/SURVIVAL_TRAINONLY_PROTOCOL.md`. "
          "The earlier Table S5 (`Table_S5_survival.md`; pooled 5×5 CV) is superseded."]
    txt = "\n".join(md)
    check("no patient identifiers in the output", "mmrf_" not in txt and "public_id" not in txt)
    long_df = d.reset_index()
    long_df.to_csv(OUT / "Table_S5_survival_trainonly.csv", index=False)
    (OUT / "Table_S5_survival_trainonly.md").write_text(txt, encoding="utf-8")
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "table_s5_trainonly_verification.csv", index=False)
    failed = ck[~ck.passed]
    print(f"{len(ck)} checks: {int(ck.passed.sum())} passed, {len(failed)} failed")
    if len(failed):
        print(failed.to_string(index=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
