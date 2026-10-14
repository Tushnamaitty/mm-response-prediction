"""
Read-only verification of the Supplementary Figure S3 / Table S3 inputs (nothing is written; no model is run, no SHAP is computed).

Run from the repository root:
    python figures/verify_figS3_inputs.py

Checks
  1. SHA-256 of the S3 data files in figures/data/FigS_shap/ against figS3_ext_provenance.json and
     figures/supplementary_tables/supplementary_tables_provenance.json.
  2. Table S3 (all share/CI cells and the per-class table) against figS3A_ext_component_shares*.csv.
  3. Net-family and total-importance shares (all test pairs) recomputed from source-level SHAP: binary from the restored
     figS3_source_shap_binary_test.csv.gz (patient-level, gitignored; local only), six-class from the older branch artifact
     artifacts/results_clean_rerun/xai/shap_artifacts/multiclass_d_xgboost_* (exact TreeSHAP of the locked model, float32), compared with the CSV.
  4. Figure S3B whiskers (95% patient-bootstrap, 1,000 resamples, seed 42, repository bootstrap function) recomputed the same way and
     compared with figS3B_top15_source_features.csv.
  5. Figure S3A / Table S3 shares and 95% CIs (net family magnitude and total importance; all test pairs, patients weighted equally and
     RNA-available pairs) recomputed for both tasks with the same bootstrap and compared with the CSV (binary to 1e-9 percentage points,
     six-class to 1e-3 because the older six-class artifact is float32). Per-class six-class CIs are not recomputed here.
Exit code 1 if any check fails.
"""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("figures/data/FigS_shap")
fails, n = [], 0


def check(ok, msg):
    global n
    n += 1
    if not ok:
        fails.append(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- 1. hashes ---------------------------------------------------------------------------------------------------------------------------
ext = json.load(open(D / "figS3_ext_provenance.json", encoding="utf-8"))
for sec in ("protected_earlier_files_sha256", "outputs_sha256"):
    for name, h in ext[sec].items():
        if (D / name).exists():
            check(sha(D / name) == h, f"{name}: SHA-256 differs from figS3_ext_provenance.json")
for k, h in ext["inputs_sha256"].items():
    check(Path(k).exists() and sha(k) == h, f"input {k}: missing or hash differs")
tp = json.load(open("figures/supplementary_tables/supplementary_tables_provenance.json", encoding="utf-8"))["inputs_sha256"]
for k, h in tp.items():
    kk = k.replace("\\", "/")
    if "FigS_shap" in kk:
        check(Path(kk).exists() and sha(kk) == h, f"{kk}: missing or hash differs from supplementary_tables_provenance.json")

# ---- 2. Table S3 against the CSVs --------------------------------------------------------------------------------------------------------
x = pd.read_csv(D / "figS3A_ext_component_shares.csv")
pc = pd.read_csv(D / "figS3A_ext_component_shares_per_class.csv")
md = open("figures/supplementary_tables/Table_S3_shap_attribution.md", encoding="utf-8").read()
comp = {"Current IMWG response": 1, "Other current clinical state": 2, "Longitudinal history": 3, "Treatment context": 4,
        "RNA pathway scores": 5, "RNA timing / availability": 6}
want = {"Net family magnitude (Figure S3A) — all test pairs": ("net_family_magnitude", "primary"),
        "Net family magnitude (Figure S3A) — patients weighted equally": ("net_family_magnitude", "patient_equal_weighted"),
        "Net family magnitude (Figure S3A) — RNA-available pairs only": ("net_family_magnitude", "rna_available_rows"),
        "Total source-feature importance (not Figure S3A) — all test pairs": ("summed_source_feature_importance", "primary"),
        "Total source-feature importance (not Figure S3A) — patients weighted equally": ("summed_source_feature_importance", "patient_equal_weighted"),
        "Total source-feature importance (not Figure S3A) — RNA-available pairs only": ("summed_source_feature_importance", "rna_available_rows")}
cells = 0
for sec in re.split(r"\n## ", md):
    title = sec.split("\n")[0]
    if title in want:
        meas, an = want[title]
        for line in sec.splitlines():
            m = re.match(r"\| (.+?) \| [A-D] \| ([\d.]+)% \(([\d.]+)–([\d.]+)\) \| ([\d.]+)% \(([\d.]+)–([\d.]+)\) \|", line)
            if m:
                for task, vals in (("binary", m.groups()[1:4]), ("multiclass", m.groups()[4:7])):
                    r = x[(x.task == task) & (x.measure == meas) & (x.analysis == an) & (x.order == comp[m.group(1)])]
                    cells += 1
                    ok = len(r) == 1 and all(abs(a - float(b)) <= 0.051 for a, b in zip(
                        (r.iloc[0].share_pct_display, r.iloc[0].share_pct_ci_lower, r.iloc[0].share_pct_ci_upper), vals))
                    check(ok, f"Table S3 cell differs from the CSV: {title[:40]} / {task} / {m.group(1)}")
check(cells == 72, f"expected 72 Table S3 share cells, found {cells}")
cls = ["progressive_disease", "stable_disease", "partial_response", "very_good_partial_response", "complete_response", "stringent_complete_response"]
pcs = [s for s in re.split(r"\n## ", md) if s.startswith("Six-class net family magnitude share by class")][0]
k2 = 0
for line in pcs.splitlines():
    m = re.match(r"\| (.+?) \| (.*) \|$", line)
    if m and m.group(1) in comp:
        for c, v in zip(cls, [float(t.strip().rstrip("%")) for t in m.group(2).split("|")]):
            r = pc[(pc.task == "multiclass") & (pc["class"] == c) & (pc.measure == "net_family_magnitude") & (pc.analysis == "primary") & (pc.order == comp[m.group(1)])]
            k2 += 1
            check(len(r) == 1 and abs(r.iloc[0].share_pct_display - v) <= 0.051, f"per-class cell differs: {c} / {m.group(1)}")
check(k2 == 36, f"expected 36 per-class cells, found {k2}")

# ---- 3-4. recomputation from source-level SHAP -------------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("regen", "modeling/regenerate_main_model_cis_clean_rerun.py")
regen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regen)
fs = json.load(open("data_pipeline/model_feature_sets.json", encoding="utf-8"))
A, B, C, Dd = [set(fs["model_" + k]) for k in "abcd"]
fam = {c: "other" for c in A}
fam["vt_disease_response"] = "resp"
fam.update({c: "hist" for c in B - A})
fam.update({c: "treat" for c in C - B})
fam.update({c: ("rnaT" if c == "days_since_rna_sample" else "path") for c in Dd - C})
fam["rna_available"] = "rnaT"
groups = ["resp", "other", "hist", "treat", "path", "rnaT"]
t15 = pd.read_csv(D / "figS3B_top15_source_features.csv")

g = pd.read_csv(D / "figS3_source_shap_binary_test.csv.gz")
cols_b = [c for c in g.columns if c not in ("pair_id", "public_id")]
SB, pid_b = g[cols_b].values, g.public_id.values
b = "artifacts/results_clean_rerun/xai/shap_artifacts/multiclass_d_xgboost_"
S = np.load(b + "shap_values.npy")
mp = pd.read_csv("artifacts/results_clean_rerun/xai/rollup_mapping_multiclass_d_xgboost.csv")
src = mp.original_column.values
cols_m = list(dict.fromkeys(src))
SS = np.stack([S[:, :, np.where(src == u)[0]].sum(axis=2) for u in cols_m], axis=2)          # (class, pair, source feature)
_pm = pd.read_csv("artifacts/results_clean_rerun/multiclass_d_xgboost_predictions.csv")
pid_m = _pm.public_id.values
_rna = pd.read_csv("data/clinical/visit_pairs_with_rna.csv", low_memory=False).set_index("pair_id")["Adipogenesis"].notna()   # RNA sample at or before V(t)
for task, signed, cols, pid, tol, pair_ids in (("binary", SB[None], cols_b, pid_b, 1e-6, g.pair_id.values), ("multiclass", SS, cols_m, pid_m, 1e-3, _pm.pair_id.values)):
    miss = [c for c in cols if c not in fam]
    check(not miss, f"{task}: unmapped source features {miss}")
    idx = {k: [i for i, c in enumerate(cols) if fam[c] == k] for k in groups}
    net = np.stack([signed[:, :, idx[k]].sum(axis=2) for k in groups], axis=2)                 # signed family sums
    v = np.abs(net).mean(axis=0).mean(axis=0)
    ma = np.abs(signed).mean(axis=0).mean(axis=0)
    tot = np.array([ma[idx[k]].sum() for k in groups])
    for meas, vec in (("net_family_magnitude", v), ("summed_source_feature_importance", tot)):
        r = x[(x.task == task) & (x.measure == meas) & (x.analysis == "primary")].sort_values("order")
        check(np.abs(100 * vec / vec.sum() - r.share_pct.values).max() < tol * 100, f"{task} {meas}: recomputed shares differ from the CSV")
    # ---- S3B whiskers
    A_abs = np.abs(signed).mean(axis=0)                                                      # class-averaged |source SHAP| per pair
    rs = regen.patient_bootstrap_indices(pid, n=1000, seed=42)
    boots = np.array([A_abs[r].mean(0) for r in rs])
    lo, hi, point = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0), A_abs.mean(0)
    s = t15[t15.task == task].sort_values("rank")
    check([cols[i] for i in np.argsort(-point)[:15]] == list(s.source_feature), f"{task}: S3B top-15 set/order differs")
    d = max(max(abs(lo[cols.index(r.source_feature)] - r.ci_lower), abs(hi[cols.index(r.source_feature)] - r.ci_upper)) for r in s.itertuples())
    p = max(abs(point[cols.index(r.source_feature)] - r.mean_abs_shap) for r in s.itertuples())
    print(f"{task}: S3B max |CI limit difference| {d:.2e}; max |point difference| {p:.2e}")
    # ---- Figure S3A / Table S3 shares and CIs for all three analyses
    A_net = np.abs(net).mean(axis=0)                                                              # (pair, family): class-averaged |signed family sum|
    A_tot = np.stack([np.abs(signed).mean(axis=0)[:, idx[k]].sum(axis=1) for k in groups], axis=1)   # (pair, family): family sum of class-averaged |source SHAP|
    codes = pd.factorize(pid)[0]
    w_all = 1 / np.bincount(codes)[codes]
    ra = _rna.reindex(pair_ids).values.astype(bool)

    def sh(M, rows, w=None):
        m = M[rows]
        v = m.mean(0) if w is None else (m * w[:, None]).sum(0) / w.sum()
        return 100 * v / v.sum()

    worst = 0.0
    for meas, M in (("net_family_magnitude", A_net), ("summed_source_feature_importance", A_tot)):
        for an in ("primary", "patient_equal_weighted", "rna_available_rows"):
            r = x[(x.task == task) & (x.measure == meas) & (x.analysis == an)].sort_values("order")
            if an == "primary":
                ex, bs = sh(M, np.arange(len(pid))), np.array([sh(M, rr) for rr in rs])
            elif an == "patient_equal_weighted":
                ex, bs = sh(M, np.arange(len(pid)), w_all), np.array([sh(M, rr, w_all[rr]) for rr in rs])
            else:                                                                                 # resample all patients, keep their RNA-available rows
                ex, bs = sh(M, np.where(ra)[0]), np.array([sh(M, rr[ra[rr]]) for rr in rs])
            lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
            dd = max(np.abs(ex - r.share_pct.values).max(), np.abs(lo - r.share_pct_ci_lower.values).max(), np.abs(hi - r.share_pct_ci_upper.values).max())
            worst = max(worst, dd)
            check(dd < (1e-9 if task == "binary" else 1e-3), f"{task} {meas} {an}: shares/CIs differ from the CSV by {dd:.2e} percentage points")
    print(f"{task}: S3A/Table S3 shares and CIs, worst difference {worst:.2e} percentage points")
    check(d < (1e-9 if task == "binary" else 1e-5), f"{task}: S3B whisker limits differ from the CSV by {d:.2e}")

print(f"{n} checks, {len(fails)} failed")
for f in fails:
    print("FAIL:", f, file=sys.stderr)
sys.exit(1 if fails else 0)
