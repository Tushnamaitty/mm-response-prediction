"""
Consistency check: numbers quoted in the train-only Figure 1 (E), 4A-C, 5 and 6 captions (final numbering; earlier working labels 3A-C, 4, 5) and in the S1/S5 tables against the
verified extracted data (read-only). Run from the repository root, after the *_trainonly_extract_data.py scripts:
    python figures/check_trainonly_captions.py
Exit code 1 if any check fails. Also flags stale pooled-CV claims in the captions.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

CAP = Path("figures/FIGURE_CAPTIONS_DRAFT.md").read_text(encoding="utf-8").replace("\r\n", "\n")
F3, F45 = Path("figures/data/Fig3ABC_trainonly"), Path("figures/data/Fig45_trainonly")
M = "−"
fails, n = [], 0


def section(start, end):
    i = CAP.index(start)
    j = CAP.index(end, i + len(start))
    return " ".join(CAP[i:j].split())


# variable names keep the earlier working labels (S3 = RNA contribution, S4 = survival, S5 = pathway coefficients)
S3 = section("### Figure 4. ", "### Figure 5. ")
S4 = section("### Figure 5. ", "### Figure 6. ")
S5 = section("### Figure 6. ", "## Supplementary figures")
S1E = section("**(E)** Downstream analyses", "---")


def fx(x, nd, sign=True):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", M)


def has(sec, text, label):
    global n
    n += 1
    if " ".join(text.split()) not in sec:
        fails.append(f"{label}: '{text}' not found")


def absent(sec, pat, label):
    global n
    n += 1
    if re.search(pat, sec):
        fails.append(f"{label}: stale text matching /{pat}/ present")


# ---- Figure 4 vs fig4A/4B data ----
d = pd.read_csv(F45 / "fig4B_trainonly_delta_c.csv").set_index("row_id")
for rid in ("A1", "A2", "S2_pfs", "S2_os", "S1", "S3_pfs", "S3_os", "P_pfs", "P_os"):
    r = d.loc[rid]
    has(S4, f"ΔC {fx(r.delta_c, 3)}" if rid in ("A1", "A2") else f"{fx(r.delta_c, 3)}", f"Fig4 {rid} dC")
    has(S4, f"{fx(r.ci_lower, 3)} to {fx(r.ci_upper, 3)}" if r.ci_lower < 0 else f"{fx(r.ci_lower, 3)} to {fx(r.ci_upper, 3)}", f"Fig4 {rid} CI")
for rid in ("A1", "A2", "S2_pfs", "S2_os", "S1"):
    r = d.loc[rid]
    has(S4, f"{r.c_index_baseline:.3f} → {r.c_index_plus_rna:.3f}", f"Fig4 {rid} C")
for rid, nm in (("A1", "PFS"), ("A2", "OS")):
    r = d.loc[rid]
    has(S4, f"{r.ibs_baseline:.3f} → {r.ibs_plus_rna:.3f}", f"Fig4 {rid} IBS")
has(S4, "(p = 0.0001, the smallest attainable)", "Fig4 permutation p")
assert abs(d.loc["A1", "permutation_p_one_sided"] - 1 / 10001) < 1e-12
for rid, (np_, ne) in {"A1": (201, 149), "A2": (201, 82), "S1": (186, 137), "S3_pfs": (473, 310), "S3_os": (473, 165)}.items():
    r = d.loc[rid]
    assert (int(r.n_patients), int(r.n_events)) == (np_, ne), rid
has(S4, "186 held-out patients, 137 PFS events", "Fig4 S1 denominators")
has(S4, "101 + 100; 149 PFS events, 82 deaths", "Fig4 held-out denominators")
has(S4, "(473 train patients; 310 PFS events, 165 deaths)", "Fig4 development denominators")
part = pd.read_csv(F45 / "fig4A_trainonly_partition_descriptive.csv").set_index("partition")
has(S4, f"validation-only ({part.loc['val', 'c_C0']:.3f} → {part.loc['val', 'c_C0_RNA']:.3f}; ΔC {fx(part.loc['val', 'delta_c'], 3)})", "Fig4 val-only")
has(S4, f"test-only ({part.loc['test', 'c_C0']:.3f} → {part.loc['test', 'c_C0_RNA']:.3f}; ΔC {fx(part.loc['test', 'delta_c'], 3)})", "Fig4 test-only")
absent(S4, r"0\.601 → 0\.642 \(ΔC", "Fig4 pooled-as-primary")
absent(S4, r"624 patients|200 permutations|within each of 5 repeats of patient-level", "Fig4 stale text")

# ---- Figure 5 vs coefficient data ----
st = json.load(open(F45 / "fig5_trainonly_summary_stats.json"))
a = pd.read_csv(F45 / "fig5A_trainonly_coefficients.csv")
b = pd.read_csv(F45 / "fig5B_trainonly_concordant.csv")
dc = st["direction_counts"]
has(S5, f"(PFS: {dc['pfs']['consistently_positive']} positive, {dc['pfs']['consistently_negative']} negative, {dc['pfs']['mixed']} mixed; "
        f"OS: {dc['os']['consistently_positive']}, {dc['os']['consistently_negative']}, {dc['os']['mixed']})", "Fig5 direction counts")
has(S5, f"Twelve pathways were concordant: {st['n_concordant_higher_hazard']} associated with higher hazard and {st['n_concordant_lower_hazard']} ", "Fig5 split")
has(S5, f"Pearson r = {st['pearson_r_mean_beta_pfs_vs_os']:.2f}", "Fig5 r")
lo_names = sorted(a[a.concordance_group == "lower hazard in both"].pathway_display_name)
assert lo_names == ["IL-6/JAK/STAT3 Signaling", "Inflammatory Response", "Notch Signaling"], lo_names
has(S5, "(IL-6/JAK/STAT3 Signaling, Inflammatory Response, Notch Signaling)", "Fig5 lower-hazard names")
hp, ho = b[b.endpoint == "PFS"].HR_of_mean_beta, b[b.endpoint == "OS"].HR_of_mean_beta
has(S5, f"{hp.min():.2f}–{hp.max():.2f} for PFS and {ho.min():.2f}–{ho.max():.2f} for OS", "Fig5 HR range")
fam = a[a.figure3D_family_member].set_index("pathway_display_name").concordant_both_endpoints.to_dict()
assert sum(fam.values()) == 4 and not fam["Spermatogenesis"]
has(S5, "whereas Spermatogenesis is not concordant", "Fig5 family statement")
absent(S5, r"All five pathways of the Figure [34]D composite are among|19 concordant pathways \(|Nineteen pathways were concordant", "Fig5 stale text")

# ---- Figure 3A-C vs data ----
o = pd.read_csv(F3 / "fig3A_trainonly_overall.csv").set_index("row_id")
for rid, nd in (("direct_binary_auprc", 4), ("direct_multiclass_rps", 5), ("devcv_direct_binary_auprc", 4), ("addon_multiclass_rps", 5)):
    r = o.loc[rid]
    has(S3, f"{fx(r.delta, nd)} (", f"Fig3A {rid} delta")
    has(S3, f"{fx(r.ci_lower, nd)} to {fx(r.ci_upper, nd)}", f"Fig3A {rid} CI")
r = o.loc["devcv_direct_multiclass_rps"]
has(S3, f"{fx(r.delta, 4)} ({fx(r.ci_lower, 4)} to {fx(r.ci_upper, 4)})", "Fig3A devcv rps")
has(S3, f"one-sided permutation p of {o.loc['addon_multiclass_rps', 'p_value']:.4f}", "Fig3A add-on p")
has(S3, "(binary 2,405 visit pairs; six-class 2,689 pairs)", "Fig3 pair counts")
assert (int(o.loc["direct_binary_auprc", "n_visit_pairs"]), int(o.loc["direct_multiclass_rps", "n_visit_pairs"]), int(o.loc["direct_binary_auprc", "n_patients"])) == (2405, 2689, 213)
u = pd.read_csv(F3 / "fig3B_trainonly_uncertainty.csv").set_index(["task", "group"])
for t, nd in (("binary", 4), ("multiclass", 5)):
    q4 = u.loc[(t, "Q4_high_uncertainty")]
    ia = u.loc[(t, "Q4_minus_Q1_Q3")]
    has(S3, f"{fx(q4.delta, nd)} ({fx(q4.ci_lower, nd)} to {fx(q4.ci_upper, nd)})", f"Fig3B {t} Q4")
    has(S3, f"{fx(ia.delta, nd)} ({fx(ia.ci_lower, nd)} to {fx(ia.ci_upper, nd)}; Holm p = {ia.adjusted_p:.2f})", f"Fig3B {t} interaction")
sl = json.load(open(F3 / "fig3ABC_trainonly_definitions.json"))["figure_3B"]["continuous_slope_binary"]
has(S3, f"{fx(sl['benefit_at_90th_minus_10th_percentile'], 4)} ({fx(sl['ci_lo'], 4)} to {fx(sl['ci_hi'], 4)})", "Fig3B slope")
inter = pd.read_csv(F3 / "tableS1_trainonly_interactions.csv")
for contrast in ("worsening_minus_stable", "improving_minus_stable"):
    x = inter[(inter.task == "multiclass") & (inter.metric == "rps") & (inter.contrast == contrast)].iloc[0]
    has(S3, f"{fx(x.interaction_a_minus_b, 4)}, {fx(x.ci_lo, 4)} to {fx(x.ci_hi, 4)}", f"Fig3C {contrast}")
    has(S3, f"BH p = {x.adjusted_p:.3f}", f"Fig3C {contrast} BH")
strata = pd.read_csv(F3 / "fig3C_trainonly_strata.csv")
assert not strata[strata.metric.isin(["auprc", "rps"])].significant_after_adjustment.any()
has(S3, "No stratum changed significantly on its primary metric after multiplicity adjustment", "Fig3C no-significance claim")
absent(S3, r"6,939|7,724|permutation p = 0\.005|Holm p = 0\.045|were small \(ΔRPS \+0\.95", "Fig3 stale pooled-CV text")

# ---- Figure 1 caption panel E ----
for t in ("716 train", "213 validation and test patients with RNA", "371 train patients", "160 validation and test", "473 train patients", "201 validation and test patients (101 and 100)",
          "439 US patients", "96 such patients", "236 US patients", "116 Spain and Canada"):
    has(S1E, t, f"Fig1E '{t}'")
absent(S1E, r"fit within their own cohorts", "Fig1E stale text")

# ---- Table S5 ----
s5 = Path("figures/supplementary_tables/Table_S5_survival_trainonly.md").read_text(encoding="utf-8")
has(" ".join(s5.split()), f"{fx(d.loc['A1', 'delta_c'], 3)} ({fx(d.loc['A1', 'ci_lower'], 3)} to {fx(d.loc['A1', 'ci_upper'], 3, False)})", "S5 A1")
has(" ".join(s5.split()), "The pre-specified PFS criterion was not met", "S5 criterion")

# ---- validation-based threshold / calibration decisions are no longer asserted anywhere in the reported materials ----
STALE = re.compile(r"validation patients\s+set\s+the\s+binary|choose\s+calibration|selected\s+on\s+the\s+validation\s+set|selected\s+on\s+validation\)|"
                   r"Binary decision threshold;\s*calibration comparison|Binary thresholds and calibration use\s+validation|"
                   r"still selected on validation|decision_threshold_from_validation")
for f in ["figures/FIGURE_CAPTIONS_DRAFT.md", "figures/supplementary_tables/Table_S2_rna_available_calibration.md",
          "figures/supplementary_tables/Table_S6_full_figure2_performance.md", "figures/supplementary_tables/Table_S5_survival_trainonly.md",
          "figures/supplementary_tables/build_supplementary_tables.py", "figures/publication_redesign/fig1_new/figure1_new_plot.py",
          "figures/publication_redesign/FigS_calibration/figureS1A_plot.py", "figures/figS_calibration_extract_data.py"]:
    t = " ".join(Path(f).read_text(encoding="utf-8").split())
    absent(t, STALE.pattern.replace(r"\s+", r"\s+"), f"stale validation-decision claim in {f}")
for f, s in (("figures/FIGURE_CAPTIONS_DRAFT.md", "no decision threshold or recalibration enters any reported metric"),
             ("figures/FIGURE_CAPTIONS_DRAFT.md", "not blind to validation data"),
             ("figures/publication_redesign/fig1_new/figure1_new_plot.py", "Descriptive only: no tuning,"),
             ("figures/SUPERSEDED_ANALYSES.md", "Honest disclosure")):
    has(" ".join(Path(f).read_text(encoding="utf-8").split()), s, f"disclosure present in {f}")
# S5 legacy transcription is marked and the missing-input register exists
s5t = Path("figures/supplementary_tables/Table_S5_survival_trainonly.md").read_text(encoding="utf-8")
has(" ".join(s5t.split()), "0.205 → 0.185 (Δ −0.020; no CI) †", "S5 legacy pooled IBS marked †")
absent(" ".join(s5t.split()), r"see earlier Table S5", "S5 dangling 'see earlier Table S5'")
for f in ("figures/LEGACY_MISSING_INPUTS.md", "figures/FIGURE_NUMBERING_AUDIT.md", "figures/FIGURE_NUMBERING.md"):
    has(Path(f).read_text(encoding="utf-8")[:200], "#", f"{f} exists")

# ---- no unsupported "best algorithm" / "reporting reference" claim in the manuscript-facing text ----
BEST = r"reporting[ -]reference|\bbest (?:model|algorithm|performing|binary|multiclass)|best-performing|(?:selected|chosen) as the best"
for f in ("figures/FIGURE_CAPTIONS_DRAFT.md", "figures/supplementary_tables/Table_S6_full_figure2_performance.md",
          "figures/supplementary_tables/build_table1_and_S6.py", "figures/data/FigS_calibration/figS1_definitions.json"):
    txt = " ".join(Path(f).read_text(encoding="utf-8").split())
    # negated wording ("not a ... best model", "no algorithm ... best") is allowed; positive claims are not
    hits = [m.group(0) for m in re.finditer(BEST, txt, flags=re.I)
            if not re.search(r"(?:\bnot|\bno|does not imply|not because)[^.]{0,120}$", txt[max(0, m.start() - 130):m.start()], flags=re.I)
            and txt[max(0, m.start() - 1):m.start()] != '"']          # a quoted mention of the removed phrase is allowed
    n += 1
    if hits:
        fails.append(f"unsupported best/reporting-reference wording in {f}: {hits}")
has(" ".join(CAP.split()), "No algorithm was prespecified as primary or selected as best, and the figure does not rank them.", "Fig3 no-ranking sentence")

# ---- XGBoost-only analyses are described as XGBoost-specific (captions and tables) ----
CAPN = " ".join(CAP.split())
for s in ("These analyses are specific to XGBoost: it was used alone, no a priori rationale for choosing it is recorded",
          "The assessment is specific to XGBoost (no a priori rationale for using it alone is recorded",
          "the check is specific to this algorithm and layer, no a priori rationale for choosing them is recorded",
          "the attributions describe these XGBoost models, not Logistic Regression or LightGBM"):
    has(CAPN, s, "XGBoost-specific caption wording")
T = "figures/supplementary_tables/"
for f in ("Table_S1_rna_subgroups_trainonly.md", "Table_S2_rna_available_calibration.md", "Table_S3_shap_attribution.md", "Table_S4_geographic_robustness_origtrain.md"):
    has(" ".join(Path(T + f).read_text(encoding="utf-8").split()), "a priori rationale", f"XGBoost-specific note in {f}")

# ---- Figure 3 panel C: caption statements vs the plotted file and the independent recomputation ----
pd_ = pd.read_csv("figures/data/Fig2/fig2_paired_deltas_full_test.csv").set_index(["task", "algorithm", "comparison"])
for task, comp, nd, label in (("binary", "B - A", 3, "+0.022"), ("multiclass", "B - A", 3, "−0.014"), ("binary", "D - C", 3, "+0.008"), ("multiclass", "D - C", 3, "−0.004")):
    v = pd_.loc[(task, "xgboost", comp), "observed_diff"]
    has(CAPN, label, f"Fig3C XGBoost {task} {comp} caption value")
    n += 1
    if fx(v, nd) != label:
        fails.append(f"Fig3C {task} {comp}: plotted {fx(v, nd)} != caption {label}")
dc_rows = pd_.xs("D - C", level="comparison")
n += 1
if not ((dc_rows.ci_lower <= 0) & (dc_rows.ci_upper >= 0)).all():
    fails.append("Fig3C: a D - C interval does not include zero, contradicting the caption")
chk = Path("figures/data/Fig2/fig2_paired_deltas_independent_check.csv")      # gitignored; written by figures/verify_fig3c_paired_cis.py
if chk.exists():
    ck = pd.read_csv(chk)
    n += 3
    if len(ck) != 24 or max(ck.abs_diff_lower.max(), ck.abs_diff_upper.max()) > 1e-9:
        fails.append("Fig3C independent recomputation does not reproduce the plotted interval limits in 24 of 24 rows")
    move = max((ck.recomputed_seed42_ci_lower - ck.other_seed_ci_lower_min).abs().max(), (ck.recomputed_seed42_ci_lower - ck.other_seed_ci_lower_max).abs().max(),
               (ck.recomputed_seed42_ci_upper - ck.other_seed_ci_upper_min).abs().max(), (ck.recomputed_seed42_ci_upper - ck.other_seed_ci_upper_max).abs().max())
    if move > 0.0053:
        fails.append(f"Fig3C other-seed movement {move:.4f} exceeds the 0.0053 stated in the caption")
    flips = [(r.task, r.algorithm, r.comparison) for r in ck.itertuples()
             if (r.other_seed_ci_lower_min <= 0 <= r.other_seed_ci_upper_min) != r.ci_includes_zero_recomputed
             or (r.other_seed_ci_lower_max <= 0 <= r.other_seed_ci_upper_max) != r.ci_includes_zero_recomputed]
    if sorted(flips) != [("binary", "xgboost", "B - A"), ("binary", "xgboost", "D - A"), ("multiclass", "xgboost", "C - B")]:
        fails.append(f"Fig3C seed-sensitive rows differ from the caption: {flips}")
else:
    print("note: independent Fig3C check file absent (run figures/verify_fig3c_paired_cis.py); its caption statements were not re-checked", file=sys.stderr)

# Figure 4C stratum sizes quoted in the caption and in the unresolved items (42-207 overall; binary 42-206)
_s = strata[strata.stratum != "all"]
_rng = {tk: (int(g.n_patients.min()), int(g.n_patients.max())) for tk, g in _s.groupby("task")}
_all = (min(v[0] for v in _rng.values()), max(v[1] for v in _rng.values()))
for _lab in (f"Strata contained {_all[0]}–{_all[1]} patients (binary {_rng['binary'][0]}–{_rng['binary'][1]}; six-class {_rng['multiclass'][0]}–{_rng['multiclass'][1]})",
             f"leave {_all[0]}–{_all[1]} patients per Figure 4C stratum (binary {_rng['binary'][0]}–{_rng['binary'][1]})"):
    n += 1
    if " ".join(_lab.split()) not in CAPN:
        fails.append(f"Fig4C stratum range not in the caption as computed from the data: '{_lab}'")

print(f"{n} checks, {len(fails)} failed")
for f in fails:
    print("FAIL:", f, file=sys.stderr)
sys.exit(1 if fails else 0)
