"""
Check of the final figure numbering (Scheme A) in the manuscript-facing files (read-only).

Run from the repository root:
    python figures/check_figure_numbering.py

Scheme A: Figure 1 study overview; Figure 2 study design; Figure 3 performance; Figure 4 RNA contribution (A-C) and pathway composite (D);
Figure 5 survival; Figure 6 pathway coefficients; Supplementary Figures S1-S3 unchanged (mapping and residual working labels:
figures/FIGURE_NUMBERING.md). The checks (1) read the caption headings, (2) test context-specific cross-references in the captions, tables,
READMEs and documentation, (3) test the first docstring line and the drawn label strings of the live plot scripts, and (4) reject any main-figure number
above 6 (a double shift) or any panel that does not exist under the new numbering. Exit code 1 if any check fails.
"""

import re
import sys
from pathlib import Path

fails, n = [], 0


def rd(path):
    return " ".join(Path(path).read_text(encoding="utf-8").split())


def has(text, s, label):
    global n
    n += 1
    if " ".join(s.split()) not in text:
        fails.append(f"{label}: '{s}' not found")


def absent(text, pat, label):
    global n
    n += 1
    m = re.search(pat, text)
    if m:
        fails.append(f"{label}: stale text '{m.group(0)}' present")


CAP_PATH = "figures/FIGURE_CAPTIONS_DRAFT.md"
CAP = rd(CAP_PATH)
raw_cap = Path(CAP_PATH).read_text(encoding="utf-8").replace("\r\n", "\n")

# ---- 1. headings, in order -------------------------------------------------------------------------------------------------------------------
heads = re.findall(r"^### (Supplementary Figure S\d|Figure \d)\. (.*)$", raw_cap, flags=re.M)
expect = [("Figure 1", "Study overview"), ("Figure 2", "Study design"), ("Figure 3", "Next-visit response prediction"),
          ("Figure 4", "Exploratory analyses of the incremental contribution of RNA"), ("Figure 5", "Exploratory incremental prognostic value"),
          ("Figure 6", "Exploratory RNA pathway coefficient"), ("Supplementary Figure S1", "Calibration"),
          ("Supplementary Figure S2", "Internal geographic hold-out"), ("Supplementary Figure S3", "SHAP attribution")]
n += 1
if [h[0] for h in heads] != [e[0] for e in expect]:
    fails.append(f"caption headings {[h[0] for h in heads]} != {[e[0] for e in expect]}")
for (num, text), (_, start) in zip(heads, expect):
    n += 1
    if not text.startswith(start):
        fails.append(f"heading {num} does not start with '{start}': '{text[:60]}'")

# ---- 2. context-specific cross-references in the captions ---------------------------------------------------------------------------------
for s in ("Performance of Logistic Regression, LightGBM and XGBoost for each information layer (Figure 2B)",
          "RNA contribution (Figures 4A–C)", "Survival (Figures 5–6)", "The survival cohort (Figures 5 and 6)",
          "these results are not independent of Figures 3–4", "Coefficients come from the Figure 5 clinical baseline",
          "pathways of the Figure 4D family are ringed", "the five train-selected pathway scores of Figure 4D share 68.1%",
          "selected the Figure 4D family and overlap those of Figures 4 and 5", "agreement with the Figure 4D result is not independent evidence",
          "recomputed from the locked test predictions (`figures/verify_fig3c_paired_cis.py`)", "per Figure 4C stratum (binary 42–206) and 100–101 patients per partition in Figure 5",
          "(Figures 3 and S1, Tables S2 and S6 use threshold-free", "Figures 4A–C, 5 and 6 and Supplementary Tables S1 and S5",
          "Figure 1, Figure 4 (A–D), Figure 5, Figure 6 and Supplementary Figure S2 captions"):
    has(CAP, s, f"caption cross-reference '{s[:50]}'")
absent(CAP, r"Figure 1B|Figure 1 \(current|New Figure 1|becomes Figure 2|renumbering of the captions has not been done|Figure 3D|Figures? 3A–C, 4|Figures 4 and 5\)", "caption stale reference")

# ---- 3. tables, READMEs and documentation --------------------------------------------------------------------------------------------------
T = "figures/supplementary_tables/"
for f, s in ((T + "Table_S1_rna_subgroups_trainonly.md", "held-out evaluation (Figure 4C), train-only redesign"),
             (T + "Table_S5_survival_trainonly.md", "held-out evaluation (Figure 5), train-only redesign"),
             (T + "Table_S6_full_figure2_performance.md", "all 24 next-visit models (Figure 3)"),
             (T + "Table_S6_full_figure2_performance.md", "Information layers A–D as in Figure 2"),
             (T + "Table_S6_full_figure2_performance.md", "Paired layer-to-layer differences (Figure 3C)"),
             (T + "Table_1_cohort_characteristics.md", "used in Figures 5–6"),
             (T + "Table_1_cohort_characteristics.md", "cross-checked against Figure 2 data"),
             (T + "build_table_s1_trainonly.py", "(Figure 4C), train-only redesign"),
             (T + "build_table_s5_trainonly.py", "(Figure 5), train-only redesign"),
             (T + "build_table1_and_S6.py", "all 24 next-visit models (Figure 3)"),
             (T + "README.md", "155 Figure 3 test patients"),
             ("figures/publication_redesign/fig1_new/README.md", "(CV; Figure 6)"),
             ("figures/publication_redesign/fig1_new/README.md", "Scheme A"),
             ("figures/SUPERSEDED_ANALYSES.md", "pathway selection (Figure 4D), the RNA-contribution analyses (Figures 4A–C"),
             ("figures/SUPERSEDED_ANALYSES.md", "analyses (Figures 5–6, Table S5)"),
             ("figures/SUPERSEDED_ANALYSES.md", "Figure 6 refit within the 473 train patients"),
             ("figures/LEGACY_MISSING_INPUTS.md", "superseded Figure 6 data"),
             ("figures/FIGURE_NUMBERING.md", "Scheme A")):
    has(rd(f), s, f"{f}: '{s[:45]}'")
for f in ("figures/supplementary_tables/Table_S1_rna_subgroups_trainonly.md", "figures/supplementary_tables/Table_S5_survival_trainonly.md",
          "figures/supplementary_tables/Table_1_cohort_characteristics.md"):
    absent(rd(f), r"Figures? 1\b|\(Figure 2\)|\(Figure 3C\)|Figure 3D", f"{f}: stale reference")
absent(rd("figures/supplementary_tables/Table_S6_full_figure2_performance.md"), r"Figures? 1\b|\(Figure 2\)|Figures? 4", "Table S6: stale reference")

# ---- 4. live plot scripts: first docstring line and drawn label strings -------------------------------------------------------------------
PR = "figures/publication_redesign/"
first = {PR + "fig1_new/figure1_new_plot.py": "New Figure 1 -", PR + "Fig1/figure1_plot.py": "Figure 2 -", PR + "Fig2/figure2_plot.py": "Figure 3 -",
         PR + "Fig3/figure3A_trainonly_plot.py": "Figure 4A (", PR + "Fig3/figure3B_trainonly_plot.py": "Figure 4B (", PR + "Fig3/figure3C_trainonly_plot.py": "Figure 4C (",
         PR + "Fig3/figure3D_trainonly_plot.py": "Figure 4D (", PR + "Fig4/figure4A_trainonly_plot.py": "Figure 5A (", PR + "Fig4/figure4B_trainonly_plot.py": "Figure 5B (",
         PR + "Fig5/figure5A_trainonly_plot.py": "Figure 6A (", PR + "Fig5/figure5B_trainonly_plot.py": "Figure 6B ("}
for f, s in first.items():
    txt = Path(f).read_text(encoding="utf-8")
    lines = [l for l in txt.splitlines() if l.strip() and l.strip() != '"""']
    n += 1
    if s not in lines[0]:
        fails.append(f"{f}: first docstring line should contain '{s}' (found '{lines[0][:60]}')")
drawn = {PR + "Fig1/figure1_plot.py": ["Survival cohort (Figures 5–6)"],
         PR + "Fig2/figure2_plot.py": ["nested information layers (Figure 2)", "AUPRC in Figure 3A"],
         PR + "fig1_new/figure1_new_plot.py": ["(CV; Fig. 6)"],
         PR + "Fig5/figure5A_trainonly_plot.py": ["Figure 4D five-pathway family (4 of 5 concordant)", "(Figure 4D family) is not concordant."],
         PR + "Fig5/figure5B_trainonly_plot.py": ["Member of the Figure 4D five-pathway family"]}
for f, ss in drawn.items():
    t = rd(f)
    for s in ss:
        has(t, s, f"{f}: drawn label '{s}'")
for f in list(first) + [PR + "FigS_calibration/figureS1A_plot.py", PR + "FigS_calibration/figureS1B_plot.py", PR + "FigS_shap/figureS3A_plot.py",
                        PR + "FigS_shap/figureS3B_plot.py", PR + "FigS_country_origtrain/figureS2A_plot.py", PR + "FigS_country_origtrain/figureS2B_plot.py"]:
    absent(rd(f), r"Figure 3D|Figures? 1-5|Fig\. 5\)|\(Figures 4–5\)|\(identical to Figure 1\)", f"{f}: stale reference")

# ---- 5. every manuscript-facing file: no double shift (main figure > 6) and no panel absent from the final numbering ---------------------------
# (FIGURE_NUMBERING.md is the mapping document and names the earlier labels on purpose, so it is not scanned here)
FILES = [CAP_PATH, "figures/SUPERSEDED_ANALYSES.md", "figures/LEGACY_MISSING_INPUTS.md", T + "README.md",
         PR + "fig1_new/README.md"] + sorted(str(p).replace("\\", "/") for p in Path(T).glob("Table_*trainonly*.md")) + [
         T + "Table_1_cohort_characteristics.md", T + "Table_S6_full_figure2_performance.md"] + list(first)
pat = re.compile(r"\bFigures? ((?:S?\d[A-D]?(?:[–\-][A-D](?![A-Za-z]))?(?:[–\-]|, | and | & )?)+)|\bFig\. (S?\d)")
panels_ok = {"1": "ABCDE", "2": "ABC", "3": "ABC", "4": "ABCD", "5": "AB", "6": "AB"}
for f in FILES:
    t = rd(f)
    for m in pat.finditer(t):
        grp = m.group(1) or m.group(2)
        for tok in re.findall(r"S?\d[A-D]?", grp):
            n += 1
            if tok.startswith("S"):
                continue
            num, panel = tok[0], tok[1:]
            if num not in panels_ok:
                fails.append(f"{f}: reference to non-existent Figure {tok} ('{m.group(0)}')")
            elif panel and panel not in panels_ok[num]:
                fails.append(f"{f}: reference to non-existent panel Figure {tok} ('{m.group(0)}')")

# ---- 6. frozen protocols untouched -------------------------------------------------------------------------------------------------------------
import hashlib
for md in ("RNA_CONTRIBUTION_TRAINONLY_PROTOCOL", "SURVIVAL_TRAINONLY_PROTOCOL"):
    n += 1
    body = Path(f"explainability/{md}.md").read_bytes().replace(b"\r\n", b"\n")
    rec = Path(f"explainability/{md}.sha256").read_text(encoding="utf-8").split()[0].lower()
    if hashlib.sha256(body).hexdigest() != rec:
        fails.append(f"frozen protocol {md}.md does not match its recorded SHA-256")

print(f"{n} checks, {len(fails)} failed")
for f in fails:
    print("FAIL:", f, file=sys.stderr)
sys.exit(1 if fails else 0)
