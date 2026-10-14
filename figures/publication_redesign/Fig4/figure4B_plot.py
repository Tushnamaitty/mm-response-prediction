"""
Figure 4B - Delta C-index (clinical baseline + RNA minus clinical baseline) for the primary analysis and
two sensitivity analyses, PFS and OS.

Run from the repository root:
    python figures/publication_redesign/Fig4/figure4B_plot.py

Input (verified by figures/fig4_extract_data.py; read-only):
    figures/data/Fig4/fig4B_delta_c_forest.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/Fig4/fig4B_delta_c_forest.csv")
OUT = Path("figures/publication_redesign/Fig4/figure4B.png")

# ---- shared style (identical to Figure 1) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "savefig.facecolor": "white",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
MINUS = "−"
FIG_W, FIG_H = 7.2, 3.3

# (endpoint, analysis, label, sublabel, y)
ROWS = [
    ("PFS", "primary", "Primary", "clinical baseline · 674 patients, 459 events", 0.9),
    ("PFS", "expanded_baseline", "Expanded clinical baseline", "+ 7 baseline labs/marrow values · 674 / 459", 1.8),
    ("PFS", "rna_timing_m30_0", "RNA day −30 to 0 only", "clinical baseline · 624 patients, 423 events", 2.7),
    ("OS", "primary", "Primary", "clinical baseline · 674 patients, 247 deaths", 4.3),
    ("OS", "expanded_baseline", "Expanded clinical baseline", "+ 7 baseline labs/marrow values · 674 / 247", 5.2),
    ("OS", "rna_timing_m30_0", "RNA day −30 to 0 only", "not run (specified for PFS only)", 6.1),
]
HEADERS = [(0.15, "Progression-free survival (PFS)"), (3.55, "Overall survival (OS)")]
EXPECT = {("PFS", "primary"): (0.04049462663357797, 674, 459), ("PFS", "expanded_baseline"): (0.035005165674360114, 674, 459),
          ("PFS", "rna_timing_m30_0"): (0.02789262630916951, 624, 423), ("OS", "primary"): (0.04349834303473617, 674, 247),
          ("OS", "expanded_baseline"): (0.04867700338261301, 674, 247)}
YLIM = (6.6, -0.35)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 4B data check failed: {msg}")


def main():
    d = pd.read_csv(DATA).set_index(["endpoint", "analysis"])
    for k, (dv, n, e) in EXPECT.items():
        r = d.loc[k]
        require(bool(r.evaluated) and abs(r.delta_c - dv) < 1e-12 and (int(r.n_patients), int(r.n_events)) == (n, e), str(k))
        require(r.ci_lower < r.delta_c < r.ci_upper, f"{k} CI")
    require(not bool(d.loc[("OS", "rna_timing_m30_0"), "evaluated"]), "OS timing must be not evaluated")
    perm = d.permutation_p_one_sided.dropna()
    require(list(perm.index) == [("PFS", "primary")] and abs(perm.iloc[0] - 1 / 201) < 1e-12,
            "permutation p only for primary PFS")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "Incremental RNA prognostic performance across survival sensitivity analyses",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.93, "EXPLORATORY · internal 5×5 cross-validation in the survival cohort "
             "(not external validation)", fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold",
             ha="left", va="top")

    bottom, height = 0.19, 0.64
    axL = fig.add_axes([0.012, bottom, 0.33, height])
    axF = fig.add_axes([0.36, bottom, 0.3, height])
    axV = fig.add_axes([0.68, bottom, 0.31, height])
    for ax in (axL, axV):
        ax.set_xlim(0, 1)
        ax.set_ylim(*YLIM)
        ax.axis("off")
    for y, h in HEADERS:
        axL.text(0, y, h, fontsize=FS["body"], color=INK, fontweight="bold", va="center")
    axV.text(0, -0.3, "ΔC [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")
    axV.text(0.62, -0.3, "C-index\nbaseline → + RNA", fontsize=FS["small"], color=INK2, fontweight="bold",
             va="bottom", linespacing=1.15)

    axF.axvline(0, color=INK2, lw=0.7, zorder=1)
    axF.axhline(3.5, color=GRID, lw=0.6)
    for ep, an, lab, sub, y in ROWS:
        r = d.loc[(ep, an)]
        primary = an == "primary"
        axL.text(0.04, y - 0.13, lab, fontsize=FS["body"], color=INK, va="center",
                 fontweight="bold" if primary else "normal")
        axL.text(0.04, y + 0.25, sub, fontsize=FS["small"] - 0.3, color=MUTED, va="center")
        if not bool(r.evaluated):
            axF.text(0.035, y, "Not evaluated", ha="center", va="center", fontsize=FS["small"], color=MUTED,
                     style="italic")
            axV.text(0, y, "—", fontsize=FS["small"], color=MUTED, va="center")
            continue
        axF.plot([r.ci_lower, r.ci_upper], [y, y], color=LAYER_COLOR["D"], lw=1.4, solid_capstyle="round", zorder=2)
        axF.plot(r.delta_c, y, ls="none", marker="o", ms=6 if primary else 5.4, mec=LAYER_COLOR["D"], mew=1.1,
                 mfc=LAYER_COLOR["D"] if primary else "white", zorder=3)
        est = f"{r.delta_c:+.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
        axV.text(0, y, est, fontsize=FS["small"], color=INK, va="center", fontweight="bold" if primary else "normal")
        axV.text(0.62, y, f"{r.c_index_baseline:.3f} → {r.c_index_plus_rna:.3f}", fontsize=FS["small"],
                 color=INK2, va="center")
        if pd.notna(r.permutation_p_one_sided):
            axV.text(0, y + 0.33, f"permutation p = {r.permutation_p_one_sided:.3f} (0/200)",
                     fontsize=FS["small"] - 0.3, color=INK2, va="center")
    axF.set_xlim(-0.012, 0.085)
    axF.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    axF.set_xticklabels(["0", "0.02", "0.04", "0.06", "0.08"])
    axF.set_ylim(*YLIM)
    axF.set_yticks([])
    for s in ("top", "right", "left"):
        axF.spines[s].set_visible(False)
    axF.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    axF.set_xlabel("ΔC-index (clinical baseline + RNA − baseline)", fontsize=FS["body"], labelpad=2)
    axF.text(0.0, 1.01, "← RNA worse", transform=axF.transAxes, ha="left", va="bottom", fontsize=FS["small"],
             color=INK2)
    axF.text(1.0, 1.01, "RNA better →", transform=axF.transAxes, ha="right", va="bottom", fontsize=FS["small"],
             color=INK2)

    fig.text(0.047, 0.065, "● primary   ○ sensitivity. Bars: 95% patient-bootstrap CI on fixed out-of-fold "
             "predictions. RNA day = days from start of first treatment (CoMMpass index date).",
             fontsize=FS["small"], color=MUTED, va="center")
    fig.text(0.047, 0.03, "Rows differ in cohort or clinical baseline and were not compared statistically. Permutation "
             "test performed for primary PFS only.", fontsize=FS["small"], color=MUTED, va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
