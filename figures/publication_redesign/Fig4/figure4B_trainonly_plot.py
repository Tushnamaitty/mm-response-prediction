"""
Figure 5B (train-only redesign) - Incremental RNA prognostic performance (Delta C-index) across the held-out evaluation, its sensitivity
analyses, the development-only cross-validation reference and the earlier pooled estimate (shown for context only).

Run from the repository root:
    python figures/publication_redesign/Fig4/figure4B_trainonly_plot.py

Input (verified by figures/fig45_trainonly_extract_data.py; read-only):
    figures/data/Fig45_trainonly/fig4B_trainonly_delta_c.csv
Replaces figure4B.png (pooled 5x5 CV), which is left unchanged.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/Fig45_trainonly/fig4B_trainonly_delta_c.csv")
OUT = Path("figures/publication_redesign/Fig4/figure4B_trainonly.png")

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})
MINUS = "−"
FIG_W, FIG_H = 7.2, 5.0
# (row_id, label, sublabel, y, kind)
ROWS = [
    ("A1", "Primary (held-out)", "train-fitted · 201 patients, 149 events", 0.9, "primary"),
    ("S2_pfs", "Expanded clinical baseline", "+ 7 baseline labs / marrow values · 201 / 149", 1.8, "sens"),
    ("S1", "RNA day −30 to 0 only", "186 patients, 137 events", 2.7, "sens"),
    ("S4_val", "Validation patients only", "descriptive · 101 / 71", 3.6, "desc"),
    ("S4_test", "Test patients only", "descriptive · 100 / 78", 4.5, "desc"),
    ("S3_pfs", "Development-only 5×5 CV", "reference · train patients · 473 / 310", 5.6, "dev"),
    ("P_pfs", "Earlier pooled 5×5 CV", "all 674 patients (not leakage-free) · 674 / 459", 6.5, "pooled"),
    ("A2", "Primary (held-out)", "train-fitted · 201 patients, 82 deaths", 8.6, "primary"),
    ("S2_os", "Expanded clinical baseline", "+ 7 baseline labs / marrow values · 201 / 82", 9.5, "sens"),
    ("S1_os", "RNA day −30 to 0 only", "not run (specified for PFS only)", 10.4, "sens"),
    ("S3_os", "Development-only 5×5 CV", "reference · train patients · 473 / 165", 11.5, "dev"),
    ("P_os", "Earlier pooled 5×5 CV", "all 674 patients (not leakage-free) · 674 / 247", 12.4, "pooled"),
]
HEADERS = [(0.15, "Progression-free survival (PFS)"), (7.8, "Overall survival (OS)")]
STYLE = {"primary": dict(marker="o", ms=6.2, color=LAYER_COLOR["D"], mfc=LAYER_COLOR["D"]),
         "sens": dict(marker="o", ms=5.2, color=LAYER_COLOR["D"], mfc="white"),
         "desc": dict(marker="s", ms=4.0, color=MUTED, mfc="white"),
         "dev": dict(marker="o", ms=5.0, color=INK2, mfc="white"),
         "pooled": dict(marker="D", ms=4.6, color=MUTED, mfc="white")}
YLIM = (13.0, -0.4)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 5B data check failed: {msg}")


def main():
    d = pd.read_csv(DATA).set_index("row_id")
    require(set(r[0] for r in ROWS) == set(d.index), "row set")
    ev = d[d.evaluated.astype(bool)]
    require(bool(((ev.ci_lower < ev.delta_c) & (ev.delta_c < ev.ci_upper)).all()), "CI brackets estimates")
    require(not bool(d.loc["S1_os", "evaluated"]), "OS timing must be not evaluated")
    require((int(d.loc["A1", "n_patients"]), int(d.loc["A1", "n_events"]), int(d.loc["A2", "n_patients"]), int(d.loc["A2", "n_events"])) == (201, 149, 201, 82), "primary denominators")
    require(d.loc["A1", "ci_lower"] < 0, "caption assumes the primary PFS CI includes 0")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.988, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.986, "Incremental RNA prognostic performance: held-out evaluation, sensitivity analyses and references",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.95, "EXPLORATORY · held-out = validation + test patients (post hoc; not independent validation).\n"
             "Diamonds are earlier pooled estimates in which the held-out patients also trained the models.",
             fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top", linespacing=1.3)
    bottom, height = 0.19, 0.67
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
    axV.text(0.62, -0.3, "C-index\nbaseline → + RNA", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom", linespacing=1.15)
    axF.axvline(0, color=INK2, lw=0.7, zorder=1)
    axF.axhline(7.4, color=GRID, lw=0.6)
    for rid, lab, sub, y, kind in ROWS:
        r = d.loc[rid]
        axL.text(0.04, y - 0.13, lab, fontsize=FS["body"], color=INK, va="center", fontweight="bold" if kind == "primary" else "normal")
        axL.text(0.04, y + 0.25, sub, fontsize=FS["small"] - 0.3, color=MUTED, va="center")
        if not bool(r.evaluated):
            axF.text(0.0, y, "Not evaluated", ha="center", va="center", fontsize=FS["small"], color=MUTED, style="italic")
            axV.text(0, y, "—", fontsize=FS["small"], color=MUTED, va="center")
            continue
        st = STYLE[kind]
        axF.plot([r.ci_lower, r.ci_upper], [y, y], color=st["color"], lw=1.4 if kind == "primary" else 1.1, solid_capstyle="round", zorder=2)
        axF.plot(r.delta_c, y, ls="none", mec=st["color"], mew=1.1, zorder=3, **{k: v for k, v in st.items() if k != "color"})
        est = f"{r.delta_c:+.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]".replace("-", MINUS)
        axV.text(0, y, est, fontsize=FS["small"], color=INK if kind in ("primary", "sens") else INK2, va="center", fontweight="bold" if kind == "primary" else "normal")
        axV.text(0.62, y, f"{r.c_index_baseline:.3f} → {r.c_index_plus_rna:.3f}", fontsize=FS["small"], color=INK2, va="center")
        if pd.notna(r.permutation_p_one_sided):
            axV.text(0, y + 0.33, f"permutation p = {r.permutation_p_one_sided:.4f}", fontsize=FS["small"] - 0.3, color=INK2, va="center")
    axF.set_xlim(-0.09, 0.115)
    axF.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
    axF.set_xticklabels([f"{MINUS}0.08", f"{MINUS}0.04", "0", "0.04", "0.08"])
    axF.set_ylim(*YLIM)
    axF.set_yticks([])
    for s in ("top", "right", "left"):
        axF.spines[s].set_visible(False)
    axF.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    axF.set_xlabel("ΔC-index (clinical baseline + RNA − baseline)", fontsize=FS["body"], labelpad=2)
    axF.text(0.0, 1.01, "← RNA worse", transform=axF.transAxes, ha="left", va="bottom", fontsize=FS["small"], color=INK2)
    axF.text(1.0, 1.01, "RNA better →", transform=axF.transAxes, ha="right", va="bottom", fontsize=FS["small"], color=INK2)
    fig.text(0.047, 0.075, "Orange filled: primary · orange open: sensitivity · grey squares: validation-only / test-only (descriptive)\n"
             "dark open circle: development reference · grey diamond: earlier pooled estimate. Bars: 95% patient-bootstrap CI.",
             fontsize=FS["small"], color=MUTED, va="center", linespacing=1.3)
    fig.text(0.047, 0.018, "Held-out primary criterion (ΔC > 0, CI lower bound > 0, permutation p ≤ 0.025) not met for PFS. Rows differ in cohort and baseline and were not\n"
             "compared statistically. RNA day = days from start of first treatment.", fontsize=FS["small"], color=MUTED, va="center", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
