"""
Figure 4B (train-only redesign) - RNA contribution (Layer D - Layer C) by Layer-C prediction-uncertainty quartile, in held-out
validation + test patients with locked train-only models. Quartile cut-points are fixed from the development (train) entropy
distribution, not from the evaluated patients.

Run from the repository root:
    python figures/publication_redesign/Fig3/figure3B_trainonly_plot.py

Input (verified by figures/fig3ABC_trainonly_extract_data.py; read-only):
    figures/data/Fig3ABC_trainonly/fig3B_trainonly_uncertainty.csv
Replaces figure3B.png (pooled train + validation CV), which is left unchanged.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/Fig3ABC_trainonly/fig3B_trainonly_uncertainty.csv")
OUT = Path("figures/publication_redesign/Fig3/figure3B_trainonly.png")

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
FIG_W, FIG_H = 7.2, 3.75
ROWS = [  # (group id, label, y, style)
    ("Q1_low_uncertainty", "Q1 (lowest uncertainty)", 0, "q"),
    ("Q2", "Q2", 1, "q"),
    ("Q3", "Q3", 2, "q"),
    ("Q4_high_uncertainty", "Q4 (highest uncertainty)", 3, "q4"),
    ("Q1_Q3_lower_uncertainty", "Q1–Q3 combined (comparator)", 4.2, "ref"),
    ("Q4_minus_Q1_Q3", "Interaction: Q4 vs Q1–Q3", 5.6, "int"),
]
STYLE = {"q": dict(marker="o", ms=5.0, color=LAYER_COLOR["D"], mfc=LAYER_COLOR["D"]),
         "q4": dict(marker="o", ms=5.5, color=LAYER_COLOR["D"], mfc=LAYER_COLOR["D"]),
         "ref": dict(marker="o", ms=5.0, color=INK2, mfc=INK2),
         "int": dict(marker="D", ms=5.2, color=INK, mfc="white")}
YLIM = (6.65, -0.9)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 4B data check failed: {msg}")


def num(v, nd, sign=True):
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}").replace("-", MINUS)


def forest(ax, sub, scale, xlim, better, title, xlabel):
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    ax.axhline(4.9, color=GRID, lw=0.6)
    for gid, _, y, st in ROWS:
        r = sub[sub.group == gid].iloc[0]
        s = STYLE[st]
        ax.plot([r.ci_lower * scale, r.ci_upper * scale], [y, y], color=s["color"], lw=1.3, solid_capstyle="round", zorder=2)
        ax.plot(r.delta * scale, y, ls="none", mec=s["color"] if st == "int" else "white", mew=1.0 if st == "int" else 0.8, zorder=3, **s)
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=FS["body"], labelpad=2)
    ax.text(0.0, 1.13, title, transform=ax.transAxes, fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="bottom")
    left, right = ("RNA worse", "RNA better") if better == "higher" else ("RNA better", "RNA worse")
    ax.text(0.0, 1.01, f"← {left}", transform=ax.transAxes, ha="left", va="bottom", fontsize=FS["small"], color=INK2)
    ax.text(1.0, 1.01, f"{right} →", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS["small"], color=INK2)


def values(fig, rect, sub, scale, nd):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    ax.text(0, -0.75, "Estimate [95% CI]   pairs", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")
    for gid, _, y, st in ROWS:
        r = sub[sub.group == gid].iloc[0]
        est = f"{num(r.delta * scale, nd)} [{num(r.ci_lower * scale, nd, False)}, {num(r.ci_upper * scale, nd, False)}]"
        if st == "int":
            ax.text(0, y - 0.3, est, fontsize=FS["small"], color=INK, va="center")
            ax.text(0, y + 0.15, f"unadjusted p = {r.p_value:.2f}", fontsize=FS["small"], color=INK2, va="center")
            sig = r.adjusted_p < 0.05
            ax.text(0, y + 0.68, f"Holm p = {r.adjusted_p:.2f}: " + ("significant" if sig else "not significant"), fontsize=FS["small"],
                    color="white" if sig else INK, fontweight="bold", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=LAYER_COLOR["D"] if sig else "#efeee9", ec=INK2, lw=0.5))
        else:
            ax.text(0, y, est, fontsize=FS["small"], color=INK, va="center")
            ax.text(0.86, y, f"{int(r.n_visit_pairs):,}", fontsize=FS["small"] - 0.3, color=MUTED, va="center", ha="right")


def main():
    d = pd.read_csv(DATA)
    b = d[(d.task == "binary") & (d.metric == "auprc")]
    m = d[(d.task == "multiclass") & (d.metric == "rps")]
    for sub, n in [(b, 2405), (m, 2689)]:
        require(int(sub[sub.group.isin(["Q1_low_uncertainty", "Q2", "Q3", "Q4_high_uncertainty"])].n_visit_pairs.sum()) == n, "quartiles sum to the held-out total")
        i = sub[sub.group == "Q4_minus_Q1_Q3"].iloc[0]
        require(i.ci_lower < i.delta < i.ci_upper and 0 < i.adjusted_p <= 1, "interaction row")
    require(not any((x[x.group == "Q4_minus_Q1_Q3"].adjusted_p.iloc[0] < 0.05) for x in (b, m)), "caption text assumes no significant primary interaction")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.987, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.985, "RNA contribution by clinical prediction uncertainty in held-out patients (XGBoost Layer D − Layer C)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.94, "EXPLORATORY · locked train-only models evaluated in held-out validation + test patients with RNA "
             "(post hoc; not independent validation)", fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")
    fig.text(0.047, 0.902, "Quartiles of Layer C prediction entropy, with cut-points fixed from the train-patient entropy distribution (computed without "
             "outcomes or RNA).\nPre-specified test: interaction (D − C in Q4) − (D − C in Q1–Q3), Holm-adjusted over the two tasks.",
             fontsize=FS["small"], color=INK2, ha="left", va="top", linespacing=1.3)

    bottom, height = 0.25, 0.5
    axL = fig.add_axes([0.012, bottom, 0.2, height])
    axL.set_xlim(0, 1)
    axL.set_ylim(*YLIM)
    axL.axis("off")
    for _, lab, y, st in ROWS:
        axL.text(0.03, y, lab, fontsize=FS["body"], color=INK, va="center", fontweight="bold" if st in ("int", "q4") else "normal")
    axL.text(0.03, 5.6 + 0.42, "(Q4 change minus Q1–Q3 change)", fontsize=FS["small"] - 0.3, color=INK2, va="center")
    axL.text(0.03, 5.6 + 0.85, "Not significant after Holm\nadjustment in either task", fontsize=FS["small"], color=INK, fontweight="bold",
             va="top", linespacing=1.15)

    axB = fig.add_axes([0.215, bottom, 0.185, height])
    forest(axB, b, 1, (-0.10, 0.06), "higher", "Binary improvement: ΔAUPRC", "ΔAUPRC, Layer D vs Layer C")
    axB.set_xticks([-0.08, -0.04, 0, 0.04])
    axB.set_xticklabels([f"{MINUS}0.08", f"{MINUS}0.04", "0", "0.04"])
    values(fig, [0.41, bottom, 0.18, height], b, 1, 3)

    axR = fig.add_axes([0.6, bottom, 0.185, height])
    forest(axR, m, 1e3, (-3.0, 3.2), "lower", "Six-class IMWG response: ΔRPS", "ΔRPS $\\times10^{-3}$, Layer D vs Layer C")
    axR.set_xticks([-2, 0, 2])
    axR.set_xticklabels([f"{MINUS}2", "0", "2"])
    values(fig, [0.795, bottom, 0.2, height], m, 1e3, 2)

    fig.text(0.047, 0.115,
             "Bars: 95% patient-cluster bootstrap CIs. Quartile estimates are unadjusted; the held-out groups contain 514–717 visit pairs each, so intervals are wide.\n"
             "RPS, ranked probability score: lower is better (ΔRPS > 0 = RNA worse). Held-out: 213 patients with RNA; 2,405 binary and 2,689 six-class visit pairs.",
             fontsize=FS["small"], color=MUTED, ha="left", va="top", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
