"""
Figure 3A - Overall RNA contribution: direct XGBoost Layer D vs Layer C, and a regularized RNA add-on.

Run from the repository root:
    python figures/publication_redesign/Fig3/figure3A_plot.py

Input (verified by figures/fig3_extract_data.py; read-only):
    figures/data/Fig3/fig3A_overall_rna_contribution.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py (ink colours, layer colours,
font sizes, bold panel letter at the upper left). The script stops if a plotted value differs
from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/Fig3/fig3A_overall_rna_contribution.csv")
OUT = Path("figures/publication_redesign/Fig3/figure3A.png")

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
DIRECT = dict(marker="o", ms=5.5, color=LAYER_COLOR["D"], mfc=LAYER_COLOR["D"])
ADDON = dict(marker="D", ms=5.0, color=LAYER_COLOR["C"], mfc=LAYER_COLOR["C"])
FIG_W, FIG_H = 7.2, 3.3
MINUS = "−"
YLIM = (1.6, -0.6)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 3A data check failed: {msg}")


def num(v, nd, sign=True):
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}").replace("-", MINUS)


def column(ax, rows, scale, xlim, better, title, denom, xlabel):
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    for y, r, style in rows:
        if not bool(r.evaluated):
            ax.text(0, y, "Not evaluated\n(run for six-class task only)", ha="center", va="center", fontsize=FS["body"], color=MUTED,
                    style="italic", bbox=dict(fc="white", ec="none", pad=1.5), zorder=3)
            continue
        ax.plot([r.ci_lower * scale, r.ci_upper * scale], [y, y], color=style["color"], lw=1.4,
                solid_capstyle="round", zorder=2)
        ax.plot(r.delta * scale, y, ls="none", mec="white", mew=0.8, zorder=3, **style)
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=FS["body"], labelpad=2)
    ax.text(0.0, 1.21, title, transform=ax.transAxes, fontsize=FS["title"], fontweight="bold",
            color=INK, ha="left", va="bottom")
    ax.text(0.0, 1.11, denom, transform=ax.transAxes, fontsize=FS["small"], color=MUTED, ha="left",
            va="bottom")
    left, right = ("RNA worse", "RNA better") if better == "higher" else ("RNA better", "RNA worse")
    ax.text(0.0, 1.01, f"← {left}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS["small"], color=INK2)
    ax.text(1.0, 1.01, f"{right} →", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS["small"], color=INK2)


def value_column(fig, rect, entries):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    ax.text(0, -0.6, "Estimate [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold",
            va="bottom")
    for y, txt in entries:
        ax.text(0, y, txt, fontsize=FS["small"], color=INK, va="center", ha="left", linespacing=1.25)


def main():
    d = pd.read_csv(DATA).set_index("row_id")
    dB, dR, aR = d.loc["direct_binary_auprc"], d.loc["direct_multiclass_rps"], d.loc["addon_multiclass_rps"]
    aB = d.loc["addon_binary_auprc"]
    require(abs(dB.delta - 0.004764462117) < 1e-10 and abs(dR.delta - 0.0007515795024) < 1e-12, "direct deltas")
    require(abs(aR.delta + 0.00033976481221190634) < 1e-15, "add-on delta")
    require(not bool(aB.evaluated), "binary add-on must be not evaluated")
    require((int(dB.n_visit_pairs), int(dB.n_patients), int(dR.n_visit_pairs), int(dR.n_patients),
             int(aR.n_visit_pairs), int(aR.n_patients)) == (6939, 596, 7724, 600, 7724, 600), "denominators")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "Overall RNA contribution: direct Layer D model vs a regularized RNA add-on",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.93, "EXPLORATORY · 5×5 patient-grouped cross-validation, train + validation "
             "patients (not the held-out test set)",
             fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")

    bottom, height = 0.29, 0.43
    axL = fig.add_axes([0.012, bottom, 0.27, height])
    axL.set_xlim(0, 1)
    axL.set_ylim(*YLIM)
    axL.axis("off")
    for y, head, sub, st in [
        (0, f"Direct: XGBoost Layer D {MINUS} Layer C",
         "Layer D = Layer C + 50 Hallmark RNA pathway\nscores + RNA age; both refitted in each fold", DIRECT),
        (1, f"Regularized RNA add-on {MINUS} Layer C",
         "Ridge-penalized correction of Layer C\nprobabilities using the 50 pathway scores", ADDON),
    ]:
        axL.plot(0.02, y - 0.1, ls="none", mec="white", mew=0.8, **st)
        axL.text(0.07, y - 0.1, head, fontsize=FS["body"], color=INK, fontweight="bold", va="center")
        axL.text(0.07, y + 0.1, sub, fontsize=FS["small"], color=INK2, va="top", linespacing=1.2)

    axB = fig.add_axes([0.305, bottom, 0.19, height])
    column(axB, [(0, dB, DIRECT), (1, aB, ADDON)], 1, (-0.012, 0.022), "higher",
           "Binary improvement task",
           f"{int(dB.n_visit_pairs):,} visit pairs · {int(dB.n_patients)} patients",
           "ΔAUPRC vs Layer C")
    axB.set_xticks([-0.01, 0, 0.01, 0.02])
    axB.set_xticklabels([f"{MINUS}0.01", "0", "0.01", "0.02"])
    value_column(fig, [0.505, bottom, 0.13, height], [
        (0, f"{num(dB.delta, 4)}\n[{num(dB.ci_lower, 4, False)}, {num(dB.ci_upper, 4, False)}]\np = {dB.p_value:.2f}"),
        (1, "—"),
    ])

    axR = fig.add_axes([0.66, bottom, 0.19, height])
    column(axR, [(0, dR, DIRECT), (1, aR, ADDON)], 1e3, (-0.9, 1.6), "lower",
           "Six-class IMWG response task",
           f"{int(dR.n_visit_pairs):,} visit pairs · {int(dR.n_patients)} patients",
           "ΔRPS $\\times10^{-3}$ vs Layer C")
    axR.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
    axR.set_xticklabels([f"{MINUS}0.5", "0", "0.5", "1.0", "1.5"])

    def rps_txt(r, ptxt):
        return (f"{num(r.delta * 1e3, 2)} [{num(r.ci_lower * 1e3, 2, False)}, {num(r.ci_upper * 1e3, 2, False)}]"
                f"\n{num(r.pct_change_vs_layer_c, 2)}% of Layer C RPS\n{ptxt}")

    value_column(fig, [0.86, bottom, 0.135, height], [
        (0, rps_txt(dR, f"p = {dR.p_value:.3f}")),
        (1, rps_txt(aR, f"permutation p = {aR.p_value:.3f}†")),
    ])

    fig.text(0.047, 0.115,
             "Paired on identical visit pairs. Bars: 95% patient-cluster bootstrap CIs (2,000 resamples). RPS, ranked "
             "probability score (lower is better; ΔRPS > 0 = RNA worsened\npredictions). Direct-model p: "
             "two-sided bootstrap. †One-sided RNA-permutation test, 0 of 200 permutations as good as observed "
             "(smallest possible p = 1/201).",
             fontsize=FS["small"], color=MUTED, ha="left", va="top", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
