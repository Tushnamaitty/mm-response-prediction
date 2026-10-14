"""
Supplementary Figure S1A - Reliability curves for binary next-visit improvement, XGBoost Layer C vs
Layer D, held-out test set.

Run from the repository root:
    python figures/publication_redesign/FigS_calibration/figureS1A_plot.py

Inputs (verified by figures/figS_calibration_extract_data.py; read-only):
    figures/data/FigS_calibration/figS1A_reliability_bins.csv
    figures/data/FigS_calibration/figS1A_prediction_histogram.csv
    figures/data/FigS_calibration/figS1_binary_calibration_metrics.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIR = Path("figures/data/FigS_calibration")
OUT = Path("figures/publication_redesign/FigS_calibration/figureS1A.png")

# ---- shared style (identical to Figure 1) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})
MINUS = "−"
FIG_W, FIG_H = 7.2, 3.7
STYLE = {"C": dict(color=LAYER_COLOR["C"], marker="o", label="Layer C (clinical + history + treatment)"),
         "D": dict(color=LAYER_COLOR["D"], marker="D", label="Layer D (Layer C + RNA)")}
LIM = 0.7


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure S1A data check failed: {msg}")


def main():
    b = pd.read_csv(DIR / "figS1A_reliability_bins.csv")
    h = pd.read_csv(DIR / "figS1A_prediction_histogram.csv")
    m = pd.read_csv(DIR / "figS1_binary_calibration_metrics.csv").set_index(["layer", "metric"])
    for L in "CD":
        bl = b[b.layer == L]
        require(len(bl) == 10 and (bl.n_rows == 193).all() and bl.n_rows.sum() == 1930, f"{L} bins")
        require(bl.n_events.sum() == 353, f"{L} events")
        require((bl.observed_rate - bl.n_events / bl.n_rows).abs().max() < 1e-12, f"{L} observed rates")
        require(int(h[h.layer == L].n_rows.sum()) == 1930, f"{L} histogram")
        require(bl.mean_predicted.max() < LIM and bl.observed_rate.max() < LIM, "inside limits")
    exp = {("C", "calibration_slope"): 1.086692, ("D", "calibration_slope"): 1.082874,
           ("C", "calibration_intercept"): -0.065946, ("D", "calibration_intercept"): -0.075087}
    for k, v in exp.items():
        require(abs(m.loc[k, "estimate"] - v) < 1e-6, f"{k}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "Calibration of binary next-visit improvement predictions (XGBoost, Layer C vs Layer D)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.935, "Held-out test set · 155 patients · 1,930 visit pairs · uncalibrated "
             "probabilities (selected on the validation set)", fontsize=FS["small"], color=INK2,
             fontweight="bold", ha="left", va="top")

    W = 2.0
    ax = fig.add_axes([0.1, 0.3, W / FIG_W, W / FIG_H])
    axh = fig.add_axes([0.1, 0.11, W / FIG_W, 0.42 / FIG_H], sharex=ax)
    ax.plot([0, LIM], [0, LIM], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for L, st in STYLE.items():
        bl = b[b.layer == L].sort_values("bin")
        ax.plot(bl.mean_predicted, bl.observed_rate, color=st["color"], lw=1.0, zorder=2, alpha=0.9)
        ax.plot(bl.mean_predicted, bl.observed_rate, ls="none", marker=st["marker"], ms=4.6 if L == "C" else 4.2,
                mfc=st["color"], mec="white", mew=0.6, zorder=3)
        hl = h[h.layer == L]
        axh.step(hl.bin_lower, hl.n_rows, where="post", color=st["color"], lw=1.0)
    ax.set_xlim(0, LIM)
    ax.set_ylim(0, LIM)
    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in ticks])
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Observed improvement rate", fontsize=FS["body"], labelpad=2)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.text(0.545, 0.48, "perfect\ncalibration", fontsize=FS["small"], color=MUTED, ha="left", va="top",
            style="italic", linespacing=1.1, rotation=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    axh.set_xticks(ticks)
    axh.set_xticklabels([f"{t:.1f}" for t in ticks])
    axh.set_yticks([])
    for sp in ("top", "right", "left"):
        axh.spines[sp].set_visible(False)
    axh.set_xlabel("Predicted probability of improvement (points: bin means)", fontsize=FS["body"], labelpad=2)
    above = {L: int(h[(h.layer == L) & (h.bin_lower >= LIM)].n_rows.sum()) for L in "CD"}
    axh.text(1.0, 1.02, f"predictions > {LIM}: C {above['C']}, D {above['D']} pairs (beyond axis)",
             transform=axh.transAxes, ha="right", va="bottom", fontsize=FS["small"] - 0.3, color=MUTED)
    axh.text(-0.02, 0.5, "pairs", transform=axh.transAxes, ha="right", va="center", fontsize=FS["small"], color=MUTED)

    # key and calibration statistics (right)
    kx, ky = 0.5, 0.78
    for i, (L, st) in enumerate(STYLE.items()):
        y = ky - i * 0.085
        fig.lines.append(plt.Line2D([kx, kx + 0.035], [y, y], color=st["color"], lw=1.0, transform=fig.transFigure))
        fig.lines.append(plt.Line2D([kx + 0.0175], [y], marker=st["marker"], ms=4.6, mfc=st["color"], mec="white",
                                    mew=0.6, ls="none", transform=fig.transFigure))
        fig.text(kx + 0.05, y, st["label"], fontsize=FS["body"], color=INK, va="center")
    fig.text(kx, ky - 0.24, "Calibration slope (ideal 1)", fontsize=FS["body"], color=INK, fontweight="bold", va="center")
    fig.text(kx, ky - 0.32, "Calibration intercept (ideal 0)", fontsize=FS["body"], color=INK, fontweight="bold",
             va="center")
    for j, L in enumerate("CD"):
        x = kx + 0.27 + j * 0.13
        fig.text(x, ky - 0.17, f"Layer {L}", fontsize=FS["small"], color=STYLE[L]["color"], fontweight="bold",
                 va="center", ha="center")
        for k, (met, yy) in enumerate([("calibration_slope", ky - 0.24), ("calibration_intercept", ky - 0.32)]):
            r = m.loc[(L, met)]
            fig.text(x, yy, f"{r.estimate:.2f}\n[{r.ci_lower:.2f}, {r.ci_upper:.2f}]".replace("-", MINUS),
                     fontsize=FS["small"], color=INK, va="center", ha="center", linespacing=1.15)
    oc = m.loc[("C", "observed_rate"), "estimate"]
    fig.text(kx, ky - 0.43, f"Mean predicted: C {m.loc[('C', 'mean_predicted'), 'estimate']:.3f}, "
             f"D {m.loc[('D', 'mean_predicted'), 'estimate']:.3f}; observed {oc:.3f}",
             fontsize=FS["small"], color=INK2, va="center")
    fig.text(kx, ky - 0.52, "Each point: one of 10 equal-count bins (193 visit pairs).\nBottom strip: distribution of "
             "predicted probabilities.", fontsize=FS["small"], color=INK2, va="top", linespacing=1.3)
    fig.text(kx, 0.07, "Brackets: 95% patient-bootstrap CIs. No intervals are shown for individual bins.",
             fontsize=FS["small"], color=MUTED, va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
