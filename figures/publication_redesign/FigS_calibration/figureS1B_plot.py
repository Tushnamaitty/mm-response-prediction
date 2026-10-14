"""
Supplementary Figure S1B - Layer D minus Layer C differences in calibration metrics (XGBoost, binary
next-visit improvement, held-out test set), each metric on its own axis.

Run from the repository root:
    python figures/publication_redesign/FigS_calibration/figureS1B_plot.py

Inputs (verified by figures/figS_calibration_extract_data.py; read-only):
    figures/data/FigS_calibration/figS1B_d_minus_c_calibration.csv
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
OUT = Path("figures/publication_redesign/FigS_calibration/figureS1B.png")

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
FIG_W, FIG_H = 7.2, 2.35
# metric, title, ideal / direction note, x half-range, number format
PANELS = [
    ("calibration_slope", "Calibration slope", "ideal 1", 0.04, "{:+.3f}", "{:.3f}"),
    ("calibration_intercept", "Calibration intercept", "ideal 0", 0.03, "{:+.3f}", "{:.3f}"),
    ("brier", "Brier score", "lower is better", 0.002, "{:+.4f}", "{:.4f}"),
    ("log_loss", "Log loss", "lower is better", 0.004, "{:+.4f}", "{:.4f}"),
]
EXPECT = {"calibration_slope": (-0.003818, -0.028852, 0.023676),
          "calibration_intercept": (-0.009141, -0.023524, 0.005231),
          "brier": (0.000102, -0.001029, 0.00105), "log_loss": (0.000521, -0.002291, 0.002967)}


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure S1B data check failed: {msg}")


def fmt(s):
    return s.replace("-", MINUS)


def main():
    d = pd.read_csv(DIR / "figS1B_d_minus_c_calibration.csv")
    d = d[d.analysis_set == "all_test_rows"].set_index("metric")
    m = pd.read_csv(DIR / "figS1_binary_calibration_metrics.csv").set_index(["layer", "metric"])
    for k, (e, lo, hi) in EXPECT.items():
        r = d.loc[k]
        require(abs(r.delta - e) < 1e-6 and abs(r.ci_lower - lo) < 1e-6 and abs(r.ci_upper - hi) < 1e-6, k)
        require(bool(r.ci_includes_0), f"{k} CI includes 0")
        require(abs((m.loc[("D", k), "estimate"] - m.loc[("C", k), "estimate"]) - r.delta) < 2e-6, f"{k} = D - C")
    for k, _, _, half, _, _ in PANELS:
        require(max(abs(d.loc[k, "ci_lower"]), abs(d.loc[k, "ci_upper"])) < half, f"{k} inside axis")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.975, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.972, "No clear overall calibration difference between Layers C and D "
             "(XGBoost, Layer D − Layer C)", fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.885, "Held-out test set · 155 patients · 1,930 visit pairs · uncalibrated "
             "probabilities · each metric on its own scale", fontsize=FS["small"], color=INK2,
             fontweight="bold", ha="left", va="top")

    w, gap, x0 = 0.205, 0.035, 0.047
    for i, (k, title, note, half, f1, f2) in enumerate(PANELS):
        r = d.loc[k]
        ax = fig.add_axes([x0 + i * (w + gap), 0.27, w, 0.3])
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.plot([r.ci_lower, r.ci_upper], [0, 0], color=LAYER_COLOR["D"], lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(r.delta, 0, "o", ms=5.5, mfc=LAYER_COLOR["D"], mec="white", mew=0.7, zorder=3)
        ax.set_xlim(-half, half)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xticks([-half / 2, 0, half / 2])
        ax.set_xticklabels([fmt(f"{v:g}") for v in (-half / 2, 0, half / 2)])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.text(0, 1.72, title, transform=ax.transAxes, fontsize=FS["body"], fontweight="bold", color=INK,
                ha="left", va="bottom")
        c0, c1 = m.loc[("C", k), "estimate"], m.loc[("D", k), "estimate"]
        ax.text(0, 1.42, fmt(f"C {f2.format(c0)} → D {f2.format(c1)} ({note})"), transform=ax.transAxes,
                fontsize=FS["small"], color=INK2, ha="left", va="bottom")
        ax.text(0.5, 1.08, fmt(f"Δ {f1.format(r.delta)} [{f2.format(r.ci_lower)}, {f2.format(r.ci_upper)}]"),
                transform=ax.transAxes, fontsize=FS["small"], color=INK, ha="center", va="bottom", fontweight="bold")
        ax.set_xlabel("D − C", fontsize=FS["small"], labelpad=1.5)

    fig.text(0.047, 0.04, "Point: Layer D − Layer C on identical test pairs. Bar: 95% paired patient-bootstrap CI "
             "(1,000 resamples). Zero line = no difference; all CIs include 0.", fontsize=FS["small"], color=MUTED,
             ha="left", va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
