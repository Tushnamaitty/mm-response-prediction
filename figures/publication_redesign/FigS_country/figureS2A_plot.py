"""
Supplementary Figure S2A - Binary improvement AUPRC by geographic cohort (XGBoost Layer C trained on US
patients; internal geographic hold-out within CoMMpass).

Run from the repository root:
    python figures/publication_redesign/FigS_country/figureS2A_plot.py

Input (verified by figures/figS_country_extract_data.py; read-only):
    figures/data/FigS_country/figS2A_binary_auprc.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/FigS_country/figS2A_binary_auprc.csv")
OUT = Path("figures/publication_redesign/FigS_country/figureS2A.png")

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
FIG_W, FIG_H = 7.2, 2.95
COL = LAYER_COLOR["C"]
ROWS = [("spain_canada_pooled", 0.0), ("spain_only", 1.0), ("canada_only", 2.0), ("us_val", 3.35)]
EXPECT = {"spain_canada_pooled": (116, 1588, 283, 0.4514, 0.4012, 0.5071),
          "spain_only": (80, 949, 185, 0.4626, 0.4066, 0.5318),
          "canada_only": (36, 639, 98, 0.4435, 0.3691, 0.5413)}
YLIM = (4.05, -0.65)
XLIM = (0.1, 0.6)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure S2A data check failed: {msg}")


def main():
    d = pd.read_csv(DATA).set_index("cohort_key")
    for k, (np_, nv, ne, e, lo, hi) in EXPECT.items():
        r = d.loc[k]
        require((int(r.n_patients), int(r.n_visit_pairs), int(r.n_improvement_events)) == (np_, nv, ne), f"{k} counts")
        require((round(r.auprc, 4), round(r.ci_lower, 4), round(r.ci_upper, 4)) == (e, lo, hi), f"{k} AUPRC")
        require(abs(r.event_rate - ne / nv) < 1e-12, f"{k} event rate")
    u = d.loc["us_val"]
    require(pd.isna(u.ci_lower) and pd.isna(u.ci_upper) and round(u.auprc, 4) == 0.5247, "US validation (no CI)")
    require((int(u.n_patients), int(u.n_visit_pairs), int(u.n_improvement_events)) == (139, 1911, 351), "US counts")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.98, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.978, "Binary next-visit improvement: AUPRC by geographic cohort (XGBoost, information layer C)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.905, "Internal geographic hold-out within CoMMpass — NOT independent external validation "
             "· model trained on US patients only", fontsize=FS["small"], color=LAYER_COLOR["D"],
             fontweight="bold", ha="left", va="top")

    bottom, height = 0.25, 0.54
    axL = fig.add_axes([0.012, bottom, 0.3, height])
    ax = fig.add_axes([0.33, bottom, 0.38, height])
    axV = fig.add_axes([0.73, bottom, 0.26, height])
    for a in (axL, axV):
        a.set_xlim(0, 1)
        a.set_ylim(*YLIM)
        a.axis("off")
    axV.text(0, -0.55, "AUPRC [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")

    for k, y in ROWS:
        r = d.loc[k]
        ref = k == "us_val"
        lab = {"spain_canada_pooled": "Spain + Canada (pooled)", "spain_only": "Spain", "canada_only": "Canada",
               "us_val": "US validation (reference)"}[k]
        axL.text(0.0 if k == "spain_canada_pooled" or ref else 0.05, y - 0.13, lab, fontsize=FS["body"],
                 color=MUTED if ref else INK, va="center", fontweight="bold" if k == "spain_canada_pooled" else "normal")
        sub = (f"{int(r.n_patients)} patients · {int(r.n_visit_pairs):,} pairs · "
               f"{int(r.n_improvement_events)} events ({r.event_rate:.1%})")
        if ref:
            sub = "informed calibration/threshold choices\n" + sub
        axL.text(0.0 if k == "spain_canada_pooled" or ref else 0.05, y + 0.12, sub, fontsize=FS["small"] - 0.4,
                 color=MUTED, va="top", linespacing=1.2)
        # cohort-specific chance level (event rate)
        ax.plot([r.event_rate, r.event_rate], [y - 0.28, y + 0.28], color=MUTED, lw=1.0, zorder=2)
        if ref:
            ax.plot(r.auprc, y, "o", ms=5.5, mfc=MUTED, mec=MUTED, zorder=3)
            axV.text(0, y, f"{r.auprc:.3f}  (no CI computed)", fontsize=FS["small"], color=MUTED, va="center")
            continue
        filled = k == "spain_canada_pooled"
        ax.plot([r.ci_lower, r.ci_upper], [y, y], color=COL, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(r.auprc, y, "o", ms=6 if filled else 5.4, mfc=COL if filled else "white", mec=COL, mew=1.1, zorder=3)
        axV.text(0, y, f"{r.auprc:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]", fontsize=FS["small"], color=INK,
                 va="center", fontweight="bold" if filled else "normal")
    ax.axhline(2.68, color=GRID, lw=0.6)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("AUPRC (higher is better)", fontsize=FS["body"], labelpad=2)
    ax.text(0.0, 1.01, "| = cohort event rate (AUPRC of a non-informative model)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS["small"], color=MUTED)

    fig.text(0.047, 0.075, "● pooled   ○ country subset. Bars: 95% patient-bootstrap CI. Countries were not "
             "compared statistically.\nHyperparameters were tuned in the main analysis on data including 80 of the 116 "
             "evaluated patients.", fontsize=FS["small"], color=MUTED, ha="left", va="center", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
