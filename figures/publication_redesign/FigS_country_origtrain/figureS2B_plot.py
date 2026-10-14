"""
Supplementary Figure S2B - Six-class IMWG response macro-F1 by geographic cohort (XGBoost Layer C trained on
US patients; internal geographic hold-out within CoMMpass; original-train-only revision).

Run from the repository root:
    python figures/publication_redesign/FigS_country_origtrain/figureS2B_plot.py

Input (verified by figures/figS_country_origtrain_extract_data.py; read-only):
    figures/data/FigS_country_origtrain/figS2B_sixclass_macro_f1.csv
Cross-checked against artifacts/results_clean_rerun/geographic_origtrain/full_run_audit_and_summary.json.
Earlier-version figures (FigS_country/, FigS_country_ustuned/) are left unchanged.

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/FigS_country_origtrain/figS2B_sixclass_macro_f1.csv")
SRC = Path("artifacts/results_clean_rerun/geographic_origtrain/full_run_audit_and_summary.json")
OUT = Path("figures/publication_redesign/FigS_country_origtrain/figureS2B.png")

# ---- shared style (identical to Figure 2) ----
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
LABEL = {"spain_canada_pooled": "Spain + Canada (pooled)", "spain_only": "Spain", "canada_only": "Canada",
         "us_val": "US calibration set (reference)"}
# cohort sizes (definition of the experiment); metric values are checked against the result JSON instead
EXPECT = {"spain_canada_pooled": (116, 1694), "spain_only": (80, 1034), "canada_only": (36, 660)}
YLIM = (4.05, -0.65)
XLIM = (0.45, 0.75)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure S2B data check failed: {msg}")


def main():
    d = pd.read_csv(DATA).set_index("cohort_key")
    j = json.load(open(SRC, encoding="utf-8"))["tasks"]["multiclass"]
    for k, (np_, nv) in EXPECT.items():
        r = d.loc[k]
        ci = j[f"{k}_bootstrap_ci"]["macro_f1"]
        require((int(r.n_patients), int(r.n_visit_pairs)) == (np_, nv), f"{k} counts")
        require(abs(r.macro_f1 - j[f"{k}_metrics"]["macro_f1"]) < 1e-12, f"{k} macro-F1 = result file")
        require(abs(r.ci_lower - ci["ci_lower_2.5"]) < 1e-12 and abs(r.ci_upper - ci["ci_upper_97.5"]) < 1e-12,
                f"{k} CI = result file")
        require(XLIM[0] < r.ci_lower and r.ci_upper < XLIM[1], f"{k} inside axis")
    u = d.loc["us_val"]
    require(pd.isna(u.ci_lower) and pd.isna(u.ci_upper)
            and abs(u.macro_f1 - j["us_val_metrics"]["macro_f1"]) < 1e-12, "US validation (no CI)")
    require((int(u.n_patients), int(u.n_visit_pairs)) == (96, 1467), "calibration-set counts")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.98, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.978, "Six-class IMWG response: macro-F1 by geographic cohort (XGBoost, information layer C)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.905, "Internal geographic hold-out within CoMMpass — NOT independent external validation "
             "· model developed on US original-training-partition patients only", fontsize=FS["small"], color=LAYER_COLOR["D"],
             fontweight="bold", ha="left", va="top")

    bottom, height = 0.25, 0.54
    axL = fig.add_axes([0.012, bottom, 0.3, height])
    ax = fig.add_axes([0.33, bottom, 0.38, height])
    axV = fig.add_axes([0.73, bottom, 0.26, height])
    for a in (axL, axV):
        a.set_xlim(0, 1)
        a.set_ylim(*YLIM)
        a.axis("off")
    axV.text(0, -0.55, "Macro-F1 [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")

    for k, y in ROWS:
        r = d.loc[k]
        ref = k == "us_val"
        x0 = 0.0 if k == "spain_canada_pooled" or ref else 0.05
        axL.text(x0, y - 0.13, LABEL[k], fontsize=FS["body"], color=MUTED if ref else INK, va="center",
                 fontweight="bold" if k == "spain_canada_pooled" else "normal")
        sub = f"{int(r.n_patients)} patients · {int(r.n_visit_pairs):,} visit pairs"
        if ref:
            sub = "informed calibration/threshold choices\n" + sub
        axL.text(x0, y + 0.12, sub, fontsize=FS["small"] - 0.4, color=MUTED, va="top", linespacing=1.2)
        if ref:
            ax.plot(r.macro_f1, y, "o", ms=5.5, mfc=MUTED, mec=MUTED, zorder=3)
            axV.text(0, y, f"{r.macro_f1:.3f}  (no CI computed)", fontsize=FS["small"], color=MUTED, va="center")
            continue
        filled = k == "spain_canada_pooled"
        ax.plot([r.ci_lower, r.ci_upper], [y, y], color=COL, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(r.macro_f1, y, "o", ms=6 if filled else 5.4, mfc=COL if filled else "white", mec=COL, mew=1.1, zorder=3)
        axV.text(0, y, f"{r.macro_f1:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]", fontsize=FS["small"], color=INK,
                 va="center", fontweight="bold" if filled else "normal")
    ax.axhline(2.68, color=GRID, lw=0.6)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    ax.set_xticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Macro-F1 (unweighted mean F1 over the six IMWG categories; higher is better)", fontsize=FS["body"],
                  labelpad=2)

    fig.text(0.047, 0.085, "● pooled   ○ country subset. Bars: 95% patient-bootstrap CI. Countries were not "
             "compared statistically.\nTuning, fitting, preprocessing, calibration and threshold used only US "
             "original-training-partition patients;\nno Spain, Canada, validation- or test-partition patient entered "
             "development. Post hoc, exploratory.", fontsize=FS["small"], color=MUTED, ha="left", va="center", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
