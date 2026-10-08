"""
Supplementary Figure S3A - Net family SHAP attribution shares for the locked XGBoost information layer D
models (binary improvement; six-class IMWG response), held-out test set.

Run from the repository root:
    python figures/publication_redesign/FigS_shap/figureS3A_plot.py

Input (verified by figures/data/FigS_shap/figS_shap_extend.py; read-only):
    figures/data/FigS_shap/figS3A_ext_component_shares.csv
        (measure = net_family_magnitude, analysis = primary). Bar widths use the exact share_pct;
        every label uses share_pct_display (largest-remainder rounding to 0.1 pp), so the six
        displayed labels sum to exactly 100.0% per task.

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/FigS_shap/figS3A_ext_component_shares.csv")
OUT = Path("figures/publication_redesign/FigS_shap/figureS3A.png")

# ---- shared style (identical to Figure 1) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
RNA_LIGHT = "#f4b393"
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "savefig.facecolor": "white", "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
FIG_W, FIG_H = 7.2, 2.8
# family, short label, fill, hatch, text colour
FAM = [("Current IMWG response", "Current IMWG\nresponse (A)", LAYER_COLOR["A"], "////", INK),
       ("Other current clinical state", "Other current\nclinical (A)", LAYER_COLOR["A"], None, INK),
       ("Longitudinal history", "Longitudinal\nhistory (B)", LAYER_COLOR["B"], None, "white"),
       ("Treatment context", "Treatment\ncontext (C)", LAYER_COLOR["C"], None, "white"),
       ("RNA pathway scores", "RNA pathway\nscores (D)", LAYER_COLOR["D"], None, "white"),
       ("RNA timing / availability", "RNA timing /\navailability (D)", RNA_LIGHT, None, INK)]
TASKS = [("binary", "Binary improvement", "log-odds SHAP"),
         ("multiclass", "Six-class IMWG response", "raw-score SHAP, mean over 6 classes")]
EXPECT = {"binary": [0.581, 0.104, 0.179, 0.087, 0.033, 0.015], "multiclass": [0.532, 0.168, 0.147, 0.068, 0.073, 0.012]}


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure S3A data check failed: {msg}")


def main():
    d = pd.read_csv(DATA)
    d = d[(d.measure == "net_family_magnitude") & (d.analysis == "primary")].rename(columns={"component": "family"})
    d["share"] = d.share_pct / 100
    d = d.set_index(["task", "family"])
    for t, exp in EXPECT.items():
        require(len(d.loc[t]) == 6 and set(d.loc[t].index) == {f[0] for f in FAM}, f"{t} six components")
        sh = [d.loc[(t, f[0]), "share"] for f in FAM]
        require(abs(sum(sh) - 1) < 1e-9, f"{t} shares sum to 1")
        require([round(x, 3) for x in sh] == exp, f"{t} shares")
        disp = [d.loc[(t, f[0]), "share_pct_display"] for f in FAM]
        require(round(sum(disp), 6) == 100.0, f"{t} displayed labels sum to 100.0")
        require(max(abs(a - 100 * b) for a, b in zip(disp, sh)) <= 0.1 + 1e-9, f"{t} display rounding")
        require(int(d.loc[(t, FAM[0][0]), "n_patients_used"]) == 155, f"{t} patients")
    require((int(d.loc[("binary", FAM[0][0]), "n_rows_used"]), int(d.loc[("multiclass", FAM[0][0]), "n_rows_used"])) == (1930, 2079),
            "rows")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.98, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.978, "Where the XGBoost information layer D models place their attribution (net family SHAP share)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.905, "Held-out test set · 155 patients · shares describe model attribution, not accuracy, "
             "variance explained or causation", fontsize=FS["small"], color=INK2, fontweight="bold", ha="left", va="top")

    ax = fig.add_axes([0.2, 0.36, 0.78, 0.45])
    ys = [0, 1]
    for y, (t, name, unit) in zip(ys, TASKS):
        left = 0.0
        for fam, _, col, hatch, tc in FAM:
            s = d.loc[(t, fam), "share"]
            lab = f"{d.loc[(t, fam), 'share_pct_display']:.1f}%"
            ax.barh(y, s - 0.003, left=left, height=0.58, color=col, edgecolor="white" if hatch else "none",
                    hatch=hatch, linewidth=0.0)
            if s >= 0.05:
                ax.text(left + s / 2, y, lab, ha="center", va="center", fontsize=FS["body"], color=tc,
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", fc=col, ec="none") if hatch else None)
            left += s
        small = [f"{d.loc[(t, f[0]), 'share_pct_display']:.1f}%" for f in FAM[4:]]
        ax.text(1.0, y + 0.42, f"RNA pathways {small[0]} · RNA timing/availability {small[1]}", ha="right", va="center",
                fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold")
        ax.text(-0.012, y - 0.07, name, ha="right", va="center", fontsize=FS["body"], color=INK, fontweight="bold",
                transform=ax.get_yaxis_transform())
        ax.text(-0.012, y + 0.2, unit, ha="right", va="center", fontsize=FS["small"] - 0.3, color=MUTED,
                transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(1.62, -0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Share of net family attribution (mean |sum of signed SHAP within family|; shares sum to 100%)",
                  fontsize=FS["body"], labelpad=2)

    # legend (two rows of three)
    for i, (fam, lab, col, hatch, _) in enumerate(FAM):
        x = 0.047 + (i % 6) * 0.158
        y = 0.155
        fig.patches.append(plt.Rectangle((x, y - 0.025), 0.018, 0.05, transform=fig.transFigure, fc=col,
                                         ec="white" if hatch else "none", hatch=hatch, lw=0))
        fig.text(x + 0.024, y, lab.replace("\n", " ") if False else lab, fontsize=FS["small"], color=INK, va="center",
                 linespacing=1.1)
    fig.text(0.047, 0.045, "Letters = information layer. Locked XGBoost layer D (reporting reference algorithm, not a "
             "validation-selected best model).", fontsize=FS["small"], color=MUTED, va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
