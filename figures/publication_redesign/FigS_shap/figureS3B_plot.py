"""
Supplementary Figure S3B - Top 15 source input features by mean absolute source-feature SHAP for the locked
XGBoost information layer D models (binary improvement; six-class IMWG response), held-out test set.

Run from the repository root:
    python figures/publication_redesign/FigS_shap/figureS3B_plot.py

Input (verified by figures/data/FigS_shap/figS_shap_compute.py; read-only):
    figures/data/FigS_shap/figS3B_top15_source_features.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/FigS_shap/figS3B_top15_source_features.csv")
OUT = Path("figures/publication_redesign/FigS_shap/figureS3B.png")

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
FIG_W, FIG_H = 7.2, 3.9
STYLE = {"Current IMWG response": (LAYER_COLOR["A"], "////"), "Other current clinical state": (LAYER_COLOR["A"], None),
         "Longitudinal history": (LAYER_COLOR["B"], None), "Treatment context": (LAYER_COLOR["C"], None),
         "RNA pathway scores": (LAYER_COLOR["D"], None), "RNA timing / availability": (RNA_LIGHT, None)}
LEGEND = [("Current IMWG response (A)", "Current IMWG response"), ("Other current clinical (A)", "Other current clinical state"),
          ("Longitudinal history (B)", "Longitudinal history"), ("Treatment context (C)", "Treatment context"),
          ("RNA pathway scores (D)", "RNA pathway scores"), ("RNA timing / availability (D)", "RNA timing / availability")]
DISPLAY = {
    "vt_disease_response": "Current IMWG response", "vt_days_to_visit": "Days since index at V(t)",
    "m_protein_current": "M-protein, current", "m_protein_previous": "M-protein, previous",
    "m_protein_change": "M-protein, change", "m_protein_pct_change": "M-protein, % change",
    "age_at_diagnosis": "Age at diagnosis", "subject_had_signs_symptoms": "Signs/symptoms reported",
    "days_since_last_transplant": "Days since last transplant", "kappa_flc_pct_change": "Kappa FLC, % change",
    "kappa_flc_current": "Kappa FLC, current", "kappa_flc_change": "Kappa FLC, change",
    "kappa_flc_previous": "Kappa FLC, previous", "lambda_flc_previous": "Lambda FLC, previous",
    "line_duration_so_far": "Time on current line", "current_regimen_categories": "Current regimen drug classes",
    "regimen_duration_so_far": "Time on current regimen", "days_since_rna_sample": "RNA age (days since sample)",
    "TNF-alpha Signaling via NF-kB": "TNF-α signalling via NF-κB", "TGF-beta Signaling": "TGF-β signalling",
    "n_prior_transplants": "Number of prior transplants", "ig_heavy_chain_type": "Ig heavy-chain type",
    "pct_plasma_cells_bm": "Bone-marrow plasma cells (%)",
}
EXPECT = {("binary", 1): ("vt_disease_response", 0.7865), ("binary", 12): ("TNF-alpha Signaling via NF-kB", 0.0204),
          ("binary", 15): ("TGF-beta Signaling", 0.0181), ("multiclass", 1): ("vt_disease_response", 0.8795),
          ("multiclass", 2): ("m_protein_current", 0.1611), ("multiclass", 15): ("days_since_rna_sample", 0.0205)}
TASKS = [("binary", "Binary improvement", "Mean |SHAP| (log-odds)"),
         ("multiclass", "Six-class IMWG response", "Mean |SHAP| (raw class score, mean of 6 classes)")]


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure S3B data check failed: {msg}")


def main():
    d = pd.read_csv(DATA)
    for t in ("binary", "multiclass"):
        s = d[d.task == t].sort_values("rank")
        require(list(s["rank"]) == list(range(1, 16)), f"{t} ranks 1-15")
        require(s.mean_abs_shap.is_monotonic_decreasing, f"{t} ordering")
        require(set(s.source_feature) <= set(DISPLAY), f"{t} display names")
        require(((s.ci_lower <= s.mean_abs_shap) & (s.mean_abs_shap <= s.ci_upper)).all(), f"{t} CI brackets")
    for (t, rk), (f, v) in EXPECT.items():
        r = d[(d.task == t) & (d["rank"] == rk)].iloc[0]
        require(r.source_feature == f and round(r.mean_abs_shap, 4) == v, f"{t} rank {rk}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "Top 15 source input features by mean absolute SHAP (XGBoost information layer D)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.935, "Held-out test set · 155 patients · encoded columns summed to source features before "
             "taking |SHAP| · model attribution, not causation", fontsize=FS["small"], color=INK2,
             fontweight="bold", ha="left", va="top")

    for i, (t, title, xlab) in enumerate(TASKS):
        s = d[d.task == t].sort_values("rank")
        ax = fig.add_axes([0.205 + i * 0.5, 0.22, 0.25, 0.62])
        for y, (_, r) in enumerate(s.iterrows()):
            col, hatch = STYLE[r.family]
            ax.barh(y, r.mean_abs_shap, height=0.68, color=col, hatch=hatch, edgecolor="white" if hatch else "none", lw=0)
            ax.plot([r.ci_lower, r.ci_upper], [y, y], color=INK2, lw=0.6, zorder=3)
            ax.text(r.ci_upper + 0.012, y, f"{r.mean_abs_shap:.3f}", va="center", ha="left", fontsize=FS["small"] - 0.3,
                    color=INK)
        ax.set_yticks(range(15))
        labels = [DISPLAY[f] for f in s.source_feature]
        ax.set_yticklabels(labels, fontsize=FS["small"])
        for tl, fam in zip(ax.get_yticklabels(), s.family):
            if fam.startswith("RNA"):
                tl.set_color(LAYER_COLOR["D"])
                tl.set_fontweight("bold")
            if fam == "Current IMWG response":
                tl.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.tick_params(axis="y", length=0, pad=3)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlab, fontsize=FS["small"], labelpad=2)
        ax.set_title(title, loc="left", fontsize=FS["body"], fontweight="bold", color=INK, pad=4, x=-0.72)

    lx = 0.047
    for j, (lab, fam) in enumerate(LEGEND):
        col, hatch = STYLE[fam]
        x = lx + (j % 3) * 0.25
        yy = 0.085 - (j // 3) * 0.045
        fig.patches.append(plt.Rectangle((x, yy - 0.016), 0.016, 0.032, transform=fig.transFigure, fc=col,
                                         ec="white" if hatch else "none", hatch=hatch, lw=0))
        fig.text(x + 0.022, yy, lab, fontsize=FS["small"] - 0.3, color=INK, va="center")
    fig.text(0.79, 0.063, "Thin lines: 95% patient-\nbootstrap CI. Orange labels:\nRNA features. Letters =\n"
             "information layer.",
             fontsize=FS["small"] - 0.3, color=MUTED, va="center", linespacing=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
