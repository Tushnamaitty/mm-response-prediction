"""
Figure 3D - Proliferation/MYC RNA programme and the next IMWG response: pathway-selection evidence,
discovery estimate, and post-hoc internal split-sample test.

Run from the repository root:
    python figures/publication_redesign/Fig3/figure3D_plot.py

Inputs (verified by figures/fig3_extract_data.py; read-only):
    figures/data/Fig3/fig3D_proliferation_myc_association.csv
    figures/data/Fig3/fig3D_composite_loadings.csv
    figures/data/Fig3/fig3D_primary_matched_pair_global_test.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DIR = Path("figures/data/Fig3")
OUT = Path("figures/publication_redesign/Fig3/figure3D.png")

# ---- shared style (identical to Figure 1) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
RNA_LIGHT = "#f4b393"
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "savefig.facecolor": "white",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
MINUS = "−"
FIG_W, FIG_H = 7.2, 3.75
FAMILY5 = ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle", "Myc Targets V1", "Myc Targets V2"]
GROUPS = [  # (stage, header, sub, y_header, rows y, style)
    ("1_selection_evidence", "1  Pathway-selection evidence (single pathways)",
     "pooled, all patients incl. test; motivated the family", 0.0, [0.95, 1.6, 2.25, 2.9, 3.55], "sel"),
    ("2_discovery", "2  Discovery (composite)", "train + validation patients",
     4.45, [5.35], "disc"),
    ("3_split_sample_test", "3  Post-hoc internal split-sample test (composite)",
     "held-out test patients only; frozen composite", 6.25, [7.15], "test"),
]
DIVIDERS = [4.0, 5.8]
STYLE = {"sel": dict(marker="o", ms=4.6, color=MUTED, mfc=MUTED),
         "disc": dict(marker="s", ms=5.0, color=LAYER_COLOR["D"], mfc="white"),
         "test": dict(marker="D", ms=5.2, color=LAYER_COLOR["D"], mfc=LAYER_COLOR["D"])}
YLIM = (7.65, -0.55)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 3D data check failed: {msg}")


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main():
    d = pd.read_csv(DIR / "fig3D_proliferation_myc_association.csv")
    load = pd.read_csv(DIR / "fig3D_composite_loadings.csv")
    glob = pd.read_csv(DIR / "fig3D_primary_matched_pair_global_test.csv").set_index("family")
    disc = d[d.evidence_stage == "2_discovery"].iloc[0]
    test = d[d.evidence_stage == "3_split_sample_test"].iloc[0]
    sel = d[d.evidence_stage == "1_selection_evidence"].set_index("estimate_label")
    require(abs(test.odds_ratio - 0.8158568037456768) < 1e-12 and abs(test.or_ci_upper - np.exp(-0.029619044949844076)) < 1e-12,
            "test odds ratio")
    require(abs(disc.beta + 0.2786100802788457) < 1e-15, "discovery beta")
    require((int(test.n_visit_pairs), int(test.n_patients), int(disc.n_visit_pairs), int(disc.n_patients))
            == (1221, 84, 6212, 448), "denominators")
    require(bool(sel.includes_test_patients.all()) and not bool(disc.includes_test_patients), "population flags")
    require(abs(glob.loc["ordinal", "global_p"] - 0.14908509) < 1e-8, "primary global test")
    require(sorted(load.pathway) == sorted(FAMILY5) and (load.loading > 0).all(), "loadings")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.988, "D", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.986, "Proliferation/MYC RNA programme and the next-visit IMWG response (association, not "
             "predictive gain)", fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.94, "EXPLORATORY · post-hoc pathway family; stage 3 is NOT independent replication",
             fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")
    fig.text(0.047, 0.897, "Prespecified primary analysis (matched-pair global test, all 50 pathways): null, "
             f"six-class p = {glob.loc['ordinal', 'global_p']:.3f}; binary p = {glob.loc['binary', 'global_p']:.3f}.",
             fontsize=FS["small"], color=INK, ha="left", va="top", fontweight="bold")

    bottom, height = 0.2, 0.59
    axL = fig.add_axes([0.012, bottom, 0.3, height])
    axL.set_xlim(0, 1)
    axL.set_ylim(*YLIM)
    axL.axis("off")
    axF = fig.add_axes([0.33, bottom, 0.22, height])
    axV = fig.add_axes([0.565, bottom, 0.2, height])
    axV.set_xlim(0, 1)
    axV.set_ylim(*YLIM)
    axV.axis("off")
    axV.text(0, -0.5, "Odds ratio [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")

    axF.axvline(1, color=INK2, lw=0.7, zorder=1)
    for stage, head, sub, yh, ys, st in GROUPS:
        axL.text(0, yh, head, fontsize=FS["body"], color=INK, fontweight="bold", va="center")
        axL.text(0.035, yh + 0.42, sub, fontsize=FS["small"] - 0.3, color=INK2, va="center", style="italic")
        rows = d[d.evidence_stage == stage]
        if stage == "1_selection_evidence":
            rows = rows.set_index("estimate_label").loc[[f"{p} (single pathway)" for p in FAMILY5]].reset_index()
        for y, (_, r) in zip(ys, rows.iterrows()):
            s = STYLE[st]
            lab = r.estimate_label.replace(" (single pathway)", "") if st == "sel" else \
                ("Composite score" if st == "disc" else "Composite score (frozen)")
            axL.text(0.07, y, lab, fontsize=FS["body"], color=INK, va="center",
                     fontweight="bold" if st == "test" else "normal")
            axF.plot([r.or_ci_lower, r.or_ci_upper], [y, y], color=s["color"], lw=1.3, solid_capstyle="round", zorder=2)
            axF.plot(r.odds_ratio, y, ls="none", mec=s["color"], mew=1.0, zorder=3,
                     **{k: v for k, v in s.items() if k != "color"})
            est = f"{r.odds_ratio:.2f} [{r.or_ci_lower:.2f}, {r.or_ci_upper:.2f}]"
            if st == "test":
                est += f"   one-sided {fmt_p(r.p_value)}"
            axV.text(0, y, est, fontsize=FS["small"], color=INK if st != "sel" else INK2, va="center",
                     fontweight="bold" if st == "test" else "normal")
        if stage == "3_split_sample_test":
            axL.text(0.07, ys[0] + 0.42, f"{int(rows.iloc[0].n_patients)} patients", fontsize=FS["small"] - 0.3,
                     color=MUTED, va="center")
    for yy in DIVIDERS:
        axF.axhline(yy, color=GRID, lw=0.6)
    axF.set_xscale("log")
    axF.set_xlim(0.62, 1.12)
    axF.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1])
    axF.set_xticklabels(["0.7", "0.8", "0.9", "1.0", "1.1"])
    axF.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axF.set_ylim(*YLIM)
    axF.set_yticks([])
    for s in ("top", "right", "left"):
        axF.spines[s].set_visible(False)
    axF.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    axF.set_xlabel("Odds ratio per SD of score (log scale)", fontsize=FS["body"], labelpad=2)
    axF.text(0.0, 1.01, "← worse next response", transform=axF.transAxes, ha="left", va="bottom",
             fontsize=FS["small"], color=INK2)
    axF.text(1.0, 1.01, "better →", transform=axF.transAxes, ha="right", va="bottom",
             fontsize=FS["small"], color=INK2)

    # composite loadings
    axW = fig.add_axes([0.875, bottom + 0.06, 0.11, height - 0.26])
    lo = load.set_index("pathway").loc[FAMILY5]
    axW.barh(range(5), lo.loading, color=RNA_LIGHT, edgecolor=LAYER_COLOR["D"], lw=0.5, height=0.62)
    for i, v in enumerate(lo.loading):
        axW.text(v + 0.02, i, f"{v:.2f}", fontsize=FS["small"] - 0.4, color=INK, va="center")
    axW.set_yticks(range(5))
    axW.set_yticklabels(FAMILY5, fontsize=FS["small"] - 0.3, color=INK)
    axW.invert_yaxis()
    axW.set_xlim(0, 0.7)
    axW.set_xticks([0, 0.3, 0.6])
    axW.tick_params(axis="y", length=0, pad=2)
    for s in ("top", "right", "left"):
        axW.spines[s].set_visible(False)
    axW.set_xlabel("Loading", fontsize=FS["small"], labelpad=1.5)
    fig.text(0.79, bottom + height + 0.005, "Composite weights\n(first principal component)", fontsize=FS["small"],
             color=INK, fontweight="bold", ha="left", va="bottom", linespacing=1.15)
    fig.text(0.79, bottom + height - 0.045, f"{lo.variance_explained_by_pc1.iloc[0]:.1%} of variance; fixed\n"
             "before stage 3 was run", fontsize=FS["small"] - 0.3, color=INK2, ha="left", va="top", linespacing=1.2)

    fig.text(0.047, 0.1,
             "Ordinal model of the next IMWG category (PD < SD < PR < VGPR < CR < sCR); odds ratio < 1 = higher score, "
             "worse next response. Bars: robust Wald 95% CIs.\nStages differ in patients and adjustment models, so "
             "estimates are not directly comparable. The one-sided direction was fixed after stage 1, before stages "
             "2–3 (not prespecified).",
             fontsize=FS["small"], color=MUTED, ha="left", va="top", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
