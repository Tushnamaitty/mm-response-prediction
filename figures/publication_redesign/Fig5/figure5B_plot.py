"""
Figure 5B - The 19 RNA pathways consistently directed for both PFS and OS: mean ridge-Cox coefficient
per endpoint with the min-max range across 25 overlapping cross-validation fits (not a CI).

Run from the repository root:
    python figures/publication_redesign/Fig5/figure5B_plot.py

Input (verified by figures/fig5_extract_data.py; read-only):
    figures/data/Fig5/fig5B_concordant_pathways.csv

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

DATA = Path("figures/data/Fig5/fig5B_concordant_pathways.csv")
OUT = Path("figures/publication_redesign/Fig5/figure5B.png")

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
FIG_W, FIG_H = 7.2, 4.9
EP_STYLE = {"PFS": dict(color=LAYER_COLOR["C"], marker="o", dy=-0.17, mfc=LAYER_COLOR["C"]),
            "OS": dict(color=LAYER_COLOR["D"], marker="D", dy=0.17, mfc=LAYER_COLOR["D"])}
GROUPS = ["higher hazard in both", "lower hazard in both"]


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 5B data check failed: {msg}")


def main():
    d = pd.read_csv(DATA)
    require(len(d) == 38 and d.pathway_source_name.nunique() == 19, "19 pathways x 2 endpoints")
    require((d.n_fits == 25).all(), "25 fits")
    require(d.groupby("concordance_group").pathway_source_name.nunique().to_dict()
            == {"higher hazard in both": 13, "lower hazard in both": 6}, "13 / 6 split")
    require(((d.beta_min_across_fits > 0) | (d.beta_max_across_fits < 0)).all(), "ranges exclude 0")
    require(((d.beta_min_across_fits <= d.beta_mean) & (d.beta_mean <= d.beta_max_across_fits)).all(), "means in ranges")
    e2f = d[(d.pathway_source_name == "E2F Targets") & (d.endpoint == "OS")].iloc[0]
    require(abs(e2f.beta_mean - 0.0890758190021487) < 1e-12, "E2F OS mean")

    # row order: within each group, by mean of the two endpoint means (largest |beta| first)
    order = []
    for g in GROUPS:
        sub = d[d.concordance_group == g].groupby("pathway_display_name").beta_mean.mean()
        order.append(list(sub.sort_values(ascending=(g != "higher hazard in both")).index))
    ys, y = {}, 0.0
    head_y = []
    for gi, names in enumerate(order):
        head_y.append(y)
        y += 0.8
        for n in names:
            ys[n] = y
            y += 1.0
        y += 0.5
    ymax = y - 0.5

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.988, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.986, "The 19 RNA pathways with the same direction in every fit for both PFS and OS",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.952, "EXPLORATORY · ridge Cox (clinical baseline + RNA) · 25 overlapping internal "
             "cross-validation fits · association, not causation", fontsize=FS["small"],
             color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")

    ax = fig.add_axes([0.3, 0.16, 0.52, 0.68])
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    for n, yy in ys.items():
        for ep, st in EP_STYLE.items():
            r = d[(d.pathway_display_name == n) & (d.endpoint == ep)].iloc[0]
            yv = yy + st["dy"]
            ax.plot([r.beta_min_across_fits, r.beta_max_across_fits], [yv, yv], color=st["color"], lw=1.1,
                    alpha=0.55, solid_capstyle="round", zorder=2)
            ax.plot(r.beta_mean, yv, ls="none", marker=st["marker"], ms=4.2 if ep == "OS" else 4.6,
                    mfc=st["mfc"], mec="white", mew=0.5, zorder=3)
    ax.set_ylim(ymax + 0.2, -0.5)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels(list(ys.keys()), fontsize=FS["body"], color=INK)
    for hy, g in zip(head_y, GROUPS):
        ax.text(-0.013, hy + 0.15, g.capitalize() + f" ({len(order[GROUPS.index(g)])})", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=FS["body"], fontweight="bold", color=INK)
    ax.axhline(head_y[1] - 0.25, color=GRID, lw=0.6)
    ax.set_xlim(-0.15, 0.12)
    ticks = [-0.15, -0.1, -0.05, 0, 0.05, 0.1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}".replace("-", MINUS) for t in ticks])
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Coefficient β (log-hazard per training-fold SD of pathway score)", fontsize=FS["body"], labelpad=2)
    top = ax.secondary_xaxis("top", functions=(np.exp, lambda v: np.log(np.clip(v, 1e-9, None))))
    hr = [0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12]
    top.set_xticks(hr)
    top.set_xticklabels([f"{h:.2f}" for h in hr], fontsize=FS["tick"])
    top.set_xlabel("Hazard ratio per SD (HR = e$^{β}$)", fontsize=FS["small"], color=INK2, labelpad=2)
    top.spines["top"].set_color(AXIS)
    ax.text(0.0, -0.115, "← lower hazard", transform=ax.transAxes, ha="left", va="top", fontsize=FS["small"],
            color=INK2)
    ax.text(1.0, -0.115, "higher hazard →", transform=ax.transAxes, ha="right", va="top", fontsize=FS["small"],
            color=INK2)

    # key
    kx = 0.84
    for i, (ep, st) in enumerate(EP_STYLE.items()):
        yk = 0.8 - i * 0.06
        fig.lines.append(plt.Line2D([kx, kx + 0.03], [yk, yk], color=st["color"], lw=1.1, alpha=0.55,
                                    transform=fig.transFigure))
        fig.lines.append(plt.Line2D([kx + 0.015], [yk], marker=st["marker"], ms=4.4, mfc=st["mfc"], mec="white",
                                    mew=0.5, ls="none", transform=fig.transFigure))
        fig.text(kx + 0.04, yk, "PFS" if ep == "PFS" else "OS", fontsize=FS["body"], color=INK, va="center")
    fig.text(kx, 0.69, "Point: mean of\n25 fits\nLine: min–max\nacross fits\n(NOT a confidence\ninterval)",
             fontsize=FS["small"], color=INK2, va="top", linespacing=1.25)
    fig.text(kx, 0.42, "Fits overlap\n(5 repeats ×\n5 folds): not\nindependent\nreplication", fontsize=FS["small"],
             color=MUTED, va="top", linespacing=1.25)
    fig.text(0.047, 0.022, "674 patients (PFS 459 events; OS 247 deaths). No pathway-level confidence intervals or "
             "significance tests exist; correlated pathways share weight in the ridge model.",
             fontsize=FS["small"], color=MUTED, va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
