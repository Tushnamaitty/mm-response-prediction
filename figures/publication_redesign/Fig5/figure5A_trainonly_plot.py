"""
Figure 6A (train-only redesign) - Mean PFS vs mean OS ridge-Cox coefficients of the 50 Hallmark RNA pathways, from 25 overlapping
internal cross-validation fits within the 473 train patients (held-out patients are not used).

Run from the repository root:
    python figures/publication_redesign/Fig5/figure5A_trainonly_plot.py

Inputs (verified by figures/fig45_trainonly_extract_data.py; read-only):
    figures/data/Fig45_trainonly/fig5A_trainonly_coefficients.csv
    figures/data/Fig45_trainonly/fig5_trainonly_summary_stats.json
Replaces figure5A.png (50 pathways fitted on all 674 patients; 19 concordant), which is left unchanged.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIR = Path("figures/data/Fig45_trainonly")
OUT = Path("figures/publication_redesign/Fig5/figure5A_trainonly.png")

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
FIG_W, FIG_H = 7.2, 3.8
LIM = (-0.14, 0.125)
# label positions (data coordinates) in the empty upper-left region; leader lines connect to the points
LABELS = {
    "E2F Targets": ((-0.015, 0.118), "right"),
    "Spermatogenesis": ((-0.015, 0.100), "right"),
    "G2-M Checkpoint": ((-0.015, 0.082), "right"),
    "Myc Targets V1": ((0.075, 0.034), "left"),
    "Mitotic Spindle": ((-0.015, 0.052), "right"),
    "Notch Signaling": ((-0.035, -0.112), "left"),
}
FAMILY = ["E2F Targets", "G2-M Checkpoint", "Myc Targets V1", "Mitotic Spindle", "Spermatogenesis"]


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 6A data check failed: {msg}")


def main():
    a = pd.read_csv(DIR / "fig5A_trainonly_coefficients.csv")
    st = json.load(open(DIR / "fig5_trainonly_summary_stats.json"))
    require(len(a) == 50 and int(a.concordant_both_endpoints.sum()) == 12, "50 pathways / 12 concordant")
    require(sorted(a[a.figure3D_family_member].pathway_display_name) == sorted(FAMILY), "Figure 4D family flagged")
    fam_conc = a[a.figure3D_family_member].set_index("pathway_display_name").concordant_both_endpoints.to_dict()
    require(fam_conc == {"E2F Targets": True, "G2-M Checkpoint": True, "Myc Targets V1": True, "Mitotic Spindle": True, "Spermatogenesis": False},
            "only Spermatogenesis of the family is not concordant")
    require(set(LABELS) <= set(a.pathway_display_name), "labelled pathways exist")
    r = st["pearson_r_mean_beta_pfs_vs_os"]
    require(abs(r - 0.8120231) < 5e-7, "Pearson r")
    require(a.pfs_beta_mean.between(*LIM).all() and a.os_beta_mean.between(*LIM).all(), "points inside limits")
    hi = a[a.concordance_group == "higher hazard in both"]
    lo = a[a.concordance_group == "lower hazard in both"]
    nc = a[~a.concordant_both_endpoints]
    require((len(hi), len(lo), len(nc)) == (9, 3, 38), "9 / 3 / 38 split")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "RNA pathway coefficients for PFS and OS (train-patient fits only)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.935, "EXPLORATORY · ridge Cox (clinical baseline + RNA) · mean over 25 internal cross-validation fits within the "
             "473 train patients · association, not causation", fontsize=FS["small"],
             color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")

    ax = fig.add_axes([0.1, 0.13, 2.65 / FIG_W, 2.65 / FIG_H])
    ax.axhline(0, color=AXIS, lw=0.7, zorder=1)
    ax.axvline(0, color=AXIS, lw=0.7, zorder=1)
    ax.plot(LIM, LIM, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.scatter(nc.pfs_beta_mean, nc.os_beta_mean, s=14, c="white", edgecolors=MUTED, linewidths=0.7, zorder=2)
    ax.scatter(hi.pfs_beta_mean, hi.os_beta_mean, s=20, c=LAYER_COLOR["D"], edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(lo.pfs_beta_mean, lo.os_beta_mean, s=20, c=LAYER_COLOR["C"], edgecolors="white", linewidths=0.5, zorder=3)
    fam = a[a.figure3D_family_member]
    ax.scatter(fam.pfs_beta_mean, fam.os_beta_mean, s=58, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=4)
    for name, ((tx, ty), ha) in LABELS.items():
        row = a[a.pathway_display_name == name].iloc[0]
        ax.annotate(name, xy=(row.pfs_beta_mean, row.os_beta_mean), xytext=(tx, ty), ha=ha, va="center",
                    fontsize=FS["small"], color=INK,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.5, shrinkA=1.5, shrinkB=3.5), zorder=5)
    ax.text(LIM[1] - 0.003, 0.004, "higher hazard\nin both", ha="right", va="bottom", fontsize=FS["small"], color=INK2, style="italic", linespacing=1.1)
    ax.text(LIM[0] + 0.003, -0.004, "lower hazard\nin both", ha="left", va="top", fontsize=FS["small"], color=INK2, style="italic", linespacing=1.1)
    ax.text(LIM[1] - 0.004, -0.085, f"Pearson r = {r:.2f}\n(50 pathways; descriptive)", ha="right", va="center",
            fontsize=FS["small"], color=INK, linespacing=1.2)
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ticks = [-0.1, -0.05, 0, 0.05, 0.1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    lab = [f"{t:g}".replace("-", MINUS) for t in ticks]
    ax.set_xticklabels(lab)
    ax.set_yticklabels(lab)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("PFS coefficient β (mean of 25 fits)", fontsize=FS["body"], labelpad=2)
    ax.set_ylabel("OS coefficient β (mean of 25 fits)", fontsize=FS["body"], labelpad=2)

    kx, ky = 0.58, 0.74
    items = [(LAYER_COLOR["D"], MUTED, f"Higher hazard in both endpoints ({len(hi)})", False),
             (LAYER_COLOR["C"], "white", f"Lower hazard in both endpoints ({len(lo)})", False),
             ("white", MUTED, f"Not consistently directed in both ({len(nc)})", False),
             ("white", INK, "Figure 4D five-pathway family (4 of 5 concordant)", True)]
    for i, (fc, ec, txt, ring) in enumerate(items):
        y = ky - i * 0.07
        if ring:
            fig.lines.append(plt.Line2D([kx], [y], marker="o", ms=7, mfc="none", mec=INK, mew=0.8, ls="none", transform=fig.transFigure))
        else:
            fig.lines.append(plt.Line2D([kx], [y], marker="o", ms=5, mfc=fc, mec=ec if fc == "white" else "white", mew=0.7, ls="none", transform=fig.transFigure))
        fig.text(kx + 0.02, y, txt, fontsize=FS["body"], color=INK, va="center")
    fig.text(kx, ky - 0.33, "Concordant: same sign in all 25 fits for both PFS and OS\n(12 pathways: 9 higher hazard, 3 lower hazard). Spermatogenesis\n"
             "(Figure 4D family) is not concordant.", fontsize=FS["small"], color=INK2, va="top", linespacing=1.3)
    fig.text(kx, ky - 0.5, "β: log-hazard per training-fold SD of the pathway score;\nβ > 0 = higher hazard. Dashed line: PFS β = OS β.",
             fontsize=FS["small"], color=INK2, va="top", linespacing=1.3)
    fig.text(kx, ky - 0.63, "Fits overlap (5 repeats × 5 folds; about 80% shared\ntraining patients): not independent replication. Held-out\n"
             "patients were not used.", fontsize=FS["small"], color=MUTED, va="top", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
