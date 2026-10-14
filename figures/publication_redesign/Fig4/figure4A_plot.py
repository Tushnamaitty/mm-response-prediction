"""
Figure 4A - Discrimination for progression-free and overall survival: clinical baseline model vs
clinical baseline + RNA (C-index), with Delta C and its 95% patient-bootstrap CI.

Run from the repository root:
    python figures/publication_redesign/Fig4/figure4A_plot.py

Inputs (verified by figures/fig4_extract_data.py; read-only):
    figures/data/Fig4/fig4A_cindex_summary.csv
    figures/data/Fig4/fig4A_cindex_by_repeat.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIR = Path("figures/data/Fig4")
OUT = Path("figures/publication_redesign/Fig4/figure4A.png")

# ---- shared style (identical to Figure 1) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
RNA_LIGHT = "#f4b393"
BASE_COLOR = "#6f6d68"
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})
MINUS = "−"
FIG_W, FIG_H = 7.2, 3.35
EXPECT = {"PFS": dict(n=674, e=459, c0=0.6012024837840428, cr=0.6416971104176208, d=0.04049462663357797,
                      lo=0.01624211038710264, hi=0.0640299086216969),
          "OS": dict(n=674, e=247, c0=0.6584694105324611, cr=0.7019677535671972, d=0.04349834303473617,
                     lo=0.012143686163121094, hi=0.07449073673410617)}


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 4A data check failed: {msg}")


def panel(ax, s, reps):
    ep = s.endpoint
    for rep, g in reps.groupby("repeat"):
        y0 = g[g.model == "Clinical baseline"].c_index.iloc[0]
        y1 = g[g.model == "Clinical baseline + RNA"].c_index.iloc[0]
        ax.plot([0, 1], [y0, y1], color=GRID, lw=0.7, zorder=1)
        ax.plot([0, 1], [y0, y1], ls="none", marker="o", ms=2.6, mfc="white", mec=MUTED, mew=0.6, zorder=2)
    ax.plot([0, 1], [s.c_index_clinical_baseline, s.c_index_baseline_plus_rna], color=INK2, lw=1.2, zorder=3)
    ax.plot(0, s.c_index_clinical_baseline, "o", ms=7, mfc=BASE_COLOR, mec="white", mew=0.8, zorder=4)
    ax.plot(1, s.c_index_baseline_plus_rna, "o", ms=7, mfc=LAYER_COLOR["D"], mec="white", mew=0.8, zorder=4)
    ax.text(-0.09, s.c_index_clinical_baseline, f"{s.c_index_clinical_baseline:.3f}", ha="right", va="center",
            fontsize=FS["body"], color=INK)
    ax.text(1.09, s.c_index_baseline_plus_rna, f"{s.c_index_baseline_plus_rna:.3f}", ha="left", va="center",
            fontsize=FS["body"], color=INK, fontweight="bold")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.57, 0.77)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clinical\nbaseline", "Clinical baseline\n+ RNA"], fontsize=FS["body"], color=INK)
    ax.set_yticks([0.58, 0.62, 0.66, 0.70])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    title = "Progression-free survival (PFS)" if ep == "PFS" else "Overall survival (OS)"
    ax.set_title(f"{title}\n{int(s.n_patients)} patients · {int(s.n_events)} {s.event_type}",
                 loc="left", fontsize=FS["title"], fontweight="bold", color=INK, pad=4, linespacing=1.3)
    txt = (f"ΔC = {s.delta_c:+.3f}\n95% CI {s.delta_c_ci_lower:.3f} to {s.delta_c_ci_upper:.3f}"
           .replace("-", MINUS))
    if ep == "PFS":
        txt += f"\npermutation p = {s.permutation_p_one_sided:.3f} (0/200)"
    ax.text(0.5, 0.765, txt, ha="center", va="top", fontsize=FS["body"], color=INK, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=AXIS, lw=0.6))


def main():
    summ = pd.read_csv(DIR / "fig4A_cindex_summary.csv").set_index("endpoint")
    reps = pd.read_csv(DIR / "fig4A_cindex_by_repeat.csv")
    for ep, e in EXPECT.items():
        s = summ.loc[ep]
        require((int(s.n_patients), int(s.n_events)) == (e["n"], e["e"]), f"{ep} denominators")
        for col, k in [("c_index_clinical_baseline", "c0"), ("c_index_baseline_plus_rna", "cr"), ("delta_c", "d"),
                       ("delta_c_ci_lower", "lo"), ("delta_c_ci_upper", "hi")]:
            require(abs(float(s[col]) - e[k]) < 1e-12, f"{ep} {col}")
        r = reps[reps.endpoint == ep]
        require(len(r) == 10 and r.groupby("model").size().eq(5).all(), f"{ep} repeats")
        require(abs(r[r.model == "Clinical baseline"].c_index.mean() - e["c0"]) < 1e-12, f"{ep} repeat mean")
    require(abs(summ.loc["PFS", "permutation_p_one_sided"] - 1 / 201) < 1e-12, "PFS permutation p")
    require(pd.isna(summ.loc["OS", "permutation_p_one_sided"]), "OS must have no permutation test")
    require(reps.c_index.between(0.57, 0.72).all(), "values inside the y range")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "Clinical baseline + RNA discriminates PFS and OS better than the clinical baseline (C-index; higher is better)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.93, "EXPLORATORY · internal 5×5 cross-validation in the survival cohort "
             "(not external validation)", fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold",
             ha="left", va="top")

    for i, ep in enumerate(["PFS", "OS"]):
        ax = fig.add_axes([0.1 + i * 0.45, 0.2, 0.36, 0.56])
        s = summ.loc[ep].copy()
        s["endpoint"] = ep
        panel(ax, s, reps[reps.endpoint == ep])
        if i == 0:
            ax.set_ylabel("Harrell's C-index", fontsize=FS["body"], labelpad=2)

    fig.text(0.047, 0.075, "●", fontsize=8, color=BASE_COLOR, va="center")
    fig.text(0.064, 0.075, "/", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.074, 0.075, "●", fontsize=8, color=LAYER_COLOR["D"], va="center")
    fig.text(0.092, 0.075, "mean over 5 cross-validation repeats", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.33, 0.075, "○", fontsize=7, color=MUTED, va="center")
    fig.text(0.347, 0.075, "individual repeats (not a confidence interval)", fontsize=FS["small"], color=INK2,
             va="center")
    fig.text(0.047, 0.03, "ΔC = C(clinical baseline + RNA) − C(clinical baseline); 95% patient-bootstrap CI "
             "on fixed out-of-fold predictions. Permutation test performed for PFS only.",
             fontsize=FS["small"], color=MUTED, va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
