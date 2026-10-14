"""
Figure 4C (train-only redesign) - RNA contribution (Layer D - Layer C) by treatment class active at V(t) and by response trajectory at
V(t), in held-out validation + test patients with locked train-only models. Which strata receive inference was decided from
train-patient counts before the held-out data were read.

Run from the repository root:
    python figures/publication_redesign/Fig3/figure3C_trainonly_plot.py

Inputs (verified by figures/fig3ABC_trainonly_extract_data.py; read-only):
    figures/data/Fig3ABC_trainonly/fig3C_trainonly_strata.csv
    figures/data/Fig3ABC_trainonly/tableS1_trainonly_interactions.csv
Replaces figure3C.png (pooled train + validation CV), which is left unchanged.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

D = Path("figures/data/Fig3ABC_trainonly")
OUT = Path("figures/publication_redesign/Fig3/figure3C_trainonly.png")

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})
MINUS = "−"
FIG_W, FIG_H = 7.2, 4.35
ROWS = [  # (stratum, label, role, y)
    ("all", "All RNA-available visit pairs", "reference", 0),
    ("pi", "Proteasome inhibitor (PI)", "primary", 1.95),
    ("imid", "Immunomodulatory drug (IMiD)", "primary", 2.6),
    ("steroid", "Corticosteroid", "primary", 3.25),
    ("cd38", "Anti-CD38 antibody", "exploratory", 4.2),
    ("chemo", "Chemotherapy", "exploratory", 4.85),
    ("stable", "Stable (same IMWG category)", "primary", 6.75),
    ("worsening", "Worsening (lower category)", "primary", 7.4),
    ("improving", "Improving (higher category)", "secondary", 8.35),
    ("no_history", "No previous visit", "secondary", 9.0),
]
HEADERS = [(1.02, "Treatment class active at V(t)"), (5.82, "Response trajectory at V(t)")]
SUBHEADERS = [(1.48, "primary · Holm-adjusted"), (3.73, "exploratory · BH-adjusted"),
              (6.28, "primary · Holm-adjusted"), (7.88, "secondary · BH-adjusted")]
DIVIDERS = [0.55, 5.35]
YLIM = (9.45, -0.75)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 4C data check failed: {msg}")


def num(v, nd, sign=True):
    v = 0.0 if round(v, nd) == 0 else v           # avoid "-0.000"
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}").replace("-", MINUS)


def get(d, stratum, task, metric):
    s = d[(d.stratum == stratum) & (d.task == task) & (d.metric == metric)]
    return None if s.empty else s.iloc[0]


def forest(ax, d, task, metric, scale, xlim, better, title, xlabel):
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    for yy in DIVIDERS:
        ax.axhline(yy, color=GRID, lw=0.6)
    for st, _, role, y in ROWS:
        r = get(d, st, task, metric)
        if r is None:
            ax.text(0, y, "not evaluated‡", ha="center", va="center", fontsize=FS["small"], color=MUTED, style="italic",
                    bbox=dict(fc="white", ec="none", pad=1.2), zorder=3)
            continue
        sig = bool(r.significant_after_adjustment)
        col = INK2 if role == "reference" else LAYER_COLOR["D"]
        ax.plot([r.ci_lower * scale, r.ci_upper * scale], [y, y], color=col, lw=1.3, solid_capstyle="round", zorder=2)
        ax.plot(r.delta * scale, y, ls="none", marker="o", ms=5.2, mec=col, mfc=col if (sig or role == "reference") else "white", mew=1.0, zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=FS["body"], labelpad=2)
    ax.text(0.0, 1.07, title, transform=ax.transAxes, fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="bottom")
    left, right = ("RNA worse", "RNA better") if better == "higher" else ("RNA better", "RNA worse")
    ax.text(0.0, 1.008, f"← {left}", transform=ax.transAxes, ha="left", va="bottom", fontsize=FS["small"], color=INK2)
    ax.text(1.0, 1.008, f"{right} →", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS["small"], color=INK2)


def values(fig, rect, d, task, metric, scale, nd):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    ax.text(0, -0.75, "Estimate [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")
    ax.text(0.97, -0.75, "patients", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom", ha="right")
    for st, _, role, y in ROWS:
        r = get(d, st, task, metric)
        if r is None:
            continue
        sig = bool(r.significant_after_adjustment)
        est = f"{num(r.delta * scale, nd)} [{num(r.ci_lower * scale, nd, False)}, {num(r.ci_upper * scale, nd, False)}]"
        if sig:
            est += f"   {'Holm' if str(r.adjustment).startswith('Holm') else 'BH'} p = {r.adjusted_p:.3f}"
        ax.text(0, y, est, fontsize=FS["small"], color=INK if sig or role == "reference" else INK2, va="center", fontweight="bold" if sig else "normal")
        ax.text(0.97, y, f"{int(r.n_patients)}", fontsize=FS["small"] - 0.3, color=MUTED, va="center", ha="right")


def main():
    d = pd.read_csv(D / "fig3C_trainonly_strata.csv")
    it = pd.read_csv(D / "tableS1_trainonly_interactions.csv")
    prim = d[d.metric.isin(["auprc", "rps"]) & d.stratum.isin([s for s, *_ in ROWS])]
    require(not prim.significant_after_adjustment.any(), "text assumes no within-stratum primary-metric test is significant after adjustment")
    require(get(d, "chemo", "multiclass", "rps") is None, "six-class chemotherapy stratum must be absent")
    require(get(d, "all", "multiclass", "rps").n_patients == 213 and get(d, "all", "binary", "auprc").n_patients == 213, "held-out patients")
    for st, _, role, _ in ROWS:
        for task, metric in [("binary", "auprc"), ("multiclass", "rps")]:
            r = get(d, st, task, metric)
            if r is None or role == "reference":
                continue
            require(str(r.adjustment).startswith("Holm" if role == "primary" else "BH"), f"{st} {task} adjustment is {r.adjustment}")
    si = it[(it.task == "multiclass") & (it.metric == "rps") & it.contrast.isin(["worsening_minus_stable", "improving_minus_stable"])].set_index("contrast")
    require(len(si) == 2 and (si.adjusted_p < 0.05).all(), "state-interaction rows")
    w, i_ = si.loc["worsening_minus_stable"], si.loc["improving_minus_stable"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.99, "C", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.988, "RNA contribution by treatment and response trajectory in held-out patients (XGBoost Layer D − Layer C)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.95, "EXPLORATORY · locked train-only models evaluated in held-out validation + test patients with RNA "
             "(post hoc; not independent validation)", fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")
    fig.text(0.047, 0.915, "No stratum shows a significant RNA effect after adjustment (all points open). Two exploratory six-class interaction contrasts "
             f"meet BH p < 0.05:\nworsening − stable {num(w.interaction_a_minus_b * 1e3, 2)} $\\times10^{{-3}}$ (BH p = {w.adjusted_p:.3f}) and improving − stable "
             f"{num(i_.interaction_a_minus_b * 1e3, 2)} $\\times10^{{-3}}$ (BH p = {i_.adjusted_p:.3f}); see Table S1.",
             fontsize=FS["small"], color=INK, ha="left", va="top", linespacing=1.3)

    bottom, height = 0.17, 0.6
    axL = fig.add_axes([0.012, bottom, 0.2, height])
    axL.set_xlim(0, 1)
    axL.set_ylim(*YLIM)
    axL.axis("off")
    for y, h in HEADERS:
        axL.text(0.0, y, h, fontsize=FS["body"], color=INK, fontweight="bold", va="center")
    for y, h in SUBHEADERS:
        axL.text(0.04, y, h, fontsize=FS["small"] - 0.3, color=MUTED, va="center", style="italic")
    for _, lab, role, y in ROWS:
        axL.text(0.0 if role == "reference" else 0.04, y - (0.1 if role == "reference" else 0), lab, fontsize=FS["body"], color=INK, va="center",
                 fontweight="bold" if role == "reference" else "normal")
    axL.text(0.0, 0.36, "unadjusted reference", fontsize=FS["small"] - 0.3, color=MUTED, va="center", style="italic")

    axB = fig.add_axes([0.235, bottom, 0.17, height])
    forest(axB, d, "binary", "auprc", 1, (-0.065, 0.08), "higher", "Binary improvement: ΔAUPRC", "ΔAUPRC, Layer D vs Layer C")
    axB.set_xticks([-0.05, 0, 0.05])
    axB.set_xticklabels([f"{MINUS}0.05", "0", "0.05"])
    values(fig, [0.415, bottom, 0.185, height], d, "binary", "auprc", 1, 3)

    axR = fig.add_axes([0.62, bottom, 0.165, height])
    forest(axR, d, "multiclass", "rps", 1e3, (-6.2, 6.6), "lower", "Six-class IMWG response: ΔRPS", "ΔRPS $\\times10^{-3}$, Layer D vs Layer C")
    axR.set_xticks([-4, 0, 4])
    axR.set_xticklabels([f"{MINUS}4", "0", "4"])
    values(fig, [0.795, bottom, 0.2, height], d, "multiclass", "rps", 1e3, 2)

    ky = 0.085
    fig.text(0.047, ky, "●", fontsize=8, color=LAYER_COLOR["D"], va="center")
    fig.text(0.064, ky, "adjusted p < 0.05 (none)", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.21, ky, "○", fontsize=8, color=LAYER_COLOR["D"], va="center")
    fig.text(0.227, ky, "not significant after adjustment", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.42, ky, "Bars: 95% bootstrap CIs. RPS: lower is better (ΔRPS > 0 = RNA worse). BH, Benjamini–Hochberg.", fontsize=FS["small"],
             color=MUTED, va="center")
    fig.text(0.047, 0.045, "Treatment strata overlap. Strata are small in the 213 held-out patients (patients column), so intervals are wide.\n"
             "‡Frozen support rule: too few CR/sCR outcomes for six-class chemotherapy.", fontsize=FS["small"], color=MUTED, ha="left",
             va="center", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
