"""
Figure 3C - RNA contribution (XGBoost Layer D - Layer C) by treatment class active at V(t) and by
IMWG response trajectory at V(t).

Run from the repository root:
    python figures/publication_redesign/Fig3/figure3C_plot.py

Input (verified by figures/fig3_extract_data.py; read-only):
    figures/data/Fig3/fig3C_treatment_and_trajectory_strata.csv

Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted
value differs from the value verified at extraction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/Fig3/fig3C_treatment_and_trajectory_strata.csv")
OUT = Path("figures/publication_redesign/Fig3/figure3C.png")

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
FIG_W, FIG_H = 7.2, 4.1
D3 = "Direction 3: treatment class active at V(t)"
D4 = "Direction 4: IMWG response trajectory at V(t)"

# (source, stratum, label, role, y)
ROWS = [
    (D3, "all", "All RNA-available visit pairs", "reference", 0),
    (D3, "on_pi", "Proteasome inhibitor (PI)", "primary", 1.95),
    (D3, "on_imid", "Immunomodulatory drug (IMiD)", "primary", 2.6),
    (D3, "on_steroid", "Corticosteroid", "primary", 3.25),
    (D3, "on_cd38", "Anti-CD38 antibody", "exploratory", 4.2),
    (D3, "on_chemo", "Chemotherapy", "exploratory", 4.85),
    (D4, "stable", "Stable (same IMWG category)", "primary", 6.75),
    (D4, "worsening", "Worsening (lower category)", "primary", 7.4),
    (D4, "improving", "Improving (higher category)", "secondary", 8.35),
    (D4, "no_history", "No previous visit", "secondary", 9.0),
]
HEADERS = [(1.02, "Treatment class active at V(t)"), (5.82, "Response trajectory at V(t)")]
SUBHEADERS = [(1.48, "primary · Holm-adjusted"), (3.73, "exploratory · BH-adjusted"),
              (6.28, "primary · Holm-adjusted"), (7.88, "secondary · BH-adjusted")]
DIVIDERS = [0.55, 5.35]
YLIM = (9.45, -0.75)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 3C data check failed: {msg}")


def num(v, nd, sign=True):
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}").replace("-", MINUS)


def get(d, src, stratum, task, metric):
    s = d[(d.source_analysis == src) & (d.stratum == stratum) & (d.task == task) & (d.metric == metric)]
    return None if s.empty else s.iloc[0]


def forest(ax, d, task, metric, scale, xlim, better, title, xlabel):
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    for yy in DIVIDERS:
        ax.axhline(yy, color=GRID, lw=0.6)
    for src, st, _, role, y in ROWS:
        r = get(d, src, st, task, metric)
        if r is None:
            ax.text(0, y, "not evaluated‡", ha="center", va="center", fontsize=FS["small"], color=MUTED,
                    style="italic", bbox=dict(fc="white", ec="none", pad=1.2), zorder=3)
            continue
        sig = bool(r.significant_after_adjustment)
        col = INK2 if role == "reference" else LAYER_COLOR["D"]
        ax.plot([r.ci_lower * scale, r.ci_upper * scale], [y, y], color=col, lw=1.3, solid_capstyle="round",
                zorder=2)
        ax.plot(r.delta * scale, y, ls="none", marker="o", ms=5.2, mec=col,
                mfc=col if (sig or role == "reference") else "white", mew=1.0, zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=FS["body"], labelpad=2)
    ax.text(0.0, 1.07, title, transform=ax.transAxes, fontsize=FS["title"], fontweight="bold",
            color=INK, ha="left", va="bottom")
    left, right = ("RNA worse", "RNA better") if better == "higher" else ("RNA better", "RNA worse")
    ax.text(0.0, 1.008, f"← {left}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS["small"], color=INK2)
    ax.text(1.0, 1.008, f"{right} →", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS["small"], color=INK2)


def values(fig, rect, d, task, metric, scale, nd):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    ax.text(0, -0.75, "Estimate [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")
    for src, st, _, role, y in ROWS:
        r = get(d, src, st, task, metric)
        if r is None:
            continue
        sig = bool(r.significant_after_adjustment)
        est = f"{num(r.delta * scale, nd)} [{num(r.ci_lower * scale, nd, False)}, {num(r.ci_upper * scale, nd, False)}]"
        if sig:   # adjusted p shown only for findings significant after adjustment
            method = "Holm" if str(r.adjustment).startswith("Holm") else "BH"
            est += f"   {method} p = {r.adjusted_p:.3f}"
        ax.text(0, y, est, fontsize=FS["small"], color=INK if sig or role == "reference" else INK2,
                va="center", fontweight="bold" if sig else "normal")


def main():
    d = pd.read_csv(DATA)
    for st, p in [("on_pi", 0.044978), ("on_imid", 0.041979), ("on_steroid", 0.041979)]:
        r = get(d, D3, st, "multiclass", "rps")
        require(abs(round(r.adjusted_p, 6) - p) < 1e-9 and bool(r.significant_after_adjustment), f"{st} RPS")
    r = get(d, D4, "stable", "multiclass", "rps")
    require(abs(round(r.adjusted_p, 6) - 0.001999) < 1e-6 and bool(r.significant_after_adjustment), "stable RPS")
    require(get(d, D3, "on_chemo", "multiclass", "rps") is None, "chemo six-class must be absent")
    require(not d[(d.metric == "auprc") & d.stratum.isin([s for _, s, _, _, _ in ROWS])].significant_after_adjustment.any(),
            "no binary AUPRC stratum significant")
    require(abs(get(d, D3, "all", "multiclass", "rps").delta - 0.0007515795024) < 1e-10, "reference row")
    for src, st, _, role, _ in ROWS:   # adjustment labels in the subheaders must match the data
        for task, metric in [("binary", "auprc"), ("multiclass", "rps")]:
            r = get(d, src, st, task, metric)
            if r is None or role == "reference":
                continue
            want = "Holm" if role == "primary" else "BH"
            require(str(r.adjustment).startswith(want), f"{st} {task} adjustment is {r.adjustment}")
    sig_rps = d[(d.metric == "rps") & d.significant_after_adjustment]
    lo_d, hi_d = sig_rps.delta.min(), sig_rps.delta.max()
    lo_p, hi_p = sig_rps.pct_change_vs_layer_c.min(), sig_rps.pct_change_vs_layer_c.max()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.99, "C", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.988, "RNA contribution by treatment and response trajectory (XGBoost Layer D − Layer C)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.948, "EXPLORATORY · 5×5 patient-grouped cross-validation, train + validation "
             "patients (not the held-out test set)",
             fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")
    fig.text(0.047, 0.91, "No stratum improved with RNA. Filled points: six-class RPS significantly worse with RNA "
             f"after adjustment (ΔRPS {num(lo_d * 1e3, 2)} to {num(hi_d * 1e3, 2)} $\\times10^{{-3}}$; "
             f"{lo_p:.1f}–{hi_p:.1f}% of Layer C RPS).",
             fontsize=FS["small"], color=INK, ha="left", va="top")

    bottom, height = 0.17, 0.62
    axL = fig.add_axes([0.012, bottom, 0.2, height])
    axL.set_xlim(0, 1)
    axL.set_ylim(*YLIM)
    axL.axis("off")
    for y, h in HEADERS:
        axL.text(0.0, y, h, fontsize=FS["body"], color=INK, fontweight="bold", va="center")
    for y, h in SUBHEADERS:
        axL.text(0.04, y, h, fontsize=FS["small"] - 0.3, color=MUTED, va="center", style="italic")
    for _, _, lab, role, y in ROWS:
        axL.text(0.0 if role == "reference" else 0.04, y - (0.1 if role == "reference" else 0), lab, fontsize=FS["body"], color=INK, va="center",
                 fontweight="bold" if role == "reference" else "normal")
    axL.text(0.0, 0.36, "unadjusted reference", fontsize=FS["small"] - 0.3, color=MUTED, va="center", style="italic")

    axB = fig.add_axes([0.235, bottom, 0.17, height])
    forest(axB, d, "binary", "auprc", 1, (-0.035, 0.035), "higher", "Binary improvement: ΔAUPRC",
           "ΔAUPRC, Layer D vs Layer C")
    axB.set_xticks([-0.03, 0, 0.03])
    axB.set_xticklabels([f"{MINUS}0.03", "0", "0.03"])
    values(fig, [0.415, bottom, 0.18, height], d, "binary", "auprc", 1, 4)

    axR = fig.add_axes([0.615, bottom, 0.17, height])
    forest(axR, d, "multiclass", "rps", 1e3, (-2.6, 6.4), "lower", "Six-class IMWG response: ΔRPS",
           "ΔRPS $\\times10^{-3}$, Layer D vs Layer C")
    axR.set_xticks([-2, 0, 2, 4, 6])
    axR.set_xticklabels([f"{MINUS}2", "0", "2", "4", "6"])
    values(fig, [0.795, bottom, 0.2, height], d, "multiclass", "rps", 1e3, 2)

    ky = 0.072
    fig.text(0.047, ky, "●", fontsize=8, color=LAYER_COLOR["D"], va="center")
    fig.text(0.064, ky, "adjusted p < 0.05", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.19, ky, "○", fontsize=8, color=LAYER_COLOR["D"], va="center")
    fig.text(0.207, ky, "not significant after adjustment", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.42, ky, "Bars: 95% bootstrap CIs. RPS: lower is better (ΔRPS > 0 = RNA worse). "
             "BH, Benjamini–Hochberg.", fontsize=FS["small"], color=MUTED, va="center")
    fig.text(0.047, 0.035, "Treatment strata overlap. ‡Too few CR/sCR outcomes for the six-class chemotherapy "
             "stratum.", fontsize=FS["small"], color=MUTED, ha="left", va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
