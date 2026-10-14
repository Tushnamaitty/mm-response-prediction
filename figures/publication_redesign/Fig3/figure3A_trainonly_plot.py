"""
Figure 4A (train-only redesign) - Overall RNA contribution: direct XGBoost Layer D vs Layer C, and a regularized RNA add-on,
evaluated in held-out validation + test patients with locked train-only models.

Run from the repository root:
    python figures/publication_redesign/Fig3/figure3A_trainonly_plot.py

Input (verified by figures/fig3ABC_trainonly_extract_data.py; read-only):
    figures/data/Fig3ABC_trainonly/fig3A_trainonly_overall.csv
Replaces figure3A.png (pooled train + validation cross-validation), which is left unchanged.
Style follows figures/publication_redesign/Fig1/figure1_plot.py. The script stops if a plotted value fails a consistency check.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("figures/data/Fig3ABC_trainonly/fig3A_trainonly_overall.csv")
OUT = Path("figures/publication_redesign/Fig3/figure3A_trainonly.png")

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})
DIRECT = dict(marker="o", ms=5.5, color=LAYER_COLOR["D"], mfc=LAYER_COLOR["D"])
ADDON = dict(marker="D", ms=5.0, color=LAYER_COLOR["C"], mfc=LAYER_COLOR["C"])
DEVREF = dict(marker="o", ms=5.0, color=MUTED, mfc="white")
FIG_W, FIG_H = 7.2, 3.6
MINUS = "−"
YLIM = (2.6, -0.6)


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 4A data check failed: {msg}")


def num(v, nd, sign=True):
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}").replace("-", MINUS)


def column(ax, rows, scale, xlim, better, title, denom, xlabel):
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    ax.axhline(1.55, color=GRID, lw=0.6)
    for y, r, style in rows:
        if not bool(r.evaluated):
            ax.text(0, y, "Not evaluated\n(run for six-class task only)", ha="center", va="center", fontsize=FS["body"], color=MUTED,
                    style="italic", bbox=dict(fc="white", ec="none", pad=1.5), zorder=3)
            continue
        ax.plot([r.ci_lower * scale, r.ci_upper * scale], [y, y], color=style["color"], lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(r.delta * scale, y, ls="none", mec=style["color"] if style["mfc"] == "white" else "white", mew=0.9, zorder=3, **style)
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=FS["body"], labelpad=2)
    ax.text(0.0, 1.21, title, transform=ax.transAxes, fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.0, 1.11, denom, transform=ax.transAxes, fontsize=FS["small"], color=MUTED, ha="left", va="bottom")
    left, right = ("RNA worse", "RNA better") if better == "higher" else ("RNA better", "RNA worse")
    ax.text(0.0, 1.01, f"← {left}", transform=ax.transAxes, ha="left", va="bottom", fontsize=FS["small"], color=INK2)
    ax.text(1.0, 1.01, f"{right} →", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS["small"], color=INK2)


def value_column(fig, rect, entries):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    ax.text(0, -0.6, "Estimate [95% CI]", fontsize=FS["small"], color=INK2, fontweight="bold", va="bottom")
    for y, txt in entries:
        ax.text(0, y, txt, fontsize=FS["small"], color=INK, va="center", ha="left", linespacing=1.25)


def main():
    d = pd.read_csv(DATA).set_index("row_id")
    need = ["direct_binary_auprc", "devcv_direct_binary_auprc", "direct_multiclass_rps", "devcv_direct_multiclass_rps",
            "addon_multiclass_rps", "addon_binary_auprc"]
    require(set(need) == set(d.index), "row set")
    for k in need[:5]:
        r = d.loc[k]
        require(bool(r.evaluated) and r.ci_lower < r.delta < r.ci_upper, f"{k} CI brackets estimate")
    require(not bool(d.loc["addon_binary_auprc", "evaluated"]), "binary add-on must be not evaluated")
    dB, dBd, dR, dRd, aR, aB = (d.loc[k] for k in ["direct_binary_auprc", "devcv_direct_binary_auprc", "direct_multiclass_rps",
                                                    "devcv_direct_multiclass_rps", "addon_multiclass_rps", "addon_binary_auprc"])
    require((int(dB.n_visit_pairs), int(dB.n_patients), int(dR.n_visit_pairs), int(dR.n_patients), int(aR.n_visit_pairs), int(aR.n_patients))
            == (2405, 213, 2689, 213, 2689, 213), "held-out denominators")
    require((int(dBd.n_visit_pairs), int(dBd.n_patients), int(dRd.n_visit_pairs), int(dRd.n_patients)) == (5857, 490, 6484, 494), "development denominators")
    require(aR.decision == "SUGGESTIVE OR INCONCLUSIVE" and aR.ci_upper > 0, "add-on verdict / CI must match the report")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "Overall RNA contribution in held-out patients: direct Layer D model vs a regularized RNA add-on",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.93, "EXPLORATORY · models developed on train patients only; evaluated once in held-out validation + test "
             "patients with RNA (post hoc; not independent validation)", fontsize=FS["small"], color=LAYER_COLOR["D"],
             fontweight="bold", ha="left", va="top")

    bottom, height = 0.27, 0.46
    axL = fig.add_axes([0.012, bottom, 0.27, height])
    axL.set_xlim(0, 1)
    axL.set_ylim(*YLIM)
    axL.axis("off")
    for y, head, sub, st in [
        (0, f"Direct: XGBoost Layer D {MINUS} Layer C", "Held-out; locked train-only models", DIRECT),
        (1, f"Regularized RNA add-on {MINUS} Layer C", "Held-out; five frozen train-only ridge fits", ADDON),
        (2.1, f"Direct D {MINUS} C, development reference", "Train-only 5×5 CV (reference, not evaluation)", DEVREF),
    ]:
        axL.plot(0.02, y - 0.1, ls="none", mec=st["color"] if st["mfc"] == "white" else "white", mew=0.9, **st)
        axL.text(0.07, y - 0.1, head, fontsize=FS["body"], color=INK, fontweight="bold" if y < 2 else "normal", va="center")
        axL.text(0.07, y + 0.12, sub, fontsize=FS["small"], color=INK2, va="top", linespacing=1.2)

    axB = fig.add_axes([0.305, bottom, 0.19, height])
    column(axB, [(0, dB, DIRECT), (1, aB, ADDON), (2.1, dBd, DEVREF)], 1, (-0.012, 0.03), "higher", "Binary improvement task",
           f"held-out {int(dB.n_visit_pairs):,} pairs · {int(dB.n_patients)} patients", "ΔAUPRC vs Layer C")
    axB.set_xticks([-0.01, 0, 0.01, 0.02, 0.03])
    axB.set_xticklabels([f"{MINUS}0.01", "0", "0.01", "0.02", "0.03"])
    value_column(fig, [0.505, bottom, 0.14, height], [
        (0, f"{num(dB.delta, 4)}\n[{num(dB.ci_lower, 4, False)}, {num(dB.ci_upper, 4, False)}]\np = {dB.p_value:.2f}"),
        (1, "—"),
        (2.1, f"{num(dBd.delta, 4)}\n[{num(dBd.ci_lower, 4, False)}, {num(dBd.ci_upper, 4, False)}]"),
    ])

    axR = fig.add_axes([0.665, bottom, 0.18, height])
    column(axR, [(0, dR, DIRECT), (1, aR, ADDON), (2.1, dRd, DEVREF)], 1e3, (-1.4, 2.2), "lower", "Six-class IMWG response task",
           f"held-out {int(dR.n_visit_pairs):,} pairs · {int(dR.n_patients)} patients", "ΔRPS $\\times10^{-3}$ vs Layer C")
    axR.set_xticks([-1, 0, 1, 2])
    axR.set_xticklabels([f"{MINUS}1", "0", "1", "2"])

    def rps_txt(r, ptxt):
        pct = f"\n{num(r.pct_change_vs_layer_c, 2)}% of Layer C RPS" if pd.notna(r.pct_change_vs_layer_c) else ""
        return f"{num(r.delta * 1e3, 2)} [{num(r.ci_lower * 1e3, 2, False)}, {num(r.ci_upper * 1e3, 2, False)}]{pct}{ptxt}"

    value_column(fig, [0.855, bottom, 0.14, height], [
        (0, rps_txt(dR, f"\np = {dR.p_value:.2f}")),
        (1, rps_txt(aR, f"\npermutation p = {aR.p_value:.3f}†")),
        (2.1, rps_txt(dRd, "")),
    ])

    fig.text(0.047, 0.15,
             "Paired on identical visit pairs. Bars: 95% patient-cluster bootstrap CIs (2,000 resamples). RPS, ranked probability score (lower is better; ΔRPS > 0 = RNA worse).\n"
             "Direct-model p: two-sided bootstrap. †One-sided permutation of RNA vectors among held-out units (108 of 10,000 as good as observed).\n"
             "Add-on rule (ΔRPS < 0, CI upper < 0, permutation p ≤ 0.05): two of three met, CI includes 0 → suggestive / inconclusive.",
             fontsize=FS["small"], color=MUTED, ha="left", va="top", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
