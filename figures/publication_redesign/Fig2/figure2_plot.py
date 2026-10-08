"""
Figure 2 - Next-visit prediction performance across nested information layers A-D.

Run from the repository root:
    python figures/publication_redesign/Fig2/figure2_plot.py

Inputs (verified in Step 2; read-only here):
    figures/data/Fig2/fig2_main_cis_1000boot.csv        24 locked-test point estimates + 1,000-resample 95% CIs
    figures/data/Fig2/fig2_paired_deltas_full_test.csv  shared-resample paired deltas (full locked test)
    figures/data/Fig2/fig2_point_estimates_from_json.csv  committed clean-rerun JSON values (cross-check)
    figures/data/Fig2/fig2_populations.csv              rows / patients / positives per task and split

Outputs (three independent PNGs): 2A binary AUPRC, 2B six-class macro-F1, 2C paired change from adding longitudinal history (B-A),
treatment context (C-B) and RNA pathway scores (D-C). Full locked test set only; the RNA-available-subset
analyses are separate and not drawn here. No algorithm is ranked or called best.

Style block follows figures/publication_redesign/Fig1/figure1_plot.py (shared across Figures 1-5).
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

DATA = Path("figures/data/Fig2")
OUT_DIR = Path("figures/publication_redesign/Fig2")
STEM = "figure2"

# ----------------------------------------------------------------------------------------------
# Shared style (identical to Figure 1)
# ----------------------------------------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
RNA_LIGHT = "#f4b393"
ALGO_LABEL = {"logistic": "Logistic Regression", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
ALGO_MARKER = {"logistic": "o", "lightgbm": "s", "xgboost": "D"}
ALGO_MS = {"logistic": 4.6, "lightgbm": 4.3, "xgboost": 3.9}
ALGOS = ["logistic", "lightgbm", "xgboost"]
DODGE = {"logistic": -0.24, "lightgbm": 0.0, "xgboost": 0.24}
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
FIG_W, FIG_H = 7.2, 6.65

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"],
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": SURFACE,
})


def y_in(top_in):
    return 1 - top_in / FIG_H


def add_ax(fig, left, top_in, width, height_in):
    return fig.add_axes([left, y_in(top_in + height_in), width, height_in / FIG_H])


def panel_letter(fig, letter, top_in, x=0.012):
    fig.text(x, y_in(top_in), letter, fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)


def heading(fig, text, top_in, x, sub=None):
    fig.text(x, y_in(top_in), text, fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    if sub:
        fig.text(x, y_in(top_in + 0.16), sub, fontsize=FS["body"], color=INK2, ha="left", va="top")


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 2 data check failed: {msg}")


# ----------------------------------------------------------------------------------------------
# Data + verification
# ----------------------------------------------------------------------------------------------
def load():
    ci = pd.read_csv(DATA / "fig2_main_cis_1000boot.csv")
    dl = pd.read_csv(DATA / "fig2_paired_deltas_full_test.csv")
    js = pd.read_csv(DATA / "fig2_point_estimates_from_json.csv")
    pop = pd.read_csv(DATA / "fig2_populations.csv")

    require(len(ci) == 24 and len(dl) == 24, "expected 24 CI rows and 24 delta rows")
    require((ci.n_resamples_used == 1000).all() and (dl.n_resamples_used == 1000).all(), "not 1,000 resamples")
    require(set(ci.n_patients) == {155}, "full-test CIs must use 155 patients")
    m = ci.merge(js, on=["task", "layer", "algorithm"], suffixes=("", "_json"))
    require(len(m) == 24 and (m.point_estimate - m.point_estimate_json).abs().max() < 1e-9,
            "bootstrap point estimates differ from committed clean-rerun JSONs")
    for task, n in [("binary", 1930), ("multiclass", 2079)]:
        require(set(ci[ci.task == task].n_rows) == {n} and set(dl[dl.task == task].n_rows) == {n},
                f"{task} evaluation population is not {n} pairs")
    t = pop[(pop.task == "binary") & (pop.split == "test")].iloc[0]
    require(int(t.test_positives_improved) == 353 and int(t.n_visit_pairs) == 1930, "binary test prevalence changed")
    prevalence = int(t.test_positives_improved) / int(t.n_visit_pairs)
    require(round(prevalence, 3) == 0.183, "prevalence != 0.183")
    # deltas must equal differences of the verified point estimates
    pe = ci.set_index(["task", "layer", "algorithm"]).point_estimate
    for _, r in dl.iterrows():
        hi, lo = r.comparison.split(" - ")
        require(abs(r.observed_diff - (pe[(r.task, hi, r.algorithm)] - pe[(r.task, lo, r.algorithm)])) < 1e-9,
                f"delta {r.task} {r.algorithm} {r.comparison} inconsistent with point estimates")
    # statements printed on the figure
    rna = dl[dl.comparison == "D - C"]
    require(len(rna) == 6 and ((rna.ci_lower < 0) & (rna.ci_upper > 0)).all(), "D-C CIs no longer all include 0")
    mx = {task: float(rna[rna.task == task].observed_diff.abs().max()) for task in ("binary", "multiclass")}
    require(mx["binary"] <= 0.0085 and mx["multiclass"] <= 0.0065, "D-C magnitude differs from printed statement")
    return ci, dl, prevalence, mx


# ----------------------------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------------------------
LAYER_TICK = {"A": "A\nCurrent clinical\nstate", "B": "B\nA + longitudinal\nhistory",
              "C": "C\nB + treatment\ncontext", "D": "D\nC + RNA pathway\nscores"}


def layer_panel(ax, ci, task, ylim, ylabel, prevalence=None):
    sub = ci[ci.task == task]
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for g in ALGOS:
        d = sub[sub.algorithm == g].set_index("layer").loc[list("ABCD")]
        xs = np.arange(4) + DODGE[g]
        ax.plot(xs, d.point_estimate, color=MUTED, lw=0.6, zorder=2, alpha=0.8)
        ax.vlines(xs, d.ci_lower, d.ci_upper, color=INK2, lw=0.7, zorder=3)
        for x, (layer, r) in zip(xs, d.iterrows()):
            ax.plot([x], [r.point_estimate], marker=ALGO_MARKER[g], ms=ALGO_MS[g], mfc=LAYER_COLOR[layer],
                    mec=INK, mew=0.5, ls="none", zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LAYER_TICK[l] for l in "ABCD"], fontsize=FS["small"], linespacing=1.15)
    for lab, l in zip(ax.get_xticklabels(), "ABCD"):
        lab.set_color(INK)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel(ylabel, fontsize=FS["body"], color=INK2)
    if prevalence is not None:
        ax.axhline(prevalence, color=INK2, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
        ax.text(-0.5, prevalence + 0.006, f"Test prevalence = {prevalence:.3f}",
                fontsize=FS["small"], color=INK2, va="bottom", ha="left",
                bbox=dict(fc="white", ec="none", pad=0.8))


def delta_panel(ax, dl, task, show_ylabels):
    sub = dl[dl.task == task]
    comps = [("B - A", "B"), ("C - B", "C"), ("D - C", "D")]
    ypos = {c: 2 - i for i, (c, _) in enumerate(comps)}
    ax.set_xlim(-0.055, 0.055)
    ax.set_ylim(-0.6, 2.6)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # RNA step highlighted (small, not significant)
    ax.add_patch(Rectangle((-0.055, -0.5), 0.11, 1.0, facecolor=RNA_LIGHT, alpha=0.22, lw=0, zorder=0))
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    for c, layer in comps:
        for g in ALGOS:
            r = sub[(sub.comparison == c) & (sub.algorithm == g)].iloc[0]
            y = ypos[c] - DODGE[g] * 1.0
            ax.hlines(y, r.ci_lower, r.ci_upper, color=INK2, lw=0.8, zorder=3)
            ax.plot([r.observed_diff], [y], marker=ALGO_MARKER[g], ms=ALGO_MS[g] + 0.2, mfc=LAYER_COLOR[layer],
                    mec=INK, mew=0.5, ls="none", zorder=4)
    ax.set_yticks([2, 1, 0])
    if show_ylabels:
        ax.set_yticklabels(["Longitudinal\n(B − A)", "Treatment\n(C − B)", "RNA\n(D − C)"],
                           fontsize=FS["body"], linespacing=1.15)
        for lab, (_, layer) in zip(ax.get_yticklabels(), comps):
            lab.set_color(INK)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
    ax.set_xticklabels(["−0.04", "−0.02", "0", "+0.02", "+0.04"])


# ----------------------------------------------------------------------------------------------
def new_fig(w, h):
    """Each Figure 2 panel is its own figure; y_in/add_ax read the current size."""
    global FIG_W, FIG_H
    FIG_W, FIG_H = w, h
    return plt.figure(figsize=(w, h), dpi=100)


def algo_legend(fig, x, top_in, loc="upper left"):
    handles = [Line2D([], [], marker=ALGO_MARKER[g], ms=ALGO_MS[g] + 0.8, mfc="#d9d8d2", mec=INK, mew=0.6, ls="none",
                      label=ALGO_LABEL[g]) for g in ALGOS]
    return fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x, y_in(top_in)), ncol=3, frameon=False,
                      fontsize=FS["body"], handletextpad=0.3, columnspacing=1.5, borderaxespad=0)


def caption(fig, text, top_in, width_chars, x=0.02):
    fig.text(x, y_in(top_in), "\n".join(textwrap.wrap(text, width_chars)), fontsize=FS["small"], color=MUTED,
             ha="left", va="top", linespacing=1.35)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=600)
    plt.close(fig)
    print("saved", OUT_DIR / name)


def fig_layer(ci, task, letter, title, sub, ylabel, ylim, fname, note, prevalence=None, baseline_note=False):
    fig = new_fig(4.3, 4.35)
    panel_letter(fig, letter, 0.06, x=0.02)
    heading(fig, title, 0.08, 0.095, sub)
    algo_legend(fig, 0.095, 0.50)
    ax = add_ax(fig, 0.16, 0.85, 0.80, 2.3)
    layer_panel(ax, ci, task, ylim, ylabel, prevalence)
    if baseline_note:
        ax.text(1.5, 0.375, "Logistic Regression: linear baseline", fontsize=FS["small"], color=INK2,
                ha="center", va="center", style="italic")
    caption(fig, note, 3.78, 88)
    save(fig, fname)


def fig_delta(dl):
    fig = new_fig(6.6, 3.95)
    panel_letter(fig, "C", 0.06, x=0.012)
    heading(fig, "Change when an information layer is added (paired \u0394)", 0.08, 0.06)
    fig.text(0.06, y_in(0.26), "RNA (D\u2212C): all 95% CIs include 0", fontsize=FS["body"], color="#b24a1a",
             fontweight="bold", ha="left", va="top")
    algo_legend(fig, 0.06, 0.50)
    ax_top = 1.12
    fig.text(0.40, y_in(ax_top - 0.10), "Binary improvement (AUPRC)", fontsize=FS["body"], fontweight="bold",
             color=INK, ha="center", va="bottom")
    fig.text(0.79, y_in(ax_top - 0.10), "Six-class response (macro-F1)", fontsize=FS["body"], fontweight="bold",
             color=INK, ha="center", va="bottom")
    ax1 = add_ax(fig, 0.215, ax_top, 0.35, 1.6)
    delta_panel(ax1, dl, "binary", True)
    ax1.set_xlabel("\u0394 AUPRC", fontsize=FS["body"], labelpad=3)
    ax2 = add_ax(fig, 0.615, ax_top, 0.35, 1.6)
    delta_panel(ax2, dl, "multiclass", False)
    ax2.set_xlabel("\u0394 macro-F1", fontsize=FS["body"], labelpad=3)
    caption(fig, "\u0394 = metric with the larger layer \u2212 metric with the smaller layer (B \u2212 A: add longitudinal history; "
                 "C \u2212 B: add treatment context; D \u2212 C: add RNA pathway scores). Marker colour = layer added. Whiskers: paired 95% "
                 "patient-bootstrap CI (1,000 resamples), unadjusted. Locked test set, 155 patients; binary and six-class metrics use "
                 "different evaluation pairs (1,930 vs 2,079).", 3.30, 150, x=0.06)
    save(fig, "figure2C_incremental_delta.png")


def main():
    ci, dl, prevalence, mx = load()  # verifies all numbers drawn below
    fig_layer(ci, "binary", "A", "Binary improvement: AUPRC", "1,930 visit pairs \u00b7 155 test patients (locked test set)",
              "AUPRC (higher is better)", (0.15, 0.62), "figure2A_binary_auprc.png",
              "A\u2013D: nested information layers (Figure 1); marker colour = layer. Whiskers: 95% patient-bootstrap CI "
              "(1,000 resamples). Dotted line: test improvement prevalence, 0.183 (353/1,930). AUPRC: area under the "
              "precision\u2013recall curve.", prevalence=prevalence)
    fig_layer(ci, "multiclass", "B", "Six-class response: macro-F1", "2,079 visit pairs \u00b7 155 test patients (locked test set)",
              "Macro-F1 (higher is better)", (0.28, 0.75), "figure2B_multiclass_macrof1.png",
              "A\u2013D: nested information layers (Figure 1); marker colour = layer. Whiskers: 95% patient-bootstrap CI "
              "(1,000 resamples). Macro-F1: mean F1 over the six IMWG classes; not comparable with AUPRC in Figure 2A "
              "(different evaluation pairs).", baseline_note=True)
    fig_delta(dl)


if __name__ == "__main__":
    main()
