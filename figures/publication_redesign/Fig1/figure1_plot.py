"""
Figure 1 - Study design: predicting the next IMWG response from nested information layers.

Run from the repository root:
    python figures/publication_redesign/Fig1/figure1_plot.py
    python figures/publication_redesign/Fig1/figure1_plot.py --formats png pdf svg

Inputs (verified by figures/fig1_extract_data.py; read-only here):
    figures/data/Fig1/fig1_cohort_task_counts.csv
    figures/data/Fig1/fig1_binary_improvement_prevalence.csv
    figures/data/Fig1/fig1_information_layers.csv
    figures/data/Fig1/fig1_features_by_layer_added.csv
    figures/data/Fig1/fig1_rna_timing_summary.json
    figures/data/Fig1/fig1_rna_samples_distinct.csv
    figures/data/Fig1/fig1_rna_age_by_visit_pair.csv
    figures/data/Fig1/fig1_survival_cohort_counts.csv
    figures/data/Fig1/fig1_survival_split_overlap.json
    figures/data/Fig1/fig1_verification_checks.csv

Every number drawn is read from these files; the script stops if any value differs from the
locked value verified in Step 2A, so no figure is produced from unverified data.

The STYLE block and helpers (panel_letter, clean_axis, fig_box) are the shared visual
conventions for Figures 1-5: layer colours, ink colours, font sizes and panel-letter placement.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

DATA = Path("figures/data/Fig1")
OUT_DIR = Path("figures/publication_redesign/Fig1")
STEM = "figure1"

# ----------------------------------------------------------------------------------------------
# Shared style (Figures 1-5)
# ----------------------------------------------------------------------------------------------
INK = "#0b0b0b"         # primary text
INK2 = "#52514e"        # secondary text
MUTED = "#898781"       # notes
GRID = "#e1e0d9"        # hairline grid
AXIS = "#c3c2b7"        # baseline / axis
SURFACE = "#ffffff"

# Information-layer colours: ordinal blue ramp for the clinical layers, orange accent for RNA.
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
LAYER_TEXT = {"A": INK, "B": "#ffffff", "C": "#ffffff", "D": "#ffffff"}
RNA_LIGHT = "#f4b393"   # lighter RNA step (secondary RNA quantities)
TASK_FILL = "#cde2fb"   # prediction-task visit pairs (blue step 100)
EXCL_FILL = "#e6e5df"   # excluded / not-available remainder (neutral)
# Algorithm markers (used from Figure 2 onward).
ALGO_MARKER = {"Logistic Regression": "o", "LightGBM": "s", "XGBoost": "D"}
# Patient-split shades (neutral; always direct-labelled).
SPLIT_COLOR = {"train": "#e6e5df", "val": "#bdbbb3", "test": "#6f6d68"}
SPLIT_TEXT = {"train": INK, "val": INK, "test": "#ffffff"}

FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
FIG_W, FIG_H = 7.2, 10.15   # inches (double-column width)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"],
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,      # editable text in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # editable text in SVG
    "savefig.facecolor": SURFACE,
})


def y_in(top_in):
    """Figure-fraction y for a distance in inches from the top edge."""
    return 1 - top_in / FIG_H


def add_ax(fig, left, top_in, width, height_in):
    """Axes placed by left (fig fraction), top (inches from top), width (fraction), height (inches)."""
    return fig.add_axes([left, y_in(top_in + height_in), width, height_in / FIG_H])


def panel_letter(fig, letter, top_in, x=0.012):
    fig.text(x, y_in(top_in), letter, fontsize=FS["panel"], fontweight="bold", ha="left",
             va="top", color=INK)


def heading(fig, text, top_in, x, sub=None):
    fig.text(x, y_in(top_in), text, fontsize=FS["title"], fontweight="bold", color=INK,
             ha="left", va="top")
    if sub:
        fig.text(x, y_in(top_in + 0.15), sub, fontsize=FS["body"], color=INK2, ha="left", va="top")


def clean_axis(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def arrow(ax, xy0, xy1, color=MUTED, lw=0.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=7, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0))


def fmt(n):
    return f"{int(n):,}"


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 1 data check failed: {msg}")


# ----------------------------------------------------------------------------------------------
# Data (read + verify against Step 2A locked values)
# ----------------------------------------------------------------------------------------------
def load():
    c = pd.read_csv(DATA / "fig1_cohort_task_counts.csv")
    prev = pd.read_csv(DATA / "fig1_binary_improvement_prevalence.csv").set_index("split")
    layers = pd.read_csv(DATA / "fig1_information_layers.csv").set_index("layer")
    feats = pd.read_csv(DATA / "fig1_features_by_layer_added.csv")
    timing = json.load(open(DATA / "fig1_rna_timing_summary.json"))
    samples = pd.read_csv(DATA / "fig1_rna_samples_distinct.csv")
    age = pd.read_csv(DATA / "fig1_rna_age_by_visit_pair.csv")
    surv = pd.read_csv(DATA / "fig1_survival_cohort_counts.csv").set_index("analysis")
    overlap = json.load(open(DATA / "fig1_survival_split_overlap.json"))["survival_patients_by_next_visit_split"]
    checks = pd.read_csv(DATA / "fig1_verification_checks.csv").set_index("check")

    def cnt(task, split, col):
        return int(c.query("task == @task and split == @split")[col].iloc[0])

    d = {"cnt": cnt, "prev": prev, "layers": layers, "feats": feats, "timing": timing,
         "samples": samples, "age": age, "surv": surv, "overlap": overlap}

    locked = {
        ("multiclass", "all", "n_patients"): 1025, ("multiclass", "all", "n_visit_pairs"): 13451,
        ("binary", "all", "n_patients"): 1018, ("binary", "all", "n_visit_pairs"): 12298,
        ("multiclass", "all", "n_patients_rna_available"): 707,
        ("multiclass", "all", "n_visit_pairs_rna_available"): 9173,
        ("binary", "all", "n_patients_rna_available"): 703,
        ("binary", "all", "n_visit_pairs_rna_available"): 8262,
        ("multiclass", "train", "n_patients"): 716, ("multiclass", "val", "n_patients"): 154,
        ("multiclass", "test", "n_patients"): 155, ("multiclass", "train", "n_visit_pairs"): 9539,
        ("multiclass", "val", "n_visit_pairs"): 1833, ("multiclass", "test", "n_visit_pairs"): 2079,
        ("binary", "train", "n_patients"): 709, ("binary", "val", "n_patients"): 154,
        ("binary", "test", "n_patients"): 155, ("binary", "train", "n_visit_pairs"): 8709,
        ("binary", "val", "n_visit_pairs"): 1659, ("binary", "test", "n_visit_pairs"): 1930,
    }
    for k, v in locked.items():
        require(cnt(*k) == v, f"{k} = {cnt(*k)}, expected {v}")
    require(int(prev.loc["test", "n_improved"]) == 353 and int(prev.loc["test", "n_visit_pairs"]) == 1930,
            "binary test prevalence")
    require(list(layers.source_input_features) == [78, 115, 143, 194], "source input features")
    require(list(layers.fitted_model_input_columns) == [148, 185, 302, 354], "fitted columns")
    require(int(layers.loc["D", "preprocessor_input_columns"]) == 195, "layer D preprocessor columns")
    require(feats.layer_added.value_counts().to_dict() == {"A": 78, "B": 37, "C": 28, "D": 51},
            "features added per layer")
    for k, v in {"n_distinct_rna_samples": 751, "n_patients_1_sample": 669, "n_patients_ge2_samples": 38,
                 "n_samples_within_30d_of_index": 636, "rna_age_median_days": 866.0,
                 "rna_age_q1_days": 380.0, "rna_age_q3_days": 1465.0, "n_rna_visit_pairs": 9173,
                 "n_patients_first_sample_on_or_before_index": 674}.items():
        require(timing[k] == v, f"timing {k} = {timing[k]}, expected {v}")
    require(len(samples) == 751 and len(age) == 9173, "RNA sample / age row counts")
    require((int(surv.loc["PFS_primary", "n_patients"]), int(surv.loc["PFS_primary", "n_events"]),
             int(surv.loc["OS_secondary", "n_events"])) == (674, 459, 247), "survival cohort")
    require(overlap == {"train": 473, "val": 101, "test": 100}, "survival split overlap")
    # nesting shown in panel A: every survival patient is one of the 707 RNA-available patients
    require(int(checks.loc["survival patients that are in the visit-pair RNA subset", "observed"]) == 674,
            "survival cohort nested in RNA-available patients")
    return d


# ----------------------------------------------------------------------------------------------
# Panel A - prediction unit, nested cohorts, patient-grouped split
# ----------------------------------------------------------------------------------------------
def panel_a_unit(ax):
    """Prediction unit: one consecutive visit pair."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    y = 5.6
    ax.plot([1, 52], [y, y], color=AXIS, lw=0.9, solid_capstyle="round", zorder=1)
    for x in (7, 15, 23):
        ax.plot([x, x], [y - 0.9, y + 0.9], color=AXIS, lw=0.9)
    ax.text(15, y + 1.5, "earlier visits", fontsize=FS["small"], color=MUTED, ha="center", va="bottom")
    ax.plot(32, y, "o", ms=6.5, mfc=LAYER_COLOR["B"], mec="white", mew=0.8, zorder=3)
    ax.plot(48, y, "o", ms=6.5, mfc="white", mec=INK, mew=0.9, zorder=3)
    ax.text(32, y + 1.5, "Current visit V(t)", fontsize=FS["body"], color=INK, ha="center",
            va="bottom", fontweight="bold")
    ax.text(48, y + 1.5, "Next visit V(t+1)", fontsize=FS["body"], color=INK, ha="center",
            va="bottom", fontweight="bold")
    arrow(ax, (33.4, y), (46.6, y), color=INK, lw=0.9)
    ax.text(40, y - 0.7, "predict", fontsize=FS["small"], color=INK2, ha="center", va="top",
            style="italic")
    ax.plot([1, 32], [y - 2.6, y - 2.6], color=LAYER_COLOR["B"], lw=0.9)
    for x in (1, 32):
        ax.plot([x, x], [y - 2.6, y - 1.8], color=LAYER_COLOR["B"], lw=0.9)
    ax.text(16.5, y - 3.1, "Input features: information available at or before V(t)",
            fontsize=FS["body"], color=INK2, ha="center", va="top")
    ax.text(48, y - 3.1, "Target: IMWG response at V(t+1)", fontsize=FS["body"], color=INK2,
            ha="center", va="top")
    ax.text(60, y + 2.3, "Unit of prediction: one consecutive visit pair, V(t) → V(t+1)",
            fontsize=FS["body"], color=INK, fontweight="bold", ha="left", va="top")
    ax.text(60, y + 0.4, "Each patient contributes several visit pairs. All of a\npatient's "
            "pairs stay in one data split (train, validation\nor test), so no patient is shared "
            "between splits.", fontsize=FS["body"], color=INK2, ha="left", va="top", linespacing=1.3)


def nested_bars(ax, rows, total, unit):
    """Horizontal bars on a common count axis. rows: dicts with keys
    label, n, fill, rest (remaining count drawn as an outlined remainder), rest_label, inside, right."""
    h = 0.66
    for i, r in enumerate(rows):
        ax.barh(i, r["n"], height=h, color=r["fill"], edgecolor="none", zorder=2,
                hatch=r.get("hatch"))
        if r.get("rest"):
            ax.barh(i, r["rest"], left=r["n"], height=h, color="white", edgecolor=AXIS,
                    linewidth=0.6, hatch=r.get("rest_hatch"), zorder=1)
        if r.get("inside"):
            ax.text(total * 0.012, i, r["inside"], va="center", ha="left", fontsize=FS["body"],
                    color=r.get("inside_color", INK), zorder=3)
        if r.get("right"):
            ax.text(total * 1.02, i, r["right"], va="center", ha="left", fontsize=FS["small"],
                    color=INK2, linespacing=1.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=FS["body"], color=INK)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(0, total)
    ax.tick_params(axis="y", length=0, pad=4)
    clean_axis(ax, keep=("bottom",))
    ax.set_xlabel(unit, fontsize=FS["body"], color=INK2, labelpad=1.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def panel_a_pairs(ax, d):
    cnt, prev = d["cnt"], d["prev"]
    mc, bn = cnt("multiclass", "all", "n_visit_pairs"), cnt("binary", "all", "n_visit_pairs")
    mc_r, bn_r = cnt("multiclass", "all", "n_visit_pairs_rna_available"), cnt("binary", "all", "n_visit_pairs_rna_available")
    n_imp, n_test = int(prev.loc["test", "n_improved"]), int(prev.loc["test", "n_visit_pairs"])
    rows = [
        dict(label="Six-class response task", n=mc, fill=TASK_FILL,
             inside=f"{fmt(mc)} visit pairs · {fmt(cnt('multiclass', 'all', 'n_patients'))} patients",
             right="Target: IMWG category at V(t+1)\n(PD, SD, PR, VGPR, CR or sCR)"),
        dict(label="Binary improvement task", n=bn, fill=TASK_FILL, rest=mc - bn, rest_hatch="////",
             inside=f"{fmt(bn)} visit pairs · {fmt(cnt('binary', 'all', 'n_patients'))} patients",
             right=f"Target: better IMWG category at V(t+1)\nhatched: {fmt(mc - bn)} pairs with sCR "
                   f"at V(t) excluded"),
        dict(label="RNA available (six-class)", n=mc_r, fill=LAYER_COLOR["D"], rest=mc - mc_r,
             inside=f"{fmt(mc_r)} pairs · {fmt(cnt('multiclass', 'all', 'n_patients_rna_available'))} patients",
             inside_color="white", right=f"{mc_r / mc:.1%} of six-class pairs"),
        dict(label="RNA available (binary)", n=bn_r, fill=LAYER_COLOR["D"], rest=bn - bn_r,
             inside=f"{fmt(bn_r)} pairs · {fmt(cnt('binary', 'all', 'n_patients_rna_available'))} patients",
             inside_color="white", right=f"{bn_r / bn:.1%} of binary pairs"),
    ]
    d["test_rate_text"] = f"{n_imp / n_test:.1%} ({fmt(n_imp)} of {fmt(n_test)} test pairs)"
    nested_bars(ax, rows, mc, "Visit pairs")
    ax.set_xticks([0, 4000, 8000, 12000])


def panel_a_patients(ax, d):
    cnt, surv, ov = d["cnt"], d["surv"], d["overlap"]
    n_all = cnt("multiclass", "all", "n_patients")
    n_rna = cnt("multiclass", "all", "n_patients_rna_available")
    n_s = int(surv.loc["PFS_primary", "n_patients"])
    rows = [
        dict(label="All patients", n=n_all, fill=TASK_FILL, inside=f"{fmt(n_all)} patients",
             right="MMRF CoMMpass interim analysis IA24"),
        dict(label="RNA available for ≥1 visit pair", n=n_rna, fill=LAYER_COLOR["D"], rest=n_all - n_rna,
             inside=f"{fmt(n_rna)} patients", inside_color="white",
             right="Layer D still uses all visit pairs; RNA inputs\nare imputed when no RNA is available"),
        dict(label="Survival cohort (Figures 4–5)", n=n_s, fill=RNA_LIGHT, rest=n_all - n_s,
             inside=f"{fmt(n_s)} patients with RNA on or before day 0", inside_color=INK,
             right=f"PFS {fmt(surv.loc['PFS_primary', 'n_events'])} events · OS "
                   f"{fmt(surv.loc['OS_secondary', 'n_events'])} deaths;\nanalysed by internal "
                   f"cross-validation"),
    ]
    nested_bars(ax, rows, n_all, "Patients")
    ax.set_xticks([0, 250, 500, 750, 1000])


def panel_a_split(ax, d):
    cnt = d["cnt"]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    x0, x1, bh = 13, 100, 1.0
    for row, (task, label) in enumerate([("multiclass", "Six-class\ntask"), ("binary", "Binary\ntask")]):
        yb = 1.25 - row * (bh + 0.2)
        total = cnt(task, "all", "n_visit_pairs")
        ax.text(0, yb + bh / 2, label, fontsize=FS["body"], color=INK, ha="left", va="center",
                fontweight="bold", linespacing=1.15)
        xs = x0
        for split, name in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
            n = cnt(task, split, "n_visit_pairs")
            w = (x1 - x0) * n / total
            ax.add_patch(Rectangle((xs, yb), w - 0.35, bh, fc=SPLIT_COLOR[split], ec="none"))
            # bold first line + regular lines, aligned as one 3-line block centred in the bar
            body = f"{fmt(cnt(task, split, 'n_patients'))} patients\n{fmt(n)} visit pairs"
            kw = dict(fontsize=FS["body"], color=SPLIT_TEXT[split], ha="left", va="center",
                      linespacing=1.2)
            ax.text(xs + 0.9, yb + bh / 2, f"{name}\n\n", fontweight="bold", **kw)
            ax.text(xs + 0.9, yb + bh / 2, f" \n{body}", **kw)
            xs += w


# ----------------------------------------------------------------------------------------------
# Panel B - nested information layers
# ----------------------------------------------------------------------------------------------
LAYER_ROW = {"A": "Current clinical state", "B": "A + longitudinal history",
             "C": "B + treatment context", "D": "C + RNA pathway scores"}
FAMILY_KEY = [
    ("A", "Current clinical state", "labs, IMWG response, ISS, CRAB,\nQoL, comorbidity, marrow/flow"),
    ("B", "Longitudinal history", "previous values, changes,\ndays since last measurement"),
    ("C", "Treatment context", "line, regimen, drug-class\nexposure, transplant"),
    ("D", "RNA pathway scores", "50 Hallmark ssGSEA scores\n+ RNA age"),
]


XT, XF = 220, 262   # x positions (data units) of the totals and fitted-column columns


def panel_b(ax, axk, d):
    layers, feats = d["layers"], d["feats"]
    added = feats.layer_added.value_counts().to_dict()
    order = ["A", "B", "C", "D"]
    for i, lay in enumerate(order):
        left = 0
        for fam in order[: i + 1]:
            w = added[fam]
            ax.barh(i, w - 0.8, left=left, height=0.64, color=LAYER_COLOR[fam], edgecolor="none")
            ax.text(left + w / 2, i, f"{w}" if fam == "A" else f"+{w}", ha="center", va="center",
                    fontsize=FS["body"], color=LAYER_TEXT[fam], fontweight="bold")
            left += w
        total = int(layers.loc[lay, "source_input_features"])
        ax.text(XT, i, f"{total}" + ("†" if lay == "D" else ""), ha="center", va="center",
                fontsize=FS["body"] + 0.5, color=INK, fontweight="bold")
        ax.text(XF, i, fmt(layers.loc[lay, "fitted_model_input_columns"]), ha="center",
                va="center", fontsize=FS["body"] + 0.5, color=INK2)
    ax.text(XT, -0.62, "Source input\nfeatures", ha="center", va="bottom", fontsize=FS["small"],
            color=INK, fontweight="bold", linespacing=1.1)
    ax.text(XF, -0.62, "Fitted model-\ninput columns*", ha="center", va="bottom",
            fontsize=FS["small"], color=INK2, fontweight="bold", linespacing=1.1)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Layer {l}: {LAYER_ROW[l]}" for l in order], fontsize=FS["body"], color=INK)
    ax.set_ylim(3.5, -0.55)
    ax.set_xlim(0, 280)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.spines["bottom"].set_bounds(0, 200)
    clean_axis(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Number of source input features", fontsize=FS["body"], color=INK2, labelpad=1.5)
    ax.xaxis.set_label_coords(100 / 280, -0.2)

    # compact family key (one row)
    axk.axis("off")
    axk.set_xlim(0, 100)
    axk.set_ylim(0, 10)
    for j, (fam, name, desc) in enumerate(FAMILY_KEY):
        x = j * 25
        axk.add_patch(Rectangle((x, 6.2), 1.4, 3.4, fc=LAYER_COLOR[fam], ec="none"))
        axk.text(x + 2.2, 9.7, name, fontsize=FS["body"], color=INK, fontweight="bold", va="top")
        axk.text(x + 2.2, 6.6, desc, fontsize=FS["small"], color=INK2, va="top", linespacing=1.2)


# ----------------------------------------------------------------------------------------------
# Panel C - RNA timing
# ----------------------------------------------------------------------------------------------
def panel_c1(ax, d):
    s = d["samples"].sort_values(["public_id", "rna_sample_day"]).copy()
    s["order"] = s.groupby("public_id").cumcount()
    first, extra = s[s.order == 0].rna_sample_day, s[s.order > 0].rna_sample_day
    tm = d["timing"]
    cats = [
        ("Before day −30", int((first < -30).sum())),
        ("Day −30 to 0", int(((first >= -30) & (first <= 0)).sum())),
        ("Day 1 to 30", int(((first > 0) & (first <= 30)).sum())),
        ("After day 30", int((first > 30).sum())),
    ]
    n_first, n_extra = len(first), len(extra)
    require(n_first == tm["n_rna_patients"] == 707 and sum(c[1] for c in cats) == 707, "first samples")
    require(n_extra == 751 - 707 == 44, "additional samples")
    require(cats[1][1] + cats[2][1] == tm["n_samples_within_30d_of_index"] == 636, "within +/-30 d")
    require(((extra >= -30) & (extra <= 30)).sum() == 0, "no additional sample within +/-30 d")
    require(int(extra.min()) > 30, "additional samples all after day 30")

    y_first = [1, 2, 3, 4]
    y_extra = 6.4
    for (lab, n), y in zip(cats, y_first):
        ax.barh(y, n, height=0.66, color=LAYER_COLOR["D"], edgecolor="none")
        ax.text(n + 12, y, fmt(n), va="center", ha="left", fontsize=FS["body"], color=INK)
    ax.barh(y_extra, n_extra, height=0.66, color=RNA_LIGHT, edgecolor=LAYER_COLOR["D"], linewidth=0.6,
            hatch="//////")
    ax.text(n_extra + 12, y_extra, f"{n_extra}  (day {int(extra.min())}–{fmt(extra.max())}; "
            f"{tm['n_patients_ge2_samples']} patients)", va="center", ha="left", fontsize=FS["small"],
            color=INK)
    ax.text(0, 0.05, f"First RNA sample per patient (n = {n_first})", fontsize=FS["body"], color=INK,
            fontweight="bold", ha="left", va="center")
    ax.text(0, 5.45, f"Additional samples (n = {n_extra})", fontsize=FS["body"], color=INK,
            fontweight="bold", ha="left", va="center")
    ax.set_yticks(y_first + [y_extra])
    ax.set_yticklabels([c[0] for c in cats] + ["After day 30"], fontsize=FS["body"], color=INK)
    ax.set_ylim(7.0, -0.5)
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 200, 400, 600])
    ax.spines["bottom"].set_bounds(0, 650)
    clean_axis(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("RNA samples (751 distinct samples)", fontsize=FS["body"], labelpad=1.5)
    ax.xaxis.set_label_coords(325 / 1000, -0.25)
    # bracket for the +/-30-day window (rows "day -30 to 0" and "day 1 to 30")
    xb = 735
    ax.plot([xb, xb], [1.7, 3.3], color=INK2, lw=0.7)
    for yy in (1.7, 3.3):
        ax.plot([xb - 14, xb], [yy, yy], color=INK2, lw=0.7)
    ax.text(xb + 18, 2.5, f"{tm['n_samples_within_30d_of_index']} / 751\n"
            f"({tm['pct_samples_within_30d_of_index']:.1f}%)\nwithin ±30 days\nof index",
            ha="left", va="center", fontsize=FS["small"], color=INK, linespacing=1.15)


def panel_c2(ax, d):
    a = d["age"].days_since_rna_sample.to_numpy()
    tm = d["timing"]
    bins = np.arange(0, 2880 + 90, 90)
    ax.axvspan(tm["rna_age_q1_days"], tm["rna_age_q3_days"], color="#fdeee6", zorder=0, lw=0)
    ax.hist(a, bins=bins, color=LAYER_COLOR["D"], edgecolor="white", linewidth=0.4, zorder=2)
    ax.axvline(tm["rna_age_median_days"], color=INK, lw=0.9, zorder=3)
    ymax = ax.get_ylim()[1]
    ax.text(tm["rna_age_q3_days"] + 70, ymax * 0.98,
            f"Median {fmt(tm['rna_age_median_days'])} days\nIQR {fmt(tm['rna_age_q1_days'])}–"
            f"{fmt(tm['rna_age_q3_days'])} days\n(shaded)", fontsize=FS["small"], color=INK, va="top",
            linespacing=1.2)
    ax.set_xlim(0, 2900)
    ax.set_xticks([0, 1000, 2000])
    ax.set_xticks([500, 1500, 2500], minor=True)
    ax.set_xticklabels(["0", "1,000", "2,000"])
    ax.set_xlabel("Days from the latest RNA sample\n(at or before V(t)) to V(t)", fontsize=FS["body"],
                  labelpad=1.5)
    ax.set_ylabel("Visit pairs", fontsize=FS["body"], labelpad=2)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    clean_axis(ax, keep=("bottom", "left"))


def panel_c3(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    yv, yr = 32, 74
    ax.plot([13, 99], [yv, yv], color=AXIS, lw=0.9)
    ax.text(0, yv, "Visits", fontsize=FS["small"], color=MUTED, ha="left", va="center")
    ax.text(0, yr, "RNA", fontsize=FS["small"], color=MUTED, ha="left", va="center")
    for x in (18, 34, 50):
        ax.plot([x, x], [yv - 2.5, yv + 2.5], color=AXIS, lw=0.9)
    ax.plot(68, yv, "o", ms=6, mfc=LAYER_COLOR["B"], mec="white", mew=0.8, zorder=3)
    ax.plot(90, yv, "o", ms=6, mfc="white", mec=INK, mew=0.9, zorder=3)
    ax.text(68, yv - 5, "V(t)", fontsize=FS["body"], color=INK, ha="center", va="top", fontweight="bold")
    ax.text(90, yv - 5, "V(t+1)", fontsize=FS["body"], color=INK, ha="center", va="top", fontweight="bold")
    ax.plot(18, yr, "o", ms=5.5, mfc=RNA_LIGHT, mec="white", mew=0.6)
    ax.plot(42, yr, "o", ms=6.5, mfc=LAYER_COLOR["D"], mec="white", mew=0.6)
    ax.plot(80, yr, "o", ms=6.5, mfc="white", mec=MUTED, mew=0.9)
    ax.plot([68, 68], [yv + 3, yr + 6], color=MUTED, lw=0.7, ls=(0, (2, 1.5)))
    arrow(ax, (66.5, yr), (44.2, yr), color=LAYER_COLOR["D"], lw=0.9)
    ax.text(44, yr + 6, "used: latest sample\nat or before V(t)", fontsize=FS["small"], color=INK,
            ha="center", va="bottom", linespacing=1.15)
    ax.text(18, yr - 6, "earlier sample\n(superseded)", fontsize=FS["small"], color=INK2, ha="center",
            va="top", linespacing=1.15)
    ax.text(85, yr + 6, "after V(t):\nnever used", fontsize=FS["small"], color=INK2, ha="center",
            va="bottom", linespacing=1.15)
    yb = 6
    ax.plot([42, 68], [yb, yb], color=INK2, lw=0.8)
    for x in (42, 68):
        ax.plot([x, x], [yb - 2.5, yb + 2.5], color=INK2, lw=0.8)
    ax.plot([42, 42], [yb + 2.5, yr - 4], color=GRID, lw=0.6)
    ax.text(55, yb - 4, "RNA age at V(t)", fontsize=FS["small"], color=INK, ha="center", va="top")
    ax.text(0, -14, "Visit pairs with no RNA sample at or before V(t)\nare RNA-unavailable; a later "
            "sample is never\nused to fill them.", fontsize=FS["small"], color=MUTED,
            ha="left", va="top", linespacing=1.2)


# ----------------------------------------------------------------------------------------------
def build(d):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    LX = 0.035            # left edge of headings
    PX = 0.30             # left edge of data axes in panels A and B (room for row labels)

    # ---- Panel A ----
    panel_letter(fig, "A", 0.10)
    heading(fig, "Cohort, prediction tasks and data split", 0.12, LX + 0.02)
    axU = add_ax(fig, LX, 0.30, 0.95, 0.56)
    panel_a_unit(axU)

    heading(fig, "Visit pairs for next-visit prediction (RNA-available pairs are a subset of each "
            "task's pairs; all bars share one axis)", 0.98, LX)
    axP = add_ax(fig, PX, 1.18, 0.40, 1.02)
    panel_a_pairs(axP, d)

    heading(fig, "Patients (nested sets: the survival cohort lies within the patients with RNA, "
            "who lie within all patients)", 2.56, LX)
    axQ = add_ax(fig, PX, 2.76, 0.40, 0.74)
    panel_a_patients(axQ, d)

    heading(fig, "Patient-grouped data split for next-visit prediction, identical for every "
            "information layer and algorithm", 3.86, LX)
    axS = add_ax(fig, LX, 4.03, 0.95, 0.92)
    panel_a_split(axS, d)
    fig.text(LX, y_in(4.99), "Hyperparameters were tuned by patient-grouped cross-validation within "
             "training patients only; validation patients set the binary decision threshold; test "
             "patients were scored once.\nBar widths are proportional to visit pairs. Binary test-set "
             f"improvement rate: {d['test_rate_text']}. The survival cohort is analysed separately "
             "and spans all three splits.", fontsize=FS["small"], color=MUTED, ha="left", va="top",
             linespacing=1.25)

    # ---- Panel B ----
    panel_letter(fig, "B", 5.40)
    heading(fig, "Nested information layers A\u2013D: input-feature sets, not algorithms", 5.42,
            LX + 0.02, sub="Each layer adds one family of features to the previous layer; every "
            "layer uses the same visit pairs and patient split.")
    axB = add_ax(fig, PX, 5.98, 0.62, 0.90)
    axK = add_ax(fig, LX, 7.22, 0.95, 0.32)
    panel_b(axB, axK, d)
    fig.text(LX, y_in(7.58), "Each layer was fitted with Logistic Regression, LightGBM and XGBoost "
             "(XGBoost is the reporting reference algorithm, not a validation-selected best model). "
             "\u2020Plus 1 derived RNA-\navailability indicator. Source input features are counted before "
             "one-hot encoding; *fitted model-input columns are counted after one-hot encoding and "
             "addition of missing-value indicators.",
             fontsize=FS["small"], color=MUTED, ha="left", va="top", linespacing=1.25)

    # ---- Panel C ----
    panel_letter(fig, "C", 8.00)
    top_c = 8.02
    for x, t in [(LX + 0.02, "RNA collection relative to the\nCoMMpass index date (day 0)"),
                 (0.505, "RNA age at the current visit V(t)\n(9,173 six-class pairs with RNA)"),
                 (0.75, "No-future-RNA alignment rule\n(schematic)")]:
        fig.text(x, y_in(top_c), t, fontsize=FS["title"], fontweight="bold", color=INK, ha="left",
                 va="top", linespacing=1.15)
    axC1 = add_ax(fig, 0.135, 8.40, 0.30, 0.98)
    axC2 = add_ax(fig, 0.545, 8.42, 0.17, 0.84)
    axC3 = add_ax(fig, 0.75, 8.40, 0.24, 0.84)
    panel_c1(axC1, d)
    panel_c2(axC2, d)
    panel_c3(axC3)

    fig.text(LX, 0.005,
             "IMWG, International Myeloma Working Group; PD, progressive disease; SD, stable disease; "
             "PR, partial response; VGPR, very good partial response; CR, complete response;\nsCR, "
             "stringent complete response; PFS, progression-free survival; OS, overall survival; ISS, "
             "International Staging System; CRAB, hypercalcaemia, renal insufficiency,\nanaemia, bone "
             "lesions; QoL, quality of life; ssGSEA, single-sample gene set enrichment analysis "
             "(MSigDB Hallmark gene sets); IQR, interquartile range.",
             fontsize=FS["small"] - 0.4, color=MUTED, ha="left", va="bottom", linespacing=1.3)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    fig = build(load())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in args.formats:
        p = OUT_DIR / f"{STEM}.{f}"
        fig.savefig(p, dpi=args.dpi if f == "png" else None)
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
