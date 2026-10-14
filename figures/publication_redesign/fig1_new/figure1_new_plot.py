"""
New Figure 1 - Study overview: data sources, cohorts, inputs, tasks, modelling framework,
evaluation, downstream analyses and validation status.

Run from the repository root:
    python figures/publication_redesign/fig1_new/figure1_new_plot.py
    python figures/publication_redesign/fig1_new/figure1_new_plot.py --formats png pdf svg

Inputs (tracked, aggregate-only; read-only here):
    figures/data/Fig1/fig1_cohort_task_counts.csv
    figures/data/Fig1/fig1_binary_improvement_prevalence.csv
    figures/data/Fig1/fig1_information_layers.csv
    figures/data/Fig1/fig1_survival_cohort_counts.csv
    figures/data/Fig1/fig1_survival_split_overlap.json
    figures/data/Fig1/fig1_rna_timing_summary.json

Numbers that are not in those files (1,143 subject records; the geographic-analysis counts) are
constants below. When the private data/ directory is present the script re-derives them and stops on
any mismatch; otherwise it prints a notice and uses the constants (see README.md for provenance).

The STYLE block follows figures/publication_redesign/Fig1/figure1_plot.py (shared Figure 2-6 style).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

DATA = Path("figures/data/Fig1")
OUT_DIR = Path("figures/publication_redesign/fig1_new")
STEM = "figure1_new"

# ----------------------------------------------------------------------------------------------
# Shared style (identical values to the existing Figure 2)
# ----------------------------------------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
LAYER_TEXT = {"A": INK, "B": "#ffffff", "C": "#ffffff", "D": "#ffffff"}
RNA_LIGHT = "#f4b393"
TASK_FILL = "#cde2fb"
EXCL_FILL = "#e6e5df"
SPLIT_COLOR = {"train": "#e6e5df", "val": "#bdbbb3", "test": "#6f6d68"}
SPLIT_TEXT = {"train": INK, "val": INK, "test": "#ffffff"}
ALGO_MARKER = {"Logistic Regression": "o", "LightGBM": "s", "XGBoost": "D"}
# New in this figure: one colour reserved for known validation concerns (not used for data).
FLAG = "#a4590a"
FLAG_FILL = "#fbeedd"

FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0}
FIG_W, FIG_H = 7.2, 10.08

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"],
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": SURFACE,
})

# Constants not stored in the tracked Figure 2 data files (sources in README.md).
N_SUBJECT_RECORDS = 1143              # data/clinical/subject_deid.csv
GEO = {"fit": 439, "calibration": 96, "excluded_us": 236, "evaluated": 116,
       "spain": 80, "canada": 36}      # original-train-only geographic run (audit JSON) + patient_split_assignments.csv
GEO_AUDIT = Path("artifacts/results_clean_rerun/geographic_origtrain/full_run_audit_and_summary.json")
# Train-only pathway redesign (selection + discovery in train patients; evaluation in validation + test patients).
PATHWAY = {"dev_patients": 371, "heldout_patients": 160, "heldout_val": 76, "heldout_test": 84}
PATHWAY_DIR = Path("artifacts/results_clean_rerun/direction2_replication_trainonly")
# Train-only RNA-contribution (Figures 4A-C, Table S1, add-on) and survival (Figures 5-6) redesigns.
RNAC = {"dev_patients": 716, "heldout_patients": 213}
RNAC_DIR = Path("artifacts/results_clean_rerun/rna_contribution_trainonly")
SURVT = {"dev_patients": 473, "heldout_patients": 201, "heldout_val": 101, "heldout_test": 100}
SURVT_DIR = Path("artifacts/results_clean_rerun/survival_trainonly")


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 1 (new) data check failed: {msg}")


def fmt(n):
    return f"{int(n):,}"


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load():
    c = pd.read_csv(DATA / "fig1_cohort_task_counts.csv")
    prev = pd.read_csv(DATA / "fig1_binary_improvement_prevalence.csv").set_index("split")
    layers = pd.read_csv(DATA / "fig1_information_layers.csv").set_index("layer")
    surv = pd.read_csv(DATA / "fig1_survival_cohort_counts.csv").set_index("analysis")
    overlap = json.load(open(DATA / "fig1_survival_split_overlap.json"))["survival_patients_by_next_visit_split"]
    timing = json.load(open(DATA / "fig1_rna_timing_summary.json"))

    def cnt(task, split, col):
        return int(c.query("task == @task and split == @split")[col].iloc[0])

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
    require(timing["n_distinct_rna_samples"] == 751, "distinct RNA samples")
    require((int(surv.loc["PFS_primary", "n_patients"]), int(surv.loc["PFS_primary", "n_events"]),
             int(surv.loc["OS_secondary", "n_events"])) == (674, 459, 247), "survival cohort")
    require(overlap == {"train": 473, "val": 101, "test": 100}, "survival split overlap")
    require(cnt("multiclass", "all", "n_visit_pairs") - cnt("binary", "all", "n_visit_pairs") == 1153,
            "binary exclusions")
    verify_private_constants()
    verify_result_constants()
    return {"cnt": cnt, "prev": prev, "layers": layers, "surv": surv, "overlap": overlap, "timing": timing}


def verify_private_constants():
    """Re-derive the non-tracked constants when data/ is available (never required)."""
    subj_p, split_p = Path("data/clinical/subject_deid.csv"), Path("data/splits/patient_split_assignments.csv")
    if not (subj_p.exists() and split_p.exists()):
        print("NOTE: data/ not found - subject-record and geographic constants not re-verified.")
        return
    subj = pd.read_csv(subj_p)
    require(len(subj) == N_SUBJECT_RECORDS, f"subject records {len(subj)} != {N_SUBJECT_RECORDS}")
    sp = pd.read_csv(split_p).merge(subj[["public_id", "country_of_residence_at_enrollment"]], on="public_id")
    sc = sp[sp.country_of_residence_at_enrollment.isin(["spain", "canada"])]
    require(len(sc) == GEO["evaluated"], "Spain+Canada patients")
    require(int((sc.country_of_residence_at_enrollment == "spain").sum()) == GEO["spain"], "Spain patients")
    require(int((sc.country_of_residence_at_enrollment == "canada").sum()) == GEO["canada"], "Canada patients")
    print("Private constants re-verified against data/.")


def verify_result_constants():
    """Check the geographic and pathway counts against the (tracked-eligible) result JSONs when present."""
    if GEO_AUDIT.exists():
        a = json.load(open(GEO_AUDIT, encoding="utf-8"))["audit"]
        require((a["n_us_train_patients"], a["n_us_val_patients"], a["n_us_excluded_not_original_train"],
                 a["n_spain_canada_patients"], a["n_spain_patients"], a["n_canada_patients"])
                == (GEO["fit"], GEO["calibration"], GEO["excluded_us"], GEO["evaluated"], GEO["spain"], GEO["canada"]),
                "geographic counts differ from the audit JSON")
        require(a["n_fit_or_calibration_overlap_with_original_val_or_test"] == 0, "geographic fit/calibration overlap")
        print("Geographic counts verified against", GEO_AUDIT)
    else:
        print("NOTE: geographic audit JSON not found - geographic counts not re-verified.")
    dr, hr = PATHWAY_DIR / "discovery_report.json", PATHWAY_DIR / "heldout_report.json"
    if dr.exists() and hr.exists():
        d, h = json.load(open(dr, encoding="utf-8")), json.load(open(hr, encoding="utf-8"))
        by = h["accounting"]["heldout_patients_by_split"]
        require((d["accounting"]["discovery_patients"], h["accounting"]["heldout_patients"], by["val"], by["test"])
                == (PATHWAY["dev_patients"], PATHWAY["heldout_patients"], PATHWAY["heldout_val"], PATHWAY["heldout_test"]),
                "pathway counts differ from the result JSONs")
        print("Pathway counts verified against", PATHWAY_DIR)
    else:
        print("NOTE: train-only pathway reports not found - pathway counts not re-verified.")
    rr, rd = RNAC_DIR / "heldout_report.json", RNAC_DIR / "frozen_development.json"
    if rr.exists() and rd.exists():
        require((json.load(open(rr, encoding="utf-8"))["heldout"]["rna_patients"], json.load(open(rd, encoding="utf-8"))["n_development"])
                == (RNAC["heldout_patients"], RNAC["dev_patients"]), "RNA-contribution counts differ from the result JSONs")
        print("RNA-contribution counts verified against", RNAC_DIR)
    else:
        print("NOTE: train-only RNA-contribution reports not found - counts not re-verified.")
    sr, sd = SURVT_DIR / "heldout_report.json", SURVT_DIR / "devcv_pfs_summary.json"
    if sr.exists() and sd.exists():
        h = json.load(open(sr, encoding="utf-8"))["heldout"]
        require((h["patients"], h["validation"], h["test"], json.load(open(sd, encoding="utf-8"))["n_patients"])
                == (SURVT["heldout_patients"], SURVT["heldout_val"], SURVT["heldout_test"], SURVT["dev_patients"]),
                "survival counts differ from the result JSONs")
        print("Survival counts verified against", SURVT_DIR)
    else:
        print("NOTE: train-only survival reports not found - counts not re-verified.")


# ----------------------------------------------------------------------------------------------
# Drawing helpers (single full-figure axes in inches; y grows downward)
# ----------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, fc=SURFACE, ec=AXIS, lw=0.7, r=0.04, hatch=None, z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       hatch=hatch, zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, size=None, color=INK, ha="left", va="top", weight="normal", style="normal",
         ls=1.22, z=5):
    return ax.text(x, y, s, fontsize=size or FS["body"], color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, linespacing=ls, zorder=z)


def arrow(ax, p0, p1, color=MUTED, lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=4))


def panel_letter(ax, letter, y, x=0.08):
    text(ax, x, y, letter, size=FS["panel"], weight="bold")


def heading(ax, s, y, x=0.36, sub=None):
    text(ax, x, y, s, size=FS["title"], weight="bold")
    if sub:
        text(ax, x, y + 0.15, sub, color=INK2)


def flag_tag(ax, x, y, s="Validation concern"):
    """Small amber tag marking a known, unresolved validation issue."""
    w = 0.046 * len(s) + 0.14
    box(ax, x, y, w, 0.17, fc=FLAG, ec=FLAG, r=0.03, z=6)
    text(ax, x + w / 2, y + 0.085, s, size=FS["small"], color="white", ha="center", va="center",
         weight="bold", z=7)
    return w


# ----------------------------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------------------------
def panel_a(ax, d, y0):
    cnt, surv = d["cnt"], d["surv"]
    panel_letter(ax, "A", y0)
    heading(ax, "Data sources and cohorts", y0, sub="MMRF CoMMpass, interim analysis 24 (IA24)")
    top = y0 + 0.42
    # --- sources
    box(ax, 0.12, top, 2.30, 1.42, fc="white", ec=AXIS)
    text(ax, 0.22, top + 0.07, "MMRF CoMMpass (IA24)", weight="bold")
    text(ax, 0.22, top + 0.23, f"{fmt(N_SUBJECT_RECORDS)} subject records", color=INK2)
    box(ax, 0.22, top + 0.45, 2.10, 0.40, fc=TASK_FILL, ec="none")
    text(ax, 0.29, top + 0.49, "Clinical data", weight="bold")
    text(ax, 0.29, top + 0.64, "diagnosis · laboratory · treatment ·\nresponse · adverse events · symptoms",
         size=FS["small"], color=INK2, ls=1.15)
    box(ax, 0.22, top + 0.92, 2.10, 0.42, fc=LAYER_COLOR["D"], ec="none")
    text(ax, 0.29, top + 0.96, "Bulk RNA sequencing", weight="bold", color="white")
    text(ax, 0.29, top + 1.11, "TPM → ssGSEA (MSigDB Hallmark): 50\npathway scores per RNA sample",
         size=FS["small"], color="white", ls=1.15)
    # --- visit pairs
    bx = 2.92
    box(ax, bx, top, 1.85, 0.80, fc=TASK_FILL, ec="none")
    text(ax, bx + 0.08, top + 0.07, "Visit pairs V(t) → V(t+1)", weight="bold")
    text(ax, bx + 0.08, top + 0.30,
         f"{fmt(cnt('multiclass', 'all', 'n_visit_pairs'))} pairs\n"
         f"{fmt(cnt('multiclass', 'all', 'n_patients'))} patients with ≥1 pair", color=INK, ls=1.2)
    box(ax, bx, top + 0.90, 1.85, 0.52, fc=LAYER_COLOR["D"], ec="none")
    text(ax, bx + 0.08, top + 0.96, "RNA available at V(t)", weight="bold", color="white")
    text(ax, bx + 0.08, top + 1.11,
         f"{fmt(cnt('multiclass', 'all', 'n_visit_pairs_rna_available'))} pairs · "
         f"{fmt(cnt('multiclass', 'all', 'n_patients_rna_available'))} patients\n"
         f"({cnt('multiclass', 'all', 'n_visit_pairs_rna_available') / cnt('multiclass', 'all', 'n_visit_pairs'):.0%} of pairs)",
         size=FS["small"], color="white", ls=1.15)
    # --- tasks + survival cohort
    tx, tw = 5.12, 1.96
    box(ax, tx, top, tw, 0.50, fc=TASK_FILL, ec="none")
    text(ax, tx + 0.08, top + 0.05, "Six-class response task", weight="bold")
    text(ax, tx + 0.08, top + 0.21,
         f"{fmt(cnt('multiclass', 'all', 'n_visit_pairs'))} pairs · {fmt(cnt('multiclass', 'all', 'n_patients'))} patients\n"
         "IMWG: PD/SD/PR/VGPR/CR/sCR", size=FS["small"], color=INK2, ls=1.15)
    box(ax, tx, top + 0.56, tw, 0.50, fc=TASK_FILL, ec="none")
    text(ax, tx + 0.08, top + 0.61, "Binary improvement task", weight="bold")
    text(ax, tx + 0.08, top + 0.77,
         f"{fmt(cnt('binary', 'all', 'n_visit_pairs'))} pairs · {fmt(cnt('binary', 'all', 'n_patients'))} patients\n"
         f"excludes {fmt(1153)} pairs with sCR at V(t)", size=FS["small"], color=INK2, ls=1.15)
    box(ax, tx, top + 1.12, tw, 0.40, fc=RNA_LIGHT, ec="none")
    text(ax, tx + 0.08, top + 1.16, f"Survival cohort: {fmt(surv.loc['PFS_primary', 'n_patients'])} patients",
         size=FS["small"] + 0.2, weight="bold")
    text(ax, tx + 0.08, top + 1.32, f"RNA by day 0 · PFS {fmt(surv.loc['PFS_primary', 'n_events'])} events · "
         f"OS {fmt(surv.loc['OS_secondary', 'n_events'])} deaths", size=FS["small"] - 0.2, color=INK)
    # --- arrows
    arrow(ax, (2.44, top + 0.40), (bx - 0.03, top + 0.40))
    arrow(ax, (2.44, top + 1.13), (bx - 0.03, top + 1.13))
    arrow(ax, (bx + 1.87, top + 0.40), (tx - 0.03, top + 0.25))
    arrow(ax, (bx + 1.87, top + 0.50), (tx - 0.03, top + 0.80))
    arrow(ax, (bx + 1.87, top + 1.16), (tx - 0.03, top + 1.27))
    return top + 1.52


def panel_b(ax, d, y0):
    """Visit-pair construction."""
    panel_letter(ax, "B", y0)
    heading(ax, "Visit-pair construction", y0)
    ty = y0 + 1.02
    x0, x1 = 0.30, 3.62
    ax.plot([x0, x1 - 0.1], [ty, ty], color=AXIS, lw=0.9, zorder=1)
    for x in (0.55, 0.95, 1.35):
        ax.plot([x, x], [ty - 0.06, ty + 0.06], color=AXIS, lw=0.9, zorder=1)
    text(ax, 0.95, ty + 0.12, "earlier visits", size=FS["small"], color=MUTED, ha="center")
    vt, vn = 2.15, 3.30
    ax.plot(vt, ty, "o", ms=6.5, mfc=LAYER_COLOR["B"], mec="white", mew=0.8, zorder=3)
    ax.plot(vn, ty, "o", ms=6.5, mfc="white", mec=INK, mew=0.9, zorder=3)
    arrow(ax, (vt + 0.1, ty), (vn - 0.1, ty), color=INK)
    text(ax, vt, ty - 0.16, "V(t)", weight="bold", ha="center", va="bottom")
    text(ax, vn, ty - 0.16, "V(t+1)", weight="bold", ha="center", va="bottom")
    # RNA sample marker (latest at or before V(t))
    rx = 1.75
    ax.plot(rx, ty + 0.42, "D", ms=5.0, mfc=LAYER_COLOR["D"], mec="white", mew=0.6, zorder=3)
    ax.plot([rx, rx], [ty + 0.36, ty + 0.04], color=LAYER_COLOR["D"], lw=0.8, ls=(0, (2, 1.5)), zorder=2)
    text(ax, rx - 0.10, ty + 0.34, "latest RNA sample at\nor before V(t) (never later)",
         size=FS["small"], color=INK2, ha="right", ls=1.15)
    # input / target brackets
    by = y0 + 0.42
    ax.plot([0.30, vt], [by + 0.18, by + 0.18], color=LAYER_COLOR["B"], lw=0.9)
    for x in (0.30, vt):
        ax.plot([x, x], [by + 0.18, by + 0.28], color=LAYER_COLOR["B"], lw=0.9)
    text(ax, 1.22, by, "Inputs: information available at or before V(t)", color=INK2, ha="center")
    text(ax, vn, by, "Target:\nIMWG at V(t+1)", color=INK2, ha="center", ls=1.15)
    text(ax, 0.30, y0 + 1.66,
         "Six-class: PD, SD, PR, VGPR, CR or sCR at V(t+1).\n"
         "Binary: V(t+1) is a better IMWG category than V(t).\n"
         "Every pair of a patient stays in one data partition.",
         size=FS["small"], color=INK2, ls=1.3)


def panel_c(ax, d, y0):
    """Nested information layers: each layer's bar is the cumulative count of source input
    features, drawn as segments (length proportional to the number of features)."""
    panel_letter(ax, "C", y0, x=3.88)
    heading(ax, "Information layers A–D", y0, x=4.16, sub="Nested feature sets, not algorithms")
    L = d["layers"]
    rows = [
        ("A", "Current clinical state", "labs, response, stage, symptoms"),
        ("B", "Longitudinal history", "previous values and changes"),
        ("C", "Treatment context", "line, regimen, drug classes"),
        ("D", "RNA pathway scores", "50 Hallmark scores + RNA age"),
    ]
    xl, xb, wmax, h, gap = 3.98, 5.52, 1.12, 0.34, 0.10
    ytop = y0 + 0.50
    nmax = int(L.loc["D", "source_input_features"])
    seg = {"A": 78, "B": 37, "C": 28, "D": 51}   # added source features per layer (Figure 2 data)
    for i, (k, name, sub) in enumerate(rows):
        y = ytop + i * (h + gap)
        text(ax, xl, y + 0.02, f"Layer {k}: {name}", weight="bold", size=FS["small"] + 0.2)
        text(ax, xl, y + 0.18, sub, size=FS["small"] - 0.2, color=INK2)
        start = 0.0
        for kk in "ABCD"[: i + 1]:
            w = wmax * seg[kk] / nmax
            ax.add_patch(Rectangle((xb + start, y), w - 0.012, h, fc=LAYER_COLOR[kk], ec="none", zorder=2))
            start += w
        n = int(L.loc[k, "source_input_features"])
        text(ax, xb + start + 0.06, y + h / 2, f"{n}", size=FS["body"], weight="bold", va="center")
    text(ax, xl, ytop + 4 * (h + gap) - 0.04,
         "Bars: cumulative source input features, coloured by the layer that adds them\n"
         "(D adds 50 pathway scores + RNA age, plus a derived RNA-availability indicator).",
         size=FS["small"] - 0.2, color=INK2, ls=1.25)


def panel_d(ax, d, y0):
    """Patient-level partitions and modelling framework."""
    cnt = d["cnt"]
    panel_letter(ax, "D", y0)
    heading(ax, "Patient-level partitions and modelling framework", y0,
            sub="Assigned once; shared by every layer and algorithm; no patient is in more than one partition")
    x0, x1, bh = 0.12, 4.45, 0.40
    ys = y0 + 0.50
    total = cnt("multiclass", "all", "n_patients")
    xs = x0
    for split in ("train", "val", "test"):
        n = cnt("multiclass", split, "n_patients")
        w = (x1 - x0) * n / total
        ax.add_patch(Rectangle((xs, ys), w - 0.03, bh, fc=SPLIT_COLOR[split], ec="none", zorder=2))
        nm = {"train": "Train", "val": "Val.", "test": "Test"}[split]
        lab = f"{nm}\n{n} patients" if split == "train" else f"{nm}\n{n}"
        text(ax, xs + 0.06, ys + bh / 2, lab, size=FS["small"], color=SPLIT_TEXT[split], va="center",
             weight="bold", ls=1.2)
        xs += w
    text(ax, x1, ys - 0.03, "Patients, six-class task (binary: 709 / 154 / 155). Pairs shown as six-class · binary", size=FS["small"] - 0.2,
         color=INK2, ha="right", va="bottom")
    ry = ys + bh + 0.14
    roles = [
        ("train", 0.12, 2.02, "Train: tuning and fitting",
         "Preprocessing fitted; hyperparameters chosen by\n5-fold patient-grouped CV on these patients only\n"
         f"Pairs: six-class {fmt(cnt('multiclass', 'train', 'n_visit_pairs'))} · binary {fmt(cnt('binary', 'train', 'n_visit_pairs'))}"),
        ("val", 2.20, 1.15, "Validation",
         "Descriptive only: no tuning,\nthreshold or calibration\n"
         f"Pairs: {fmt(cnt('multiclass', 'val', 'n_visit_pairs'))} · {fmt(cnt('binary', 'val', 'n_visit_pairs'))}"),
        ("test", 3.41, 1.04, "Test",
         "Scored once; patient-\nlevel bootstrap CIs\n"
         f"Pairs: {fmt(cnt('multiclass', 'test', 'n_visit_pairs'))} · {fmt(cnt('binary', 'test', 'n_visit_pairs'))}"),
    ]
    for split, rx, rw, ttl, body in roles:
        box(ax, rx, ry, rw, 0.68, fc="white", ec=AXIS, lw=0.7)
        ax.add_patch(Rectangle((rx, ry), 0.05, 0.68, fc=SPLIT_COLOR[split], ec="none", zorder=3))
        text(ax, rx + 0.12, ry + 0.05, ttl, weight="bold", size=FS["small"] + 0.2)
        text(ax, rx + 0.12, ry + 0.20, body, size=FS["small"] - 0.2, color=INK2, ls=1.2)
    mx = 4.72
    box(ax, mx, y0 + 0.50, 2.36, 1.32, fc="white", ec=AXIS)
    text(ax, mx + 0.10, y0 + 0.56, "Modelling framework", weight="bold")
    for i, (name, mk) in enumerate(ALGO_MARKER.items()):
        yy = y0 + 0.80 + i * 0.17
        ax.plot(mx + 0.17, yy + 0.05, mk, ms=4.2, mfc=INK2, mec="white", mew=0.4, zorder=5)
        text(ax, mx + 0.32, yy, name)
    text(ax, mx + 0.10, y0 + 1.34, "4 layers × 3 algorithms × 2 tasks = 24 models",
         size=FS["small"], weight="bold")
    text(ax, mx + 0.10, y0 + 1.50,
         "Binary: AUPRC (test improvement rate 18.3%)\nSix-class: macro-F1 (argmax)\nAlso AUROC, Brier score, RPS, log loss",
         size=FS["small"] - 0.2, color=INK2, ls=1.2)


def panel_e(ax, d, y0):
    """Downstream analyses and validation status."""
    panel_letter(ax, "E", y0)
    heading(ax, "Evaluation, downstream analyses and validation status", y0,
            sub="All evaluations are internal to CoMMpass; no external validation cohort was used")
    ov = d["overlap"]
    cols = [0.12, 2.00, 4.52]            # x of: analysis | patients / design | status
    wcol = [1.80, 2.45, 2.55]
    hy = y0 + 0.46
    for x, s in zip(cols, ["Analysis", "Patients and design", "Validation status"]):
        text(ax, x + 0.04, hy, s, weight="bold", color=INK2, size=FS["small"] + 0.2)
    ax.plot([0.12, 7.08], [hy + 0.17, hy + 0.17], color=AXIS, lw=0.7, zorder=1)
    rows = [
        ("Main prediction (24 models),\ncalibration and SHAP",
         "Test: 155 patients, scored once.\nTuning on train patients only",
         "Internal held-out test split;\ncalibration and SHAP descriptive", False),
        ("RNA contribution, uncertainty\nand subgroup analyses",
         f"Development: {RNAC['dev_patients']} train patients (CV reference).\n"
         f"Evaluation: {RNAC['heldout_patients']} validation + test patients\n"
         "with RNA; locked models, scored once",
         "Post hoc, exploratory; earlier pooled\nresults were seen first; not independent", False),
        ("Train-selected pathway\ncomposite (association)",
         f"Selection and discovery: {PATHWAY['dev_patients']} train patients.\n"
         f"Evaluation: {PATHWAY['heldout_patients']} validation + test patients\n"
         f"with RNA ({PATHWAY['heldout_val']} + {PATHWAY['heldout_test']})",
         "Post hoc, exploratory; earlier pooled\nresults were seen first; not independent", False),
        (f"Survival (PFS, OS) and\npathway coefficients",
         f"Development: {SURVT['dev_patients']} train patients (CV; Fig. 6).\n"
         f"Evaluation: {SURVT['heldout_patients']} validation + test patients\n"
         f"({ov['val']} + {ov['test']}); frozen models, scored once",
         "Post hoc, exploratory; earlier pooled\nresults were seen first; not independent", False),
        ("Geographic hold-out\n(US → Spain, Canada)",
         f"Tuning and fit: {GEO['fit']} US train-partition\n"
         f"patients; threshold, calibration: {GEO['calibration']} such\n"
         f"patients. Scored once: {GEO['evaluated']} Spain/Canada",
         f"Post hoc, exploratory; {GEO['excluded_us']} US validation/test-\n"
         "partition patients and Italy excluded", False),
    ]
    rh, y = 0.44, hy + 0.24
    for i, (a, p, s, flagged) in enumerate(rows):
        if flagged:
            box(ax, 0.10, y - 0.03, 6.98, rh - 0.02, fc=FLAG_FILL, ec="none", r=0.03, z=1)
        text(ax, cols[0] + 0.04, y + 0.02, a, weight="bold", size=FS["small"] + 0.2, ls=1.15)
        text(ax, cols[1] + 0.04, y + 0.02, p, size=FS["small"], color=INK2, ls=1.18)
        if flagged:
            w = flag_tag(ax, cols[2] + 0.04, y + 0.01, "Not independent")
            text(ax, cols[2] + 0.04, y + 0.20, s, size=FS["small"], color=FLAG, ls=1.18)
        else:
            text(ax, cols[2] + 0.04, y + 0.02, s, size=FS["small"], color=INK, ls=1.18)
        y += rh
    ax.plot([0.12, 7.08], [y - 0.02, y - 0.02], color=AXIS, lw=0.7, zorder=1)
    text(ax, 0.12, y + 0.04,
         "Main models and every redesigned analysis (RNA contribution, pathway composite, survival, geographic): tuning, selection, fitting and "
         "preprocessing use train-partition patients only; held-out\n"
         "patients are scored once. No decision threshold or recalibration enters any reported metric of the main models (probabilities are "
         "shown raw; validation patients are descriptive only);\n"
         "the geographic model's threshold and calibration use train-partition patients. The RNA-contribution, pathway, survival and "
         "geographic analyses are\n"
         "post hoc and exploratory (earlier pooled results were seen first); nothing here is externally validated, and CIs are "
         "descriptive and unadjusted.",
         size=FS["small"] - 0.2, color=INK2, ls=1.25)
    return y + 0.35


def build():
    d = load()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(FIG_H, 0)
    ax.axis("off")
    yA = 0.10
    yEnd = panel_a(ax, d, yA)
    yB = yEnd + 0.18
    panel_b(ax, d, yB)
    panel_c(ax, d, yB)
    yD = yB + 2.42
    panel_d(ax, d, yD)
    yE = yD + 2.12
    end = panel_e(ax, d, yE)
    print(f"content ends at {end:.2f} in of {FIG_H} in")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    fig = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in args.formats:
        out = OUT_DIR / f"{STEM}.{f}"
        fig.savefig(out, dpi=args.dpi, format=f)
        print("wrote", out)


if __name__ == "__main__":
    main()
