"""
Figure 5A (train-only redesign) - Discrimination for progression-free and overall survival in held-out validation + test patients:
clinical baseline model vs clinical baseline + RNA (C-index), with Delta C and its 95% patient-bootstrap CI. Models were fitted on the
473 train patients only and scored once on the 201 held-out patients.

Run from the repository root:
    python figures/publication_redesign/Fig4/figure4A_trainonly_plot.py

Inputs (verified by figures/fig45_trainonly_extract_data.py; read-only):
    figures/data/Fig45_trainonly/fig4A_trainonly_cindex_summary.csv
    figures/data/Fig45_trainonly/fig4A_trainonly_partition_descriptive.csv
Replaces figure4A.png (pooled 5x5 CV over all 674 patients), which is left unchanged.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIR = Path("figures/data/Fig45_trainonly")
OUT = Path("figures/publication_redesign/Fig4/figure4A_trainonly.png")

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYER_COLOR = {"A": "#86b6ef", "B": "#3987e5", "C": "#184f95", "D": "#eb6834"}
BASE_COLOR = "#52514e"
FS = {"panel": 10, "title": 7.5, "body": 6.5, "small": 6.0, "tick": 6}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS["body"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})
MINUS = "−"
FIG_W, FIG_H = 7.2, 3.6


def require(cond, msg):
    if not cond:
        sys.exit(f"Figure 5A data check failed: {msg}")


def panel(ax, s, part=None):
    ep = s.endpoint
    if part is not None:                      # descriptive partition-level lines (PFS only)
        for _, p in part.iterrows():
            ax.plot([0, 1], [p.c_C0, p.c_C0_RNA], color=GRID, lw=0.9, zorder=1)
            ax.plot([0, 1], [p.c_C0, p.c_C0_RNA], ls="none", marker="o", ms=3.2, mfc="white", mec=MUTED, mew=0.7, zorder=2)
            ax.text(1.05, p.c_C0_RNA, f"{p.partition}", ha="left", va="center", fontsize=FS["small"] - 0.4, color=MUTED)
    ax.plot([0, 1], [s.c_index_baseline, s.c_index_plus_rna], color=INK2, lw=1.2, zorder=3)
    ax.plot(0, s.c_index_baseline, "o", ms=7, mfc=BASE_COLOR, mec="white", mew=0.8, zorder=4)
    ax.plot(1, s.c_index_plus_rna, "o", ms=7, mfc=LAYER_COLOR["D"], mec="white", mew=0.8, zorder=4)
    ax.text(-0.09, s.c_index_baseline, f"{s.c_index_baseline:.3f}", ha="right", va="center", fontsize=FS["body"], color=INK)
    ax.text(1.13, s.c_index_plus_rna, f"{s.c_index_plus_rna:.3f}", ha="left", va="center", fontsize=FS["body"], color=INK, fontweight="bold")
    ax.set_xlim(-0.45, 1.6)
    ax.set_ylim(0.57, 0.775)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clinical\nbaseline", "Clinical baseline\n+ RNA"], fontsize=FS["body"], color=INK)
    ax.set_yticks([0.58, 0.62, 0.66, 0.70, 0.74])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    title = "Progression-free survival (PFS)" if ep == "PFS" else "Overall survival (OS)"
    ax.set_title(f"{title}\n{int(s.n_patients)} held-out patients · {int(s.n_events)} {s.event_type}", loc="left", fontsize=FS["title"],
                 fontweight="bold", color=INK, pad=4, linespacing=1.3)
    txt = f"ΔC = {s.delta_c:+.3f}\n95% CI {s.ci_lower:.3f} to {s.ci_upper:.3f}".replace("-", MINUS)
    if ep == "PFS":
        txt += f"\npermutation p = {s.permutation_p_one_sided:.4f}\ncriterion not met (CI includes 0)"
    ax.text(0.5, 0.772, txt, ha="center", va="top", fontsize=FS["body"], color=INK, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=AXIS, lw=0.6))


def main():
    summ = pd.read_csv(DIR / "fig4A_trainonly_cindex_summary.csv").set_index("endpoint")
    part = pd.read_csv(DIR / "fig4A_trainonly_partition_descriptive.csv")
    require((int(summ.loc["PFS", "n_patients"]), int(summ.loc["PFS", "n_events"]), int(summ.loc["OS", "n_patients"]), int(summ.loc["OS", "n_events"]))
            == (201, 149, 201, 82), "held-out denominators")
    for ep in ("PFS", "OS"):
        s = summ.loc[ep]
        require(abs(s.c_index_plus_rna - s.c_index_baseline - s.delta_c) < 1e-12 and s.ci_lower < s.delta_c < s.ci_upper, f"{ep} delta C")
    require(summ.loc["PFS", "ci_lower"] < 0, "caption assumes the PFS CI includes 0")
    require(abs(summ.loc["PFS", "permutation_p_one_sided"] - 1 / 10001) < 1e-12, "PFS permutation p = 1/10,001")
    require(pd.isna(summ.loc["OS", "permutation_p_one_sided"]), "OS has no permutation test")
    require(sorted(part.partition) == ["test", "val"] and int(part.n_patients.sum()) == 201, "partition rows")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(0.012, 0.985, "A", fontsize=FS["panel"], fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.047, 0.983, "Clinical baseline vs baseline + RNA discrimination of PFS and OS in held-out patients (C-index; higher is better)",
             fontsize=FS["title"], fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.047, 0.93, "EXPLORATORY · models fitted on train patients only; scored once in held-out validation + test patients "
             "(post hoc; not independent validation)", fontsize=FS["small"], color=LAYER_COLOR["D"], fontweight="bold", ha="left", va="top")
    for i, ep in enumerate(["PFS", "OS"]):
        ax = fig.add_axes([0.1 + i * 0.45, 0.2, 0.36, 0.54])
        s = summ.loc[ep].copy()
        s["endpoint"] = ep
        panel(ax, s, part if ep == "PFS" else None)
        if i == 0:
            ax.set_ylabel("Harrell's C-index", fontsize=FS["body"], labelpad=2)
    fig.text(0.047, 0.085, "●", fontsize=8, color=BASE_COLOR, va="center")
    fig.text(0.064, 0.085, "/", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.074, 0.085, "●", fontsize=8, color=LAYER_COLOR["D"], va="center")
    fig.text(0.092, 0.085, "held-out validation + test patients", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.33, 0.085, "○", fontsize=7, color=MUTED, va="center")
    fig.text(0.347, 0.085, "validation-only and test-only patients (PFS; descriptive, not tested)", fontsize=FS["small"], color=INK2, va="center")
    fig.text(0.047, 0.04, "ΔC = C(baseline + RNA) − C(baseline); 95% patient-bootstrap CI (2,000 resamples). Permutation: RNA vectors permuted among held-out "
             "patients (B = 10,000; PFS only).\nThe pre-specified PFS criterion (ΔC > 0, CI lower bound > 0, permutation p ≤ 0.025) was not met; "
             "this is not evidence of no effect.", fontsize=FS["small"], color=MUTED, va="center", linespacing=1.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
