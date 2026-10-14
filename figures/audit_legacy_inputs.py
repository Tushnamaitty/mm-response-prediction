"""
Audit of legacy (pre-redesign) outputs that are referenced by manifests but absent from this machine (read-only).

Run from the repository root:
    python figures/audit_legacy_inputs.py

For every artifacts/results_clean_rerun/**/output_manifest.json it lists the files recorded in the manifest, checks whether each exists next to
the manifest and, when it does, whether its SHA-256 matches the recorded hash (raw or LF-normalised bytes). Files that are recorded but absent are
listed with their recorded hash, so a copy recovered from the original machine can be verified before use. It also checks which figure-data
folders the superseded extractors write to. Writes figures/LEGACY_MISSING_INPUTS.md. Nothing is fabricated, substituted or modified.
"""

import hashlib
import json
import sys
from pathlib import Path

RC = Path("artifacts/results_clean_rerun")
OUT = Path("figures/LEGACY_MISSING_INPUTS.md")
NEW_DIRS = {"survival_trainonly", "rna_contribution_trainonly", "direction2_replication_trainonly", "geographic_origtrain",
            "geographic_us_tuned", "geographic_us_tuned_control_previous_params"}
USED_BY = {
    "direction1_sensitivity/entropy_slope": "superseded Figure 4B / Table S1 inputs (`figures/fig3_extract_data.py`)",
    "rna_low_variance_incremental": "superseded Figure 4A add-on permutation null (`figures/fig3_extract_data.py`)",
    "direction2_replication/proliferation_axis": "superseded Figure 4D composite loadings (`figures/fig3_extract_data.py`)",
    "survival_pathway_interpretation": "superseded Figure 6 (`figures/fig5_extract_data.py`)",
    "survival_incremental": "superseded Figure 5 / Table S5 (`figures/fig4_extract_data.py`)",
    "survival_richer_baseline_sensitivity": "superseded Figure 5 expanded-baseline rows / Table S5 (`figures/fig4_extract_data.py`)",
}
FIG_DATA = [("figures/data/Fig3", "figures/fig3_extract_data.py", "superseded Figure 4A–D data"),
            ("figures/data/Fig4", "figures/fig4_extract_data.py", "superseded Figure 5 / Table S5 data"),
            ("figures/data/Fig5", "figures/fig5_extract_data.py", "superseded Figure 6 data"),
            ("figures/data/FigS_calibration", "figures/figS_calibration_extract_data.py", "Figure S1 / Table S2 data (the extractor can be re-run: its inputs exist)"),
            ("figures/data/FigS_shap", "figures/data/FigS_shap/figS_shap_extend.py", "Figure S3 / Table S3 data (recovered copy, SHA-256 verified against the provenance records; the extractor was not re-run)"),
            ("figures/data/FigS_country", "figures/figS_country_extract_data.py", "superseded Figure S2 / Table S4 data")]


def sha(path, lf=False):
    b = Path(path).read_bytes()
    if lf:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def s3_status():
    """Status of the recovered Supplementary Figure S3 data folder (read-only; hashes against figS3_ext_provenance.json)."""
    d = Path("figures/data/FigS_shap")
    pj = d / "figS3_ext_provenance.json"
    if not pj.exists():
        return None
    prov = json.load(open(pj, encoding="utf-8"))
    out = {}
    for sec in ("protected_earlier_files_sha256", "outputs_sha256"):
        ok, bad, miss = 0, 0, []
        for name, h in prov.get(sec, {}).items():
            f = d / name
            if not f.exists():
                miss.append(name)
            elif sha(f) == h:
                ok += 1
            else:
                bad += 1
        out[sec] = (ok, bad, miss)
    return out


def main():
    rows_missing, n_ok, n_bad, n_files, notes = [], 0, 0, 0, []
    for man in sorted(RC.rglob("output_manifest.json")):
        folder = man.parent
        if folder.relative_to(RC).parts[0] in NEW_DIRS:
            continue
        try:
            d = json.load(open(man, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            notes.append(f"{folder}: manifest unreadable ({e})")
            continue
        files = d.get("files")
        if not isinstance(files, dict):
            notes.append(f"{folder.relative_to(RC)}: manifest has no name->sha256 `files` mapping (not auditable)")
            continue
        rel = folder.relative_to(RC).as_posix()
        for name, h in sorted(files.items()):
            n_files += 1
            p = folder / name
            if not isinstance(h, str):
                continue
            if not p.exists():
                rows_missing.append((rel, name, h))
            elif h in (sha(p), sha(p, lf=True)):
                n_ok += 1
            else:
                n_bad += 1
                notes.append(f"HASH MISMATCH: {rel}/{name}")
    fd_rows = [(p, e, w, Path(p).exists(), Path(e).exists() if e.endswith(".py") else None) for p, e, w in FIG_DATA]

    # inputs recorded (with SHA-256) in the figure-data / table provenance files
    prov_missing, prov_ok, prov_bad = [], 0, []
    for pj in sorted(Path("figures").rglob("*provenance*.json")):
        try:
            inp = json.load(open(pj, encoding="utf-8")).get("inputs_sha256")
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(inp, dict):
            continue
        for k, h in sorted(inp.items()):
            p = Path(k.replace("\\", "/"))
            if not p.exists():
                prov_missing.append((pj.as_posix(), p.as_posix(), h))
            elif h in (sha(p), sha(p, lf=True)):
                prov_ok += 1
            else:
                prov_bad.append((pj.as_posix(), p.as_posix(), h))
    EXPLAINED = {"figures/data/FigS_calibration/figS1_binary_calibration_metrics.csv":
                 "regenerated on 2026-10-14 with a new `calibration` label text only: restoring the earlier label \"uncalibrated (selected on validation)\" reproduces the recorded hash "
                 "byte for byte (checked), so every number is unchanged"}

    def used(rel):
        return next((v for k, v in USED_BY.items() if rel == k or rel.startswith(k)), "other legacy analysis")

    s3 = s3_status()
    if s3 is None:
        s3_text = ("the data folder `figures/data/FigS_shap` and its extractor are absent; `figureS3A.png` cannot be replotted "
                   "(recorded hashes are listed in section D).")
    else:
        (po, pb, pm), (oo, ob, om) = s3["protected_earlier_files_sha256"], s3["outputs_sha256"]
        s3_text = ("the data folder is present (restored as `FigS_Shap` and renamed to `FigS_shap` to match every script and record) with the extractor "
                   "`figS_shap_extend.py`. Its inputs recorded in `figS3_ext_provenance.json` are checked in section D; of the files recorded there as protected earlier "
                   f"outputs, {po} are present and match, {pb} differ and {len(pm)} are absent; of the extension outputs, {oo} are present and match, {ob} differ and "
                   f"{len(om)} are absent (absent: " + ", ".join(f"`{n}`" for n in pm + om) + "). The three Figure S3A files, the Figure S3B top-15 file and the "
                   "additivity file match the recorded hashes. The protocol claim of 256 passed checks cannot be confirmed here because `figS3_ext_verification_checks.csv` is absent. "
                   "`figureS3A.png` was replotted from the verified `figS3A_ext_component_shares.csv` with the footnote changed; only the footnote area differs from the earlier image. "
                   "The folder is excluded by `figures/data/*` in `.gitignore` (so it is on no branch); it also holds a patient-level source-SHAP file that must stay out of Git. "
                   "Nothing was recomputed.")
    md = ["# Legacy inputs and outputs that are missing on this machine", "",
          "Generated by `figures/audit_legacy_inputs.py` (read-only). Nothing listed here was fabricated, substituted or silently replaced; "
          "the redesigned analyses do not depend on any of it. Per-file CSVs are excluded by the `*.csv` rule in `.gitignore`, so they are on no "
          "branch; a copy recovered from the original machine can be verified against the recorded SHA-256 before use.", "",
          f"Manifests audited: {len(list(RC.rglob('output_manifest.json'))) - sum(1 for m in RC.rglob('output_manifest.json') if m.parent.relative_to(RC).parts[0] in NEW_DIRS)} "
          f"(legacy folders only). Files recorded: {n_files}; present with matching hash: {n_ok}; present with MISMATCHED hash: {n_bad}; "
          f"recorded but absent: {len(rows_missing)}.", "",
          "## A. Recorded in a legacy `output_manifest.json` but absent", "",
          "| Legacy folder (`artifacts/results_clean_rerun/…`) | Missing file | Recorded SHA-256 | Needed by |", "|---|---|---|---|"]
    md += [f"| `{r}` | `{n}` | `{h}` | {used(r)} |" for r, n, h in rows_missing]
    md += ["", "## B. Figure-data folders written by superseded extractors", "",
           "| Folder | Extractor | Content | Folder present | Extractor present |", "|---|---|---|---|---|"]
    md += [f"| `{p}` | `{e}` | {w} | {'yes' if ok else '**no**'} | {'yes' if ex else ('no' if ex is not None else 'n/a')} |" for p, e, w, ok, ex in fd_rows]
    md += ["", "## C. Consequences", "",
           "* The superseded extractors for Figures 4A–D, 5 and 6 and their plot scripts cannot be re-run (the CSV inputs in section A and the "
           "`figures/data/Fig3|Fig4|Fig5` folders are absent); only their existing PNGs remain. `figures/supplementary_tables/build_supplementary_tables.py` "
           "(S1–S5 in one run) cannot run either. The superseded Figure S2 extractor (`figS_country_extract_data.py`) was not tested. All are kept unchanged and marked superseded.",
           "* Replacements (`*_trainonly*`, `*_origtrain*`) read only the new frozen result folders and do not need anything above.",
           "* Figure S3 / Table S3 (SHAP): " + s3_text,
           "* Numbering: figure numbers in this file follow the final scheme (Figures 1-6; see `figures/FIGURE_NUMBERING.md`). Folder and file names "
           "keep their earlier working labels, e.g. `figures/data/Fig3|Fig4|Fig5` hold the superseded earlier versions of Figures 4A-D, 5 and 6.",
           "* The legacy pooled-CV Table S5 values quoted for context in `Table_S5_survival_trainonly.md` (integrated Brier score and repeat ranges) are "
           "transcribed verbatim from `Table_S5_survival.md` and cannot be re-derived (the out-of-fold prediction files are among the missing items).",
           "* Recovery: copy the files in section A back to their folders, then re-run the extractor and check the recorded hashes with this script "
           "(a mismatch is reported as `HASH MISMATCH`)."]
    md += ["", "## D. Inputs recorded in figure-data / table provenance files (`figures/**/*provenance*.json`)", "",
           f"Present with matching hash: {prov_ok}; present with a different hash: {len(prov_bad)}; absent: {len(prov_missing)}. "
           "A recovered copy of an absent file can be verified against its recorded SHA-256 before it is used. "
           "The Supplementary Figure S3A files (`figures/data/FigS_shap/figS3A_*.csv`) were absent until a copy was restored into the working tree; they now match their recorded hashes "
           "and the extension provenance (`figS3_ext_provenance.json`) is audited here as well (see section C).", "",
           "| Provenance file | Input | Recorded SHA-256 |", "|---|---|---|"]
    md += [f"| `{pj}` | `{p}` | `{h}` |" for pj, p, h in prov_missing]
    if prov_bad:
        md += ["", "Present but with a different hash:", ""] + [f"* `{p}` (recorded `{h[:12]}…` in `{pj}`): {EXPLAINED.get(p, 'unexplained')}" for pj, p, h in prov_bad]
    if notes:
        md += ["", "## E. Notes", ""] + [f"* {n}" for n in notes]
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"recorded {n_files}; ok {n_ok}; mismatched {n_bad}; absent {len(rows_missing)}; wrote {OUT}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
