# Final figure numbering (Scheme A, applied 2026-10-14)

Scheme A: the new study overview is **Figure 1**, the earlier Figure 1 (study design) becomes **Figure 2**, and the earlier Figures 2–5
become **Figures 3–6**. Supplementary Figures S1–S3 and Supplementary Tables S1–S6 and Table 1 keep their numbers. No analysis was rerun and no
result changed; only labels were changed (a one-off scripted shift of every main-figure reference in the manuscript-facing files, followed by a
line-by-line review). The pre-application inventory and the two candidate schemes are kept in `figures/FIGURE_NUMBERING_AUDIT.md`.

## Mapping

| Manuscript label | Content | Earlier label | Working names (unchanged: image, script, data folder) |
|---|---|---|---|
| Figure 1 | Study overview | New Figure 1 | `publication_redesign/fig1_new/figure1_new.{png,pdf,svg}`, `figure1_new_plot.py` |
| Figure 2 | Study design, information layers, RNA timing | Figure 1 | `publication_redesign/Fig1/figure1.png`, `figure1_plot.py`; `data/Fig1/`, `fig1_extract_data.py` |
| Figure 3 (A–C) | Next-visit performance across layers A–D | Figure 2 | `publication_redesign/Fig2/figure2{A_binary_auprc,B_multiclass_macrof1,C_incremental_delta}.png`, `figure2_plot.py`; `data/Fig2/`; Table S6 `Table_S6_full_figure2_performance.md` |
| Figure 4A–C | RNA contribution (train-only, held-out) | Figure 3A–C | `Fig3/figure3{A,B,C}_trainonly.png`; `data/Fig3ABC_trainonly/`, `fig3ABC_trainonly_extract_data.py`; Table S1 `Table_S1_rna_subgroups_trainonly.*` |
| Figure 4D | Train-selected pathway composite | Figure 3D | `Fig3/figure3D_trainonly.png`; `data/Fig3D_trainonly/`, `fig3D_trainonly_extract_data.py` |
| Figure 5A–B | Survival: incremental prognostic value of RNA | Figure 4A–B | `Fig4/figure4{A,B}_trainonly.png`; `data/Fig45_trainonly/`, `fig45_trainonly_extract_data.py`; Table S5 `Table_S5_survival_trainonly.*` |
| Figure 6A–B | Pathway coefficient patterns (PFS vs OS) | Figure 5A–B | `Fig5/figure5{A,B}_trainonly.png`; `data/Fig45_trainonly/` |
| Supplementary Figures S1, S2, S3 | Calibration (raw probabilities), geographic hold-out, SHAP | unchanged | `FigS_calibration/`, `FigS_country_origtrain/`, `FigS_shap/` |

Superseded earlier versions (kept unchanged, not for the manuscript): `Fig3/figure3{A,B,C,D}.png` (earlier Figure 4A–D), `Fig4/figure4{A,B}.png` (earlier
Figure 5), `Fig5/figure5{A,B}.png` (earlier Figure 6), `FigS_country*` and `FigS_country_ustuned/` (earlier Figure S2), `Table_S1_fig3C_subgroups.md`, `Table_S5_survival.md`,
`Table_S4_geographic_robustness{,_ustuned}.md`. See `figures/SUPERSEDED_ANALYSES.md`.

## What was updated

* Captions (`FIGURE_CAPTIONS_DRAFT.md`: headings, panel and cross-references, revision note, unresolved items), `SUPERSEDED_ANALYSES.md`, `LEGACY_MISSING_INPUTS.md` (and its generator
  `audit_legacy_inputs.py`), `supplementary_tables/README.md`, Table 1, Tables S1, S5 and S6 (titles, notes) and their builders, `fig1_new/README.md`, `check_trainonly_captions.py`.
* Docstrings, comments and error messages of the live plot scripts (Figures 1–6 and Supplementary Figures S1–S3) so that they use the final numbers (no logic changed).
* **Images with drawn figure labels, replotted from the existing tracked data (no data or logic change):** Figure 1 (panel E, "Fig. 6"), Figure 2 (panel A label "Figures 5–6"),
  Figure 3A and 3B (footnotes "Figure 2" and "Figure 3A"; Figure 3C draws no label and is pixel-identical), Figure 6A and 6B ("Figure 4D family"); Supplementary Figure S3A (footnote wording, replotted 2026-10-15 from the restored data). Only the label text strips differ
  (checked by pixel comparison with the previous images; at most 1,101 pixels of each image changed). Figure 2 (`Fig1/figure1_plot.py`) needs a patient-level RNA-sample file that is not in the repository
  (removed for privacy); it was regenerated locally with `fig1_extract_data.py`, used for the replot and deleted again (the tracked aggregate files in `data/Fig1/` were verified unchanged by hash).
* The phrase "reporting reference algorithm" was removed (captions, Table S6, S1 data definitions, and the footnote drawn in `figureS3A.png`); no algorithm is called best.

## Residual working labels (deliberately not changed)

1. **File, folder and data-folder names** in the table above (renaming would break script paths and hash records).
2. **Frozen protocols and hash-chained artifacts** (`explainability/*_PROTOCOL.md`, `.sha256`, `explainability/*_trainonly_*.py`, `artifacts/results_clean_rerun/**`) still say "Figures 3A-C", "Figures 4 and 5", "Figure 5 refit".
   Read them with the mapping above (their Figure 3A–C = Figure 4A–C, Figures 4–5 = Figures 5–6). They must not be edited.
3. **Extractors and their data outputs** keep working labels in docstrings and check names (for example `fig3ABC_trainonly_extract_data.py`, the row "Figure 3D family members concordant here"
   in `data/Fig45_trainonly/fig45_trainonly_verification_checks.csv`, and the superseded extractors); regenerating them only to relabel was not done.
4. **Superseded scripts, images and tables** (list above) and the legacy builder `supplementary_tables/build_supplementary_tables.py` keep their earlier labels.
5. **Supplementary Figure S3A image (resolved 2026-10-15):** its drawn footnote ("reporting reference algorithm, not a validation-selected best model") was replaced and the image regenerated from the restored,
   hash-verified `figures/data/FigS_shap/` data; only the footnote band differs from the earlier image (`SUPERSEDED_ANALYSES.md`, section "Supplementary Figure S3A"; `LEGACY_MISSING_INPUTS.md` sections C and D). Residual: the folder name `FigS_shap` and the source CSV
   names keep the working label (the restored folder was renamed from `FigS_Shap` to match every script and record); 13 protected earlier files and 8 extension outputs recorded in `figS3_ext_provenance.json` are still absent.
6. `opus_plan.md` (repository root) is a historical planning note and keeps the earlier labels.

## Check

`python figures/check_figure_numbering.py` verifies the headings and cross-references of the captions, the table titles, the live plot scripts and the text drawn in the replotted images
(it reads the plot-script strings; the PNGs were compared by pixels when they were regenerated).
