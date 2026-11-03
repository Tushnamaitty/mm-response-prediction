# MM treatment-response project — handoff (written 2026-10-26)

Project: MMRF CoMMpass next-visit IMWG response prediction, revised after Professor Song's feedback.
PC path: `C:\mm-response-prediction`. Remote: `https://github.com/Tushnamaitty/mm-response-prediction.git` (public).

Legend: **[V]** verified from repo files/commands during this chat · **[A]** assumption / inference · **[U]** not verified.
Nothing in this file contains patient-level data.

---

## 1. Git state

**Verified live on 2026-10-26 [V]**

| Item | Value |
|---|---|
| Branch | `new-revisions` |
| HEAD | `ae36c4d038cd42f45c9e71c7bebd5bad29c1d190` ("Update figures and train-only analyses") |
| `git ls-remote origin new-revisions` | `ae36c4d038cd42f45c9e71c7bebd5bad29c1d190` (same as HEAD; ahead/behind = 0/0) |
| Staged / modified tracked files | 0 / 0 |
| Remote `rna-alignment-and-model-prep` | `454709a71a7c59c433eccfd6057d4e8acece3ee2` |
| Remote `main` | `2398d69d58324a72078167baae5f9d0f73756be3` |
| Local `main` | `586e650` — 1 commit ahead of the local `origin/main` ref (`2398d69`); unrelated to this work, untouched |

**History [V earlier in chat; not re-run today]**
- `new-revisions` was created 2026-10-14 10:23:04 from `rna-alignment-and-model-prep` at `454709a`; pushed 10:24:51.
- One commit on top: `ae36c4d`, author/committer Jash Nikumbhe, 2026-10-15 02:19:26, pushed 02:19:38. 85 files, +9,561/−724. Unsigned, no co-author trailer. **It was not made by Claude.**
- All figure/doc edits made in this chat's earlier sessions are inside `ae36c4d` (working tree has 0 modified tracked files).
- PR #1 exists only for `rna-alignment-and-model-prep`; no PR for `new-revisions` was seen [PR state not re-checked today].
- Claude never staged, committed or pushed anything in this chat.

**Untracked on the PC at the end (exact `git status --short`) [V]**
```
?? "Revised_MM Treatment Response Study.pdf"
?? "Revised_MM Treatment Response Study.pptx"
?? artifacts/results_clean_rerun/direction2_replication_trainonly/
?? artifacts/results_clean_rerun/geographic_origtrain/
?? artifacts/results_clean_rerun/geographic_us_tuned/
?? artifacts/results_clean_rerun/geographic_us_tuned_control_previous_params/
?? artifacts/results_clean_rerun/rna_contribution_trainonly/
?? artifacts/results_clean_rerun/survival_trainonly/
?? figures/publication_redesign.zip
?? figures/supplementary_tables/Table_S4_geographic_robustness_ustuned.md
?? figures/supplementary_tables/build_table_s4_ustuned.py
?? modeling/us_to_spain_canada_model_c_ustuned.py
?? opus_plan.md
?? repo_structure.txt
?? MYELoma_PROJECT_HANDOFF.md        (this file, created afterwards)
```
- `figures/publication_redesign.zip`: contents not inspected [U].
- `artifacts/results_clean_rerun/survival_trainonly/frozen_models.json` is hidden by a **local-only** rule in `.git/info/exclude` (not in the clone). It stores step functions derived from the 473 train patients — do not commit. Its SHA-256 (`3b4b79a6…`) is recorded in `survival_trainonly/heldout_report.json`.
- `.gitignore` ignores `data/` and `figures/data/*` (except the tracked `figures/data/Fig1` and `Fig2` files).
- `core.autocrlf=true` on the PC: worktree text is CRLF, blobs are LF. Protocol hashes are computed on LF-normalised text.

---

## 2. Professor Song's two comments and how they are addressed

**Comment 1 (verbatim):** "Please ensure that validation patients are excluded from all model-development steps, including hyperparameter tuning and pathway selection. The geographic analysis currently includes patients previously used for tuning, and the pathway-selection analysis includes test patients. These analyses should be redesigned before being presented as independent validation."

**Comment 2 (verbatim):** "I suggest adding a new Figure 1 with an overview of the entire study, illustrating the data sources, patient cohorts, clinical and RNA inputs, prediction tasks, modeling framework, and validation strategy... The current figure 1 can be figure 2."

| Concern | How it was addressed | Status |
|---|---|---|
| Geographic analysis used tuning patients | Redesigned (`modeling/us_to_spain_canada_model_c_origtrain.py`): tune/fit/preprocess on 439 US original-training-partition patients (438 binary); calibration method + binary threshold (0.20) on 96 other US train-partition patients; 236 US validation/test-partition patients excluded; evaluated once on 116 Spain/Canada patients (80 + 36). | Counts and recorded overlap counter (0) match [V counts; counter read, not recomputed; development patient IDs are not stored] |
| Pathway selection used test patients | Selection, composite (PC1, 68.1% variance) and discovery in 371 train patients only; composite frozen, then scored once on 160 validation+test patients. Protocol hash recorded before held-out outcomes were read. | Hashes match [V] |
| RNA-contribution and survival analyses | Fitted on train only (716 patients for RNA contribution; 473 for survival), frozen, scored once on held-out (213; 201). | Hash chain, held-out checks (RNA 20/20, survival 13/13) and survival development predictions = 473 train patients [V] |
| "Independent validation" | **Not claimed.** Held-out patients appeared in earlier pooled analyses, so everything is labelled post hoc/exploratory, "not independent validation". | Labels present on figures, captions, deck [V] |
| New Figure 1 | `fig1_new/figure1_new.{png,pdf,svg}`: A data sources/cohorts, B visit-pair construction, C layers A–D, D partitions + modelling framework (3 algorithms × 4 layers × 2 tasks = 24 models), E per-analysis table of development patients/evaluation patients/validation status. Old Figure 1 became Figure 2. | Present in the deck as slides 3–4 [V] |

---

## 3. Analysis and figure revisions

**Numbering ("Scheme A"). File/folder names keep the old working labels; map is in `figures/FIGURE_NUMBERING.md`. [V]**

| Manuscript figure | Current file under `figures/publication_redesign/` |
|---|---|
| Figure 1 overview (new) | `fig1_new/figure1_new.png` (+ `.pdf`, `.svg`) |
| Figure 2 design/cohort/layers/RNA timing | `Fig1/figure1.png` |
| Figure 3A / 3B / 3C | `Fig2/figure2A_binary_auprc.png`, `figure2B_multiclass_macrof1.png`, `figure2C_incremental_delta.png` |
| Figure 4A / 4B / 4C / 4D | `Fig3/figure3{A,B,C,D}_trainonly.png` |
| Figure 5A / 5B (survival) | `Fig4/figure4{A,B}_trainonly.png` |
| Figure 6A / 6B (pathway coefficients) | `Fig5/figure5{A,B}_trainonly.png` |
| S1A / S1B (calibration) | `FigS_calibration/figureS1{A,B}.png` |
| S2A / S2B (geographic) | `FigS_country_origtrain/figureS2{A,B}.png` |
| S3A / S3B (SHAP) | `FigS_shap/figureS3{A,B}.png` |

**Superseded look-alikes still tracked — do not use:** `Fig3/figure3{A–D}.png`, `Fig4/figure4{A,B}.png`, `Fig5/figure5{A,B}.png`, `FigS_country/`, `FigS_country_ustuned/` (without `_trainonly`/`origtrain`).

**Safeguards and hashes [V]**
- Protocols (LF-normalised SHA-256): pathway `3b3c8d3c…` (**no `.sha256` sidecar**; hash recorded in artifacts), RNA `f93323a3…`, survival `8e26e0a1…`. Files: `explainability/{PATHWAY,RNA_CONTRIBUTION,SURVIVAL}_TRAINONLY_PROTOCOL.md`.
- Frozen artifacts: `survival_trainonly/frozen_models.json` `3b4b79a6…`; `rna_contribution_trainonly/frozen_development.json` `5148bed2…`; `direction2_replication_trainonly/frozen_composite_artifact.json` `d541eb70…` (no sidecar).
- Workflow: protocol + hash → Stage D (held-out outcomes unread) → hashed frozen artifact → Stage H scores once. All 46 manifest-listed result files match their hashes.
- Partitions (patients): train/val/test 716/154/155 (binary 709/154/155). Visit pairs six-class 9,539/1,833/2,079; binary 8,709/1,659/1,930.

**Documents (all tracked) [V]:** `figures/FIGURE_CAPTIONS_DRAFT.md`, `figures/FIGURE_NUMBERING.md`, `figures/SUPERSEDED_ANALYSES.md`, `figures/LEGACY_MISSING_INPUTS.md` (generated by `figures/audit_legacy_inputs.py`), `figures/supplementary_tables/` (Table 1, S1–S6 `.md`, builders, `README.md`).

**Audits completed [V]**
- 18 live plot scripts re-rendered → pixel-identical to repo images (Fig 2's `Fig1/figure1_plot.py` excluded: needs a deleted patient-level file).
- Tables S1–S6 builders reproduce the `.md` byte-for-byte; Table 1 builder cannot run (missing legacy file) but counts were checked directly.
- S3 (SHAP) shares, CIs and S3B whiskers independently recomputed (binary to 1e-14 pp; six-class to 2.5e-5); `figures/verify_figS3_inputs.py` (156 checks; needs the local patient-level `.gz`). Per-class six-class CIs not independently recomputed.
- S3A footnote corrected ("Locked XGBoost layer D, used as a single fixed model; no algorithm is claimed to perform better."); data folder renamed `figures/data/FigS_Shap` → `FigS_shap` (gitignored).
- Checks: `figures/check_figure_numbering.py`, `figures/check_trainonly_captions.py` [passed on the PC; behaviour on a fresh clone not tested — the second reads gitignored data, so expect failure there (A)].

**Key figure values (for reference) [V from captions/data]:** test improvement rate 0.183; Fig 4A direct ΔAUPRC +0.0109 [−0.0026, 0.0248], six-class ΔRPS −0.06×10⁻³ [−1.14, 1.04]; regularized add-on ΔRPS −0.18×10⁻³ [−0.63, 0.28] (permutation p 0.011, rule not met → inconclusive); Fig 4B interaction Holm p = 0.94 (both tasks); Fig 4D composite OR 0.76 (0.67–0.87), prespecified global test null; Fig 5A PFS ΔC +0.028 [−0.019, 0.074], OS +0.010 [−0.045, 0.065]; S2 pooled AUPRC 0.426 [0.374, 0.484], macro-F1 0.629 [0.585, 0.658]; S3A RNA-pathway net share 3.3% (binary) / 7.3% (six-class).

---

## 4. Research limitations and unresolved issues

1. **Exploratory, not independent.** All redesigned analyses are post hoc; held-out patients were seen in earlier pooled analyses. Binary and six-class results must be reported separately; validated vs exploratory findings kept distinct.
2. **Main-model tuning not re-verified [U].** Figure 1E states tuning used train patients only, and main results were not modified during the revision (file-time based, [A]), but tuning code/logs for the 24 main models were not re-inspected.
3. **Patient-separation checks:** geographic overlap counter read from the metrics JSON (0), not recomputed [U]; development patient IDs are not stored.
4. **Validation-informed presentation choice:** showing raw (uncalibrated) probabilities in S1 followed an earlier exploratory comparison within validation patients (disclosed in the S1 caption; not yet on slide 20).
5. **No recorded a priori rationale for using XGBoost** for Figures 4–6/S1–S3; results do not show XGBoost is better. Never call any algorithm "best".
6. **Table 1 small-cell disclosure** (8 cells with 1–4, 7 cells with 5–9): MMRF data-use terms not available → **decision needed from user**; not resolved.
7. **Missing/unrecoverable inputs:** 47 legacy inputs absent and ~18 provenance-recorded inputs absent (`figures/LEGACY_MISSING_INPUTS.md`); 13 protected earlier S3 files + 8 extension outputs absent (the earlier "256-check" claim is unconfirmable); original Fig 3C script absent; origin of the restored S3 `.gz` unrecorded.
8. **Weak spots in results:** 3 seed-sensitive rows in Fig 3C; imprecise held-out strata (42–207 patients; binary 42–206); Italy excluded post hoc (disclosed in caption).
9. **Reproducibility from a fresh clone:** `requirements.txt` is the UTF-16 July version with no statsmodels/formulaic/xgboost/lightgbm pins, although `SUPERSEDED_ANALYSES.md` says statsmodels is pinned (discrepancy, cause unknown). The PC `.venv` has xgboost 3.4.1 and must stay unmodified. Result artifacts, `figures/data/*` and `data/` are not on the branch, so Stage D/H cannot be rerun and most plot scripts fail on a fresh clone [V by clone test].
10. **Dangling references:** docs cite 75 paths — 44 in the commit, 10 local-only (untracked/ignored), 21 absent everywhere. `ustuned` files are untracked but referenced.
11. Pathway protocol and frozen composite artifact have no `.sha256` sidecar.

---

## 5. Slide deck

**Files (PC repo root, untracked):** `Revised_MM Treatment Response Study.pptx` (17,058,326 bytes) and `Revised_MM Treatment Response Study.pdf` (19,508,666 bytes), both saved 2026-10-15 23:58. Canva export (PDF metadata keywords `DAHYF11vUWQ,BAGupyaeE4I`, creator Canva). Text, slide count (24) and images are identical between PPTX and PDF [V]. Canva design is presumably `DAHYF11vUWQ` [A].

**Latest audit [V]: READY AFTER MINOR FIXES.**
- All 11 requested replacements are present word-for-word in both files; no stale phrase remains; slide 12 no longer overflows (text ends 26 pt above the bottom).
- All 21 required images on all slides are the current train-only/origtrain variants; no clipped figure content.
- Revised claims were checked against the repo data (Holm p 0.94153 both tasks; 0 of 48 strata significant after adjustment; Table S2 subset values; S3 shares; geographic counts).

**Replacements applied (exact new text)**

| Slide / figure | New text |
|---|---|
| 3 and 4 — Figure 1 title | `Figure 1 - Study Overview and Analysis Framework` |
| 11 — Fig 4B takeaway | `Key takeaway: No evidence that the RNA effect differs by clinical prediction uncertainty (interaction Holm p = 0.94 in both tasks); quartile-level estimates are imprecise.` |
| 12 — Fig 4C takeaway | `Key takeaway: No treatment or response-trajectory subgroup showed a significant RNA effect after multiplicity adjustment; subgroups are small and intervals wide.` |
| 14 — Fig 5A text | `In post hoc held-out survival analyses, RNA addition was associated with small increases in PFS and OS C-index, but both 95% confidence intervals included zero; the prespecified PFS criterion was not met.` |
| 18 — bullet 1 | `Current clinical state, notably current IMWG response, carries most of the model attribution; adding longitudinal history or treatment context gave small, inconsistent changes in performance.` |
| 18 — bullet 3 | `Train-only fitted survival models showed small, post hoc increases in held-out C-index after RNA addition, but confidence intervals included zero and the prespecified PFS criterion was not met. Independent validation is required.` |
| 20 — S1B bullet 2 | `Overall calibration is broadly similar; differences in the RNA-available subset (Supplementary Table S2) are small, unadjusted and descriptive.` |
| 22 — S2B bullet 1 | `Summarize performance in Spain and Canada for a model developed only on US original-training-partition patients; the US calibration set is a descriptive reference.` |
| 22 — S2B bullet 2 | `Country-stratified results are post hoc robustness checks. Tuning, fitting, calibration and threshold used only US original-training-partition patients; no Spain, Canada, validation or test patient entered development.` |
| 22 — takeaway | **Unchanged** (correct): `KEY TAKEAWAY \| This is internal geographic robustness, not independent external validation.` |
| 23 — S3A bullet 2 | `RNA pathway attribution is relatively small: 3.3% of binary and 7.3% of six-class net SHAP attribution.` |

Superseded text that was removed: "varied across clinical prediction-uncertainty groups"; "several subgroups showed worse six-class RPS"; "historical hyperparameter tuning overlaps some evaluated patients"; "US development/validation"; "remain central"; "Figure 1-" spacing.

**Remaining layout fixes**
- **Slide 14 (needed):** text box starts at x = 0 (touches left edge) and ends 5 pt above the bottom; in the PPTX the box extends to y = 827 on an 810-pt slide, so PowerPoint may clip it. Move right ~20 pt and shorten/shrink. Suggested shorter wording (proposal, not applied, wording to be re-checked): `Post hoc held-out analysis: RNA addition was associated with small C-index increases (PFS +0.028, OS +0.010); both 95% CIs included zero and the prespecified PFS criterion was not met.`
- **Slide 11 (minor):** PPTX text box extends to y = 819 (slide bottom 810); text itself ends 15 pt above bottom in the PDF. Shrink the box.
- **Slide 22 (acceptable):** text ends 10 pt above bottom; tight.
- **Optional, slide 20:** add `Raw probabilities are shown; no recalibration was applied.` (discloses the S1 caption note about the earlier validation-patient comparison).
- **Optional, slide 12:** append `Two exploratory six-class interaction contrasts had BH p < 0.05 and are descriptive.` — mind the overflow that slide 12 previously had.
- **Not verified [U]:** how the PPTX renders in PowerPoint (checked only through the PDF and PPTX geometry); projected legibility of small embedded footnotes (slides 12, 15, 24).
- **Canva leftovers (PC-side Canva account) [V]:** copy `DAHYA0q1PI4` with an unsaved transaction `9006592345136747463` on slide 3; one uploaded image `MAHYBKPkC5k`. The user ordered all Canva editing stopped; deck edits have since been applied by the user.

---

## 6. Figure assets and crops

All 21 deck image files are tracked, on `origin/new-revisions`, not gitignored, and identical in a real fresh clone (10.2 MB total) [V]. Paths are those in section 3. Embedded deck images match repo files by visual signature (Canva re-encodes, so not byte-identical) [V].

Crops used in the deck (full-figure images clipped on the slide):
- Figure 1 `fig1_new/figure1_new.png` (4320×6048): slide 3 = panels A–C (≈ rows 67–2836); slide 4 = panels D–E (from ≈ row 2768).
- Figure 2 `Fig1/figure1.png` (4320×5670): slide 5 = panel A (rows 0–≈2970); slide 6 = panel B (≈ 2982–4377); slide 7 = panel C (from ≈ 4323 to end). The tracked crop script does not exist; my crop script was in a scratch folder on the PC only.
- Figures 3A and 3B share slide 8 (two images side by side); Figure 3C is slide 9.

Deck order: 1 title, 2 objectives, 3–4 Fig 1, 5–7 Fig 2, 8–9 Fig 3, 10–13 Fig 4A–D, 14–15 Fig 5A–B, 16–17 Fig 6A–B, 18 conclusions, 19–20 S1A–B, 21–22 S2A–B, 23–24 S3A–B.

---

## 7. Prioritized next steps

**Required before sending the deck**
1. Fix slide 14 layout (move/shorten text). Optionally shrink slide 11's box.
2. Open the PPTX once in PowerPoint and flip through slides 11–14 and 22.
3. Decide whether to tell Professor Song plainly that the redesigned analyses are train-only but still post hoc (not independent validation).

**Strongly recommended**
4. Add the slide 20 raw-probability line (disclosure of the validation-informed choice).
5. Be ready to answer how the 24 main models were tuned (unverified here) — locate tuning code/logs.

**Optional**
6. Slide 12 interaction-contrast sentence; slide 23 mention that summed-|SHAP| gives 6.7% / 12.9% for RNA pathways (Table S3).
7. Decide whether to trim the overlap between Fig 1B–D and Fig 2A–B (manuscript decision).
8. Table 1 small-cell policy (needs MMRF data-use terms).
9. Pin `requirements.txt` (statsmodels, xgboost, lightgbm …); decide whether to push the figure data/results (without patient-level files) so a fresh clone can replot or rerun.
10. Record an XGBoost rationale or keep the "no a priori rationale" wording.
11. Delete or finish the Canva leftovers.

**Git (only with the user's explicit permission):** decide what to commit (e.g., `MYELoma_PROJECT_HANDOFF.md`, deck files, `ustuned` files). Never `git add .` / `git add artifacts` — see section 8.

---

## 8. Laptop continuation

**Get the repo and branch**
```
git clone --branch new-revisions https://github.com/Tushnamaitty/mm-response-prediction.git
cd mm-response-prediction
git log --oneline -1          # expect: ae36c4d Update figures and train-only analyses
git status --short            # expect clean
```
(Equivalent: `git clone <url>` then `git checkout new-revisions`.) Do not switch or modify other branches.

**Add the local-only exclude rule right away** (it lives in `.git/info/exclude`, which is not cloned):
```
echo "/artifacts/results_clean_rerun/survival_trainonly/frozen_models.json" >> .git/info/exclude
```

**Available after cloning [V]:** all 21 figure images, plot scripts, protocols, docs, caption/numbering/tables, check scripts, `modeling/us_to_spain_canada_model_c_origtrain.py`, `figures/data/Fig1` and `Fig2`. `figure1_new_plot.py` and `figure2_plot.py` ran in a clone; S3A and Fig 4D plot scripts failed with FileNotFoundError.

**Exist only on the PC — copy over manually if needed**

| What | Path (PC) | Note |
|---|---|---|
| Slide deck + PDF | repo root: `Revised_MM Treatment Response Study.pptx`, `.pdf` | Untracked. Copy via cloud/USB, or re-export from Canva (design `DAHYF11vUWQ` [A]). |
| Train-only results/manifests | `artifacts/results_clean_rerun/{direction2_replication_trainonly,geographic_origtrain,geographic_us_tuned,geographic_us_tuned_control_previous_params,rna_contribution_trainonly,survival_trainonly}/` | Untracked but **not ignored**; contain patient-level prediction files with patient IDs. Never commit wholesale. |
| Figure source data | `figures/data/*` (ignored), incl. `figures/data/FigS_shap/figS3_source_shap_binary_test.csv.gz` | The `.gz` is patient-level: keep local only. |
| Raw/derived patient data | `data/` (ignored) | Patient-level; do not copy to unsecured locations. |
| Python env | `.venv` (xgboost 3.4.1) | Leave unmodified; `requirements.txt` is incomplete. To capture the env, run once on the PC: `.venv\Scripts\python.exe -m pip freeze` and save the output **outside** the repo. |
| Local-only/misc | `opus_plan.md`, `repo_structure.txt`, `figures/publication_redesign.zip` (not inspected), `modeling/us_to_spain_canada_model_c_ustuned.py`, `figures/supplementary_tables/build_table_s4_ustuned.py`, `figures/supplementary_tables/Table_S4_geographic_robustness_ustuned.md` | Untracked. The `ustuned` files are referenced by docs but absent from the branch. |
| This handoff | `MYELoma_PROJECT_HANDOFF.md` | Untracked; copy it over or paste its text. |
| Claude memory | `C:\Users\Jash\.claude\projects\C--mm-response-prediction\memory\` | PC-only; the constraints below replace it. |

---

## Appendix: standing constraints (tell any new Claude session)
Never stage, commit or push without explicit permission; never switch or modify other branches (read others only via `git show`/`git ls-tree`); do not retrain models or alter existing results; protect patient privacy (no patient-level rows, IDs or dates in committed files; do not widen `.gitignore` for patient-level files); never call any algorithm "best"; always distinguish binary from six-class and validated from exploratory; do not claim RNA has no biological value; do not call post hoc analyses "validated" or "independent"; use only `artifacts/results_clean_rerun/`; frozen protocols and hash-chained artifacts must not be edited; do not fabricate legacy data; no broad filesystem searches (an 18-minute search earlier annoyed the user); no Canva/deck editing unless the user says so; do not draft Methods/Results unless asked. Environment quirks: Windows `core.autocrlf=true`; Bash tool halves doubled backslashes (use the Write tool); project `.venv` must stay unmodified.
