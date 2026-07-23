# MM Response Prediction

Explainable ML model predicting visit-to-visit treatment response (Improved / Not Improved) 
for multiple myeloma patients using the MMRF CoMMpass dataset.

## Repo structure

- **data_pipeline/** — scripts to pull and clean GDC + MMRF clinical, lab, and treatment data
- **transcriptomics/** — RNA-seq → pathway-score pipeline (ssGSEA/GSVA)
- **modeling/** — feature assembly, training, and evaluation (binary + secondary three-class model)
- **explainability/** — SHAP, Permutation Feature Importance, PDP, ALE analyses

## Decision log

All modeling/labeling/preprocessing decisions are logged here before implementation:
https://docs.google.com/document/d/1P3W-0vnKLaZoj_bbXeEh4LVYDO4odhWKN2DmchHonUE/edit?tab=t.0

## Branching convention

- `main` is always stable — no direct commits.
- Create a branch per task: `<folder>-<short-description>` (e.g. `modeling-baseline-logreg`)
- Open a Pull Request to merge into `main`; get at least one review before merging.

## Environment setup

See `requirements.txt`. To set up:
Python version : Python 3.13.5

\`\`\`
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`