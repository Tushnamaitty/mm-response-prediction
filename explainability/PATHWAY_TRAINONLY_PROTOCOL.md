# Protocol: train-only pathway selection and held-out evaluation of the proliferation/MYC axis

Fixed BEFORE the selection script is run (its SHA-256 is written into every output of the two scripts below).
Responds to Professor Song's feedback that validation and test patients must not take part in pathway selection or any other
model-development step.

## Why a redesign
The earlier analysis chose its five-pathway family as the five smallest p-values of the pooled S3 ordinal regression
(`S3_adjusted`, all 532 retained patients, including validation and test patients), z-scored the pathway scores over all 751
RNA samples, and then "replicated" the effect in the 84 retained test patients. Test patients therefore influenced pathway
selection and preprocessing. The earlier outputs are kept unchanged as a documented earlier exploratory analysis.

## Populations
* **Development set:** patients with `split == "train"` in `data/splits/visit_pair_splits.csv` (identical to the patient-level
  partition in `data/splits/patient_split_assignments.csv`; patient-disjoint, asserted).
* **Held-out set:** patients with `split` in {`val`, `test`}. Validation and test patients are pooled because neither may enter
  development; no other held-out definition is evaluated (no test-only, validation-only or other re-cut).
* Spain/Canada and the geographic analysis are unrelated and unchanged.

## Stage 1: pathway selection (development set only) -- `direction2_pathway_selection_trainonly.py`
1. Universe: the frozen ordinal universe (`select_universe(master, "ordinal")`, 9,172 rows, whole-cohort count checks still
   enforced), then restricted to development-set rows.
2. Pathway scores are z-scored with the mean and SD (ddof = 1) of the distinct (patient, RNA-sample) vectors **among
   development-set RNA-available rows only**.
3. Regression: the frozen D2 `regression_analysis` (`S3_adjusted`: ordinal GEE, cumulative logit, independence working
   correlation, patient-clustered robust SEs; 16 continuous Model-C covariates transformed / winsorised / z-scored on the final
   rows, log1p RNA age, exact strata, strata < 20 rows excluded, strata without outcome variation excluded, complete-case),
   run on development-set rows only, one pathway per model, BH q over the 50 pathways reported.
4. **Selection rule (unchanged from the earlier analysis):** the five pathways with the smallest robust Wald p-values. No
   direction, q-value, biological-coherence or other filter is added or removed. The earlier family equalled this rule applied
   to the pooled data (re-verified).
5. Stop conditions: the regression analysis is not `complete`, or fewer than five pathways have valid estimates. No
   alternative rule, model or pathway count is tried.
6. The selected family and the rule are written to a hashed `selection_artifact.json` before Stage 2 is run.

## Stage 2: composite, discovery and one held-out test (`direction2_proliferation_replication_trainonly.py`)
Identical to the earlier replication script except for the populations and the family:
* composite = PC1 of the correlation matrix of the five z-scored selected pathways (development-set means/SDs; loadings oriented
  to sum > 0; unit variance in discovery);
* discovery model = frozen S3 ordinal formulation on development-set rows only (composite + 16 covariates + log1p RNA age + exact
  strata), covariate transformation parameters fitted on the development rows;
* **acceptance criteria (unchanged):** all five loadings positive; all six outcome classes represented in discovery; design full
  rank; model identifiable and converged; composite coefficient negative with one-sided p < 0.025. If any fails the script stops
  **before any held-out outcome is read** and nothing is retried;
* the frozen discovery artifact is hashed before held-out rows are scored; held-out outcomes are blanked in the data used for
  Stage 1 and discovery and are attached only after the hash;
* single primary directional test in the held-out RNA-available patients: OrdinalGEE of next class on composite + clinical linear
  predictor eta (discovery coefficients, discovery transformation parameters); H1: beta < 0; one-sided robust Wald p; support
  checks (>= 80 held-out patients, all six classes, eta non-degenerate); validity flag (eta coefficient > 0); interpretation
  rule and the descriptive 2,000-draw patient-cluster bootstrap exactly as before.

## Reporting rules
* The earlier pooled results, the earlier family and the earlier test result were seen before this redesign. The new analysis is
  therefore **post hoc / exploratory**, not untouched confirmation and not external validation; "independent" and "validated"
  are not used.
* The old and new families are compared only after Stage 1 has been written, and the comparison does not feed back into selection.
* Held-out results are never used to change the family, the rule, the composite or the model.
