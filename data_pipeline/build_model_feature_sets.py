"""
Step 10: Define and validate Model A/B/C/D feature sets from the final
merged clinical+RNA table.

Input:  data/clinical/visit_pairs_with_rna.csv  (202 columns)
        data/clinical/visit_pairs_master.csv    (152 columns, frozen -
            used only to programmatically identify the RNA columns as a
            set difference, not to modify anything)
Output: data_pipeline/model_feature_sets.json    (column lists per model,
            plus identifiers/targets/excluded, for reuse in Step 13+)

This script does NOT modify the frozen clinical or RNA pipelines, does
NOT split data, does NOT preprocess, and does NOT train anything. It only
defines and validates which columns belong to which model stage, and
saves that definition to a reusable file.

Every column is assigned to EXACTLY ONE of 7 mutually exclusive buckets:
  identifiers, targets, excluded_leakage, model_a, model_b_only,
  model_c_only, model_d_only
Models are then assembled as unions: A, B=A+temporal, C=B+treatment,
D=C+RNA - enforcing A subset B subset C subset D by construction.

EXCLUDED AS LEAKAGE-PRONE (flagged explicitly, not silently dropped):
  vt1_study_visit    - Vt+1's visit label/identity, unknowable at Vt
  vt1_days_to_visit  - Vt+1's timing, unknowable at Vt
  time_gap_days      - derived from vt1_days_to_visit; the realized gap
                       to the next visit isn't known in advance at
                       prediction time, and a shorter gap could itself be
                       a symptom of clinical concern (e.g. an early
                       check-in scheduled due to suspected relapse) -
                       a subtle proxy-leakage risk, not a legitimate
                       temporal feature. Excluded by default; revisit if
                       this project's clinical framing later justifies a
                       different call.
  vt1_disease_response - functionally a duplicate of the exact_next_response
                       target (same content, pre-cleaning), so treated as
                       a target-adjacent field, never a predictor.
"""

import json
import pandas as pd

CLINICAL_DIR = "data/clinical"
OUTPUT_PATH = "data_pipeline/model_feature_sets.json"

IDENTIFIERS = ["pair_id", "public_id", "vt_study_visit"]

TARGETS = ["improved", "exact_next_response", "vt1_disease_response"]

EXCLUDED_LEAKAGE = ["vt1_study_visit", "vt1_days_to_visit", "time_gap_days"]

MODEL_A = [
    # current disease state at Vt
    "vt_disease_response",
    "vt_ecog_ps_numeric",
    "vt_reported_signs_symptoms",
    "vt_reported_adverse_event",
    "vt_reported_clinical_event",
    # current lab values
    "m_protein_current", "kappa_flc_current", "lambda_flc_current",
    "calcium_current", "creatinine_current", "hemoglobin_current",
    "b2m_current", "albumin_current", "ldh_current",
    # diagnosis-time static variables
    "iss_stage", "r_iss_stage", "age_at_diagnosis",
    "height_at_diagnosis_m", "weight_at_diagnosis_kg", "bmi_at_diagnosis",
    # current CRAB symptoms
    "subject_had_signs_symptoms", "bone_pain", "hypercalcemia",
    "renal_insufficiency", "anemia", "lytic_bone_lesion",
    "spinal_cord_compression", "recurrent_bacteria_infection",
    "amyloidosis", "soft_tissue_plasmacytoma_single",
    "soft_tissue_plasmacytoma_multi",
    # bone disease
    "bone_lytic_lesion_present", "n_lytic_lesions_ordinal",
    "pathologic_fracture",
    # current adverse-event state
    "has_peripheral_neuropathy", "peripheral_neuropathy_max_grade",
    "had_other_ae",
    # current supportive-care state
    "had_transfusion", "had_dialysis", "had_bisphosphonate",
    "had_opioid", "had_radiation_therapy", "had_wbc_growth_factor",
    "had_esa", "had_tumor_lysis_meds", "had_anticoagulation",
    # current QoL
    "qol_global_health_status", "qol_fatigue", "qol_pain",
    "qol_physical_functioning", "qol_disease_symptoms",
    "qol_side_effects_treatment",
    # comorbidities (static)
    "has_hypertension_or_chf", "has_chronic_pulmonary_disease",
    "has_renal_comorbidity", "has_borderline_diabetes", "has_cva",
    "has_mi", "comorbidity_count_total",
    # current acute-care state at Vt
    "had_hospitalization", "had_ed_visit",
    # current flow-cytometry values
    "cd38_detected", "cd56_detected", "cd138_detected",
    "cd45_typical_detected", "cd38_pct", "cd56_pct",
    "pct_plasma_cells_bm", "pct_plasma_cells_pb", "ig_heavy_chain_type",
    "ig_light_chain_type", "has_braf_mutation", "dna_index",
    # family history (static)
    "has_family_history_of_cancer", "has_family_history_of_myeloma",
    "count_relatives_with_cancer",
    # current CMMC
    "cmmc_number", "cmmc_detected",
]

MODEL_B_ONLY = [
    "vt_days_to_visit",  # time since baseline - a trajectory feature
    "m_protein_previous", "m_protein_change", "m_protein_pct_change",
    "m_protein_days_since_previous",
    "kappa_flc_previous", "kappa_flc_change", "kappa_flc_pct_change",
    "kappa_flc_days_since_previous",
    "lambda_flc_previous", "lambda_flc_change", "lambda_flc_pct_change",
    "lambda_flc_days_since_previous",
    "calcium_previous", "calcium_change", "calcium_pct_change",
    "calcium_days_since_previous",
    "creatinine_previous", "creatinine_change", "creatinine_pct_change",
    "creatinine_days_since_previous",
    "hemoglobin_previous", "hemoglobin_change", "hemoglobin_pct_change",
    "hemoglobin_days_since_previous",
    "b2m_previous", "b2m_change", "b2m_pct_change",
    "b2m_days_since_previous",
    "albumin_previous", "albumin_change", "albumin_pct_change",
    "albumin_days_since_previous",
    "ldh_previous", "ldh_change", "ldh_pct_change",
    "ldh_days_since_previous",
]

MODEL_C_ONLY = [
    "current_line_number", "current_line_is_active", "line_duration_so_far",
    "n_prior_lines_completed",
    "current_regimen_name", "current_regimen_type",
    "current_regimen_categories", "regimen_duration_so_far",
    "n_prior_regimens", "prior_treatment_failure",
    "has_prior_transplant", "has_prior_allo_transplant",
    "n_prior_transplants", "days_since_last_transplant",
    "currently_on_pi", "prior_exposure_pi",
    "currently_on_imid", "prior_exposure_imid",
    "currently_on_cd38", "prior_exposure_cd38",
    "currently_on_slamf7", "prior_exposure_slamf7",
    "currently_on_bcma", "prior_exposure_bcma",
    "currently_on_steroid", "prior_exposure_steroid",
    "currently_on_chemo", "prior_exposure_chemo",
]


def get_rna_columns() -> list[str]:
    """Identify the 50 pathway-score columns PROGRAMMATICALLY as the set
    difference between the RNA-merged table and the frozen clinical
    table's columns - not hardcoded, so this stays correct even if the
    RNA pipeline's pathway set changes."""
    rna_df_cols = pd.read_csv(
        f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", nrows=0
    ).columns
    clinical_df_cols = pd.read_csv(
        f"{CLINICAL_DIR}/visit_pairs_master.csv", nrows=0
    ).columns
    rna_cols = [c for c in rna_df_cols if c not in set(clinical_df_cols)]
    return rna_cols


def validate_and_build(df_columns: list[str], rna_cols: list[str]) -> dict:
    buckets = {
        "identifiers": IDENTIFIERS,
        "targets": TARGETS,
        "excluded_leakage": EXCLUDED_LEAKAGE,
        "model_a": MODEL_A,
        "model_b_only": MODEL_B_ONLY,
        "model_c_only": MODEL_C_ONLY,
        "model_d_only": rna_cols,
    }

    # --- Check 1: no overlaps between mutually exclusive buckets ---
    seen = {}
    overlaps = []
    for bucket_name, cols in buckets.items():
        for c in cols:
            if c in seen:
                overlaps.append((c, seen[c], bucket_name))
            seen[c] = bucket_name
    if overlaps:
        raise ValueError(f"Column(s) assigned to multiple buckets: {overlaps}")

    # --- Check 2: every column in the actual file is accounted for ---
    all_categorized = set(seen.keys())
    all_actual = set(df_columns)
    missing = all_actual - all_categorized
    extra = all_categorized - all_actual
    if missing:
        raise ValueError(
            f"UNCATEGORIZED COLUMNS FOUND (present in file, not classified "
            f"anywhere): {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"Categorized columns that DON'T EXIST in the actual file "
            f"(stale/typo'd entries): {sorted(extra)}"
        )

    # --- Build the nested model column lists ---
    model_a_cols = list(MODEL_A)
    model_b_cols = model_a_cols + MODEL_B_ONLY
    model_c_cols = model_b_cols + MODEL_C_ONLY
    model_d_cols = model_c_cols + rna_cols

    # --- Check 3: nesting A subset B subset C subset D ---
    assert set(model_a_cols) <= set(model_b_cols), "A is not a subset of B"
    assert set(model_b_cols) <= set(model_c_cols), "B is not a subset of C"
    assert set(model_c_cols) <= set(model_d_cols), "C is not a subset of D"

    # --- Check 4: no target/identifier/excluded column leaked into any model ---
    forbidden = set(IDENTIFIERS) | set(TARGETS) | set(EXCLUDED_LEAKAGE)
    for name, cols in [
        ("Model A", model_a_cols), ("Model B", model_b_cols),
        ("Model C", model_c_cols), ("Model D", model_d_cols),
    ]:
        leaked = forbidden & set(cols)
        if leaked:
            raise ValueError(f"{name} contains forbidden column(s): {leaked}")

    # --- Check 5: RNA never appears before D ---
    for name, cols in [("Model A", model_a_cols), ("Model B", model_b_cols), ("Model C", model_c_cols)]:
        leaked_rna = set(rna_cols) & set(cols)
        if leaked_rna:
            raise ValueError(f"{name} contains RNA column(s) - should only appear in D: {leaked_rna}")

    return {
        "identifiers": IDENTIFIERS,
        "targets": TARGETS,
        "excluded_leakage": EXCLUDED_LEAKAGE,
        "model_a": model_a_cols,
        "model_b": model_b_cols,
        "model_c": model_c_cols,
        "model_d": model_d_cols,
    }


def print_report(result: dict, total_columns: int) -> None:
    print(" Step 10: Model A/B/C/D feature set validation \n")
    print(f"Identifiers/metadata:  {len(result['identifiers'])}")
    print(f"Targets:               {len(result['targets'])}")
    print(f"Excluded (leakage):    {len(result['excluded_leakage'])}")
    print()
    print(f"Model A (current clinical):        {len(result['model_a'])} predictor columns")
    print(f"Model B (A + temporal):             {len(result['model_b'])} predictor columns")
    print(f"Model C (B + treatment):            {len(result['model_c'])} predictor columns")
    print(f"Model D (C + RNA):                  {len(result['model_d'])} predictor columns")
    print()

    accounted = (
        len(result["identifiers"]) + len(result["targets"])
        + len(result["excluded_leakage"]) + len(result["model_d"])
    )
    print(f"Total accounted for: {accounted} / {total_columns} actual columns")
    if accounted == total_columns:
        print("ALL COLUMNS ACCOUNTED FOR - no uncategorized, no duplicates.")
    else:
        print("MISMATCH - see errors above.")


if __name__ == "__main__":
    df_cols = list(pd.read_csv(f"{CLINICAL_DIR}/visit_pairs_with_rna.csv", nrows=0).columns)
    rna_cols = get_rna_columns()

    print(f"Identified {len(rna_cols)} RNA pathway columns programmatically:")
    print(f"  {rna_cols[:3]} ... {rna_cols[-3:]}\n")

    result = validate_and_build(df_cols, rna_cols)
    print_report(result, len(df_cols))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")