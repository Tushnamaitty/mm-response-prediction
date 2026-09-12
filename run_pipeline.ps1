# Runs the full clinical feature pipeline, in order.
# Usage: from the project root, run:  .\run_pipeline.ps1

$scripts = @(
    "build_timelines.py",
    "build_visit_pairs.py",
    "build_clinical_features.py",
    "build_treatment_features.py",
    "build_diagnosis_features.py",
    "build_signs_symptoms_features.py",
    "build_bone_assessment_features.py",
    "build_drug_exposure_features.py",
    "build_adverse_event_features.py",
    "build_supportive_care_features.py",
    "build_qol_features.py",
    "build_comorbidity_features.py",
    "build_flow_cytometry_features.py",
    "build_family_history_features.py",
    "build_cmmc_features.py"
)

foreach ($script in $scripts) {
    Write-Host "`n=== Running $script ===" -ForegroundColor Cyan
    python "data_pipeline/$script"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nFAILED at $script - stopping here." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n All 15 scripts completed successfully " -ForegroundColor Green