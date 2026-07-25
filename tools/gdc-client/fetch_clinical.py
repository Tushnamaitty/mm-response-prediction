import requests
import pandas as pd
import json

cases_endpt = "https://api.gdc.cancer.gov/cases"

filters = {
    "op": "in",
    "content": {
        "field": "project.project_id",
        "value": ["MMRF-COMMPASS"]
    }
}

fields = [
    "case_id",
    "submitter_id",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.primary_diagnosis",
    "diagnoses.iss_stage",
    "diagnoses.days_to_diagnosis",
    "diagnoses.morphology",
    "diagnoses.tumor_grade",
    # treatment history
    "diagnoses.treatments.treatment_type",
    "diagnoses.treatments.treatment_outcome",
    "diagnoses.treatments.days_to_treatment_start",
    "diagnoses.treatments.days_to_treatment_end",
    "diagnoses.treatments.therapeutic_agents",
    "diagnoses.treatments.regimen_or_line_of_therapy",
    "diagnoses.treatments.number_of_cycles",
    # follow-up / per-visit labs and response
    "follow_ups.days_to_follow_up",
    "follow_ups.disease_response",
    "follow_ups.progression_or_recurrence",
    "follow_ups.days_to_progression",
    "follow_ups.molecular_tests.test_analyte_type",
    "follow_ups.molecular_tests.test_result",
    "follow_ups.labs.test_result",
    "exposures.tobacco_smoking_status",
]

params = {
    "filters": json.dumps(filters),
    "fields": ",".join(fields),
    "format": "TSV",
    "size": "2000"
}

response = requests.get(cases_endpt, params=params)

with open("clinical_data_full.tsv", "wb") as f:
    f.write(response.content)

df = pd.read_csv("clinical_data_full.tsv", sep="\t")
print(df.shape)
print(list(df.columns))