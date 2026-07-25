import pandas as pd
df = pd.read_csv("clinical_data_full.tsv", sep="\t", low_memory=False)
print(df['follow_ups.0.disease_response'].notna().sum())
print(df['demographic.vital_status'].notna().sum())