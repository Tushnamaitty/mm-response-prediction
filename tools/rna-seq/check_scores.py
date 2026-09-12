import pandas as pd

scores = pd.read_csv(
    "data/rna/pathway_scores_hallmark.csv",
    index_col=0
)
print(scores.shape)
print(scores.describe().T)

print("Missing values:", scores.isna().sum().sum())