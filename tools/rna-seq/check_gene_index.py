import pandas as pd

expr = pd.read_csv(
    "data/rna/expression_matrix_tpm.csv",
    index_col=0
)

print(expr.index[:10])