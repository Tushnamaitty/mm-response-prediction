import pandas as pd
from pybiomart import Dataset

# Load expression matrix
expr = pd.read_csv("data/rna/expression_matrix_tpm.csv", index_col=0)

# Strip version suffix (ENSG00000000003.15 -> ENSG00000000003)
expr.index = expr.index.str.split(".").str[0]
expr = expr[~expr.index.duplicated(keep="first")]

# Query biomart for Ensembl -> gene symbol mapping
dataset = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
mapping = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
mapping = mapping.dropna().drop_duplicates(subset="Gene stable ID")
id_to_symbol = dict(zip(mapping["Gene stable ID"], mapping["Gene name"]))

# Map and drop unmapped genes
expr["gene_symbol"] = expr.index.map(id_to_symbol)
expr = expr.dropna(subset=["gene_symbol"])
expr = expr.set_index("gene_symbol")
expr = expr[~expr.index.duplicated(keep="first")]

print(expr.shape)
print(expr.index[:10])
expr.to_csv("data/rna/expression_matrix_tpm_symbols.csv")