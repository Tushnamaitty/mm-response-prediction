import pandas as pd
import glob
import os

files = glob.glob("data_raw/rnaseq/**/*star_gene_counts.tsv", recursive=True)
print(f"Found {len(files)} files")

expr_dict = {}

for f in files:
    # case UUID is typically the parent folder name in GDC downloads
    case_id = os.path.basename(os.path.dirname(f))
    df = pd.read_csv(f, sep="\t", comment="#")
    # keep gene_id and a chosen expression column (tpm_unstranded is standard for pathway scoring)
    df = df[["gene_id", "tpm_unstranded"]].dropna()
    df = df[~df["gene_id"].str.startswith("N_")]  # drop QC rows like N_unmapped
    expr_dict[case_id] = df.set_index("gene_id")["tpm_unstranded"]

expr_matrix = pd.DataFrame(expr_dict)
print(expr_matrix.shape)
expr_matrix.to_csv("data_raw/rnaseq/expression_matrix_tpm.csv")