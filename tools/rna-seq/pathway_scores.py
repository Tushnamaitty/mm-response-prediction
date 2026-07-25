import pandas as pd
import gseapy as gp

expr = pd.read_csv("data_raw/rnaseq/expression_matrix_tpm_symbols.csv", index_col=0)

# ssGSEA expects a genes x samples matrix (which we have) with gene symbols as index
ss = gp.ssgsea(
    data=expr,
    gene_sets="MSigDB_Hallmark_2020",  # confirm exact name from Step 1 output
    outdir=None,           # don't write per-sample plots/files
    sample_norm_method='rank',
    min_size=5,
    max_size=1000,
    permutation_num=0,     # no permutation testing needed, just scores
    no_plot=True,
    threads=8
)

pathway_scores = ss.res2d.pivot(index='Name', columns='Term', values='ES')
print(pathway_scores.shape)
print(pathway_scores.head())

pathway_scores.to_csv("data_raw/rnaseq/pathway_scores_hallmark.csv")