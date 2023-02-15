import pandas as pd
import numpy as np
from scipy.stats import variation


#df = pd.read_csv("human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_samples-columns.csv",nrows=500)
# Load data
intron_matrix = pd.read_csv("human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_intron-matrix.csv",usecols=range(500))
exon_matrix = pd.read_csv("human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_exon-matrix.csv",usecols=range(500))
full_matrix = intron_matrix.add(exon_matrix, fill_value=0)

# Compute log2CPM values
full_matrix_log2cpm = np.log2(full_matrix.div(full_matrix.sum(axis=0), axis=1) * 1e6 + 1)
gene_counts = full_matrix_log2cpm.iloc[:, 1:].apply(lambda x: np.sum(x > 0), axis=1)
# Replace first column with gene IDs
full_matrix.iloc[:, 0] = intron_matrix.iloc[:, 0]
full_matrix_log2cpm.iloc[:, 0] = intron_matrix.iloc[:, 0]

# Load sample annotations
samples = pd.read_csv("human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_samples-columns.csv", index_col=0,nrows=500)
labels = samples["cluster"]
# Filter out "no class" cells
labels = labels[labels != "no class"]

# Extract cell types and corresponding sample names
cell_types = labels["cluster"].unique()
celltype_samples = {ct: labels.index[labels["cluster"] == ct].tolist() for ct in cell_types}

# Load gene annotations
gene_rows = pd.read_csv("human_MTG_2018-06-14_genes-rows.csv", index_col=0)

# Filter out genes with zero expression across all cells
gene_counts = full_matrix_log2cpm.iloc[:, 1:].apply(lambda x: np.sum(x > 0), axis=1)
keep_genes = gene_counts > 0
kept_genes = full_matrix_log2cpm.loc[keep_genes, :]

# Compute median expression for each cell type
cg_median = pd.DataFrame(index=kept_genes.index)
cg_counts = pd.DataFrame(index=kept_genes.index)
for ct in cell_types:
    celltype_subset = kept_genes[celltype_samples[ct]]
    cg_median[ct] = celltype_subset.median(axis=1)
    cg_counts[ct] = (celltype_subset > 0).sum(axis=1)

# Remove genes with zero median expression variance
cg_median_var = cg_median.apply(lambda x: variation(x), axis=1)
keep_var_genes = cg_median_var != 0
cg_median_var = cg_median_var[keep_var_genes]
cg_median_var_genes = cg_median.loc[keep_var_genes, :]

# Subset full matrix to keep only genes with non-zero variance
keep_genes = cg_median_var_genes.index
final_matrix = full_matrix_log2cpm.loc[keep_genes, :]

# Write final matrix to file
final_matrix.to_csv("everything.csv", index=True, header=True)